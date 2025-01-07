# This file is part of multiprofit.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "InvalidProposalError",
    "fit_methods_linear",
    "LinearGaussians",
    "make_image_gaussians",
    "make_psf_model_null",
    "FitInputsBase",
    "FitInputsDummy",
    "ModelFitConfig",
    "FitResult",
    "Modeller",
]

from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging
import sys
import time
from typing import Any, ClassVar, Iterable, TypeAlias

import lsst.gauss2d as g2
import lsst.gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import numpy as np
import pydantic
import scipy.optimize as spopt

from .model_utils import make_image_gaussians, make_psf_model_null
from .utils import arbitrary_allowed_config, frozen_arbitrary_allowed_config, get_params_uniq

_has_py_11_plus = sys.version_info >= (3, 11, 0)
if _has_py_11_plus:
    from typing import Self
else:
    from typing import TypeVar

    Self = TypeVar("Self", bound="LinearGaussians")  # type: ignore

try:
    # TODO: try importlib.util.find_spec
    from fastnnls import fnnls  # noqa

    has_fastnnls = True
except ImportError:
    has_fastnnls = False

try:
    # TODO: try importlib.util.find_spec
    import pygmo as pg  # noqa

    has_pygmo = True
except ImportError:
    has_pygmo = False


Model: TypeAlias = g2f.ModelD | g2f.ModelF


class InvalidProposalError(ValueError):
    """Error for an invalid parameter proposal."""


fit_methods_linear = {
    "scipy.optimize.nnls": {},
    "scipy.optimize.lsq_linear": {"bounds": (1e-5, np.inf), "method": "bvls"},
    "numpy.linalg.lstsq": {"rcond": 1e-3},
}
if has_fastnnls:
    fit_methods_linear["fastnnls.fnnls"] = {}


class LinearGaussians(pydantic.BaseModel):
    """Helper for linear least-squares fitting of Gaussian mixtures."""

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    gaussians_fixed: g2.Gaussians = pydantic.Field(title="Fixed Gaussian components")
    gaussians_free: tuple[tuple[g2.Gaussians, g2f.ParameterD], ...] = pydantic.Field(
        title="Free Gaussian components"
    )

    @staticmethod
    def make(
        component_mixture: g2f.ComponentMixture,
        channel: g2f.Channel = None,
        is_psf: bool = False,
    ) -> Self:
        """Make a LinearGaussians from a ComponentMixture.

        Parameters
        ----------
        component_mixture
            A component mixture to initialize Gaussians from.
        channel
            The channel all Gaussians are applicable for.
        is_psf
            Whether the components are a smoothing kernel.

        Returns
        -------
        lineargaussians
            A LinearGaussians instance initialized with the appropriate
            fixed/free gaussians.
        """
        if channel is None:
            channel = g2f.Channel.NONE
        components = component_mixture.components
        if len(components) == 0:
            raise ValueError(f"Can't get linear Source from {component_mixture=} with no components")

        gaussians_free = []
        gaussians_fixed = []

        for component in components:
            gaussians: g2.Gaussians = component.gaussians(channel)
            # TODO: Support multi-Gaussian components if sensible
            # The challenge would be in mapping linear param values back onto
            # non-linear IntegralModels
            if is_psf:
                n_g = len(gaussians)
                if n_g != 1:
                    raise ValueError(f"{component=} has {gaussians=} of len {n_g=}!=1")
            param_fluxes = component.parameters(paramfilter=g2f.ParamFilter(nonlinear=False, channel=channel))
            if len(param_fluxes) != 1:
                raise ValueError(f"Can't make linear source from {component=} with {len(param_fluxes)=}")
            param_flux: g2f.ParameterD = param_fluxes[0]
            if param_flux.fixed:
                gaussians_fixed.append(gaussians.at(0))
            else:
                gaussians_free.append((gaussians, param_flux))

        return LinearGaussians(
            gaussians_fixed=g2.Gaussians(gaussians_fixed), gaussians_free=tuple(gaussians_free)
        )


class FitInputsBase(ABC):
    """Interface for inputs to a model fit."""

    @abstractmethod
    def validate_for_model(self, model: Model) -> list[str]:
        """Check that this FitInputs is valid for a Model.

        Parameters
        ----------
        model
            The model to validate with.

        Returns
        -------
        errors
            A list of validation errors, if any.
        """


class FitInputsDummy(FitInputsBase):
    """A dummy FitInputs that always fails to validate.

    This class can be used to initialize a FitInputsBase that may be
    reassigned to a non-dummy derived instance in a loop.
    """

    def validate_for_model(self, model: Model) -> list[str]:
        return [
            "This is a dummy FitInputs and will never validate",
        ]


if not _has_py_11_plus:
    Self = TypeVar("Self", bound="FitInputs")  # type: ignore


class FitInputs(FitInputsBase, pydantic.BaseModel):
    """Model fit inputs for gauss2dfit."""

    model_config: ClassVar[pydantic.ConfigDict] = arbitrary_allowed_config

    jacobian: np.ndarray = pydantic.Field(None, title="The full Jacobian array")
    jacobians: list[list[g2.ImageD]] = pydantic.Field(
        title="Jacobian arrays (views) for each observation",
    )
    outputs_prior: tuple[g2.ImageD, ...] = pydantic.Field(
        title="Jacobian arrays (views) for each free parameter's prior",
    )
    residual: np.ndarray = pydantic.Field(title="The full residual (chi) array")
    residuals: list[g2.ImageD] = pydantic.Field(
        default_factory=list,
        title="Residual (chi) arrays (views) for each observation",
    )
    residuals_prior: g2.ImageD = pydantic.Field(
        title="Shared residual array for all Prior instances",
    )

    @classmethod
    def get_sizes(
        cls,
        model: Model,
    ) -> tuple[int, int, int, np.ndarray]:
        """Initialize Jacobian and residual arrays for a model.

        Parameters
        ----------
        model : `lsst.gauss2d.fit.Model`
            The model to initialize arrays for.

        Returns
        -------
        n_obs
            The number of observations initialized.
        n_params_jac
            The number of Jacobian matrix columns, which is the number of free
            parameters plus one validation column.
        n_prior_residuals
            The number of residual array values required for priors.
        shapes
            An ndarray containing the number of rows and columns for each
            observation in rows.
        """
        priors = model.priors
        n_prior_residuals = sum(len(p) for p in priors)
        params_free = tuple(get_params_uniq(model, fixed=False))
        n_params_free = len(params_free)
        # gauss2d_fit reserves the zeroth index of the jacobian array for
        # validation, i.e. it can be used to dump terms for fixed params
        n_params_jac = n_params_free + 1
        if not (n_params_jac > 1):
            raise ValueError("Can't fit model with no free parameters")

        n_obs = len(model.data)
        shapes = np.zeros((n_obs, 2), dtype=int)
        ranges_params = [None] * n_obs

        for idx_obs in range(n_obs):
            observation = model.data[idx_obs]
            shapes[idx_obs, :] = (observation.image.n_rows, observation.image.n_cols)
            # Get the free parameter indices for each observation
            params = tuple(get_params_uniq(model, fixed=False, channel=observation.channel))
            n_params_obs = len(params)
            ranges_params_obs = [0] * (n_params_obs + 1)
            for idx_param in range(n_params_obs):
                ranges_params_obs[idx_param + 1] = params_free.index(params[idx_param]) + 1
            ranges_params[idx_obs] = ranges_params_obs

        n_free_first = len(ranges_params[0])
        # Ensure that there are the same number of free parameters in each obs
        # They don't need to be the same set, but the counts should equal
        # (this assumption may be violated by future IntegralModels - TBD)
        assert all([len(rp) == n_free_first for rp in ranges_params[1:]])

        return n_obs, n_params_jac, n_prior_residuals, shapes

    @classmethod
    def from_model(
        cls,
        model: Model,
    ) -> Self:
        """Initialize Jacobian and residual arrays for a model.

        Parameters
        ----------
        model : `gauss2d.fit.Model`
            The model to initialize arrays for.
        """
        n_obs, n_params_jac, n_prior_residuals, shapes = cls.get_sizes(model)
        n_pixels_cumsum = np.cumsum(np.prod(shapes, axis=1))
        n_pixels_total = n_pixels_cumsum[-1]
        size_data = n_pixels_total + n_prior_residuals
        shape_jacobian = (size_data, n_params_jac)
        jacobian = np.zeros(shape_jacobian)
        jacobians = [None] * n_obs
        outputs_prior = [None] * n_params_jac
        for idx in range(n_params_jac):
            outputs_prior[idx] = g2.ImageD(jacobian[n_pixels_total:, idx].reshape((1, n_prior_residuals)))

        residual = np.zeros(size_data)
        residuals = [None] * n_obs
        residuals_prior = g2.ImageD(residual[n_pixels_total:].reshape(1, n_prior_residuals))

        offset = 0
        for idx_obs in range(n_obs):
            shape = shapes[idx_obs, :]
            size_obs = shape[0] * shape[1]
            end = offset + size_obs
            jacobians_obs = [None] * n_params_jac
            for idx_jac in range(n_params_jac):
                jacobians_obs[idx_jac] = g2.ImageD(jacobian[offset:end, idx_jac].reshape(shape))
            jacobians[idx_obs] = jacobians_obs
            residuals[idx_obs] = g2.ImageD(residual[offset:end].reshape(shape))
            offset = end
            if offset != n_pixels_cumsum[idx_obs]:
                raise RuntimeError(f"Assigned {offset=} data points != {n_pixels_cumsum[idx_obs]=}")
        return cls(
            jacobian=jacobian,
            jacobians=jacobians,
            residual=residual,
            residuals=residuals,
            outputs_prior=tuple(outputs_prior),
            residuals_prior=residuals_prior,
        )

    def validate_for_model(self, model: Model) -> list[str]:
        n_obs, n_params_jac, n_prior_residuals, shapes = self.get_sizes(model)
        n_pixels_total = np.sum(np.prod(shapes, axis=1))
        size_data = n_pixels_total + n_prior_residuals
        shape_jacobian = (size_data, n_params_jac)

        errors = []

        if self.jacobian.shape != shape_jacobian:
            errors.append(f"{self.jacobian.shape=} != {shape_jacobian=}")

        if len(self.jacobians) != n_obs:
            errors.append(f"{len(self.jacobians)=} != {n_obs=}")

        if len(self.residuals) != n_obs:
            errors.append(f"{len(self.residuals)=} != {n_obs=}")

        if not errors:
            for idx_obs in range(n_obs):
                shape_obs = shapes[idx_obs, :]
                jacobian_obs = self.jacobians[idx_obs]
                n_jacobian_obs = len(jacobian_obs)
                if n_jacobian_obs != n_params_jac:
                    errors.append(f"len(self.jacobians[{idx_obs}])={n_jacobian_obs} != {n_params_jac=}")
                else:
                    for idx_jac in range(n_jacobian_obs):
                        if not all(jacobian_obs[idx_jac].shape == shape_obs):
                            errors.append(f"{jacobian_obs[idx_jac].shape=} != {shape_obs=}")
                if not all(self.residuals[idx_obs].shape == shape_obs):
                    errors.append(f"{self.residuals[idx_obs].shape=} != {shape_obs=}")

        shape_residual_prior = [1, n_prior_residuals]
        if len(self.outputs_prior) != n_params_jac:
            errors.append(f"{len(self.outputs_prior)=} != {n_params_jac=}")
        elif n_prior_residuals > 0:
            for idx in range(n_params_jac):
                if self.outputs_prior[idx].shape != shape_residual_prior:
                    errors.append(f"{self.outputs_prior[idx].shape=} != {shape_residual_prior=}")

        if n_prior_residuals > 0:
            if self.residuals_prior.shape != shape_residual_prior:
                errors.append(f"{self.residuals_prior.shape=} != {shape_residual_prior=}")

        return errors


class ModelFitConfig(pexConfig.Config):
    """Configuration for model fitting."""

    eval_residual = pexConfig.Field[bool](
        doc="Whether to evaluate the residual every iteration before the Jacobian, which can improve "
        "performance if most steps do not call the Jacobian function. Must be set to True if the "
        "optimizer does not always evaluate the residual first, before the Jacobian.",
        default=True,
    )
    fit_linear_iter = pexConfig.Field[int](
        doc="The number of iterations to wait before performing a linear fit during optimization."
        " Default 0 disables the feature.",
        default=0,
    )
    optimization_library = pexConfig.ChoiceField[str](
        doc="The optimization library to use when fitting",
        allowed={
            "pygmo": "Pygmo2",
            "scipy": "scipy.optimize",
        },
        default="scipy",
    )

    def validate(self) -> None:
        if not self.fit_linear_iter >= 0:
            raise ValueError(f"{self.fit_linear_iter=} must be >=0")


class FitResult(pydantic.BaseModel):
    """Results from a Modeller fit, including metadata."""

    model_config: ClassVar[pydantic.ConfigDict] = arbitrary_allowed_config

    chisq_best: float = pydantic.Field(default=0, title="The chi-squared (sum) of the best-fit parameters")
    # TODO: Why does setting default=ModelFitConfig() cause a circular import?
    config: ModelFitConfig = pydantic.Field(None, title="The configuration for fitting")
    inputs: FitInputs | None = pydantic.Field(None, title="The fit input arrays")
    result: Any | None = pydantic.Field(
        None,
        title="The result object of the fit, directly from the optimizer",
    )
    params: tuple[g2f.ParameterD, ...] | None = pydantic.Field(
        None,
        title="The parameter instances corresponding to params_best",
    )
    params_best: tuple[float, ...] | None = pydantic.Field(
        None,
        title="The best-fit parameter array (un-transformed)",
    )
    params_free_missing: tuple[g2f.ParameterD, ...] | None = pydantic.Field(
        None,
        title="Free parameters that were fixed during fitting - usually an"
              " IntegralParameterD for a band with missing data",
    )
    n_eval_resid: int = pydantic.Field(0, title="Total number of self-reported residual function evaluations")
    n_eval_func: int = pydantic.Field(
        0, title="Total number of optimizer-reported fitness function evaluations"
    )
    n_eval_jac: int = pydantic.Field(
        0, title="Total number of optimizer-reported Jacobian function evaluations"
    )
    time_eval: float = pydantic.Field(0, title="Total runtime spent in model/Jacobian evaluation")
    time_run: float = pydantic.Field(0, title="Total runtime spent in fitting, excluding initial setup")


def set_params(params: Iterable[g2f.ParameterD], params_new: Iterable[float], model_loglike: Model):
    """Set new parameter values from an optimizer proposal.

    Parameters
    ----------
    params
        An iterable of ParameterD instances.
    params_new
        An iterable of new untransformed values for params.
    model_loglike
        A model instance configured to compute the log-likelihood.

    Raises
    ------
    InvalidProposalError
        Raised if a new value is nan, or if a RuntimeError is raised when
        setting the new value.
    RuntimeError
        Raised if the new transformed value is not finite.
    """
    try:
        for param, value in zip(params, params_new, strict=True):
            if np.isnan(value):
                raise InvalidProposalError(
                    f"optimizer for {model_loglike=} proposed non-finite {value=} for {param=}"
                )
            param.value_transformed = value
            if not np.isfinite(param.value):
                raise RuntimeError(f"{param=} set to (transformed) non-finite {value=}")
    except RuntimeError as e:
        raise InvalidProposalError(f"optimizer for {model_loglike=} proposal generated error={e}")


def residual_scipy(
    params_new: np.ndarray,
    model_jacobian: Model,
    model_loglike: Model,
    params: tuple[g2f.ParameterD],
    result: FitResult,
    jacobian: np.ndarray | None,
    never_evaluate_jacobian: bool = False,
    return_loglike: bool = False,
) -> np.ndarray:
    """Compute the residual for a scipy optimizer.

    Parameters
    ----------
    params_new
        An array of new parameter values.
    model_jacobian
        A model instance configured to compute the Jacobian.
    model_loglike
        A model instance configured to compute the log-likelihood.
    params
        A tuple of the free parameters. The length and order must be identical
        to params_new.
    result
        A FitResult instance to update.
    jacobian
        The Jacobian array. Unused in this function.
    never_evaluate_jacobian
        If True, the jacobian will never be evaluated, taking precedence
        over result.config.eval_residual.
    return_loglike
        If False, will return the negative of the residual instead of the
        log-likelihood.

    Returns
    -------
    The log-likehood if return_loglike, otherwise the negative of the
    residual from result.inputs.residual. kwargs are for the convenience of
    libraries other than scipy and will not be changed by scipy itself.

    Notes
    -----
    Scipy requires that this function have the same args as the jacobian
    function (jacobian_scipy), so unused args must not be removed.

    Scipy generally calls this function every iteration, but only conditionally
    calls the jacobian_scipy function (e.g. if the proposal is accepted). If
    users expect proposals to (almost) always be accepted, it is more efficient
    to compute the Jacobian here (and skip evaluating it again when
    jacobian_scipy is called), because evaluating model_jacobian also updates
    the residual array, and so there is no need to evaluate model_loglike.

    To summarize, if never_evaluate_jacobian or config_fit.eval_residual:
        There is ALWAYS one call to model_jacobian.evaluate,
        and ZERO calls to model_loglike.evaluate.
    else:
        There is always one call to model_loglike.evaluate,
        and MAYBE one call to model_jacobian.evaluate.

    """
    set_params(params, params_new, model_loglike)
    config_fit = result.config
    fit_linear_iter = config_fit.fit_linear_iter
    if (fit_linear_iter > 0) and ((result.n_eval_resid + 1) % fit_linear_iter == 0):
        Modeller.fit_model_linear(model_loglike, ratio_min=1e-6)
    time_init = time.process_time()

    if never_evaluate_jacobian or config_fit.eval_residual:
        try:
            loglike = model_loglike.evaluate()
        except Exception:
            loglike = None
        result.n_eval_resid += 1
    else:
        loglike = model_jacobian.evaluate()
        result.n_eval_jac += 1
    result.time_eval += time.process_time() - time_init
    return loglike if return_loglike else -result.inputs.residual


def jacobian_scipy(
    params_new: np.ndarray,
    model_jacobian: Model,
    model_loglike: Model,
    params: tuple[g2f.ParameterD],
    result: FitResult,
    jacobian: np.ndarray,
    always_evaluate_jacobian: bool = False,
) -> np.ndarray:
    """Compute the Jacobian for a scipy optimizer.

    Parameters
    ----------
    params_new
        An array of new parameter values. Unused here.
    model_jacobian
        A model instance configured to compute the Jacobian.
    model_loglike
        A model instance configured to compute the log-likelihood. Unused here.
    params
        A tuple of the free parameters. Unused here.
    result
        A FitResult instance to update.
    jacobian
        The Jacobian array. Unused in this function.
    always_evaluate_jacobian
        If True, the jacobian will always be evaluated, taking precedence
        over result.config.eval_residual.

    Returns
    -------
    A reference to jacobian, whose values may have been updated.

    Notes
    -----
    Scipy requires that this function have the same args as the residual
    function (residual_scipy), so unused args must not be removed. kwargs are
    for the convenience of libraries other than scipy and will not be changed
    by scipy itself.

    Parameter objects and new values are unused here as they will have already
    been set by the residual funciton.

    Scipy generally does not call this function every iteration. If it is
    configured to skip evaluating the Jacobian, it is presumed to have been
    updated by the residual function already.
    """
    if always_evaluate_jacobian or result.config.eval_residual:
        time_init = time.process_time()
        model_jacobian.evaluate()
        result.time_eval += time.process_time() - time_init
        result.n_eval_jac += 1
    return jacobian


if has_pygmo:
    class PygmoUDP:
        """A Pygmo User-Defined Problem for a MultiProFit model.

        Pygmo optimizers take a class with a fitness function
        (i.e. the negative log-likelihood, although one could use some other
        arbitrary fitness function if it made snese to do so), with a
        fitness function and a gradient function returning the derivative of
        the fitness w.r.t. each free parameters.

        Pygmo optimizers do not appear to use the full residual array or
        Jacobian the way scipy optimizers do. The gradient of the
        log-likelihood is cheaper to compute than the full Jacobian; however,
        using only the gradient of the fitness may cause slower convergence.

        Parameters
        ----------
        params
            A tuple of the free parameters.
        model_loglike
            A model configured to compute the log-likelihood.
        model_loglike_grad
            A model configured to compute the gradient of the
            log-likelihood w.r.t. each free parameter.
        bounds_lower
            A tuple of the lower bounds of the transformed value for each
            free parameter in params.
        bounds_upper
            A tuple of the upper bounds of the transformed value for each
            free parameter in params.
        result
            A result object to update and read configuration from.
        """

        def __init__(
            self,
            params: tuple[g2f.ParameterD],
            model_loglike: Model,
            model_loglike_grad: Model,
            bounds_lower: tuple[float],
            bounds_upper: tuple[float],
            result: FitResult,
        ):
            self.params = params
            self.model_loglike = model_loglike
            self.model_loglike_grad = model_loglike_grad
            self.bounds_lower = bounds_lower
            self.bounds_upper = bounds_upper
            self.result = result

        def fitness(self, x):
            loglike = residual_scipy(
                x,
                model_jacobian=self.model_loglike_grad,
                model_loglike=self.model_loglike,
                params=self.params,
                result=self.result,
                jacobian=None,
                never_evaluate_jacobian=True,
                return_loglike=True,
            )
            return [-sum(loglike),]

        def get_bounds(self):
            return self.bounds_lower, self.bounds_upper

        def gradient(self, x):
            set_params(params=self.params, params_new=x, model_loglike=self.model_loglike)
            time_init = time.process_time()
            loglike_grad = -np.array(self.model_loglike_grad.compute_loglike_grad())
            self.result.time_eval += time.process_time() - time_init
            self.result.n_eval_jac += 1
            return loglike_grad

        def __deepcopy__(self, memo):
            """Make a deep copy of a model with a shallow copy of the data
            which should not be duplicated.

            Pygmo optimizers always make at least one copy of this class, and
            some (like particle swarm) will make many more. The input data
            must be shallow copies, both to avoid excess memory usage and
            because Model instances cannot be deep copied.
            """
            fitinputs = FitInputs.from_model(self.model_loglike)
            model_loglike, model_loglike_grad = (g2f.ModelD(
                data=model.data,
                psfmodels=model.psfmodels,
                sources=model.sources,
                priors=model.priors,
            ) for model in (self.model_loglike, self.model_loglike_grad))
            model_loglike.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike)
            model_loglike_grad.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike_grad)

            copied = self.__class__(
                params=self.params,
                model_loglike=model_loglike,
                model_loglike_grad=model_loglike_grad,
                bounds_lower=self.bounds_lower,
                bounds_upper=self.bounds_upper,
                result=FitResult(inputs=fitinputs, config=self.result.config),
            )
            memo[id(self)] = copied
            return copied


class Modeller:
    """Fit lsst.gauss2d.fit Model instances using Python optimizers.

    Parameters
    ----------
    logger : `logging.Logger`
        The logger. Defaults to calling `_getlogger`.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        if logger is None:
            logger = self._get_logger()
        self.logger = logger

    @staticmethod
    def _get_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        return logger

    @staticmethod
    def compute_variances(
        model: Model, use_diag_only: bool = False, use_svd: bool = False, **kwargs: Any
    ) -> np.ndarray:
        """Compute model free parameter variances from the inverse Hessian.

        Parameters
        ----------
        model
            The model to compute parameter variances for.
        use_diag_only
            Whether to use diagonal terms only, i.e. ignore covariance.
        use_svd
            Whether to use singular value decomposition to compute the inverse
            Hessian.
        **kwargs
            Additional keyword arguments to pass to model.compute_hessian.

        Returns
        -------
        variances
            The free parameter variances.
        """
        hessian = model.compute_hessian(**kwargs).data
        if use_diag_only:
            return -1 / np.diag(hessian)
        if use_svd:
            u, s, v = np.linalg.svd(-hessian)
            inverse = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
        else:
            inverse = np.linalg.inv(-hessian)
        return np.diag(inverse)

    @staticmethod
    def fit_gaussians_linear(
        gaussians_linear: LinearGaussians,
        observation: g2f.ObservationD,
        psf_model: g2f.PsfModel = None,
        fit_methods: dict[str, dict[str, Any]] | None = None,
        plot: bool = False,
    ) -> dict[str, FitResult]:
        """Fit normalizations for a Gaussian mixture model.

        Parameters
        ----------
        gaussians_linear
            The Gaussian components - fixed or otherwise - to fit.
        observation
            The observation to fit against.
        psf_model
            A PSF model for the observation, if fitting sources.
        fit_methods
            A dictionary of fitting methods to employ, keyed by method name,
            with a value of a dict of options (kwargs) to pass on. Default
            is "scipy.optimize.nnls".
        plot
            Whether to generate fit residual/diagnostic plots.

        Returns
        -------
        results
            Fit results for each method, keyed by the fit method name.
        """
        if psf_model is None:
            psf_model = make_psf_model_null()
        if fit_methods is None:
            fit_methods = {"scipy.optimize.nnls": fit_methods_linear["scipy.optimize.nnls"]}
        else:
            for fit_method in fit_methods:
                if fit_method not in fit_methods_linear:
                    raise ValueError(f"Unknown linear {fit_method=}")
        n_params = len(gaussians_linear.gaussians_free)
        if not (n_params > 0):
            raise ValueError(f"!({len(gaussians_linear.gaussians_free)=}>0); can't fit with no free params")
        image = observation.image
        shape = image.shape
        coordsys = image.coordsys

        mask_inv = observation.mask_inv.data
        sigma_inv = observation.sigma_inv.data
        bad = ~(sigma_inv > 0)
        n_bad = np.sum(bad)
        if n_bad > 0:
            mask_inv &= ~bad

        sigma_inv = sigma_inv[mask_inv]
        size = np.sum(mask_inv)

        gaussians_psf = psf_model.gaussians(g2f.Channel.NONE)
        if len(gaussians_linear.gaussians_fixed) > 0:
            image_fixed = make_image_gaussians(
                gaussians_source=gaussians_linear.gaussians_fixed,
                gaussians_kernel=gaussians_psf,
                n_rows=shape[0],
                n_cols=shape[1],
            ).data
            image_fixed = image_fixed[mask_inv]
        else:
            image_fixed = None

        x = np.zeros((size, n_params))

        params = [None] * n_params
        for idx_param, (gaussians_free, param) in enumerate(gaussians_linear.gaussians_free):
            image_free = make_image_gaussians(
                gaussians_source=gaussians_free,
                gaussians_kernel=gaussians_psf,
                n_rows=shape[0],
                n_cols=shape[1],
                coordsys=coordsys,
            ).data
            x[:, idx_param] = ((image_free if mask_inv is None else image_free[mask_inv]) * sigma_inv).flat
            params[idx_param] = param

        y = observation.image.data
        if plot:
            import matplotlib.pyplot as plt

            plt.imshow(y, origin="lower")
            plt.show()
        if mask_inv is not None:
            y = y[mask_inv]
        if image_fixed is not None:
            y -= image_fixed
        y = (y * sigma_inv).flat

        results = {}

        for fit_method, kwargs in fit_methods.items():
            kwargs = kwargs if kwargs is not None else fit_methods_linear[fit_method]
            if fit_method == "scipy.optimize.nnls":
                values = spopt.nnls(x, y)[0]
            elif fit_method == "scipy.optimize.lsq_linear":
                values = spopt.lsq_linear(x, y, **kwargs).x
            elif fit_method == "numpy.linalg.lstsq":
                values = np.linalg.lstsq(x, y, **kwargs)[0]
            elif fit_method == "fastnnls.fnnls":
                y = x.T.dot(y)
                x = x.T.dot(x)
                values = fnnls(x, y)
            else:
                raise RuntimeError(f"Unknown linear {fit_method=} not caught earlier (logic error)")
            results[fit_method] = values
        return results

    def fit_model(
        self,
        model: Model,
        fitinputs: FitInputs | None = None,
        printout: bool = False,
        config: ModelFitConfig | None = None,
        **kwargs: Any,
    ) -> FitResult:
        """Fit a model with a nonlinear optimizer.

        Parameters
        ----------
        model
            The model to fit.
        fitinputs
            An existing FitInputs with jacobian/residual arrays to reuse.
        printout
            Whether to print diagnostic information.
        config
            Configuration settings for model fitting.
        **kwargs
            Keyword arguments to pass to the optimizer.

        Returns
        -------
        result
            The results from running the fitter.

        Notes
        -----
        The only supported fitter is scipy.optimize.least_squares.
        """
        if config is None:
            config = ModelFitConfig()
        config.validate()

        use_pygmo = config.optimization_library == "pygmo"
        model_loglike = g2f.ModelD(
            data=model.data,
            psfmodels=model.psfmodels,
            sources=model.sources,
            priors=model.priors,
        ) if (use_pygmo or config.eval_residual) else None

        if use_pygmo:
            model.setup_evaluators(g2f.EvaluatorMode.loglike_grad, force=True)
            model_loglike.setup_evaluators(g2f.EvaluatorMode.loglike, force=True)
        else:
            if fitinputs is None:
                fitinputs = FitInputs.from_model(model)
            else:
                errors = fitinputs.validate_for_model(model)
                if errors:
                    newline = "\n"
                    raise ValueError(f"fitinputs validation got errors:\n{newline.join(errors)}")
            model.setup_evaluators(
                evaluatormode=g2f.EvaluatorMode.jacobian,
                outputs=fitinputs.jacobians,
                residuals=fitinputs.residuals,
                outputs_prior=fitinputs.outputs_prior,
                residuals_prior=fitinputs.residuals_prior,
                print=printout,
                force=True,
            )

        params_psf_free = []
        for psfmodel in model.psfmodels:
            params_psf_free.extend(get_params_uniq(psfmodel, fixed=False))
        if params_psf_free:
            params_psf_free = {k: None for k in params_psf_free}
            raise ValueError(
                f"Model has free PSF model params: {list(params_psf_free.keys())}."
                f" All PSF model parameters must be fixed before fitting."
            )

        offsets_params = dict(model.offsets_parameters())
        params_offsets = {v: k for (k, v) in offsets_params.items()}
        params_free = tuple(params_offsets[idx] for idx in range(1, len(offsets_params) + 1))
        params_free_sorted_all = tuple(get_params_uniq(model, fixed=False))
        params_free_sorted = []
        params_free_sorted_missing = []

        # If we were forced to drop an observation, re-generate the modeller
        # Only integral parameters should be missing
        for param in params_free_sorted_all:
            if param in params_offsets.values():
                params_free_sorted.append(param)
            else:
                if not isinstance(param, g2f.IntegralParameterD):
                    raise RuntimeError(f"non-integral {param=} missing from {offsets_params=}")
                param.limits = g2f.LimitsD(param.min, param.max)
                param.value = param.min
                param.fixed = True
                params_free_sorted_missing.append(param)

        try:
            if not use_pygmo:
                if params_free_sorted_missing:
                    fitinputs = FitInputs.from_model(model)
                    params_free_sorted = tuple(params_free_sorted)
                    model.setup_evaluators(
                        evaluatormode=g2f.EvaluatorMode.jacobian,
                        outputs=fitinputs.jacobians,
                        residuals=fitinputs.residuals,
                        outputs_prior=fitinputs.outputs_prior,
                        residuals_prior=fitinputs.residuals_prior,
                        print=printout,
                        force=True,
                    )
                else:
                    params_free_sorted = params_free_sorted_all
                if config.eval_residual:
                    model_loglike.setup_evaluators(
                        evaluatormode=g2f.EvaluatorMode.loglike,
                        residuals=fitinputs.residuals,
                        residuals_prior=fitinputs.residuals_prior,
                    )

                jac = fitinputs.jacobian[:, 1:]
                # Assert that this is a view, otherwise this won't work
                assert id(jac.base) == id(fitinputs.jacobian)

            n_params_free = len(params_free)
            bounds = ([None] * n_params_free, [None] * n_params_free)
            params_init = [None] * n_params_free

            for idx, param in enumerate(params_free):
                limits = param.limits
                # If the transform has more restrictive limits, use those
                if hasattr(param.transform, "limits"):
                    limits_transform = param.transform.limits
                    n_within = limits.check(limits_transform.min) + limits.check(limits_transform.min)
                    if n_within == 2:
                        limits = limits_transform
                    elif n_within != 0:
                        raise ValueError(
                            f"{param=} {param.limits=} and {param.transform.limits=}"
                            f" intersect; one must be a subset of the other"
                        )
                bounds[0][idx] = param.transform.forward(limits.min)
                bounds[1][idx] = param.transform.forward(limits.max)
                if not limits.check(param.value):
                    raise RuntimeError(f"{param=}.value_transformed={param.value} not within {limits=}")
                params_init[idx] = param.value_transformed

            results = FitResult(inputs=fitinputs, config=config)
            time_init = time.process_time()
            if use_pygmo:
                uda = pg.nlopt("lbfgs")
                uda.ftol_abs = 1e-4
                algo = pg.algorithm(uda)

                # pygmo seems to make proposals right at the limits
                # parameter limits are currently set as untransformed values
                # and sometimes the proposal exceeds those when transformed
                bounds_lower = tuple(np.nextafter(x, np.inf) for x in bounds[0])
                bounds_upper = tuple(np.nextafter(x, -np.inf) for x in bounds[1])

                udp = PygmoUDP(
                    params=params_free,
                    model_loglike=model_loglike,
                    model_loglike_grad=model,
                    bounds_lower=bounds_lower,
                    bounds_upper=bounds_upper,
                    result=results,
                )

                # if the initial value was at one of the bounds, reset it to
                # one percent of the range away from the bound
                for idx, value_init in enumerate(params_init):
                    bound_lower = bounds_lower[idx]
                    bound_upper = bounds_upper[idx]
                    if value_init >= bound_upper:
                        params_init[idx] = bound_lower + 0.99*(bound_upper - bound_lower)
                    elif value_init <= bound_lower:
                        params_init[idx] = bound_lower + 0.01*(bound_upper - bound_lower)

                problem = pg.problem(udp)
                pop = pg.population(prob=problem, size=0)
                pop.push_back(np.array(params_init))
                result_opt = algo.evolve(pop)
                x_best = result_opt.champion_x
                results.n_eval_func = pop.problem.get_fevals()
                results.n_eval_jac = pop.problem.get_gevals()
                results.chisq_best = 2*result_opt.champion_f
            else:
                # The initial evaluate will fill in jac for the next line
                # _ll_init is assigned just for convenient debugging
                _ll_init = model.evaluate()  # noqa: F841
                x_scale_jac_clipped = np.clip(1.0 / (np.sum(jac**2, axis=0) ** 0.5), 1e-5, 1e19)
                result_opt = spopt.least_squares(
                    residual_scipy,
                    params_init,
                    jac=jacobian_scipy,
                    bounds=bounds,
                    args=(model, model_loglike, params_free, results, jac),
                    x_scale=x_scale_jac_clipped,
                    **kwargs,
                )
                x_best = result_opt.x
                results.n_eval_func = result_opt.nfev
                results.n_eval_jac = result_opt.njev if result_opt.njev else 0
                results.chisq_best = 2*result_opt.cost

            results.time_run = time.process_time() - time_init
            results.result = result_opt
            if params_free_sorted_missing:
                params_best = []
                for param in params_free_sorted_all:
                    if param in params_free_sorted_missing:
                        params_best.append(param.value)
                        param.fixed = False
                    else:
                        params_best.append(x_best[offsets_params[param] - 1])
                results.params_best = tuple(params_best)
                results.params = params_free_sorted_all
            else:
                results.params_best = tuple(
                    x_best[offsets_params[param] - 1] for param in params_free_sorted
                )
                results.params = params_free_sorted
            results.params_free_missing = tuple(params_free_sorted_missing)
        except Exception as e:
            # Any missing params we fixed must be set free again
            for param in params_free_sorted_missing:
                param.fixed = False
            raise e

        return results

    # TODO: change to staticmethod if requiring py3.10+
    @classmethod
    def fit_model_linear(
        cls,
        model: Model,
        idx_obs: int | Sequence[int] | None = None,
        ratio_min: float = 0,
        validate: bool = False,
        limits_interval_min: float = 0.01,
        limits_interval_max: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a model's linear parameters (integrals).

        Parameters
        ----------
        model
            The model to fit parameters for.
        idx_obs
            An index or sequence of indices of observations to fit.
            The default is to fit all observations.
        ratio_min
            The minimum ratio of the previous value to set any parameter to.
            This can prevent setting parameters to zero.
        validate
            If True, check that the model log-likelihood improves and restore
            the original parameter values if not.
        limits_interval_min
            A value 0<=x<limits_interval_max<=1 specifying the lower bound to
            clip parameter values to, as a ratio of each parameter's limits.
        limits_interval_max
            A value 0<=limits_interval_min<x<=1 specifying the upper bound to
            clip parameter values to, as a ratio of each parameter's limits.

        Returns
        -------
        loglike_init
            The initial log likelihood if validate is True, otherwise None.
        loglike_final
            The post-fit log likelihood if validate is True, otherwise None.

        Notes
        -----
        The purpose of limits_interval is to slightly offset parameters from
        the extrema of their limits. This is typically most useful for
        integral parameters with a minimum of zero, which might otherwise be
        stuck at zero in a subsequent nonlinear fit.
        """
        if (
            not (0 <= limits_interval_min <= 1)
            or not (0 <= limits_interval_max <= 1)
            or not (limits_interval_min < limits_interval_max)
        ):
            raise ValueError(f"Must have 0 <= {limits_interval_min} < {limits_interval_max} <= 1")
        n_data = len(model.data)
        n_sources = len(model.sources)
        if n_sources != 1:
            raise ValueError("fit_model_linear does not yet support models with >1 sources")
        if idx_obs is not None:
            if isinstance(idx_obs, int):
                if not ((idx_obs >= 0) and (idx_obs < n_data)):
                    raise ValueError(f"{idx_obs=} not >=0 and < {len(model.data)=}")
                indices = range(idx_obs, idx_obs + 1)
            else:
                if len(set(idx_obs)) != len(idx_obs):
                    raise ValueError(f"{idx_obs=} has duplicate values")
                indices = tuple(idx_obs)
                if not all(((idx_obs >= 0) and (idx_obs < n_data) for idx_obs in indices)):
                    raise ValueError(f"idx_obs={indices} has values not >=0 and < {len(model.data)=}")
        else:
            indices = range(n_data)

        if validate:
            model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike)
            loglike_init = model.evaluate()
        else:
            loglike_init = None
        values_init = {}
        values_new = {}

        for idx_obs in indices:
            obs = model.data[idx_obs]
            gaussians_linear = LinearGaussians.make(model.sources[0], channel=obs.channel)
            result = cls.fit_gaussians_linear(gaussians_linear, obs, psf_model=model.psfmodels[idx_obs])
            values = list(result.values())[0]

            for (_, parameter), ratio in zip(gaussians_linear.gaussians_free, values):
                values_init[parameter] = float(parameter.value)
                if not (ratio >= ratio_min):
                    ratio = ratio_min
                value_new = max(ratio * parameter.value, parameter.limits.min)
                values_new[parameter] = value_new

        for parameter, value in values_new.items():
            value_min, value_max = parameter.limits.min, parameter.limits.max
            min_is_inf = value_min == -np.inf
            max_is_inf = value_max == np.inf
            if min_is_inf:
                if max_is_inf:
                    parameter.value = value
                    continue
                if not value < value_max:
                    value = value_max - 1e-5
            elif max_is_inf:
                if not value > value_min:
                    value = value_min + 1e-5
            else:
                limits_interval = parameter.limits.max - parameter.limits.min
                value = np.clip(
                    value,
                    value_min + limits_interval_min * limits_interval,
                    value_min + limits_interval_max * limits_interval,
                )
            parameter.value = value

        if validate:
            loglike_new = model.evaluate()
            if not (sum(loglike_new) > sum(loglike_init)):
                for parameter, value in values_init.items():
                    parameter.value = value
        else:
            loglike_new = None
        return loglike_init, loglike_new

    @staticmethod
    def make_components_linear(
        component_mixture: g2f.ComponentMixture,
    ) -> list[g2f.GaussianComponent]:
        """Make a list of fixed Gaussian components from a ComponentMixture.

        Parameters
        ----------
        component_mixture
            A component mixture to create a component list for.

        Returns
        -------
        gaussians
            A list of Gaussians components with fixed parameters and values
            matching those in the original component mixture.
        """
        components = component_mixture.components
        if len(components) == 0:
            raise ValueError(f"Can't get linear Source from {component_mixture=} with no components")
        components_new = [None] * len(components)
        for idx, component in enumerate(components):
            gaussians = component.gaussians(g2f.Channel.NONE)
            # TODO: Support multi-Gaussian components if sensible
            # The challenge would be in mapping linear param values back onto
            # non-linear IntegralModels
            n_g = len(gaussians)
            if not n_g == 1:
                raise ValueError(f"{component=} has {gaussians=} of len {n_g=}!=1")
            gaussian = gaussians.at(0)
            component_new = g2f.GaussianComponent(
                g2f.GaussianParametricEllipse(
                    g2f.SigmaXParameterD(gaussian.ellipse.sigma_x, fixed=True),
                    g2f.SigmaYParameterD(gaussian.ellipse.sigma_y, fixed=True),
                    g2f.RhoParameterD(gaussian.ellipse.rho, fixed=True),
                ),
                g2f.CentroidParameters(
                    g2f.CentroidXParameterD(gaussian.centroid.x, fixed=True),
                    g2f.CentroidYParameterD(gaussian.centroid.y, fixed=True),
                ),
                g2f.LinearIntegralModel(
                    [
                        (g2f.Channel.NONE, g2f.IntegralParameterD(gaussian.integral.value)),
                    ]
                ),
            )
            components_new[idx] = component_new
        return components_new
