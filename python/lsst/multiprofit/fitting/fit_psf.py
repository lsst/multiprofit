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
    "CatalogExposurePsfABC",
    "CatalogPsfFitterConfig",
    "CatalogPsfFitterConfigData",
    "CatalogPsfFitter",
]

from abc import abstractmethod
from functools import cached_property
import logging
import math
import time
from typing import Any, ClassVar, Mapping, Type

import astropy
import lsst.gauss2d as g2
import lsst.gauss2d.fit as g2f
import lsst.pex.config as pexConfig
from lsst.utils.logging import PeriodicLogger
import numpy as np
import pydantic

from ..componentconfig import (
    Fluxes,
    FluxFractionParameterConfig,
    FluxParameterConfig,
    GaussianComponentConfig,
    ParameterConfig,
)
from ..errors import PsfRebuildFitFlagError
from ..modeller import FitInputsDummy, LinearGaussians, Modeller, make_psf_model_null
from ..sourceconfig import ComponentConfigs, ComponentGroupConfig, SourceConfig
from ..utils import frozen_arbitrary_allowed_config, get_params_uniq
from .fit_catalog import CatalogExposureABC, CatalogFitterConfig, ColumnInfo

SourceRow = astropy.table.Row | Mapping[str, Any]


class CatalogPsfFitterConfig(CatalogFitterConfig):
    """Configuration for MultiProFit PSF image fitter."""

    model = pexConfig.ConfigField[SourceConfig](
        default=SourceConfig(
            component_groups={
                "": ComponentGroupConfig(
                    components_gauss={
                        "gauss1": GaussianComponentConfig(
                            size_x=ParameterConfig(value_initial=1.5),
                            size_y=ParameterConfig(value_initial=1.5),
                            fluxfrac=FluxFractionParameterConfig(value_initial=0.5),
                            flux=FluxParameterConfig(value_initial=1.0, fixed=True),
                        ),
                        "gauss2": GaussianComponentConfig(
                            size_x=ParameterConfig(value_initial=3.0),
                            size_y=ParameterConfig(value_initial=3.0),
                            fluxfrac=FluxFractionParameterConfig(value_initial=1.0, fixed=True),
                        ),
                    },
                    is_fractional=True,
                )
            }
        ),
        doc="PSF model configuration",
    )
    prior_axrat_mean = pexConfig.Field[float](default=0.95, doc="Mean for axis ratio prior")
    sigma_min = pexConfig.Field[float](
        default=0.8,
        doc="Minimum sigma in pixels for PSF components. Must be >=0.8 to avoid undersampling.",
        check=lambda x: x >= 0.8,
    )

    def make_psf_model(
        self,
        component_group_fluxes: list[list[Fluxes]] | None = None,
    ) -> [g2f.PsfModel, list[g2f.Prior]]:
        """Make a PsfModel object for a given source.

        Parameters
        ----------
        component_group_fluxes
            Initial fluxes for each constituent ComponentGroup.

        Returns
        -------
        psf_model
            The rebuilt PSF model.

        Notes
        -----
        This function does not initialize the PSF model.
        """
        if component_group_fluxes is None:
            channels = (g2f.Channel.NONE,)
            component_group_fluxes = [
                component_group.get_fluxes_default(
                    channels=channels,
                    component_configs=component_group.get_component_configs(),
                    is_fractional=component_group.is_fractional,
                )
                for component_group in self.model.component_groups.values()
            ]

        psf_model, _ = self.model.make_psf_model(component_group_fluxes=component_group_fluxes)
        return psf_model

    def schema_configurable(self) -> list[ColumnInfo]:
        columns = []
        if self.config_fit.eval_residual:
            columns.append(ColumnInfo(key="n_eval_jac", dtype="i4"))
        return columns

    def schema(
        self,
        bands: list[str] | None = None,
    ) -> list[ColumnInfo]:
        """Return the schema as an ordered list of columns.

        Parameters
        ----------
        bands
            The bands to add band-dependent columns for.
        """
        if bands is not None:
            if len(bands) != 1:
                raise ValueError("CatalogPsfFitter must have exactly one band")
        schema = super().schema(bands)
        parameters = CatalogPsfFitterConfigData(config=self).parameters
        unit_size = "pix"
        units = {
            g2f.ReffXParameterD: unit_size,
            g2f.ReffYParameterD: unit_size,
            g2f.SizeXParameterD: unit_size,
            g2f.SizeYParameterD: unit_size,
        }
        schema.extend(
            [
                ColumnInfo(key=key, dtype="f8", unit=units.get(type(param)))
                for key, param in parameters.items()
            ]
        )
        schema.extend(self.schema_configurable())

        return schema

    def setDefaults(self) -> None:
        self.prefix_column = "mpf_psf_"
        self.compute_errors = "NONE"


class CatalogPsfFitterConfigData(pydantic.BaseModel):
    """A PSF fit configuration that can initialize models and images thereof.

    This class relies on cached properties being computed once, mostly shortly
    after initialization. Therefore, it and the config field must be frozen to
    ensure that the model remains unchanged.
    """

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    config: CatalogPsfFitterConfig = pydantic.Field(title="A CatalogPsfFitterConfig to be frozen")

    @pydantic.field_validator("config")
    @classmethod
    def validate_config(cls, v: CatalogPsfFitterConfig) -> CatalogPsfFitterConfig:
        v.validate()
        return v

    @cached_property
    def components(self) -> dict[str, g2f.Component]:
        """Return the components of the PSF model by name."""
        components = self.psf_model.components
        names = self.component_configs.keys()
        if len(components) != len(names):
            raise RuntimeError(f"{len(components)=} != {len(names)=}")
        components_names = {name: component for name, component in zip(names, components)}
        return components_names

    @cached_property
    def component_configs(self) -> ComponentConfigs:
        """Return the config for each component in the PSF model."""
        return self.config.model.get_component_configs()

    @cached_property
    def componentgroup_configs(self) -> dict[str, ComponentGroupConfig]:
        """Return the config for each component group in the PSF model
        by name.
        """
        return {k: v for k, v in self.config.model.component_groups.items()}

    def init_psf_model(
        self,
        params: SourceRow,
    ) -> None:
        """Initialize the PSF model for a single source.

        Parameters
        ----------
        params : astropy.table.Row | typing.Mapping[str, typing.Any]
            A mapping with parameter values for the best-fit PSF model at the
            centroid of a single source.
        """
        # TODO: Improve _flag checking (add a total _flag column)
        for flag in (col for col in params.keys() if col.endswith("_flag")):
            if params[flag]:
                raise PsfRebuildFitFlagError(f"Failed to rebuild PSF; {flag} set")

        for name, param in self.parameters.items():
            param.value = params[f"{self.config.prefix_column}{name}"]

    def model_post_init(self, __context: Any) -> None:
        self.config.freeze()
        n_component_configs = len(self.component_configs)
        n_components = len(self.psf_model.components)
        if n_components != n_component_configs:
            raise AssertionError(f"{n_components=} != {n_component_configs=}")

    @cached_property
    def parameters(self) -> dict[str, g2f.ParameterD]:
        """Return the free parameters for the PSF model by name."""
        parameters = {}
        config = self.config
        has_prefix_group = config.model.has_prefix_group()
        components = self.psf_model.components
        idx_comp_first = 0

        label_cen = config.get_key_cen()
        label_rho = config.get_key_rho()
        label_flux = config.get_key_flux("")
        label_fluxfrac = f"{label_flux}frac"
        suffix_x, suffix_y = config.get_suffix_x(), config.get_suffix_y()

        # Iterate over each component group
        for name_group, config_group in self.componentgroup_configs.items():
            prefix_group = f"{name_group}_" if has_prefix_group else ""
            is_fractional = config_group.is_fractional
            # Does this group have independent centroids?
            # Should generally be False if not is_fractional, since there is
            # little reason to make a group otherwise.
            multicen = len(config_group.centroids) > 1
            configs_comp = config_group.get_component_configs()
            idx_last = len(configs_comp) - 1
            n_params_flux_frac = 0

            # Iterate over each component in the group
            for idx_comp_group, (name_comp, config_comp) in enumerate(configs_comp.items()):
                # The last component needs special handling if is_fractional
                is_last = idx_comp_group == idx_last
                component = components[idx_comp_first + idx_comp_group]
                prefix_comp = f"{prefix_group}{name_comp}"
                key_size = config.get_prefixed_label(
                    config.get_key_size(config_comp.get_size_label()),
                    prefix_comp,
                )
                key_rho = config.get_prefixed_label(label_rho, prefix_comp)

                # Give the centroid parameters an appropriate prefix
                if multicen or (idx_comp_group == 0):
                    prefix_cen = prefix_comp if multicen else prefix_group
                    # Avoid redundant -underscores if there's nothing to prefix
                    # or an existing prefix starting with an underscore
                    key_cen = config.get_prefixed_label(label_cen, prefix_cen)
                    parameters[f"{key_cen}{suffix_x}"] = component.centroid.x_param
                    parameters[f"{key_cen}{suffix_y}"] = component.centroid.y_param
                # Add each free shape parameter
                if not config_comp.size_x.fixed:
                    parameters[f"{key_size}{suffix_x}"] = component.ellipse.size_x_param
                if not config_comp.size_y.fixed:
                    parameters[f"{key_size}{suffix_y}"] = component.ellipse.size_y_param
                if not config_comp.rho.fixed:
                    parameters[key_rho] = component.ellipse.rho_param

                # TODO: return this to component.integralmodel
                # when binding for g2f.FractionalIntegralModel is fixed
                params_flux = get_params_uniq(component, fixed=False, nonlinear=False)
                # Fractional models should have one fixed flux parameter
                # (with value=1), and hence an empty params_flux
                has_params_flux = not config_comp.flux.fixed and (
                    (not is_fractional) or (idx_comp_group == 0)
                )
                n_params_flux_expect = int(has_params_flux)
                if len(params_flux) != n_params_flux_expect:
                    raise RuntimeError(
                        f"{params_flux=} has len={len(params_flux)} but expected {n_params_flux_expect}"
                    )
                if has_params_flux:
                    parameters[f"{prefix_comp}{label_flux}"] = params_flux[0]
                # TODO: return this to component.integralmodel
                # when binding for g2f.FractionalIntegralModel is fixed
                params_fluxfrac = [
                    param
                    for param in get_params_uniq(component, fixed=False, linear=False)
                    if isinstance(param, g2f.ProperFractionParameterD)
                ]
                if is_fractional:
                    # The last flux fraction must be fixed at 1.0, since it
                    # by definition contains all of the remaining flux
                    if is_last:
                        if (config_comp.fluxfrac.value_initial != 1.0) or (not config_comp.fluxfrac.fixed):
                            raise ValueError(
                                f"{config_comp=} {is_last=} and must be fixed with value_initial==1.0"
                            )
                    else:
                        if not config_comp.fluxfrac.fixed:
                            parameters[f"{prefix_comp}{label_fluxfrac}"] = params_fluxfrac[-1]
                            n_params_flux_frac += 1
                    if len(params_fluxfrac) != n_params_flux_frac:
                        raise RuntimeError(
                            f"{config_comp=} has {params_fluxfrac=} but expected {n_params_flux_frac=}"
                        )
                else:
                    if len(params_fluxfrac) > 0:
                        raise RuntimeError(f"{config_group=} has {params_fluxfrac=} but {is_fractional=}")

        return parameters

    @cached_property
    def psf_model(self) -> g2f.PsfModel:
        """Return the PSF model."""
        psf_model = self.config.make_psf_model()
        return psf_model

    @cached_property
    def psf_model_gaussians(self) -> g2.Gaussians:
        """Return the Gaussians comprising the PSF model."""
        gaussians = self.psf_model.gaussians()
        return gaussians


class CatalogExposurePsfABC(CatalogExposureABC):
    """A CatalogExposure for PSF fitting."""

    @abstractmethod
    def get_psf_image(
        self,
        source: SourceRow,
    ) -> np.ndarray:
        """Get a PSF image for a specific source.

        Parameters
        ----------
        source
            The source row/dict.

        Returns
        -------
        psf
           The image of the PSF.

        Notes
        -----
        The PSF image should be normalized, and centered in a 2D array of odd
        dimensions on both sides.
        """


class CatalogPsfFitter:
    """Fit a Gaussian mixture model to a pixelated PSF image.

    Parameters
    ----------
    modeller : `multiprofit.Modeller`
        A Modeller instance to use for fitting.
    errors_expected : dict[Type[Exception], str]
        A dictionary keyed by an Exception type, with a string value of the
        flag column key to assign if this Exception is raised.

    Notes
    -----
    Any exceptions raised and not in errors_expected will be logged in a
    generic unknown_flag failure column.
    """

    def __init__(
        self,
        modeller: Modeller | None = None,
        errors_expected: dict[Type[Exception], str] | None = None,
    ):
        if modeller is None:
            modeller = Modeller()
        if errors_expected is None:
            errors_expected = {}
        self.errors_expected = errors_expected
        self.modeller = modeller

    @staticmethod
    def _get_data_default(img_psf: np.ndarray, gain: float = 1e5) -> g2f.DataD:
        """Return a default-initialized data for a given PSF image.

        Parameters
        ----------
        img_psf
            A normalized image of the PSF, ideally with noise.
        gain
            The multiplicative factor needed to change the image to counts.

        Returns
        -------
        data
            A DataD instance than can be used to fit a PSF model.
        """
        # TODO: Try to improve these ad-hoc settings after DM-49008 merges
        # e.g. estimate PSF model noise from S/N of PSF stars?
        # If more than 1% of pixels are negative, try to estimate noise
        if np.sum(img_psf < 0) / img_psf.size > 0.01:
            background = np.std(img_psf[img_psf < 2 * np.abs(np.min(img_psf))])
        # otherwise, try to estimate noise from the lowest-values pixels
        # There might not actually be much noise; in that case, the
        # background value is not going to matter much anyway
        else:
            background = np.max((1e-10, np.std(img_psf[img_psf <= np.percentile(img_psf, 5.0)])))
        min_psf = np.min(img_psf)
        # ensure there are no negative values before sqrt
        if not (background > -min_psf):
            background = -1.1 * min_psf
        img_sig_inv = np.sqrt(gain / (img_psf + background))
        return g2f.DataD(
            [
                g2f.ObservationD(
                    channel=g2f.Channel.NONE,
                    image=g2.ImageD(img_psf),
                    sigma_inv=g2.ImageD(img_sig_inv),
                    mask_inv=g2.ImageB(np.ones_like(img_psf)),
                )
            ]
        )

    def _get_data(self, img_psf: np.ndarray, gain: float = 1e5) -> g2f.DataD:
        """Build a Model-able gauss2d.fit.Data from a normalized PSF image.

        Parameters
        ----------
        img_psf
            A normalized PSF image array.
        gain
            The number of counts in the image, used as a multiplicative
            factor for the inverse variance.

        Returns
        -------
        data
            A Data object that can be passed to a Model(ler).
        """
        return self._get_data_default(img_psf=img_psf, gain=gain)

    @staticmethod
    def _get_logger() -> logging.Logger:
        """Return a suitably-named and configured logger."""
        logger = logging.getLogger(__name__)
        return logger

    @abstractmethod
    def check_source(self, source: SourceRow, config: CatalogPsfFitterConfig) -> None:
        """Check whether a source can have its PSF model fit.

        Parameters
        ----------
        source
            The source row to check.
        config
            The fitter config.

        Notes
        -----
        Derived classes may use the source row as they deem fit. For example,
        if the source has poor quality flags, a fitter may choose not to fit
        the PSF model if it will not end up being used anyway.
        """

    def fit(
        self,
        catexp: CatalogExposurePsfABC,
        config_data: CatalogPsfFitterConfigData | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> astropy.table.Table:
        """Fit PSF models for a catalog with MultiProFit.

        Each source has its PSF fit with a configureable Gaussian mixture PSF
        model, given a pixellated PSF image from the CatalogExposure.

        Parameters
        ----------
        catexp
            An exposure to fit a model PSF at the position of all
            sources in the corresponding catalog.
        config_data
            Configuration settings for fitting and output.
        logger
            The logger. Defaults to calling `_getlogger`.
        **kwargs
            Additional keyword arguments to pass to self.modeller.

        Returns
        -------
        catalog
            A table with fit parameters for the PSF model at the location
            of each source.
        """
        if config_data is None:
            config_data = CatalogPsfFitterConfigData(config=CatalogPsfFitterConfig())
        if logger is None:
            logger = CatalogPsfFitter._get_logger()
        config = config_data.config
        if config.compute_errors != "NONE":
            raise ValueError("CatalogPsfFitter doesn't support computing errors")

        errors_expected = set(self.errors_expected.values())
        n_errors_expected = len(errors_expected)
        if n_errors_expected != len(self.errors_expected):
            raise ValueError(f"{self.errors_expected=} has duplicate values; they must be unique")
        if n_errors_expected != len(config.flag_errors):
            raise ValueError(f"len({self.errors_expected=}) != len({config.flag_errors=})")

        priors = []
        sigmas = [
            np.linalg.norm((comp.size_x.value_initial, comp.size_y.value_initial))
            for comp in config_data.component_configs.values()
        ]

        psf_model = config_data.psf_model
        model_source = g2f.Source(psf_model.components)

        for idx, (comp, config_comp) in enumerate(
            zip(psf_model.components, config_data.component_configs.values())
        ):
            prior = config_comp.make_shape_prior(comp.ellipse)
            if prior:
                if prior_size := prior.prior_size:
                    prior_size.mean = sigmas[idx]
                if prior_axrat := prior.prior_axrat:
                    prior_axrat.mean = config.prior_axrat_mean
                priors.append(prior)

        params = config_data.parameters
        flux_total = tuple(get_params_uniq(psf_model, nonlinear=False, channel=g2f.Channel.NONE))
        if len(flux_total) != 1:
            raise RuntimeError(f"len({flux_total=}) != 1; PSF model is badly-formed")
        flux_total = flux_total[0]
        gaussians_linear = None
        if config.fit_linear_init:
            # The total flux must be freed first or else LinearGaussians.make
            # will fail to find the required number of free linear params
            flux_total.fixed = False
            gaussians_linear = LinearGaussians.make(model_source, is_psf=True)
            flux_total.fixed = True

        # TODO: Remove isinstance when channel filtering is fixed: DM-45665
        fluxfracs = tuple(
            param
            for param in get_params_uniq(model_source, linear=False, channel=g2f.Channel.NONE, fixed=False)
            if isinstance(param, g2f.ProperFractionParameterD)
        )
        # We're fitting the PSF, so make a single Gaussian
        model_psf = make_psf_model_null()
        # Set the size to the minimum sigma to avoid undersampling
        ellipse = model_psf.components[0].ellipse
        ellipse.sigma_x_param.value = config.sigma_min
        ellipse.sigma_y_param.value = config.sigma_min
        sigma_min_sq = config.sigma_min**2

        catalog = catexp.get_catalog()
        n_rows = len(catalog)
        range_idx = range(n_rows)

        results, columns = config.make_catalog(n_rows)
        prefix = config.prefix_column
        columns_param = {}
        for key, param in params.items():
            is_sigma = isinstance(param, g2f.SigmaXParameterD) or isinstance(param, g2f.SigmaYParameterD)
            columns_param[f"{prefix}{key}"] = param, is_sigma
            if is_sigma:
                param.value = math.sqrt(max(param.value**2 - sigma_min_sq, 0.1))

        # dummy size for first iteration
        size, size_new = 0, 0
        fitInputs = FitInputsDummy()
        time_init_all = time.process_time()
        logger_periodic = PeriodicLogger(logger)
        n_skipfail = 0

        for idx in range_idx:
            time_init = time.process_time()
            row = results[idx]
            source = catalog[idx]
            id_source = source[config.column_id]
            row[config.column_id] = id_source
            time_final = time_init

            try:
                self.check_source(source, config=config)
                img_psf = catexp.get_psf_image(source)
                data = self._get_data(img_psf)
                model = g2f.ModelD(data=data, psfmodels=[model_psf], sources=[model_source], priors=priors)
                self.initialize_model(model=model, config_data=config_data)

                # Caches the jacobian residual if the kernel size is unchanged
                if img_psf.size != size:
                    fitInputs = None
                    size = int(img_psf.size)
                # Some algorithms might not even use fitInputs
                elif fitInputs is not None:
                    fitInputs = fitInputs if not fitInputs.validate_for_model(model) else None

                if config.fit_linear_init:
                    result = self.modeller.fit_gaussians_linear(gaussians_linear, data[0])
                    result = list(result.values())[0]
                    # Re-normalize fluxes (hopefully close already)
                    result = np.clip(
                        result * np.array([x[1].value for x in gaussians_linear.gaussians_free]),
                        1e-2,
                        0.99,
                    )
                    result /= np.sum(result)
                    for idx_param, param in enumerate(fluxfracs):
                        param.value = result[idx_param]
                        # fluxfracs are the fraction of the remaining flux
                        # this renormalizes to the remaining parameters
                        result /= np.sum(result[idx_param + 1 :])

                result_full = self.modeller.fit_model(model, fitinputs=fitInputs, **kwargs)
                fitInputs = result_full.inputs
                results[f"{prefix}n_iter"][idx] = result_full.n_eval_func
                results[f"{prefix}time_eval"][idx] = result_full.time_eval
                results[f"{prefix}time_fit"][idx] = result_full.time_run
                results[f"{prefix}chisq_reduced"][idx] = result_full.chisq_best / size
                if config.config_fit.eval_residual:
                    results[f"{prefix}n_eval_jac"][idx] = result_full.n_eval_jac

                for (key, (param, is_sigma)), value in zip(columns_param.items(), result_full.params_best):
                    param.value_transformed = value
                    value = param.value
                    if is_sigma:
                        value = math.sqrt(sigma_min_sq**2 + value**2)
                    results[key][idx] = value

                time_final = time.process_time()
                results[f"{prefix}time_full"][idx] = time_final - time_init
            except Exception as e:
                n_skipfail += 1
                size = 0 if fitInputs is None else size_new
                column = self.errors_expected.get(e.__class__, "")
                if column:
                    row[f"{prefix}{column}"] = True
                    logger.debug(
                        "id_source=%i (idx=%i/%i) PSF fit failed with known exception: %s",
                        id_source,
                        idx,
                        n_rows,
                        e,
                    )
                else:
                    row[f"{prefix}unknown_flag"] = True
                    logger.info(
                        "id_source=%i (idx=%i/%i) PSF fit failed with unexpected exception: %s",
                        id_source,
                        idx,
                        n_rows,
                        e,
                        exc_info=1,
                    )
            logger_periodic.log(
                "Fit idx=%i/%i PSFs (%i skipped/failed) in %.2f",
                idx,
                n_rows,
                n_skipfail,
                time_final - time_init_all,
            )

        n_unknown = np.sum(row[f"{prefix}unknown_flag"])
        if n_unknown > 0:
            logger.warning("%i/%i PSF fits failed with unexpected exceptions", n_unknown, n_rows)

        return results

    def initialize_model(
        self,
        model: g2f.ModelD,
        config_data: CatalogPsfFitterConfigData,
        limits_x: g2f.LimitsD | None = None,
        limits_y: g2f.LimitsD | None = None,
    ) -> None:
        """Initialize a Model for a single source row.

        Parameters
        ----------
        model
            The model object to initialize.
        config_data
            The fitter config with cached data.
        limits_x
            Hard limits for the source's x centroid. Default (0, n_columns).
        limits_y
            Hard limits for the source's y centroid. Default (0, n_rows).
        """
        n_rows, n_cols = model.data[0].image.data.shape
        cen_x, cen_y = n_cols / 2.0, n_rows / 2.0
        centroids = set()
        if limits_x is None:
            limits_x = g2f.LimitsD(0, n_cols)
        if limits_y is None:
            limits_y = g2f.LimitsD(0, n_rows)

        for component, config_comp in zip(
            config_data.components.values(), config_data.component_configs.values()
        ):
            centroid = component.centroid
            if centroid not in centroids:
                centroid.x_param.value = cen_x
                centroid.x_param.limits = limits_x
                centroid.y_param.value = cen_y
                centroid.y_param.limits = limits_y
                centroids.add(centroid)
            ellipse = component.ellipse
            ellipse.size_x_param.limits = limits_x
            ellipse.size_x = config_comp.size_x.value_initial
            ellipse.size_y_param.limits = limits_y
            ellipse.size_y = config_comp.size_y.value_initial
            ellipse.rho = config_comp.rho.value_initial
