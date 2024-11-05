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
    "CatalogBootstrapConfig",
    "CatalogExposurePsfBootstrap",
    "CatalogExposureSourcesBootstrap",
    "CatalogPsfBootstrapConfig",
    "CatalogSourceBootstrapConfig",
    "CatalogSourceFitterBootstrap",
    "NoisyObservationConfig",
]

from functools import cached_property
import logging
from typing import Any, ClassVar, Mapping, Sequence

import astropy
import lsst.gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import numpy as np
import pydantic

from ..model_utils import make_image_gaussians
from ..observationconfig import ObservationConfig
from ..utils import frozen_arbitrary_allowed_config, get_params_uniq, set_config_from_dict
from .fit_psf import CatalogExposurePsfABC, CatalogPsfFitterConfig, CatalogPsfFitterConfigData
from .fit_source import CatalogExposureSourcesABC, CatalogSourceFitterABC, CatalogSourceFitterConfigData


class CatalogBootstrapConfig(pexConfig.Config):
    """Configuration for a bootstrap source catalog fitter."""

    n_sources = pexConfig.Field[int](doc="Number of sources", default=1)

    @cached_property
    def catalog(self) -> astropy.table.Table:
        catalog = astropy.table.Table({"id": np.arange(self.n_sources)})
        return catalog


class ObservationNoiseConfig(pexConfig.Config):
    """Configuration for noise to be added to an Observation.

    The background level is in user-defined flux units, should be multiplied
    by the gain to obtain counts.
    """

    background = pexConfig.Field[float](doc="Background flux per pixel", default=1e-4)
    gain = pexConfig.Field[float](doc="Multiplicative factor to convert flux to counts", default=1.0)


class NoisyObservationConfig(ObservationConfig, ObservationNoiseConfig):
    """Configuration for an observation with noise."""


class NoisyPsfObservationConfig(ObservationConfig, ObservationNoiseConfig):
    """Configuration for a PSF observation with noise."""


class CatalogPsfBootstrapConfig(CatalogBootstrapConfig):
    """Configuration for a catalog of noisy PSF observations for bootstrapping.

    Each row is a stacked and normalized image of any number of point sources.
    """

    observation = pexConfig.ConfigField[NoisyPsfObservationConfig](
        doc="The PSF image configuration",
        default=NoisyPsfObservationConfig,
    )


class CatalogSourceBootstrapConfig(CatalogBootstrapConfig):
    """Configuration for a catalog of noisy source observations
     for bootstrapping.

    Each row is a PSF-convolved observation of the sources in one band.
    """

    observation = pexConfig.ConfigField[NoisyObservationConfig](
        doc="The source image configuration",
        default=NoisyObservationConfig,
    )


class CatalogExposurePsfBootstrap(CatalogExposurePsfABC, CatalogPsfFitterConfigData, pydantic.BaseModel):
    """Dataclass for a PSF-convolved bootstrap fitter."""

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    config_boot: CatalogPsfBootstrapConfig = pydantic.Field(title="The configuration for bootstrapping")

    @cached_property
    def image(self) -> np.ndarray:
        psf_model_init = self.config.make_psf_model()
        # A hacky way to initialize the psf_model property to the same values
        # TODO: Include this functionality in fit_psf.py
        for param_init, param in zip(get_params_uniq(psf_model_init), get_params_uniq(self.psf_model)):
            param.value = param_init.value
        image = make_image_gaussians(
            psf_model_init.gaussians(g2f.Channel.NONE),
            n_rows=self.config_boot.observation.n_rows,
            n_cols=self.config_boot.observation.n_cols,
        )
        return image.data

    def get_catalog(self) -> astropy.table.Table:
        return self.config_boot.catalog

    def get_psf_image(
        self, source: astropy.table.Row | Mapping[str, Any], config: CatalogPsfFitterConfig | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(source["id"])
        image = self.image
        config_obs = self.config_boot.observation
        return image + rng.standard_normal(image.shape) * np.sqrt(
            (image + config_obs.background) / config_obs.gain
        )

    def model_post_init(self, __context: Any) -> None:
        self.config_boot.freeze()


class CatalogExposureSourcesBootstrap(CatalogExposureSourcesABC, pydantic.BaseModel):
    """A CatalogExposure for bootstrap fitting of source catalogs."""

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    config_boot: CatalogSourceBootstrapConfig = pydantic.Field(
        title="A CatalogSourceBootstrapConfig to be frozen"
    )
    table_psf_fits: astropy.table.Table = pydantic.Field(title="PSF fit parameters for the catalog")

    @cached_property
    def channel(self) -> g2f.Channel:
        channel = g2f.Channel.get(self.config_boot.observation.band)
        return channel

    def get_catalog(self) -> astropy.table.Table:
        return self.config_boot.catalog

    def get_psf_model(self, params: Mapping[str, Any]) -> g2f.PsfModel:
        psf_model = self.psf_model_data.psf_model
        self.psf_model_data.init_psf_model(self.table_psf_fits[params["id"]])
        return psf_model

    def get_source_observation(self, source: Mapping[str, Any]) -> g2f.ObservationD:
        obs = self.config_boot.observation.make_observation()
        return obs

    def model_post_init(self, __context: Any) -> None:
        config_dict = self.table_psf_fits.meta["config"]
        config = CatalogPsfFitterConfig()
        set_config_from_dict(config, config_dict)
        config_data = CatalogPsfFitterConfigData(config=config)
        object.__setattr__(self, "psf_model_data", config_data)


class CatalogSourceFitterBootstrap(CatalogSourceFitterABC, pydantic.BaseModel):
    """A catalog fitter that bootstraps a single model.

    This fitter generates a different noisy image of the specified model for
    each row. The resulting catalog can be used to examine performance and
    statistics of the best-fit parameters.
    """

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float) -> tuple[float, float]:
        return float(cen_x), float(cen_y)

    def initialize_model(
        self,
        model: g2f.ModelD,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        centroid_pixel_offset: float = 0,
        **kwargs: Any,
    ) -> None:
        if values_init is None:
            values_init = {}
        min_x, max_x = np.inf, -np.inf
        min_y, max_y = np.inf, -np.inf
        for idx_obs, observation in enumerate(model.data):
            x_min = observation.image.coordsys.x_min
            min_x = min(min_x, x_min)
            max_x = max(max_x, x_min + observation.image.n_cols * observation.image.coordsys.dx1)
            y_min = observation.image.coordsys.y_min
            min_y = min(min_y, y_min)
            max_y = max(max_y, y_min + observation.image.n_rows * observation.image.coordsys.dy2)

        cen_x = (min_x + max_x) / 2.0
        cen_y = (min_y + max_y) / 2.0

        # One could consider setting initial values from estimated moments
        # here, like a real fitter would

        # An R_eff larger than the box size is problematic. This should also
        # stop unreasonable size proposals; a log10 transform isn't enough.
        limits_size = max(5.0, 2.0 * np.hypot(max_x - min_x, max_y - min_y))
        limits_xy = (1e-6, limits_size)
        params_limits_init = {
            g2f.CentroidXParameterD: (cen_x, (min_x, max_x)),
            g2f.CentroidYParameterD: (cen_y, (min_y, max_y)),
            g2f.ReffXParameterD: (None, limits_xy),
            g2f.ReffYParameterD: (None, limits_xy),
            g2f.SigmaXParameterD: (None, limits_xy),
            g2f.SigmaYParameterD: (None, limits_xy),
        }

        params_free = get_params_uniq(model, fixed=False)
        for param in params_free:
            value_init, limits_new = params_limits_init.get(type(param), (values_init.get(param), None))
            if value_init is not None:
                param.value = value_init
            if limits_new:
                param.limits.min = -np.inf
                param.limits.max = limits_new[1]
                param.limits.min = limits_new[0]

        # Should be done in get_source_observation, but it gets called first
        # ... and therefore does not have the initialization above
        # Also, this must be done per-iteration because PSF parameters vary
        model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.image)
        model.evaluate()

        # The offset is to keep the rng seed different from the PSF image seed
        # It doesn't really need to be so large but it's reasonably safe
        rng = np.random.default_rng(source["id"] + 10000000)

        for idx_obs, observation in enumerate(model.data):
            config_obs = catexps[idx_obs].config_boot.observation
            image_data, sigma_inv_data = observation.image.data, observation.sigma_inv.data
            output_data = model.outputs[idx_obs].data
            # numpy does not warn if these are different lengths, so assert
            assert image_data.size == output_data.size
            # This should definitely never fail
            assert image_data.size == sigma_inv_data.size
            image_data.flat = output_data.flat
            sigma_inv_data.flat = np.sqrt((image_data + config_obs.background) / config_obs.gain)
            image_data.flat += sigma_inv_data.flat * rng.standard_normal(image_data.size)
            sigma_inv_data.flat = (1.0 / sigma_inv_data).flat
            # This is mandatory because C++ construction does no initialization
            # (could instead initialize in get_source_observation)
            # TODO: Do some timings to see which is more efficient
            observation.mask_inv.data.flat = 1

    def validate_fit_inputs(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData = None,
        logger: logging.Logger = None,
        **kwargs: Any,
    ) -> None:
        errors = []
        for idx, catexp in enumerate(catexps):
            if not (
                (config_boot := getattr(catexp, "config_boot", None))
                and isinstance(config_boot, CatalogSourceBootstrapConfig)
            ):
                errors.append(
                    f"catexps[{idx=}] = {catexp} does not have a config_boot attr of type"
                    f"{CatalogSourceBootstrapConfig}"
                )
