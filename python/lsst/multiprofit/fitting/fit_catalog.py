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

__all__ = ["CatalogExposureABC", "ColumnInfo", "CatalogFitterConfig"]

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

import astropy.units as u
import lsst.pex.config as pexConfig
import pydantic

from ..componentconfig import GaussianComponentConfig, SersicComponentConfig
from ..modeller import ModelFitConfig
from ..utils import frozen_arbitrary_allowed_config


class CatalogExposureABC(ABC):
    """Interface for catalog-exposure pairs."""

    # TODO: add get_exposure (with Any return type?)

    @abstractmethod
    def get_catalog(self) -> Iterable:
        """Return a row-iterable catalog covering an exposure."""


class ColumnInfo(pydantic.BaseModel):
    """Metadata for a column in a catalog."""

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    dtype: str = pydantic.Field(title="Column data type name (numpy or otherwise)")
    key: str = pydantic.Field(title="Column key (name)")
    description: str = pydantic.Field("", title="Column description")
    unit: u.UnitBase | None = pydantic.Field(None, title="Column unit (astropy)")


class CatalogFitterConfig(pexConfig.Config):
    """Configuration for generic MultiProFit fitting tasks."""

    column_id = pexConfig.Field[str](default="id", doc="Catalog index column key")
    compute_errors = pexConfig.ChoiceField[str](
        default="INV_HESSIAN_BESTFIT",
        doc="Whether/how to compute sqrt(variances) of each free parameter",
        allowed={
            "NONE": "no errors computed",
            "INV_HESSIAN": "inverse hessian using noisy image as data",
            "INV_HESSIAN_BESTFIT": "inverse hessian using best-fit model as data",
        },
    )
    compute_errors_from_jacobian = pexConfig.Field[bool](
        default=True,
        doc="Whether to estimate the Hessian from the Jacobian first, with finite differencing as a backup",
    )
    compute_errors_no_covar = pexConfig.Field[bool](
        default=True,
        doc="Whether to compute parameter errors independently, ignoring covariances",
    )
    config_fit = pexConfig.ConfigField[ModelFitConfig](default=ModelFitConfig, doc="Fitter configuration")
    fit_centroid = pexConfig.Field[bool](default=True, doc="Fit centroid parameters")
    fit_linear_init = pexConfig.Field[bool](default=True, doc="Fit linear parameters after initialization")
    fit_linear_final = pexConfig.Field[bool](default=True, doc="Fit linear parameters after optimization")
    flag_errors = pexConfig.DictField(
        default={},
        keytype=str,
        itemtype=str,
        doc="Flag column names to set, keyed by name of exception to catch",
    )
    naming_scheme = pexConfig.ChoiceField[str](
        doc="Naming scheme for column names",
        allowed={
            "default": "snake_case with {component_name}[_{band}]_{parameter}[_err]",
            "camel": "CamelCase with {component_name}[_{band}]_{parameter}[Err]",
            "lsst": "snake_case with [{band}_]{component_name}_{parameter}[Err]",
        },
        default="default",
    )
    prefix_column = pexConfig.Field[str](default="mpf_", doc="Column name prefix")
    suffix_error = pexConfig.Field[str](
        default="_err",
        doc="Default suffix for error columns. Can be overridden by naming_scheme.",
    )

    _format_flux = {
        "default": "{label}{band}_flux",
        "lsst": "{band}_{label}Flux",
        "camel": "{label}{band}Flux",
    }
    _key_cen = {"default": "_cen", "lsst": "_cen", "camel": "Cen"}
    _key_reff = {"default": f"_{SersicComponentConfig._size_label}", "lsst": "_reff", "camel": "Reff"}
    _key_rho = {"default": "_rho", "lsst": "_rho", "camel": "Rho"}
    _key_sigma = {"default": f"_{GaussianComponentConfig._size_label}", "lsst": "_sigma", "camel": "Sigma"}
    _key_sersicindex = {"default": "_sersic_index", "lsst": "sersic_index", "camel": "SersicIndex"}
    _suffix_dec = {"default": "_dec", "lsst": "_dec", "camel": "Dec"}
    _suffix_ra = {"default": "_ra", "lsst": "_ra", "camel": "Ra"}
    _suffix_x = {"default": "_x", "lsst": "_x", "camel": "X"}
    _suffix_y = {"default": "_y", "lsst": "_y", "camel": "Y"}

    def _get_label(self, format_name: str, values: dict[str, str]) -> str:
        """Get the label for part of a column name for a given format.

        Parameters
        ----------
        format_name
            The name of the format to get the label for.
        values
            The values of the name by format.

        Returns
        -------
        label
            The formatted label, if specified for that format, else the
            value for the default format.
        """
        return values.get(format_name, values["default"])

    def get_key_cen(self) -> str:
        """Get the key for centroid columns."""
        return self._get_label(self.naming_scheme, self._key_cen)

    def get_key_flux(self, band: str, label: str = "") -> str:
        """Get the key for a flux column.

        Parameters
        ----------
        band
            The band of the flux column.
        label
            A label for this flux, e.g. a component name.

        Returns
        -------
        key_flux
            The flux column key.
        """
        return self._get_label(self.naming_scheme, self._format_flux).format(band=band, label=label)

    def get_key_reff(self) -> str:
        """Get the key for Sersic effective radius columns."""
        return self._get_label(self.naming_scheme, self._key_reff)

    def get_key_rho(self) -> str:
        """Get the key for ellipse rho columns."""
        return self._get_label(self.naming_scheme, self._key_rho)

    def get_key_sersicindex(self) -> str:
        """Get the key for Sersic index columns."""
        return self._get_label(self.naming_scheme, self._key_sersicindex)

    def get_key_sigma(self) -> str:
        """Get the key for Gaussian sigma columns."""
        return self._get_label(self.naming_scheme, self._key_sigma)

    def get_key_size(self, label_size: str) -> str:
        """Get the key for a size column by its label.

        Parameters
        ----------
        label_size
            The label of the size, usually specified in a ComponentConfig.

        Returns
        -------
        key_size
            The size column key.
        """
        if label_size == GaussianComponentConfig._size_label:
            return self._get_label(self.naming_scheme, self._key_sigma)
        elif label_size == SersicComponentConfig._size_label:
            return self._get_label(self.naming_scheme, self._key_reff)
        return label_size

    def get_prefixed_label(self, label: str, prefix: str) -> str:
        """Get a prefixed label with redundant underscores removed.

        Parameters
        ----------
        label
            The label to format.
        prefix
            The prefix to prepend.

        Returns
        -------
        label_prefixed
            The prefixed label, with redundant underscores removed.
        """
        if label.startswith("_") and ((prefix == "") or (prefix[-1] == "_")):
            return f"{prefix}{label[1:]}"
        return f"{prefix}{label}"

    def get_suffix_dec(self) -> str:
        """Get the suffix for declination columns."""
        return self._get_label(self.naming_scheme, self._suffix_dec)

    def get_suffix_ra(self) -> str:
        """Get the suffix for right ascension columns."""
        return self._get_label(self.naming_scheme, self._suffix_ra)

    def get_suffix_x(self) -> str:
        """Get the suffix for x-axis columns."""
        return self._get_label(self.naming_scheme, self._suffix_x)

    def get_suffix_y(self) -> str:
        """Get the suffix for y-axis columns."""
        return self._get_label(self.naming_scheme, self._suffix_y)

    def schema(
        self,
        bands: list[str] | None = None,
    ) -> list[ColumnInfo]:
        """Return the schema as an ordered list of columns.

        Parameters
        ----------
        bands
            A list of band names to prefix band-dependent columns with.
            Band prefixes should not be used if None.

        Returns
        -------
        schema
            An ordered list of ColumnInfo instances.
        """
        schema = [
            ColumnInfo(key=self.column_id, dtype="i8"),
            ColumnInfo(key="n_iter", dtype="i4"),
            ColumnInfo(key="time_eval", dtype="f8", unit=u.s),
            ColumnInfo(key="time_fit", dtype="f8", unit=u.s),
            ColumnInfo(key="time_full", dtype="f8", unit=u.s),
            ColumnInfo(key="chisq_reduced", dtype="f8"),
            ColumnInfo(key="unknown_flag", dtype="bool"),
        ]
        schema.extend([ColumnInfo(key=key, dtype="bool") for key in self.flag_errors.keys()])
        # Subclasses should always write out centroids even if not fitting
        # They are helpful for reconstructing models
        return schema
