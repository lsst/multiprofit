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

__all__ = ["CatalogError", "NoDataError", "PsfRebuildFitFlagError"]

from abc import abstractmethod


class CatalogError(RuntimeError):
    """RuntimeError that can be caught and flagged in a column."""

    @classmethod
    @abstractmethod
    def column_name(cls) -> str:
        """Return the standard column name for this error."""


class NoDataError(CatalogError):
    """RuntimeError for when there is no data to fit."""

    @classmethod
    @abstractmethod
    def column_name(cls) -> str:
        return "no_data_flag"


class PsfRebuildFitFlagError(RuntimeError):
    """RuntimeError for when a PSF can't be rebuilt because the fit failed."""

    @classmethod
    def column_name(cls) -> str:
        return "psf_fit_flag"
