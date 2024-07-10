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

__all__ = ["plot_catalog_bootstrap"]

from collections import defaultdict
from typing import Any, Iterable

import astropy.table
import matplotlib.pyplot as plt
import numpy as np

ln10 = np.log(10)


def plot_catalog_bootstrap(
    catalog_bootstrap: astropy.table.Table,
    n_bins: int | None = None,
    paramvals_ref: Iterable[np.ndarray] | None = None,
    plot_total_fluxes: bool = False,
    plot_colors: bool = False,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a bootstrap catalog for a single source model.

    Parameters
    ----------
    catalog_bootstrap
        A bootstrap catalog, as returned by
        `multiprofit.fit_bootstrap_model.CatalogSourceFitterBootstrap`.
    n_bins
        The number of bins for parameter value histograms. Default
        is sqrt(N) with a minimum of 10.
    paramvals_ref
        Reference parameter values to plot, if any.
    plot_total_fluxes
        Whether to plot total fluxes, not just component.
    plot_colors
        Whether to plot colors in addition to fluxes.
    **kwargs
        Keyword arguments to pass to matplotlib hist calls.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis handles, as returned by plt.subplots.
    """
    n_sources = len(catalog_bootstrap)
    if n_bins is None:
        n_bins = np.max([int(np.ceil(np.sqrt(n_sources))), 10])

    config = catalog_bootstrap.meta["config"]
    prefix = config["prefix_column"]
    suffix_err = "_err"

    # TODO: There are probably better ways of doing this
    colnames_err = [col for col in catalog_bootstrap.colnames if col.endswith(suffix_err)]
    colnames_meas = [col[:-4] for col in colnames_err]
    n_params_init = len(colnames_meas)
    if paramvals_ref is not None and (len(paramvals_ref) != n_params_init):
        raise ValueError(f"{len(paramvals_ref)=} != {n_params_init=}")

    results_good = catalog_bootstrap[catalog_bootstrap["mpf_n_iter"] > 0]

    if plot_total_fluxes or plot_colors:
        if paramvals_ref:
            paramvals_ref = {
                colname: paramval_ref for colname, paramval_ref in zip(colnames_meas, paramvals_ref)
            }
        results_dict = {}
        for colname_meas, colname_err in zip(colnames_meas, colnames_err):
            results_dict[colname_meas] = results_good[colname_meas]
            results_dict[colname_err] = results_good[colname_err]

        colnames_flux = [colname for colname in colnames_meas if colname.endswith("_flux")]

        colnames_flux_band = defaultdict(list)
        colnames_flux_comp = defaultdict(list)

        for colname in colnames_flux:
            colname_short = colname.partition(prefix)[-1]
            comp, band = colname_short[:-5].split("_")
            colnames_flux_band[band].append(colname)
            colnames_flux_comp[comp].append(colname)

        band_prev = None
        for band, colnames_band in colnames_flux_band.items():
            for suffix, target in (("", colnames_meas), ("_err", colnames_err)):
                is_err = suffix == "_err"
                colname_flux = f"{prefix}{band}_flux{suffix}"
                total = np.sum(
                    [results_good[f"{colname}{suffix}"] ** (1 + is_err) for colname in colnames_band], axis=0
                )
                if is_err:
                    total = np.sqrt(total)
                elif paramvals_ref:
                    paramvals_ref[colname_flux] = sum((paramvals_ref[colname] for colname in colnames_band))
                results_dict[colname_flux] = total
                if plot_total_fluxes:
                    target.append(colname_flux)

            if band_prev:
                flux_prev, flux = (results_dict[f"{prefix}{b}_flux"] for b in (band_prev, band))
                mag_prev, mag = (-2.5 * np.log10(flux_b) for flux_b in (flux_prev, flux))
                mag_err_prev, mag_err = (
                    results_dict[f"{prefix}{b}_flux{suffix_err}"] / (-0.4 * flux_b * ln10)
                    for b, flux_b in ((band_prev, flux_prev), (band, flux))
                )
                colname_color = f"{prefix}{band_prev}-{band}_flux"
                colnames_meas.append(colname_color)
                colnames_err.append(f"{colname_color}{suffix_err}")

                results_dict[colname_color] = mag_prev - mag
                results_dict[f"{colname_color}{suffix_err}"] = 2.5 / ln10 * np.hypot(mag_err, mag_err_prev)
                if paramvals_ref:
                    mag_prev_ref, mag_ref = (
                        -2.5 * np.log10(paramvals_ref[f"{prefix}{b}_flux"]) for b in (band_prev, band)
                    )
                    paramvals_ref[colname_color] = mag_prev_ref - mag_ref

            band_prev = band

        results_good = results_dict
        if paramvals_ref:
            paramvals_ref = tuple(paramvals_ref.values())

    n_colnames = len(colnames_err)
    n_cols = 3
    n_rows = int(np.ceil(n_colnames / n_cols))

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, constrained_layout=True)
    idx_row, idx_col = 0, 0

    for idx_colname in range(n_colnames):
        colname_meas = colnames_meas[idx_colname]
        colname_short = colname_meas.partition(prefix)[-1]
        values = results_good[colname_meas]
        errors = results_good[colnames_err[idx_colname]]
        median = np.median(values)
        std = np.std(values)

        median_err = np.median(errors)

        axis = ax[idx_row][idx_col]
        axis.hist(values, bins=n_bins, color="b", label="fit values", **kwargs)

        label = "median +/- stddev"
        for offset in (-std, 0, std):
            axis.axvline(median + offset, label=label, color="k")
            label = None
        if paramvals_ref is not None:
            value_ref = paramvals_ref[idx_colname]
            label_value = f" {value_ref=:.3e} bias={median - value_ref:.3e}"
            axis.axvline(value_ref, label="reference", color="k", linestyle="--")
        else:
            label_value = f" {median=:.3e}"
        axis.hist(median + errors, bins=n_bins, color="r", label="median + error", **kwargs)
        axis.set_title(f"{colname_short} {std=:.3e} vs {median_err=:.3e}")
        axis.set_xlabel(f"{colname_short} {label_value}")
        axis.legend()

        idx_col += 1

        if idx_col == n_cols:
            idx_row += 1
            idx_col = 0

    return fig, ax
