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

__all__ = ["Interpolator", "plot_sersicmix_interp"]

from typing import Any, Type, TypeAlias

import lsst.gauss2d.fit as g2f
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .types import FigureAxes

Interpolator: TypeAlias = g2f.SersicMixInterpolator | tuple[Type, dict[str, Any]]


def plot_sersicmix_interp(
    interps: dict[str, tuple[Interpolator, str | tuple]], n_ser: np.ndarray, **kwargs: Any
) -> FigureAxes:
    """Plot Gaussian mixture Sersic profile interpolated values.

    Parameters
    ----------
    interps
        Dict of interpolators by name.
    n_ser
        Array of Sersic index values to plot interpolated quantities for.
    **kwargs
        Keyword arguments to pass to matplotlib.pyplot.subplots.

    Returns
    -------
    figure
        The resulting figure.
    """
    orders = {
        name: interp.order
        for name, (interp, _) in interps.items()
        if isinstance(interp, g2f.SersicMixInterpolator)
    }
    order = set(orders.values())
    if not len(order) == 1:
        raise ValueError(f"len(set({orders})) != 1; all interpolators must have the same order")
    order = tuple(order)[0]

    cmap = mpl.cm.get_cmap("tab20b")
    colors_ord = [None] * order
    for i_ord in range(order):
        colors_ord[i_ord] = cmap(i_ord / (order - 1.0))

    n_ser_min = np.min(n_ser)
    n_ser_max = np.max(n_ser)
    knots = g2f.sersic_mix_knots(order=order)
    n_knots = len(knots)
    integrals_knots = np.empty((n_knots, order))
    sigmas_knots = np.empty((n_knots, order))
    n_ser_knots = np.empty(n_knots)

    i_knot_first = None
    i_knot_last = n_knots
    for i_knot, knot in enumerate(knots):
        if i_knot_first is None:
            if knot.sersicindex > n_ser_min:
                i_knot_first = i_knot
            else:
                continue
        if knot.sersicindex > n_ser_max:
            i_knot_last = i_knot
            break
        n_ser_knots[i_knot] = knot.sersicindex
        for i_ord in range(order):
            values = knot.values[i_ord]
            integrals_knots[i_knot, i_ord] = values.integral
            sigmas_knots[i_knot, i_ord] = values.sigma
    range_knots = range(i_knot_first, i_knot_last)
    integrals_knots = integrals_knots[range_knots, :]
    sigmas_knots = sigmas_knots[range_knots, :]
    n_ser_knots = n_ser_knots[range_knots]

    n_values = len(n_ser)
    integrals, dintegrals, sigmas, dsigmas = (
        {name: np.empty((n_values, order)) for name in interps} for _ in range(4)
    )

    for name, (interp, _) in interps.items():
        if not isinstance(interp, g2f.SersicMixInterpolator):
            kwargs = interp[1] if interp[1] is not None else {}
            interp = interp[0]
            x = [knot.sersicindex for knot in knots]
            for i_ord in range(order):
                integrals_i = np.empty(n_knots, dtype=float)
                sigmas_i = np.empty(n_knots, dtype=float)
                for i_knot, knot in enumerate(knots):
                    integrals_i[i_knot] = knot.values[i_ord].integral
                    sigmas_i[i_knot] = knot.values[i_ord].sigma
                interp_int = interp(x, integrals_i, **kwargs)
                dinterp_int = interp_int.derivative()
                interp_sigma = interp(x, sigmas_i, **kwargs)
                dinterp_sigma = interp_sigma.derivative()
                for i_val, value in enumerate(n_ser):
                    integrals[name][i_val, i_ord] = interp_int(value)
                    sigmas[name][i_val, i_ord] = interp_sigma(value)
                    dintegrals[name][i_val, i_ord] = dinterp_int(value)
                    dsigmas[name][i_val, i_ord] = dinterp_sigma(value)

    for i_val, value in enumerate(n_ser):
        for name, (interp, _) in interps.items():
            if isinstance(interp, g2f.SersicMixInterpolator):
                values = interp.integralsizes(value)
                derivs = interp.integralsizes_derivs(value)
                for i_ord in range(order):
                    integrals[name][i_val, i_ord] = values[i_ord].integral
                    sigmas[name][i_val, i_ord] = values[i_ord].sigma
                    dintegrals[name][i_val, i_ord] = derivs[i_ord].integral
                    dsigmas[name][i_val, i_ord] = derivs[i_ord].sigma

    fig, axes = plt.subplots(2, 2, **kwargs)
    for idx_row, (yv, yd, yk, y_label) in (
        (0, (integrals, dintegrals, integrals_knots, "integral")),
        (1, (sigmas, dsigmas, sigmas_knots, "sigma")),
    ):
        is_label_row = idx_row == 1
        for idx_col, y_i, y_prefix in ((0, yv, ""), (1, yd, "d")):
            is_label_col = idx_col == 0
            make_label = is_label_col and is_label_row
            axis = axes[idx_row, idx_col]
            if is_label_col:
                for i_ord in range(order):
                    axis.plot(
                        n_ser_knots,
                        yk[:, i_ord],
                        "kx",
                        label="knots" if make_label and (i_ord == 0) else None,
                    )
            for name, (_, lstyle) in interps.items():
                for i_ord in range(order):
                    label = f"{name}" if make_label and (i_ord == 0) else None
                    axis.plot(n_ser, y_i[name][:, i_ord], c=colors_ord[i_ord], label=label, linestyle=lstyle)
                axis.set_xlim((n_ser_min, n_ser_max))
                axis.set_ylabel(f"{y_prefix}{y_label}")
            if make_label:
                axis.legend(loc="upper left")
    return fig, axes
