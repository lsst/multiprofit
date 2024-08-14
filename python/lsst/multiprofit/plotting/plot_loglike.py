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

__all__ = ["plot_loglike"]

import itertools

import lsst.gauss2d.fit as g2f
import matplotlib.pyplot as plt
import numpy as np

from ..utils import get_params_uniq
from .config import linestyles_default
from .errorvalues import ErrorValues


def plot_loglike(
    model: g2f.ModelD,
    params: list[g2f.ParameterD] | None = None,
    n_values: int = 15,
    errors: dict[str, ErrorValues] | None = None,
    values_reference: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the loglikehood and derivatives vs free parameter values around
       best-fit values.

    Parameters
    ----------
    model
        The model to evaluate.
    params
        Free parameters to plot marginal loglikelihood for.
    n_values
        The number of evaluations to make on either side of each param value.
    errors
        A dict keyed by label of uncertainties to plot. Values must be the same
        length as `params`.
    values_reference
        Reference values to plot (e.g. true parameter values). Must be the same
        length as `params`.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis handles, as returned by plt.subplots.
    """
    if errors is None:
        errors = {}
    loglike_grads = np.array(model.compute_loglike_grad())
    loglike_init = np.array(model.evaluate())

    if params is None:
        params = tuple(get_params_uniq(model, fixed=False))

    n_params = len(params)

    if values_reference is not None and len(values_reference) != n_params:
        raise ValueError(f"{len(values_reference)=} != {n_params=}")

    n_rows = n_params
    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(10, 3 * n_rows))
    axes = [ax] if (n_rows == 1) else ax

    n_loglikes = len(loglike_init)
    labels = [channel.name for channel in model.data.channels]
    labels.extend(["prior", "total"])

    for param in params:
        param.fixed = True

    for row, param in enumerate(params):
        value_init = param.value
        param.fixed = False
        values = [value_init]
        loglikes = [loglike_init * 0]
        dlls = [loglike_grads[row]]

        diff_init = 1e-4 * np.sign(loglike_grads[row])
        diff = diff_init

        # TODO: This entire scheme should be improved/replaced
        # It sometimes takes excessively large steps
        # Option: Try to fit a curve once there are a couple of points
        # on each side of the peak
        idx_prev = -1
        for idx in range(2 * n_values):
            try:
                param.value_transformed += diff
                loglikes_new = np.array(model.evaluate()) - loglike_init
                dloglike_actual = np.sum(loglikes_new) - np.sum(loglikes[idx_prev])
                values.append(param.value)
                loglikes.append(loglikes_new)
                dloglike_actual_abs = np.abs(dloglike_actual)
                if dloglike_actual_abs > 1:
                    diff /= dloglike_actual_abs
                elif dloglike_actual_abs < 0.5:
                    diff /= np.clip(dloglike_actual_abs, 0.2, 0.5)
                dlls.append(model.compute_loglike_grad()[0])
                if idx == n_values:
                    diff = -diff_init
                    param.value = value_init
                    idx_prev = 0
                else:
                    idx_prev = -1
            except RuntimeError:
                break
        param.value = value_init
        param.fixed = True

        subplot = axes[row][0]
        sorted = np.argsort(values)
        values = np.array(values)[sorted]
        loglikes = [loglikes[idx] for idx in sorted]
        dlls = np.array(dlls)[sorted]

        for idx in range(n_loglikes):
            subplot.plot(values, [loglike[idx] for loglike in loglikes], label=labels[idx])
        subplot.plot(values, np.sum(loglikes, axis=1), label=labels[-1])
        vline_kwargs = dict(ymin=np.min(loglikes) - 1, ymax=np.max(loglikes) + 1, color="k")
        subplot.vlines(value_init, **vline_kwargs)

        suffix = f" {param.label}" if param.label else ""
        subplot.legend()
        subplot.set_title(f"{param.name}{suffix}")
        subplot.set_ylabel("loglike")
        subplot.set_ylim(vline_kwargs["ymin"], vline_kwargs["ymax"])

        subplot = axes[row][1]
        subplot.plot(values, dlls)
        subplot.axhline(0, color="k")
        subplot.set_ylabel("dloglike/dx")

        vline_kwargs = dict(ymin=np.min(dlls), ymax=np.max(dlls))
        subplot.vlines(value_init, **vline_kwargs, color="k", label="fit")
        if values_reference is not None:
            subplot.vlines(values_reference[row], **vline_kwargs, color="b", label="ref")

        cycler_linestyle = itertools.cycle(linestyles_default)
        for name_error, valerr in errors.items():
            linestyle = valerr.kwargs_plot.pop("linestyle", next(cycler_linestyle))
            for idx_ax in range(2):
                axes[row][idx_ax].vlines(
                    [value_init - valerr.values[row], value_init + valerr.values[row]],
                    linestyles=[linestyle, linestyle],
                    label=name_error if (idx_ax == 1) else None,
                    **valerr.kwargs_plot,
                    **vline_kwargs,
                )
        subplot.legend()

    for param in params:
        param.fixed = False

    return fig, ax
