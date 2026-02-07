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

__all__ = ["plot_model_singleband"]

import lsst.gauss2d.fit as g2f
import matplotlib.pyplot as plt
import numpy as np

from .types import Axes, Figure


def plot_model_singleband(
    model: g2f.ModelD,
    idx_obs: int,
    percentile_scaling: float = 98.0,
) -> tuple[Figure, Axes]:
    """Plot a model and its residuals compared to a single observation.

    Parameters
    ----------
    model
        The model to plot.
    idx_obs
        The index of the observation to plot.
    percentile_scaling
        The percentile of the non-nan data values to use as a maximum for
        arcsinh scaling.

    Returns
    -------
    fig
        The Figure for the grayscale plots.
    ax
        The Axes for the grayscale plots.
    """
    if not model.outputs:
        model.setup_evaluators(g2f.EvaluatorMode.image)
        model.evaluate()

    obs = model.data[idx_obs]
    band = obs.channel.name
    img_data = obs.image.data
    img_model = model.outputs[idx_obs].data

    value_max = np.nanpercentile(img_model, percentile_scaling)
    offset = 1 / np.nanmedian(obs.sigma_inv.data)

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0][0].imshow(np.arcsinh((img_data + offset) / value_max), cmap="gray", origin="lower")
    ax[0][0].tick_params(labelleft=False)
    ax[0][0].set_title(f"{band}-band Image")
    ax[0][1].imshow(np.arcsinh((img_model + offset) / value_max), cmap="gray", origin="lower")
    ax[0][1].tick_params(labelleft=False)
    ax[0][1].set_title(f"{band}-band Model")
    ax[1][0].imshow(np.arcsinh((img_data - img_model) / value_max), cmap="gray", origin="lower")
    ax[1][0].tick_params(labelleft=False)
    ax[1][0].set_title(f"{band}-band Residual")
    ax[1][1].imshow((img_data - img_model) * obs.sigma_inv.data, cmap="gray", origin="lower")
    ax[1][1].tick_params(labelleft=False)
    ax[1][1].set_title(f"{band}-band Residual")

    return fig, ax
