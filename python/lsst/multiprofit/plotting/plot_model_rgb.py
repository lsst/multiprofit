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

__all__ = ["plot_model_rgb"]

import math
from typing import Any

import astropy.visualization as apVis
import lsst.gauss2d as g2
import lsst.gauss2d.fit as g2f
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .types import Axes, Figure


def plot_model_rgb(
    model: g2f.ModelD | None,
    weights: dict[str, float] | None = None,
    high_sn_threshold: float | None = None,
    plot_singleband: bool = True,
    plot_chi_hist: bool = True,
    chi_max: float = 5.0,
    rgb_min_auto: bool = False,
    rgb_stretch_auto: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Axes, Figure, Axes, np.ndarray]:
    """Plot RGB images of a model, its data and residuals thereof.

    Parameters
    ----------
    model
        The model to plot. If None, a dict of observations by band may be
        passed as an additional kwarg; otherwise, only the data will be
        plotted.
    weights
        Linear weights to multiply each band's image by. The default is a
        weight of one for each band.
    high_sn_threshold
        If non-None and given a model, this will return an image with the
        pixels having a model S/N above this threshold in every band.
    plot_singleband
        Whether to make grayscale plots for each band.
    plot_chi_hist
        Whether to plot histograms of the chi (scaled residual) values.
    chi_max
        The maximum absolute value of chi in residual plots. Values of 3-5 are
        suitable for good models while inadequate ones may need larger values.
    rgb_min_auto
        Whether to set the minimum in RGB plots automatically. Cannot supply
        minimum in kwargs if enabled.
    rgb_stretch_auto
        Whether to set the stretch in RGB plots automatically. Cannot supply
        stretch in kwargs if enabled.
    **kwargs
        Additional keyword arguments to pass to make_lupton_rgb when creating
        RGB images.

    Returns
    -------
    fig_rgb
        The Figure for the RGB plots.
    ax_rgb
        The Axes for the RGB plots.
    fig_gs
        The Figure for the grayscale plots.
    ax_gs
        The Axes for the grayscale plots.
    mask_inv_highsn
        The inverse mask (1=selected) if high_sn_threshold was specified.
    """
    if rgb_min_auto and "minimum" in kwargs:
        raise ValueError(f"Cannot set rgb_min_auto and pass {kwargs['minimum']=}")
    if rgb_stretch_auto and "stretch" in kwargs:
        raise ValueError(f"Cannot set rgb_stretch_auto and pass {kwargs['stretch']=}")
    if not (chi_max > 0):
        raise ValueError(f"{chi_max=} not >0")
    if weights is None:
        if model is None:
            weights = {band: 1.0 for band in kwargs["observations"].keys()}
        else:
            bands_set = set()
            bands = []
            weights = {}
            for obs in model.data:
                band = obs.channel.name
                if band not in bands_set:
                    bands_set.add(band)
                    bands.append(band)
                    weights[band] = 1.0

    n_data = len(model.data)
    has_model = model is not None
    observations = {}
    models = {}

    if has_model and (n_data < 3):
        if n_data == 1:
            # pretend this is three bands
            obs, output_data = model.data[0], model.outputs[0].data
            band = obs.channel.name
            weights = {}
            for idx in range(1, 4):
                key = f"{band}{idx}"
                weights[key] = 1.0
                observations[key] = obs
                models[key] = output_data
        elif n_data == 2:
            raise NotImplementedError("RGB images for two-band data are not supported (yet)")

    bands = tuple(weights.keys())
    band_str = ",".join(bands)
    n_bands = len(bands)

    if has_model and (not model.outputs or any([output is None for output in model.outputs])):
        model.setup_evaluators(g2f.EvaluatorMode.image)
        model.evaluate()

    if not has_model:
        if plot_chi_hist:
            raise ValueError("Cannot plot chi histograms without a model")
        obs_kwarg = kwargs.pop("observations")
        for band in bands:
            observations[band] = obs_kwarg[band]

    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    coordsys_last = None
    if has_model and n_data >= 3:
        for obs, output in zip(model.data, model.outputs):
            band = obs.channel.name
            if band in bands:
                if band in observations:
                    raise ValueError(f"Cannot plot {model=} because {band=} has multiple observations")
                observations[band] = obs
                models[band] = output.data

    for band, obs in observations.items():
        coordsys = obs.image.coordsys
        if coordsys:
            coordsys_last = coordsys
            x_min = int(round(min(x_min, coordsys.x_min), 0))
            x_max = int(round(max(x_max, coordsys.x_min + obs.image.n_cols), 0))
            y_min = int(round(min(y_min, coordsys.y_min), 0))
            y_max = int(round(max(y_max, coordsys.y_min + obs.image.n_rows), 0))
        elif coordsys_last is not None:
            raise ValueError(
                f"coordinate system for {band=} is None but last was not; they must either "
                f"all be None or all non-None"
            )

    if coordsys_last:
        shape_new = (y_max - y_min, x_max - x_min)
        keys = ("image", "mask_inv", "sigma_inv")
        if has_model:
            keys += ("model",)
        for band, obs in observations.items():
            coordsys = obs.image.coordsys
            x_min_c = int(round(coordsys.x_min, 0)) - x_min
            y_min_c = int(round(coordsys.y_min, 0)) - y_min
            x_min_o, x_max_o = x_min_c, x_min_c + obs.image.n_cols
            y_min_o, y_max_o = y_min_c, y_min_c + obs.image.n_rows
            if x_min_o or x_max_o or y_min_o or y_max_o:
                # zero-pad the relevant images into a new observation
                data_new = {}
                for key in keys:
                    img = np.zeros(shape_new)
                    img[y_min_o:y_max_o, x_min_o:x_max_o] = (
                        models[band] if (key == "model") else getattr(obs, key).data
                    )
                    if key == "model":
                        models[band] = img
                    else:
                        data_new[key] = (g2.ImageB if (key == "mask_inv") else g2.ImageD)(img)
                observations[band] = g2f.ObservationD(channel=obs.channel, **data_new)

    extent = (x_min, x_max, y_min, y_max)

    images_data = [None] * 3
    images_data_unweighted = [None] * 3 if has_model else None
    images_model = [None] * 3 if has_model else None
    images_model_unweighted = [None] * 3 if has_model else None
    images_sigma_inv = [None] * 3 if has_model else None
    masks_inv_rgb = [None] * 3

    weights_channel = np.linspace(0, 3, len(weights) + 1)[1:]
    idx_channel = 0
    weight_channel = 0

    def add_if_not_none(array: np.ndarray, index: int, arg: float | None) -> None:
        if array[index] is not None:
            array[index] += arg
        else:
            array[index] = arg

    chis_unweighted = {}

    for idx_band, (band, weight) in enumerate(weights.items()):
        observation = observations[band]
        if has_model:
            model_band = models[band]
            sigma_inv = observation.sigma_inv.data
            sigma_inv_good = sigma_inv > 0
            variance_band = np.empty_like(sigma_inv)
            variance_band[sigma_inv_good] = sigma_inv[sigma_inv_good] ** -2
            variance_band[~sigma_inv_good] = np.nan
            if plot_chi_hist:
                chi_good = (sigma_inv > 0) & np.isfinite(sigma_inv)
                chi_unweighted = (observation.image.data[chi_good] - model_band[chi_good]) * sigma_inv[
                    chi_good
                ]
                chis_unweighted[band] = chi_unweighted
        weight_channel_new = weights_channel[idx_band]
        idx_channel_new = int(weight_channel_new // 1)
        if idx_channel_new == idx_channel:
            weight_low = weight_channel_new - weight_channel
            weight_high = 0.0
        else:
            weight_low = idx_channel_new - weight_channel
            weight_high = weight_channel_new - idx_channel_new
        assert weight_high >= 0
        assert weight_low >= 0
        if weight_low > 0:
            data_band = observation.image.data * weight_low
            add_if_not_none(images_data, idx_channel, data_band * weight)
            add_if_not_none(masks_inv_rgb, idx_channel, observation.mask_inv.data * weight_low)
            if has_model:
                add_if_not_none(images_data_unweighted, idx_channel, data_band)
                model_sub = model_band * weight_low
                add_if_not_none(images_model, idx_channel, model_sub * weight)
                add_if_not_none(images_model_unweighted, idx_channel, model_sub)
                add_if_not_none(images_sigma_inv, idx_channel, variance_band * weight_low)
        if (idx_channel_new != idx_channel) and (weight_high > 0):
            data_band = observation.image.data * weight_high
            images_data[idx_channel_new] = data_band * weight
            masks_inv_rgb[idx_channel_new] = observation.mask_inv.data * weight_low
            if has_model:
                images_model_unweighted[idx_channel_new] = data_band
                model_sub = model_band * weight_high
                images_model[idx_channel_new] = model_sub * weight
                images_model_unweighted[idx_channel_new] = model_sub
                images_sigma_inv[idx_channel_new] = variance_band * weight_high
        weight_channel = weight_channel_new
        idx_channel = idx_channel_new

    # convert variance to 1/sigma
    if has_model:
        for idx in range(3):
            images_sigma_inv[idx] = 1 / np.sqrt(images_sigma_inv[idx])

    if rgb_min_auto or rgb_stretch_auto:
        # The model won't have negative pixels, so it ought to stretch fine
        # the max/stretch is not as important anyway
        rgb_min, rgb_max = np.nanpercentile(
            np.concatenate([image[mask_inv != 0] for mask_inv, image in zip(masks_inv_rgb, images_data)]),
            (5, 95),
        )
        if rgb_min_auto:
            kwargs["minimum"] = rgb_min
        if rgb_stretch_auto:
            kwargs["stretch"] = 2 * (rgb_max - rgb_min)

    img_rgb = apVis.make_lupton_rgb(*images_data, **kwargs)
    if has_model:
        img_model_rgb = apVis.make_lupton_rgb(*images_model, **kwargs)
    aspect = np.clip((y_max - y_min) / (x_max - x_min), 0.25, 4)

    n_rows = 1 + has_model
    n_cols_gs = 1 + has_model
    n_cols_rgb = 1 + has_model * (1 + plot_chi_hist)
    figsize_y = 8 * n_rows * aspect

    fig_rgb, ax_rgb = plt.subplots(nrows=n_rows, ncols=n_cols_rgb, figsize=(8 * n_cols_rgb, figsize_y))
    fig_gs, ax_gs = (
        (None, None)
        if not plot_singleband
        else plt.subplots(
            nrows=n_bands,
            ncols=n_cols_gs,
            figsize=(8 * n_cols_gs, 8 * aspect * n_bands),
        )
    )
    (ax_rgb[0][0] if has_model else ax_rgb).imshow(img_rgb, extent=extent, origin="lower")
    (ax_rgb[0][0] if has_model else ax_rgb).set_title("Data")
    if has_model:
        ax_rgb[1][0].imshow(img_model_rgb, extent=extent, origin="lower")
        ax_rgb[1][0].set_title(f"Model ({band_str})")

    masks_inv = {}
    # Create a mask of high-sn pixels (based on the model)
    mask_inv_highsn = np.ones(img_rgb.shape[:1], dtype="bool") if high_sn_threshold else None

    for idx, band in enumerate(bands):
        obs = observations[band]
        mask_inv = obs.mask_inv.data
        masks_inv[band] = mask_inv
        img_data = obs.image.data
        img_sigma_inv = obs.sigma_inv.data
        if plot_singleband:
            if has_model:
                img_model = models[band]
                if mask_inv_highsn:
                    mask_inv_highsn *= (img_model * np.nanmedian(img_sigma_inv)) > high_sn_threshold
                residual = (img_data - img_model) * mask_inv
                value_max = np.nanpercentile(np.abs(residual), 98)
                ax_gs[idx][0].imshow(residual, cmap="gray", vmin=-value_max, vmax=value_max, origin="lower")
                ax_gs[idx][0].tick_params(labelleft=False)
                ax_gs[idx][0].set_title(f"{band}-band Residual (abs.)")
                ax_gs[idx][1].imshow(
                    np.clip(residual * img_sigma_inv, -chi_max, chi_max),
                    cmap="gray",
                    origin="lower",
                )
                ax_gs[idx][1].tick_params(labelleft=False)
                ax_gs[idx][1].set_title(f"{band}-band Residual (chi, +/- {chi_max:.2f})")
            else:
                ax_gs[idx].imshow(img_data * mask_inv * (img_sigma_inv > 0), cmap="gray", origin="lower")
                ax_gs[idx].set_title(band)

    if has_model:
        # TODO: Draw masks in each channel? or draw the combined mask, like:
        # mask_inv_all = np.prod(list(masks_inv.values()), axis=0)
        residuals = [(images_model_unweighted[idx] - images_data_unweighted[idx]) for idx in range(3)]
        resid_max = np.nanpercentile(
            np.abs(np.concatenate([residual[np.isfinite(residual)] for residual in residuals])), 98
        )

        # This may or may not be equivalent to make_lupton_rgb
        # I just can't figure out how to get that scaled so zero = 50% gray
        stretch = 3
        residual_rgb = np.stack(
            [np.arcsinh(np.clip(residuals[idx], -resid_max, resid_max) * stretch) for idx in range(3)],
            axis=-1,
        )
        residual_rgb /= 2 * np.arcsinh(resid_max * stretch)
        residual_rgb += 0.5

        ax_rgb[0][1].imshow(residual_rgb, origin="lower")
        ax_rgb[0][1].set_title(f"Residual (abs., += {resid_max:.3e})")
        ax_rgb[0][1].tick_params(labelleft=False)

        if plot_chi_hist:
            cmap = mpl.colormaps["coolwarm"]
            residuals_rgb = np.concatenate(tuple(chis_unweighted.values()))
            residuals_abs = np.abs(residuals_rgb)
            n_resid = len(residuals_abs)
            chi_max = 5 + 2.5 * (
                (np.sum(residuals_abs > 5) / n_resid > 0.1) + (np.sum(residuals_abs > 7.5) / n_resid > 0.1)
            )
            n_bins = int(math.ceil(np.clip(n_resid / 50, 2, 20)) * chi_max)
            # ax_rgb[0][2].set_adjustable('box')
            ax_rgb[0][2].hist(
                np.clip(residuals_rgb, -chi_max, chi_max),
                bins=n_bins,
                histtype="step",
                label="all",
            )
            band_colors = cmap(np.linspace(0, 1, n_bands))
            for band, band_color in zip(bands, band_colors):
                ax_rgb[0][2].hist(
                    np.clip(residuals_rgb, -chi_max, chi_max),
                    bins=n_bins,
                    histtype="step",
                    label=band,
                )
                ax_rgb[0][2].legend()

        # TODO: Plot unscaled residuals in ax_rgb[1][2]? It's unused now.
        residual_rgb = np.stack(
            [
                (np.clip(residuals[idx] * images_sigma_inv[idx], -chi_max, chi_max) + chi_max) / (2 * chi_max)
                for idx in range(3)
            ],
            axis=-1,
        )

        ax_rgb[1][1].imshow(residual_rgb, origin="lower")
        ax_rgb[1][1].set_title(f"Residual (chi, +/- {chi_max:.2f})")
        ax_rgb[1][1].tick_params(labelleft=False)

    return fig_rgb, ax_rgb, fig_gs, ax_gs, mask_inv_highsn
