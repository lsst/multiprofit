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

import lsst.gauss2d.fit as g2f
from lsst.multiprofit import (
    ComponentGroupConfig,
    GaussianComponentConfig,
    ModelConfig,
    ModelFitConfig,
    SourceConfig,
)
from lsst.multiprofit.fitting.fit_source import CatalogSourceFitterConfig, CatalogSourceFitterConfigData
from lsst.multiprofit.utils import get_params_uniq
import pytest


@pytest.fixture(scope="module")
def channels() -> tuple[g2f.Channel]:
    channels = tuple(g2f.Channel.get(band) for band in ("R", "G", "B"))
    return channels


@pytest.fixture(scope="module")
def fitter_config() -> CatalogSourceFitterConfig:
    config = CatalogSourceFitterConfig(
        config_fit=ModelFitConfig(),
        config_model=ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            components_gauss=({"gauss": GaussianComponentConfig()}),
                        )
                    }
                ),
            },
        ),
        fit_psmodel_final=True,
    )
    return config


@pytest.fixture(scope="module")
def fitter_config_data(channels, fitter_config) -> CatalogSourceFitterConfigData:
    config_data = CatalogSourceFitterConfigData(channels=channels, config=fitter_config)
    return config_data


def test_fitter_config_data(fitter_config_data):
    parameters = fitter_config_data.parameters
    assert len(parameters) > 0
    sources, priors = fitter_config_data.sources_priors
    same = (p1 is p2 for p1, p2 in zip(parameters.values(), get_params_uniq(sources[0], fixed=False)))
    assert all(same)
