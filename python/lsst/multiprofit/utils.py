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
    "arbitrary_allowed_config",
    "frozen_arbitrary_allowed_config",
    "get_params_uniq",
    "set_config_from_dict",
]

import logging
from typing import Any

import lsst.gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import pydantic

_LOG = logging.getLogger(__name__)

# Pydantic config to allow arbitrary typed Fields.
arbitrary_allowed_config = pydantic.ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
)

# As above, but frozen
frozen_arbitrary_allowed_config = pydantic.ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    frozen=True,
)


def get_params_uniq(parametric: g2f.Parametric, **kwargs: Any) -> list[g2f.ParameterD]:
    """Get a sorted set of parameters matching a filter.

    Parameters
    ----------
    parametric
        The parametric object to get parameters from.
    **kwargs
        Keyword arguments to pass to g2f.ParamFilter.

    Returns
    -------
    params
        The unique parameters from the parametric object matching the filter.
    """
    params = parametric.parameters(paramfilter=g2f.ParamFilter(**kwargs))
    # This should always return the same list as:
    # list({p: None for p in }.keys())
    return g2f.params_unique(params)


def set_config_from_dict(
    config: pexConfig.Config | pexConfig.dictField.Dict | pexConfig.configDictField.ConfigDict | dict,
    overrides: dict[str, Any],
) -> None:
    """Set `lsst.pex.config` params from a dict.

    Parameters
    ----------
    config
        A config, dictField or configDictField object.
    overrides
        A dict of key-value pairs to override in the config.
    """
    is_config_dict = hasattr(config, "__getitem__")
    if is_config_dict:
        keys = tuple(config.keys())
        for key in keys:
            if key not in overrides:
                del config[key]
    for key, value in overrides.items():
        if isinstance(value, dict):
            attr = config[key] if is_config_dict else getattr(config, key)
            set_config_from_dict(attr, value)
        else:
            try:
                if is_config_dict:
                    config[key] = value
                else:
                    setattr(config, key, value)
            # TODO: Check if a narrower Exception type is better in DM-45577
            except Exception as e:
                _LOG.warning(e)
