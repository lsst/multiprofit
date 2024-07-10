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
    "abs_mag_sol_lsst",
    "bands_weights_lsst",
]

# See Wilmer 2018 (https://iopscience.iop.org/article/10.3847/1538-4365/aabfdf)
# LSST ugrizy ABmags are:
abs_mag_sol_lsst = {
    "y": 4.50,
    "z": 4.51,
    "i": 4.52,
    "r": 4.64,
    "g": 5.06,
    "u": 6.27,
}
# fluxes = u.ABmag.to(u.nanojansky, list(abs_mag_sol_lsst.values()))
# # = ['5.754e+10', '5.702e+10', '5.649e+10',
# #    '5.058e+10', '3.436e+10', '1.127e+10']
# weights = 6*(1/fluxes)/np.sum(1/fluxes)

bands_weights_lsst = {
    "y": 0.5481722621482569,
    "z": 0.553244437640313,
    "i": 0.5583635453943578,
    "r": 0.6236157227514114,
    "g": 0.9181572253205194,
    "u": 2.798446806745142,
}
