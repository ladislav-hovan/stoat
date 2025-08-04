# Copyright (C) 2025 Ladislav Hovan <ladislav.hovan@ncmbm.uio.no>
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Public License as published
# by the Free Software Foundation; either version 3 of the License,
# or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Public License along
# with this library. If not, see <https://www.gnu.org/licenses/>.

### Imports ###
import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Union

from stoat.config import DISTANCE_KERNEL

### Functions ###
def calculate_gaussian(
    r: Union[float, np.array],
    sigma: float,
) -> float:
    """
    Calculates the value of the Gaussian PDF with standard deviation
    sigma at distance r from the mean.

    Parameters
    ----------
    r : float
        The distance from the mean
    sigma : float
        The standard deviation of the Gaussian distribution

    Returns
    -------
    float
        The value of the Gaussian PDF at distance r
    """

    # Normalisation is irrelevant because of the finite discretised
    # scope, it will be done based on the sum of contributing parts
    return np.exp(-np.power(r, 2)/(2 * sigma**2))


def weigh_by_distance(
    adata: AnnData,
    kernel: DISTANCE_KERNEL = 'uniform',
    sigma: float = 0.5,
) -> pd.DataFrame:

    is_neigh = adata.obsp['spatial_connectivities'].copy()
    for i in range(adata.n_obs):
        is_neigh[i,i] = 1
    if kernel == 'uniform':
        # The contribution of every cell to the average is independent of
        # the distance from the central cell
        d_weights = is_neigh
    elif kernel == 'gaussian':
        # The contribution is based on the distance from the central cell
        # and decreases proportionally to exp(-r**2)
        # Define a gaussian distribution with a provided sigma
        calculate_gaussian_fixed = lambda r: calculate_gaussian(r, sigma)
        # Provide the function as an input to distance weighting template
        d_weights = spatial.apply(lambda row: get_distance_to_neighbours(
            row.name, spatial).apply(calculate_gaussian_fixed).values, axis=1)
    else:
        raise NotImplementedError(f'Unrecognised kernel: {kernel}'
            f'\nOptions are: {", ".join(DISTANCE_KERNEL.__args__)}')
    # resc_weights = rescale_weights_by_row(d_weights)

    return d_weights


# def weigh_by_correlation(
#     adata: AnnData,
# ) -> pd.DataFrame:

#     corr = expression.T.corr()
#     c_weights = spatial.apply(lambda row: corr.loc[row.name][
#         spatial.loc[row.name]['ValNeighbours']].values, axis=1)

#     return c_weights