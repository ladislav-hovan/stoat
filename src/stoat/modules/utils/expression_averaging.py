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

from anndata import AnnData
from scipy.sparse import csr_matrix

from stoat.config import DISTANCE_KERNEL

### Functions ###
def calculate_gaussian_m1(
    A: csr_matrix,
    sigma: float,
) -> csr_matrix:
    # TODO: Update description
    """
    Calculates the value of the Gaussian PDF with standard deviation
    sigma at distance r from the mean, minus 1 (so that it works
    easily with a sparse matrix).

    Parameters
    ----------
    r : float
        The distance from the mean
    sigma : float
        The standard deviation of the Gaussian distribution

    Returns
    -------
    float
        The value of the Gaussian PDF at distance r, -1
    """

    # Normalisation is irrelevant because of the finite discretised
    # scope, it will be done based on the sum of contributing parts
    return A.power(2).multiply(-1 / (2 * sigma**2)).expm1()


def calculate_pearson_r(
    A: csr_matrix,
) -> np.ndarray:

    A = A.astype(np.float64)
    n = A.shape[1]

    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))

    return coeffs.A


def rescale_weights_by_row(
    weights: csr_matrix,
) -> csr_matrix:
    
    weight_sums = weights.sum(axis=1)
    for pos in range(len(weight_sums)):
        if weight_sums[pos][0] == 0:
            weight_sums[pos][0] = 1

    return weights / weight_sums


def weigh_by_distance(
    adata: AnnData,
    kernel: DISTANCE_KERNEL = 'uniform',
    sigma: float = 0.5,
) -> csr_matrix:

    is_neigh = adata.obsp['spatial_neighbours']
    if kernel == 'uniform':
        # The contribution of every cell to the average is independent of
        # the distance from the central cell
        d_weights = is_neigh
    elif kernel == 'gaussian':
        # The contribution is based on the distance from the central cell
        # and decreases proportionally to exp(-r**2)
        dists = adata.obsp['spatial_distances'].copy()
        for i in range(adata.n_obs):
            dists[i,i] = 0
        # Add the 1 again from is_neigh (doesn't affect sparse zeroes)
        d_weights = calculate_gaussian_m1(dists, sigma) + is_neigh
    else:
        raise NotImplementedError(f'Unrecognised kernel: {kernel}'
            f'\nOptions are: {", ".join(DISTANCE_KERNEL.__args__)}')

    return d_weights


def weigh_by_correlation(
    adata: AnnData,
) -> np.ndarray:

    c_weights = calculate_pearson_r(adata.X)

    return c_weights