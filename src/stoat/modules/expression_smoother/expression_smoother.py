# Copyright (C) 2026 Ladislav Hovan <ladislav.hovan@ncmbm.uio.no>
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
import squidpy as sq

from anndata import AnnData
from collections import Counter
from numpy.random import default_rng
from scipy.sparse import csr_matrix, eye
from typing import Callable, Optional

from stoat.modules.utils import rescale_weights_by_row, weigh_by_distance

### Class definition ###
class ExpressionSmoother:
    ### Initialisation ###
    def __init__(
        self,
        spatial_table: AnnData,
        n_rings: int = 1,
        n_neighs: int = 6,
        coord_type: Optional[str] = None,
        random_connections: bool = False,
        random_seed: Optional[int] = None,
    ):

        self.st = spatial_table
        if coord_type == 'grid':
            sq.gr.spatial_neighbors_grid(
                self.st,
                n_rings=n_rings,
                n_neighs=n_neighs,
            )
        else: 
            raise ValueError('The coord_type specified is not currently '
                'supported. Please use "grid" for now.')
        if random_connections:
            self._reshuffle_connections(random_seed)
        self.st.obs['valid'] = self.st.obs['in_tissue']
        # A definition of neighbour that includes self
        self.st.obsp['spatial_neighbours'] = (
            self.st.obsp['spatial_connectivities'] +
            eye(self.st.n_obs, format='csr')
        )

    ### Class methods ###
    def _reshuffle_connections(
        self,
        random_seed: Optional[int] = None,
        max_attempts: int = 10,
    ) -> None:

        rng = default_rng(seed=random_seed)
        sp_conn = self.st.obsp['spatial_connectivities']
        sp_dists = self.st.obsp['spatial_distances']
        n = sp_conn.shape[0]
        in_tissue = self.st.obs['in_tissue'].astype(int).values.reshape(
            (-1, 1))
        sp_conn_it = sp_conn.multiply(in_tissue).multiply(in_tissue.T)
        sp_dists_it = sp_dists.multiply(in_tissue).multiply(in_tissue.T
            ).tocsr()
        n_it_neighs = sp_conn_it.sum(axis=1).A1.astype(int)
        sp_conn_it.eliminate_zeros()
        sp_dists_it.eliminate_zeros()
        initial_conn = sp_conn.todense() - sp_conn_it.todense()
        initial_dists = sp_dists.todense() - sp_dists_it.todense()
        # This way of trying to make sure everything is conserved may fail
        # Hence, we try multiple times
        for attempt in range(max_attempts):
            new_conn = initial_conn.copy()
            new_dists = initial_dists.copy()
            counts = Counter()
            try:
                for i in range(n):
                    if not in_tissue[i][0]:
                        continue
                    n_left = n_it_neighs[i] - counts[i]
                    if n_left == 0:
                        continue
                    arr = np.array([j for j in range(i + 1, n)
                        if (counts[j] < n_it_neighs[j])
                        and in_tissue[j][0]])
                    new_idx = rng.choice(arr, n_left, replace=False)
                    old_idx = sp_dists_it[i, :].indices
                    dists = sp_dists_it[i, old_idx].todense().A1
                    for pos,j in enumerate(new_idx):
                        counts[j] += 1
                        new_conn[i, j] = 1
                        new_conn[j, i] = 1
                        new_dists[i, j] = dists[pos]
                        new_dists[j, i] = dists[pos]
            except ValueError:
                if attempt == max_attempts - 1:
                    print ('Could not successfully reshuffle connections '
                        'while preserving the constraints.')
                    # No assignment happens - connections remain as before
                continue
            else:
                self.st.obsp['spatial_connectivities'] = csr_matrix(new_conn)
                self.st.obsp['spatial_distances'] = csr_matrix(new_dists)
                break


    def filter_edges(
        self,
    ) -> None:

        connections = self.st.obsp['spatial_connectivities']
        conn_counts = connections.sum(axis=1)
        max_conns = conn_counts.max()
        to_keep = (conn_counts == max_conns)
        self.st.obs['valid'] = self.st.obs['valid'] & to_keep.A1


    def enforce_max_invalid(
        self,
        max_invalid: int = 0,
    ) -> None:

        connections = self.st.obsp['spatial_connectivities']
        all_neighs = connections.sum(axis=0)
        neighs_in_tissue = connections * self.st.obs['in_tissue']
        inv_neighs = all_neighs - neighs_in_tissue
        to_keep = (inv_neighs <= max_invalid)
        self.st.obs['valid'] = self.st.obs['valid'] & to_keep.A1


    def smooth_expression(
        self,
        avg_function: Callable = weigh_by_distance,
        *args,
        **kwargs,
    ) -> None:

        self.st.layers['averaged'] = rescale_weights_by_row(
            self.st.obsp['spatial_neighbours'].multiply(
                avg_function(self.st, *args, **kwargs)
            ).multiply(
                self.st.obs['in_tissue']
            )
        # Specify matrix multiplication
        ) @ self.st.X