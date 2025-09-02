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

### Imports and settings ###
import squidpy as sq

from anndata import AnnData
from scipy.sparse import eye
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
    ):

        self.st = spatial_table
        sq.gr.spatial_neighbors(
            self.st,
            n_rings=n_rings,
            n_neighs=n_neighs,
            coord_type=coord_type,
        )
        self.st.obs['valid'] = self.st.obs['in_tissue']
        # A definition of neighbour that includes self
        self.st.obsp['spatial_neighbours'] = (
            self.st.obsp['spatial_connectivities'] +
            eye(self.st.n_obs, format='csr')
        )


    ### Class methods ###
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