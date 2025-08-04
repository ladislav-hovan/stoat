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

# This file contains the implementation of plotting functions for STOAT
# They can be called directly and certain STOAT functions call them

### Imports and settings ###
import squidpy as sq

from spatialdata import SpatialData
from typing import Callable

from stoat.modules.utils import weigh_by_distance

### Class definition ###
class ExpressionSmoother:
    ### Initialisation ###
    def __init__(
        self,
        spatial: SpatialData,
        n_rings: int = 1,
    ):

        self.spatial = spatial
        st = self.spatial['table']
        sq.gr.spatial_neighbors(st, n_rings=n_rings)
        st.obs['valid'] = st.obs['in_tissue']

    ### Class methods ###
    def filter_edges(
        self,
    ) -> None:

        connections = self.spatial['table'].obsp['spatial_connectivities']
        conn_counts = connections.sum(axis=1)
        max_conns = conn_counts.max()
        to_keep = (conn_counts == max_conns)
        st = self.spatial['table']
        st.obs['valid'] = st.obs['valid'] & to_keep.A1


    def enforce_max_invalid(
        self,
        max_invalid: int = 0,
    ) -> None:

        connections = self.spatial['table'].obsp['spatial_connectivities']
        all_neighs = connections.sum(axis=0)
        st = self.spatial['table']
        neighs_in_tissue = connections * st.obs['in_tissue']
        inv_neighs = all_neighs - neighs_in_tissue
        to_keep = (inv_neighs <= max_invalid)
        st.obs['valid'] = st.obs['valid'] & to_keep.A1


    def smooth_expression(
        self,
        avg_function: Callable = weigh_by_distance,
        *args,
        **kwargs,
    ) -> None:

        st = self.spatial['table']
        st.layers['averaged'] = avg_function(st, *args, **kwargs) * st.X