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
from pathlib import Path

from stoat.config import FORMAT
from stoat.modules.clustering import determine_cluster_labels
from stoat.modules.utils import load_into_df
from stoat.stoat import Stoat

from typing import Any, Dict, Literal, Optional

### Class definition ###
class StoatAnalysis(Stoat):
    ### Initialisation ###
    def __init__(
        self,
        stoat_obj: Optional[Stoat] = None,
    ) -> None:

        if stoat_obj is not None:
            self.spatial = stoat_obj.spatial
            self.table = stoat_obj.table
            self.coord_type = stoat_obj.coord_type
            self.n_neighs = stoat_obj.n_neighs

    ### Methods ###
    def load_degrees(
        self,
        degree_file: Path,
        format: FORMAT,
        label: str = 'indegree',
    ) -> None:

        st = self.spatial[self.table]
        id_df = load_into_df(degree_file, format).T
        st.obsm[label] = id_df.reindex(st.obs.index)


    # def determine_clusters(
    #     self,
    #     on_df: Literal['expr', 'ind', 'both'] = 'both',
    #     clust_settings: Dict[Any, Any] = {},
    # ) -> None:

    #     if on_df in ['expr', 'both']:
    #         classes, ordering, n_classes = determine_cluster_labels(
    #             self.expression, self.spatial, self.validity, **clust_settings)
    #         self.classes_expr = classes
    #         self.ordering_expr = ordering
    #         self.n_classes_expr = n_classes
    #     if on_df in ['ind', 'both']:
    #         classes, ordering, n_classes = determine_cluster_labels(
    #             self.indegrees, self.spatial, self.validity, **clust_settings)
    #         self.classes_ind = classes
    #         self.ordering_ind = ordering
    #         self.n_classes_ind = n_classes