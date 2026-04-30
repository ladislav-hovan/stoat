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
from pathlib import Path

from stoat.config import FORMAT
from stoat.modules.analysis import calculate_ari
from stoat.modules.clustering import determine_cluster_labels
from stoat.modules.utils import load_into_df
from stoat.stoat import Stoat

from typing import Optional

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
        st.layers[label] = id_df.reindex_like(st.to_df())


    def cluster_on_degrees(
        self,
        label: str = 'indegree',
        validity: str = 'valid',
        cluster_label: str = 'clusters_in',
        **kwargs,
    ) -> None:

        determine_cluster_labels(
            self.spatial[self.table],
            layer=label,
            validity=validity,
            key_added=cluster_label,
            **kwargs,
        )


    def calculate_ari(
        self,
        label1: str,
        label2: str,
        unclassified_label: int = -1,
    ) -> float:

        st = self.spatial[self.table]

        return calculate_ari(
            st.obs[label1],
            st.obs[label2],
            unclassified_label=unclassified_label,
        )