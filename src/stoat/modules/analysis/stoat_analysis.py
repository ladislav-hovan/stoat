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
from stoat.modules.clustering import *
from stoat.stoat import Stoat

from typing import Optional, Literal, Dict, Any

### Class definition ###
class StoatAnalysis(Stoat):
    ### Initialisation ###
    def __init__(
        self,
        stoat_obj: Optional[Stoat] = None,
    ) -> None:

        if stoat_obj is not None:
            self.spatial = stoat_obj.spatial

    ### Class methods ###
    def determine_clusters(
        self,
        on_df: Literal['expr', 'ind', 'both'] = 'both',
        clust_settings: Dict[Any, Any] = {},
    ) -> None:

        if on_df in ['expr', 'both']:
            classes, ordering, n_classes = determine_cluster_labels(
                self.expression, self.spatial, self.validity, **clust_settings)
            self.classes_expr = classes
            self.ordering_expr = ordering
            self.n_classes_expr = n_classes
        if on_df in ['ind', 'both']:
            classes, ordering, n_classes = determine_cluster_labels(
                self.indegrees, self.spatial, self.validity, **clust_settings)
            self.classes_ind = classes
            self.ordering_ind = ordering
            self.n_classes_ind = n_classes