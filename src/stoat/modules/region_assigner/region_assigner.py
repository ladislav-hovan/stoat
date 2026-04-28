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
import pandas as pd

from anndata import AnnData
from typing import Optional

from stoat.modules.clustering import determine_cluster_labels
from stoat.modules.utils import (create_sparse_dataframe, get_layer,
    get_validity)

### Class definition ###
class RegionAssigner:
    ### Initialisation ###
    def __init__(
        self,
        spatial_table: AnnData,
    ):

        self.st = spatial_table

    ### Class methods ###
    def assign_regions(
        self,
        mapping: Optional[pd.Series] = None,
        from_expression: bool = False,
        layer: Optional[str] = None,
        key_added: str = 'region_id',
        **kwargs,
    ) -> None:

        if from_expression and mapping is not None:
            raise ValueError('Both from_expression and mapping were specified,'
                ' please provide only one of them.')

        validity = ('in_tissue' if layer is None else 'valid')
        if from_expression:
            # Default options for clustering
            clustering_opts = {
                'validity': validity,
            }
            # Allow overwriting them through kwargs
            clustering_opts.update(**kwargs)
            # Assign regions based on expression clustering
            determine_cluster_labels(
                spatial_table=self.st,
                layer=layer,
                key_added=key_added,
                **clustering_opts,
            )
            id_col = self.st.obs[key_added]
            self.st.obs[key_added] = id_col.replace(-1, None)
        elif mapping is not None:
            # Assignment of spots to regions, fills in NaN for missing
            self.st.obs[key_added] = mapping
        else:
            # Every valid spot is its own region
            self.st.obs[key_added] = [
                ind if val else None
                for ind,val in self.st.obs[validity].items()
            ]


    def collapse_expression(
        self,
        key: str = 'region_id',
    ) -> None:

        if 'averaged' in self.st.layers:
            expr_df = create_sparse_dataframe(self.st, layer='averaged')
        else:
            expr_df = create_sparse_dataframe(self.st)

        valid = get_validity(self.st)

        region_to_spot = pd.get_dummies(
            self.st.obs[key],
            sparse=True,
            dtype=int,
        ).T
        sum_df = (region_to_spot * valid.astype(int)) @ expr_df
        self.st.varm['collapsed'] = sum_df.T