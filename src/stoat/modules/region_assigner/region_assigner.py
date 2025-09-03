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
import pandas as pd

from anndata import AnnData
from typing import Optional

from stoat.modules.utils import create_sparse_dataframe, get_validity

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
    ) -> None:

        if mapping is not None:
            # Assignment of spots to regions
            self.st.obs['region'] = mapping
        else:
            # Every spot is its own region
            self.st.obs['region'] = self.st.obs.index


    def collapse_expression(
        self,
    ) -> None:

        if 'averaged' in self.st.layers:
            expr_df = create_sparse_dataframe(self.st, layer='averaged')
        else:
            expr_df = create_sparse_dataframe(self.st)

        valid = get_validity(self.st)

        region_to_spot = pd.get_dummies(
            self.st.obs['region'],
            sparse=True,
            dtype=int,
        ).T
        sum_df = (region_to_spot * valid.astype(int)) @ expr_df
        self.st.varm['collapsed'] = sum_df.T