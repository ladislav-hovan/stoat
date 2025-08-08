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
            self.st['region'] = mapping
        else:
            self.st['region'] = self.st.index