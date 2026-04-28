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

from pathlib import Path
from typing import List, Union

### Functions ###
def process_regions(
    expr_data: pd.DataFrame,
    regions: Union[Path, List[str], None] = None,
) -> List[str]:

    if regions is None:
        # Calculate a STOAT network for every region with valid spots
        regions = expr_data.columns
    elif type(regions) == list:
        # Keep as is
        pass
    else:
        # Load the regions from a file
        f = open(regions, 'r')
        lines = f.readlines()
        regions = [i.split('\n')[0] for i in lines]
        f.close()

    return regions