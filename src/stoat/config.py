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
from typing import Literal, Union

### Definitions ###
# Typing literals
CLUSTERING = Literal['leiden', 'hdbscan']
DISTANCE_KERNEL = Literal['uniform', 'gaussian']
FORMAT = Literal['tsv', 'feather', 'parquet']
FILE_LIKE = Union[bytes, Path]
# Plotting defaults
DIMENSIONS = pd.DataFrame({
    'type': ['deg', 'gsea'],
    'overhead': [2.5, 2.5],
    'width_per_col': [3, 8],
    'height_per_line': [0.3, 0.5],
}).set_index('type')
COL_TO_TITLE = {
    'n_genes_by_counts': 'Number of expressed genes',
    'total_counts': 'Total number of counts',
    'pct_counts_mt': 'Mitochondrial gene percentage',
}