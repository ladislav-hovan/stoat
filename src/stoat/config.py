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
from io import BytesIO, StringIO
from pandas import DataFrame
from pathlib import Path
from typing import Literal, Union

### Definitions ###
# Typing literals
CLUSTERING = Literal['hdbscan', 'leiden', 'spagcn']
DISTANCE_KERNEL = Literal['gaussian', 'uniform']
FILE_LIKE = Union[BytesIO, Path, StringIO]
FORMAT = Literal['feather', 'parquet', 'tsv']
# Plotting default settings and mappings
COL_TO_TITLE = {
    'n_genes_by_counts': 'Number of expressed genes',
    'total_counts': 'Total number of counts',
    'pct_counts_mt': 'Mitochondrial gene percentage',
}
DIMENSIONS = DataFrame({
    'type': ['deg', 'gsea', 'match'],
    'overhead': [2.5, 2.5, 3.5],
    'width_per_col': [3, 8, 3],
    'height_per_line': [0.3, 0.5, 1],
}).set_index('type')
P_VAL_MAPPING = {
    'Adjusted P-value': 'FDR',
    'FDR q-val': 'FDR',
    'NOM p-val': 'Pval',
    'P-value': 'Pval',
}
# Ignored warnings when loading data
IGNORED_WARNINGS = {
    'Converting .* to categorical dtype',
    'Variable names are not unique',
}