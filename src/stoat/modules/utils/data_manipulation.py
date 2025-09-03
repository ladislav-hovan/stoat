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

### Functions ###
def get_validity(
    adata: AnnData,
) -> pd.Series:

    if 'valid' in adata.obs.columns:
        valid = adata.obs['valid']
    else:
        valid = adata.obs['in_tissue']

    return valid


def get_layer(
    adata: AnnData,
    layer: Optional[str] = None,
) -> any:

    if layer is None:
        data = adata.X
    else:
        data = adata.layers[layer]

    return data


def create_sparse_dataframe(
    adata: AnnData,
    layer: Optional[str] = None,
) -> pd.DataFrame:

    return pd.DataFrame.sparse.from_spmatrix(
        get_layer(adata, layer),
        index=adata.obs_names,
        columns=adata.var_names,
    )