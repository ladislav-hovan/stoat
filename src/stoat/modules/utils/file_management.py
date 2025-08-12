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

from typing import Union

from stoat.config import EXTENSION

### Functions ###
def get_full_name(
    base_filename: str,
    extension: str,
) -> str:

    return f'{base_filename}.{extension}'


def save_dataframe(
    df: Union[pd.DataFrame, pd.Series],
    base_filename: str,
    extension: EXTENSION,
) -> None:

    if extension == 'tsv':
        df.to_csv(f'{base_filename}.tsv', sep='\t')
    elif extension == 'feather':
        # Resetting the index will convert to DataFrame
        df.reset_index().to_feather(f'{base_filename}.feather')
    elif extension == 'parquet':
        if type(df) == pd.DataFrame:
            df.to_parquet(f'{base_filename}.parquet')
        else:
            df.to_frame().to_parquet(f'{base_filename}.parquet')
    else:
        raise NotImplementedError(f'Cannot save extension {extension}, '
            'not implemented')