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

from stoat.config import FILE_LIKE, FORMAT

### Functions ###
def save_df_into_file(
    df: Union[pd.DataFrame, pd.Series],
    base_filename: str,
    format: FORMAT,
) -> None:

    save_df_into_filelike(df, f'{base_filename}.{format}', format)


def save_df_into_filelike(
    df: Union[pd.DataFrame, pd.Series],
    filename: FILE_LIKE,
    format: FORMAT,
) -> None:
    # Saves the df into a file with proper extension
    if format == 'tsv':
        df.to_csv(filename, sep='\t')
    elif format == 'feather':
        # Resetting the index will convert to DataFrame
        df.reset_index().to_feather(filename)
    elif format == 'parquet':
        if type(df) == pd.DataFrame:
            df.to_parquet(filename)
        else:
            df.to_frame().to_parquet(filename)
    else:
        raise NotImplementedError(f'Cannot save format {format}, '
            'not implemented')


def load_into_df(
    filename: FILE_LIKE,
    format: FORMAT,
) -> pd.DataFrame:
    # Loads the df from a file with a given format
    if format == 'tsv':
        df = pd.read_csv(filename, sep='\t', index_col=0)
    elif format == 'feather':
        # Resetting the index will convert to DataFrame
        df = pd.read_feather(filename).set_index('index')
        df.index.rename(None, inplace=True)
    elif format == 'parquet':
        df = pd.read_parquet(filename)
    else:
        raise NotImplementedError(f'Cannot read format {format}, '
            'not implemented')

    return df