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

from typing import Union

from stoat.config import FILE_LIKE, FORMAT
from stoat.modules.utils import format_docstring

### Functions ###
@format_docstring(formats=', '.join(FORMAT.__args__))
def save_df_into_file(
    df: Union[pd.DataFrame, pd.Series],
    base_filename: str,
    format: FORMAT,
) -> None:
    """
    Saves a pandas DataFrame into a file.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        DataFrame to be saved
    base_filename : str
        Prefix for the output file (extension will be added)
    format : FORMAT
        Format of the output file, one of:
        {formats}
    """

    save_df_into_filelike(df, f'{base_filename}.{format}', format)


@format_docstring(formats=', '.join(FORMAT.__args__))
def save_df_into_filelike(
    df: Union[pd.DataFrame, pd.Series],
    filename: FILE_LIKE,
    format: FORMAT,
) -> None:
    """
    Saves a pandas DataFrame into a file or an IO buffer.

    Parameters
    ----------
    df : Union[pd.DataFrame, pd.Series]
        DataFrame to be saved
    filename : FILE_LIKE
        Path to a file or an IO buffer
    format : FORMAT
        Format of the output file, one of:
        {formats}

    Raises
    ------
    NotImplementedError
        If the selected format is not implemented
    """

    if format == 'feather':
        # Resetting the index will convert to DataFrame
        df.reset_index().to_feather(filename)
    elif format == 'parquet':
        if type(df_f) == pd.Series:
            df_f = df
        else:
            df_f = df.to_frame()
        df_f.to_parquet(filename)
    elif format == 'tsv':
        df.to_csv(filename, sep='\t')
    else:
        raise NotImplementedError(
            f'Saving into format {format} is not implemented.\n'
            f'Implemented formats: {", ".join(FORMAT.__args__)}'
        )


@format_docstring(formats=', '.join(FORMAT.__args__))
def load_into_df(
    filename: FILE_LIKE,
    format: FORMAT,
) -> pd.DataFrame:
    """
    Loads the content of a file or an IO buffer into a pandas DataFrame.

    Parameters
    ----------
    filename : FILE_LIKE
        Path to a file or an IO buffer
    format : FORMAT
        Format of the output file, one of:
        {formats}

    Returns
    -------
    pd.DataFrame
        DataFrame with the data

    Raises
    ------
    NotImplementedError
        If the selected format is not implemented
    """

    if format == 'feather':
        # Resetting the index will convert to DataFrame
        df = pd.read_feather(filename).set_index('index')
        df.index.rename(None, inplace=True)
    elif format == 'parquet':
        df = pd.read_parquet(filename)
    elif format == 'tsv':
        df = pd.read_csv(filename, sep='\t', index_col=0)
    else:
        raise NotImplementedError(
            f'Reading from format {format} is not implemented.\n'
            f'Implemented formats: {", ".join(FORMAT.__args__)}'
        )

    return df