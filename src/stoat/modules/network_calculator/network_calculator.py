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
from netZooPy.panda import Panda
from pathlib import Path
from scanpy.preprocessing import log1p
from typing import Optional, Union

from stoat.config import EXTENSION

### Class definition ###
class NetworkCalculator:
    ### Initialisation ###
    def __init__(
        self,
        spatial_table: AnnData,
        motif_prior: Optional[Union[Path, pd.DataFrame]] = None,
        ppi_prior: Optional[Union[Path, pd.DataFrame]] = None,
    ):

        # TODO: Implement network methods other than PANDA (PUMA, DRAGON?)
        self.st = spatial_table
        self.motif_prior = motif_prior
        self.ppi_prior = ppi_prior
        self.panda_obj = None

    ### Class methods ###
    def ensure_compatibility(
        self,
    ) -> None:

        if self.motif_prior is None:
            print ('No motif prior provided, skipping prior adjustments')
            return

        if type(self.motif_prior) == str:
            # Names are chosen for compatibility with PANDA later
            motif_df = pd.read_csv(self.motif_prior, sep='\t', header=None,
                names=['source', 'target', 'weight'])
        else:
            motif_df = self.motif_prior
        prior_genes = set(motif_df['target'])
        expr_genes = set(self.st.var.index)
        in_common = prior_genes.intersection(expr_genes)

        print (f'The prior contains {len(prior_genes)} genes.')
        print (f'The expression contains {len(expr_genes)} genes.')
        print (f'These two sets have {len(in_common)} gene names in common.')

        to_keep = sorted(in_common)
        self.st = self.st[:,to_keep]
        expr_genes = set(self.st.var.index)
        motif_df = motif_df[motif_df['target'].isin(expr_genes)]
        self.motif_prior = motif_df


    def log1p_transform(
        self,
    ) -> None:

        if 'averaged' in self.st.layers:
            self.st.layers['averaged_log1p'] = log1p(
                self.st.layers['averaged'],
                copy=True,
            )
        else:
            self.st.layers['log1p'] = log1p(
                self.st.X,
                copy=True,
            )


    def calculate_panda(
        self,
        *args,
        **kwargs,
    ) -> None:

        expr_data = self.st.to_df()
        # Order of layers to be tried
        preferences = ['averaged_log1p', 'log1p', 'averaged']
        for layer in preferences:
            if layer in self.st.layers:
                expr_data = self.st.to_df(layer=layer)
                break

        self.panda_obj = Panda(
            expression_file=expr_data,
            motif_file=self.motif_prior,
            ppi_file=self.ppi_prior,
            *args,
            **kwargs,
        )


    def calculate(
        self,
        save_dir: Path = './',
        extension: EXTENSION = 'feather',
    ) -> None:

        print ('TODO: calculate is not implemented yet')