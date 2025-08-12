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
import os

import pandas as pd

from anndata import AnnData
from netZooPy.panda import Panda
from pathlib import Path
from scanpy.preprocessing import log1p
from typing import Iterable, Optional, Union

from stoat.config import EXTENSION
from stoat.modules.utils import get_full_name, process_regions, save_dataframe

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
        self.expr_data = self.st.to_df()
        self.motif_prior = motif_prior
        self.ppi_prior = ppi_prior
        self.panda_network = None

    ### Class methods ###
    def ensure_compatibility(
        self,
    ) -> None:

        if self.motif_prior is None:
            print ('No motif prior provided, skipping prior adjustments.')
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

        # Order of layers to be tried
        preferences = ['averaged_log1p', 'log1p', 'averaged']
        for layer in preferences:
            if layer in self.st.layers:
                self.expr_data = self.st.to_df(layer=layer)
                break

        panda_obj = Panda(
            expression_file=self.expr_data.T,
            motif_file=self.motif_prior,
            ppi_file=self.ppi_prior,
            *args,
            **kwargs,
        )
        self.panda_network = panda_obj.panda_network


    def calculate(
        self,
        save_dir: Path = './',
        extension: EXTENSION = 'feather',
        regions: Union[Path, Iterable[str], None] = None,
        save_panda: bool = False,
        save_degrees: bool = False,
        overwrite_old = True,
        *args,
        **kwargs,
    ) -> None:

        if not hasattr(self, 'panda_network'):
            print ('The full PANDA network has not been calculated, '
                'calculating it now.')
            self.calculate_panda()

        # Create output directory if nonexistent
        os.makedirs(save_dir, exist_ok=True)

        regions = process_regions(self.st, regions)

        panda_input = self.expr_data.T
        n_regions = len(panda_input.columns)

        for r in regions:
            # Names of output files (base, without extension)
            panda_outfile = os.path.join(save_dir, f'panda_{r}')
            stoat_outfile = os.path.join(save_dir, f'stoat_{r}')

            # Check if we're overwriting
            if not overwrite_old and (
                os.path.exists(get_full_name(stoat_outfile, extension)) or
                (save_panda and
                os.path.exists(get_full_name(panda_outfile, extension)))):
                print (f'Skipping region {r} because the STOAT or '
                    'PANDA file already exists in the target directory.')
                continue

            print (f'Calculating the STOAT network for region {r}.')

            # PANDA network with the current spot missing
            panda_obj = Panda(
                expression_file=panda_input.drop(r, axis=1),
                motif_file=self.motif_prior,
                ppi_file=self.ppi_prior,
                *args,
                **kwargs,
            )

            panda_net = panda_obj.panda_network

            if save_panda:
                print ('Saving the intermediate PANDA network to '
                    f'{get_full_name(panda_outfile, extension)}.')
                save_dataframe(panda_net, panda_outfile, extension)

            # Equation for deriving the region-specific network
            stoat_net = (n_regions * (self.panda_network - panda_net) +
                panda_net)

            print ('Saving the STOAT network to '
                f'{get_full_name(stoat_outfile, extension)}.')
            save_dataframe(stoat_net, stoat_outfile, extension)

            if save_degrees:
                # Names of output files
                in_outfile = os.path.join(save_dir, f'indegree_{r}')
                out_outfile = os.path.join(save_dir, f'outdegree_{r}')

                print ('Saving the indegrees to '
                    f'{get_full_name(in_outfile, extension)}.')
                save_dataframe(stoat_net.sum().rename('Indegrees'),
                    in_outfile, extension)
                print ('Saving the outdegrees to '
                    f'{get_full_name(out_outfile, extension)}.')
                save_dataframe(stoat_net.sum(axis=1).rename('Outdegrees'),
                    out_outfile, extension)