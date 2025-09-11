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
from numpy import log, log1p
from pathlib import Path
from typing import Iterable, Optional, Union

from stoat.config import EXTENSION
from stoat.modules.utils import (create_sparse_dataframe, get_network,
    get_validity, process_regions, save_dataframe)

### Class definition ###
class NetworkCalculator:
    ### Initialisation ###
    def __init__(
        self,
        spatial_table: AnnData,
        grn_generator: type = Panda,
        motif_prior: Optional[Union[Path, pd.DataFrame]] = None,
        ppi_prior: Optional[Union[Path, pd.DataFrame]] = None,
    ):

        self.st = spatial_table
        self.generator = grn_generator
        self.expr_data = self.st.to_df()
        self.motif_prior = motif_prior
        self.ppi_prior = ppi_prior
        self.basis_network = None

    ### Class methods ###
    def ensure_compatibility(
        self,
    ) -> None:

        if self.motif_prior is None:
            print ('No motif prior provided, skipping prior adjustments.')
            return

        if type(self.motif_prior) == str:
            # Names are chosen for compatibility
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
        self.st._inplace_subset_var(to_keep)
        expr_genes = set(self.st.var.index)
        motif_df = motif_df[motif_df['target'].isin(expr_genes)]
        self.motif_prior = motif_df


    def log1p_transform(
        self,
    ) -> None:
        # Converted to base 2

        if 'collapsed' in self.st.varm:
            self.st.varm['collapsed_log1p'] = log1p(
                self.st.varm['collapsed']
            ) / log(2)
        elif 'averaged' in self.st.layers:
            self.st.layers['averaged_log1p'] = log1p(
                self.st.layers['averaged']
            ) / log(2)
        else:
            self.st.layers['log1p'] = log1p(self.st.X) / log(2)


    def calculate_basis(
        self,
        **kwargs,
    ) -> None:

        # Order of layers to be tried
        self.expr_data = self.st.to_df()
        preferences = ['collapsed_log1p', 'collapsed', 'averaged_log1p',
            'log1p', 'averaged']
        for layer in preferences:
            if layer in self.st.varm:
                self.expr_data = self.st.varm[layer]
                break
            elif layer in self.st.layers:
                valid = get_validity(self.st.obs)
                self.expr_data = create_sparse_dataframe(
                    self.st,
                    layer=layer,
                ).loc[valid].T
                break

        grn_obj = self.generator(
            expression_file=self.expr_data,
            motif_file=self.motif_prior,
            ppi_file=self.ppi_prior,
            **kwargs,
        )

        self.basis_network = get_network(grn_obj)


    def calculate(
        self,
        save_dir: Path = './',
        extension: EXTENSION = 'feather',
        regions: Union[Path, Iterable[str], None] = None,
        save_network: bool = False,
        save_degrees: bool = False,
        overwrite_old = True,
        **kwargs,
    ) -> None:

        if not hasattr(self, 'basis_network'):
            print ('The full basis network has not been calculated, '
                'calculating it now.')
            self.calculate_basis()

        def get_full_name(
            base_filename: str,
        ) -> str:

            return f'{base_filename}.{extension}'

        # Create output directory if nonexistent
        os.makedirs(save_dir, exist_ok=True)

        grn_input = self.expr_data
        regions = process_regions(grn_input, regions)
        n_regions = len(grn_input.columns)

        for r in regions:
            # Names of output files (base, without extension)
            net_outfile = os.path.join(save_dir, f'net_{r}')
            stoat_outfile = os.path.join(save_dir, f'stoat_{r}')

            # Check if we're overwriting
            if not overwrite_old and (
                os.path.exists(get_full_name(stoat_outfile)) or
                (
                    save_network and
                    os.path.exists(get_full_name(net_outfile))
                )
            ):
                print (f'Skipping region {r} because the STOAT or '
                    'network file already exists in the target directory.')
                continue

            print (f'Calculating the STOAT network for region {r}.')

            # Network with the current spot missing
            grn_obj = self.generator(
                expression_file=grn_input.drop(r, axis=1),
                motif_file=self.motif_prior,
                ppi_file=self.ppi_prior,
                **kwargs,
            )

            net = get_network(grn_obj)

            if save_network:
                print ('Saving the intermediate network to '
                    f'{get_full_name(net_outfile)}.')
                save_dataframe(net, net_outfile, extension)

            # Equation for deriving the region-specific network
            stoat_net = (n_regions * (self.basis_network - net) + net)

            print ('Saving the STOAT network to '
                f'{get_full_name(stoat_outfile)}.')
            save_dataframe(stoat_net, stoat_outfile, extension)

            if save_degrees:
                # Names of output files
                in_outfile = os.path.join(save_dir, f'indegree_{r}')
                out_outfile = os.path.join(save_dir, f'outdegree_{r}')

                print ('Saving the indegrees to '
                    f'{get_full_name(in_outfile)}.')
                save_dataframe(stoat_net.sum().rename('Indegrees'),
                    in_outfile, extension)
                print ('Saving the outdegrees to '
                    f'{get_full_name(out_outfile)}.')
                save_dataframe(stoat_net.sum(axis=1).rename('Outdegrees'),
                    out_outfile, extension)