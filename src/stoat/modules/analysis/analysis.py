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

### Imports and definitions ###
import glob
import os

import gseapy as gp
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from pathlib import Path
from typing import Literal, Optional, Union

from stoat import Stoat
from stoat.config import FORMAT, FILE_LIKE
from stoat.modules.utils import get_layer, load_into_df

### Functions ###
def describe_expression(
    stoat_obj: Stoat,
    layer: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    # Provides details about the expression sparsity
    st = stoat_obj.spatial[stoat_obj.table]
    print ('Proportion of spots with tissue: '
        f'{100 * st.obs["in_tissue"].sum() / len(st.obs):.2f} %')
    expr_mat = get_layer(st, layer)
    avg_sparsity = (expr_mat[st.obs['in_tissue'].values,:] != 0).mean(
        axis=1).mean()
    print ('Average sparsity of genes in a spot with tissue: '
        f'{100 * (1 - avg_sparsity):.2f} %')

    gene_coverage = {}
    gene_coverage[0] = (expr_mat[st.obs['in_tissue'].values,:] != 0).mean(
        axis=1).A1
    for i in range(1, 4):
        stoat_obj.average_expression(n_rings=i)
        avg_success = st.layers['averaged'][st.obs['in_tissue'].values,:]
        gene_coverage[i] = (avg_success != 0).mean(axis=1).A1

    if ax is None:
        _,ax = plt.subplots(figsize=(8,8))
    labels = ['Original data', '1st neighbours',
        '2nd neighbours', '3rd neighbours']
    for label,data in zip(labels, gene_coverage.values()):
        ax.hist(data, bins=120, range=(0,1), alpha=0.5, label=label)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of genes with non-zero reads', size=16)
    ax.set_ylabel('Spot count', size=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    return ax


def calculate_degrees(
    stoat_folder: Path,
    extension: str,
    which: Literal['in', 'out', 'both'] = 'in',
    output_file_base: FILE_LIKE = 'final_indegree',
) -> None:
    # Calculates indegrees and saves them for every STOAT network in
    # the folder
    pass


def save_into_file(
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
        print ('Format not recognised')


def collate_indegrees(
    stoat_folder: Path,
    format: FORMAT,
    output_file: FILE_LIKE,
) -> None:
    # Gathers the data from all indegree files in a folder and puts
    # it into a single file
    id_files = glob.glob(os.path.join(stoat_folder,
        f'indegree_*.{format}'))
    collection = []
    for file in id_files:
        temp = load_into_df(file, format)
        base_name = file.split('.')[-2].split('_')[-1]
        temp.rename(columns={'Indegrees': base_name}, inplace=True)
        collection.append(temp.copy())
    df = pd.concat(collection, axis=1)
    save_into_file(df, output_file, format)


def perform_gsea(
    anndata: sc.AnnData,
    gene_set: str,
    **kwargs,
) -> dict:
    # Perform GSEA for every identified cluster in the anndata object
    # Compares to all other clusters
    # Assumes clusters are present at obs['clusters'], data is log1p
    # transformed and gene names are used instead of Ensembl IDs
    # kwargs are passed to the gsea function
    for i in range(anndata.obs['clusters'].nunique()):
        anndata.obs[f'is_{i}'] = (anndata.obs['clusters'] == f'{i}').astype(int)
    res_all = {}
    for i in range(anndata.obs['clusters'].nunique()):
        in_cluster = anndata.obs[f'is_{i}'].copy()
        in_cluster.sort_values(ascending=False, inplace=True)
        res_all[i] = gp.gsea(
            # row -> genes, column -> samples
            data=anndata.to_df().reindex(in_cluster.index).T,
            gene_sets=gene_set,
            cls=in_cluster,
            **kwargs,
        )

    return res_all


def perform_deg_analysis(
    anndata: sc.AnnData,
) -> sc.AnnData:
    # Perform DEG for every identified cluster in the anndata object
    # Compares to all other clusters
    # Assumes clusters are present at obs['clusters'], data is log1p
    # transformed and gene names are used instead of Ensembl IDs
    deg_adata = sc.tl.rank_genes_groups(anndata, groupby='clusters',
        method='wilcoxon', tie_correct=True, copy=True)

    return deg_adata