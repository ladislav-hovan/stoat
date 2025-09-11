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
import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from stoat.config import EXTENSION, FILE_LIKE
from stoat.stoat import Stoat

### Functions ###
# TODO: This whole thing should be replaced by StoatAnalysis function
# def analyse_fully(
#     stoat_obj: Stoat,
#     stoat_folder: Path,
#     indegree_file: Optional[FILE_LIKE],
#     extension: str,
#     output_folder: Path,
# ) -> None:
#     # Runs the entire pipeline
#     if not os.path.isdir(output_folder):
#         os.makedirs(output_folder)

#     describe_expression(stoat_obj)

#     if indegree_file is None:
#         if len(glob.glob(os.path.join(stoat_folder, 'indegree_*'))) == 0:
#             calculate_indegrees(stoat_folder, extension)
#         indegree_file = os.path.join(output_folder,
#             f'final_indegree.{extension}')
#         collate_indegrees(stoat_folder, extension, indegree_file)

#     # Clustering on expression - unfiltered/filtered
#     # Clustering on indegree - unfiltered/filtered


def describe_expression(
    stoat_obj: Stoat,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    # Provides details about the expression sparsity
    expr_df = stoat_obj.expression
    spatial_df = stoat_obj.spatial

    print ('Proportion of spots with tissue: '
        f'{100 * spatial_df["isTissue"].sum() / len(spatial_df):.2f} %')
    avg_sparsity = (expr_df.loc[spatial_df['isTissue']] == 0).mean(
        axis=1).mean()
    print ('Average sparsity of genes in a spot with tissue: '
        f'{100 * avg_sparsity:.2f} %')

    gene_coverage = {}
    cov_lambda = lambda row: np.mean(row > 0)
    gene_coverage[-1] = expr_df.loc[spatial_df['in_tissue']].apply(
        cov_lambda, axis=1)
    stoat_obj.filter_genes()
    success = stoat_obj.expression.loc[spatial_df['in_tissue']]
    gene_coverage[0] = success.apply(cov_lambda, axis=1)
    for i in range(1, 4):
        stoat_obj.average_expression(neighbours=i)
        avg_success = stoat_obj.avg_expression.loc[spatial_df['in_tissue']]
        gene_coverage[i] = avg_success.apply(cov_lambda, axis=1)

    if ax is None:
        _,ax = plt.subplots(figsize=(8,8))
    labels = ['Raw data', 'Filtered genes', 'Filtered + 1 neighbour',
        'Filtered + 2 neighbours', 'Filtered + 3 neighbours']
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


def load_into_df(
    filename: FILE_LIKE,
    extension: str,
) -> pd.DataFrame:
    # Loads the df from a file with a given extension
    if extension == 'tsv':
        df = pd.read_csv(filename, sep='\t', index_col=0)
    elif extension == 'feather':
        # Resetting the index will convert to DataFrame
        df = pd.read_feather(filename).set_index('index')
        df.index.rename(None, inplace=True)
    elif extension == 'parquet':
        df = pd.read_parquet(filename)
    else:
        print ('Extension not recognised')

    return df


def save_into_file(
    df: Union[pd.DataFrame, pd.Series],
    filename: FILE_LIKE,
    extension: EXTENSION,
) -> None:
    # Saves the df into a file with proper extension
    if extension == 'tsv':
        df.to_csv(filename, sep='\t')
    elif extension == 'feather':
        # Resetting the index will convert to DataFrame
        df.reset_index().to_feather(filename)
    elif extension == 'parquet':
        if type(df) == pd.DataFrame:
            df.to_parquet(filename)
        else:
            df.to_frame().to_parquet(filename)
    else:
        print ('Extension not recognised')


def collate_indegrees(
    stoat_folder: Path,
    extension: EXTENSION,
    output_file: FILE_LIKE,
) -> None:
    # Gathers the data from all indegree files in a folder and puts
    # it into a single file
    id_files = glob.glob(os.path.join(stoat_folder,
        f'indegree_*.{extension}'))
    collection = []
    for file in id_files:
        temp = load_into_df(file, extension)
        base_name = file.split('.')[-2].split('_')[-1]
        temp.rename(columns={'Indegrees': base_name}, inplace=True)
        collection.append(temp.copy())
    df = pd.concat(collection, axis=1)
    save_into_file(df, output_file, extension)


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