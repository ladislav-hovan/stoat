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
from stoat.modules.utils import (create_sparse_dataframe, get_layer,
    load_into_df, save_df_into_filelike)

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
    format: FORMAT,
    output_file_base_in: Optional[str] = 'final_indegree',
    output_file_base_out: Optional[str] = 'final_outdegree',
) -> None:
    # Calculates indegrees and saves them for every STOAT network in
    # the folder
    process_in = output_file_base_in is not None
    process_out = output_file_base_out is not None
    if not process_in and not process_out:
        print ('Neither indegrees nor outdegrees were chosen for processing, '
            'exiting')
        return
    # Match the files in the folder and count them
    to_process = glob.glob(stoat_folder + '/stoat_*.' + format)
    n_files = len(to_process)
    print ('Found {} files to process'.format(n_files))
    in_frames = []
    out_frames = []
    # The first threshold on which to report progress (in %)
    REPORT_EVERY = 5
    threshold = REPORT_EVERY
    # Iterate over the files
    for pos, file in enumerate(to_process):
        # Load the STOAT network
        df = load_into_df(file)
        # Extract the barcode from the filename for output file naming
        bc = file.split('.')[-2].split('_')[-1]
        # Calculate and add the indegrees
        if process_in:
            indegree = df.sum()
            in_frames.append(indegree.rename(bc))
        if process_out:
            outdegree = df.sum(axis=1)
            out_frames.append(outdegree.rename(bc))
        # Report on the progress regularly
        while (100 * (pos+1) // n_files) >= threshold:
            print ('{}% complete'.format(threshold))
            threshold += REPORT_EVERY
    # Concatenate and save the DataFrames
    if process_in:
        in_df = pd.concat(in_frames, axis=1)
        save_df_into_filelike(in_df, output_file_base_in, format)
    if process_out:
        out_df = pd.concat(out_frames, axis=1)
        save_df_into_filelike(out_df, output_file_base_out, format)


def collate_degrees(
    stoat_folder: Path,
    format: FORMAT,
    output_file: FILE_LIKE,
    base_to_match: str = 'indegree_*',
    col_name: str = 'Indegrees',
    overwrite: bool = False,
) -> None:
    # Gathers the data from all indegree files in a folder and puts
    # it into a single file
    if not overwrite and os.path.exists(output_file):
        print ('The output file already exists! Specify overwrite=True '
            'if you want it replaced.')
        return
    id_files = glob.glob(os.path.join(stoat_folder,
        f'{base_to_match}.{format}'))
    collection = []
    for file in id_files:
        temp = load_into_df(file, format)
        base_name = file.split('.')[-2].split('_')[-1]
        temp.rename(columns={col_name: base_name}, inplace=True)
        collection.append(temp.copy())
    df = pd.concat(collection, axis=1)
    save_df_into_filelike(df, output_file, format)


def perform_gsea(
    anndata: sc.AnnData,
    gene_set: str,
    validity: str,
    layer: Optional[str] = None,
    cluster_col: str = 'clusters',
    exclude_vals: set = {-1},
    **kwargs,
) -> dict:
    # Perform GSEA for every identified cluster in the anndata object
    # Compares to all other clusters
    # Assumes clusters are present at obs['clusters'], data is log1p
    # transformed and gene names are used instead of Ensembl IDs
    # kwargs are passed to the gsea function

    valid_vals = sorted(v for v in anndata.obs[cluster_col].unique()
        if v not in exclude_vals)
    anndata_filt = anndata[anndata.obs[validity]]
    for i in valid_vals:
        anndata_filt.obs[f'is_{i}'] = (anndata_filt.obs[cluster_col] == i
            ).astype(int)
    data = anndata_filt.to_df(layer=layer)
    res_all = {}
    for i in valid_vals:
        in_cluster = anndata_filt.obs[f'is_{i}'].copy()
        in_cluster.sort_values(ascending=False, inplace=True)
        res_all[i] = gp.gsea(
            # row -> genes, column -> samples
            data=data.reindex(in_cluster.index).T,
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