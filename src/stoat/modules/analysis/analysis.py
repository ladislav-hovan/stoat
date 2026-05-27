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
import glob
import os

import gseapy as gp
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from anndata import AnnData
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from typing import Optional, Tuple

from stoat import Stoat
from stoat.config import FILE_LIKE, FORMAT
from stoat.modules.utils import (format_docstring, get_layer, load_into_df,
    save_df_into_filelike)

### Functions ###
def describe_expression(
    stoat_obj: Stoat,
    layer: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Axes:
    """
    Shows details about the expression sparsity within a Stoat object,
    including if it was averaged up to three neighbouring layers out.

    Parameters
    ----------
    stoat_obj : Stoat
        Stoat object with the expression data already loaded
    layer : Optional[str], optional
        Layer of the spatial data to be used or None to use the
        raw data, by default None
    ax : Optional[plt.Axes], optional
        Axes to plot on or None to generate new ones, by default None
    figsize : Tuple[float, float], optional
        Size of the figure if it is to be created, by default (8, 8)

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    # Print basic information without the need for averaging
    st = stoat_obj.spatial[stoat_obj.table]
    print ('Proportion of spots with tissue: '
        f'{100 * st.obs["in_tissue"].sum() / len(st.obs):.2f} %')
    expr_mat = get_layer(st, layer)
    avg_sparsity = (expr_mat[st.obs['in_tissue'].values,:] != 0).mean(
        axis=1).mean()
    print ('Average sparsity of genes in a spot with tissue: '
        f'{100 * (1 - avg_sparsity):.2f} %')
    # Perform averaging for layers and save the information
    gene_coverage = {}
    gene_coverage[0] = (expr_mat[st.obs['in_tissue'].values,:] != 0).mean(
        axis=1).A1
    for i in range(1, 4):
        stoat_obj.average_expression(n_rings=i)
        avg_success = st.layers['averaged'][st.obs['in_tissue'].values,:]
        gene_coverage[i] = (avg_success != 0).mean(axis=1).A1
    # Plot the histograms
    if ax is None:
        # Generate new Axes if needed
        _,ax = plt.subplots(figsize=figsize)
    labels = ['Original data', '1st neighbours', '2nd neighbours',
        '3rd neighbours']
    for label,data in zip(labels, gene_coverage.values()):
        ax.hist(data, bins=120, range=(0, 1), alpha=0.5, label=label)
    # Adjust the plot parameters
    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of genes with non-zero reads', size=16)
    ax.set_ylabel('Spot count', size=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    return ax


@format_docstring(formats=', '.join(FORMAT.__args__))
def calculate_degrees(
    stoat_folder: Path,
    format: FORMAT,
    output_file_base_in: Optional[str] = 'final_indegree',
    output_file_base_out: Optional[str] = 'final_outdegree',
    overwrite: bool = False,
) -> None:
    """
    Calculates degrees for every STOAT network in the given folder and
    saves them into output files.

    Parameters
    ----------
    stoat_folder : Path
        Folder in which to look for STOAT network files
    format : FORMAT
        Format of the STOAT networks, it will also be the one to save
        the degrees in, one of: {formats}
    output_file_base_in : Optional[str], optional
        Prefix for the indegree output file (extension will be added) or
        None to not save indegrees, by default 'final_indegree'
    output_file_base_out : Optional[str], optional
        Prefix for the outdegree output file (extension will be added)
        or None to not save outdegrees, by default 'final_outdegree'
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    """

    # Check which files need to be generated
    process_in = output_file_base_in is not None
    process_out = output_file_base_out is not None
    if not overwrite and process_in and os.path.exists(
        f'{os.path.join(stoat_folder, output_file_base_in)}.{format}'):
        print ('The output file for indegrees already exists! Specify '
            'overwrite=True if you want it replaced.')
        process_in = False
    if not overwrite and process_out and os.path.exists(
        f'{os.path.join(stoat_folder, output_file_base_out)}.{format}'):
        print ('The output file for outdegrees already exists! Specify '
            'overwrite=True if you want it replaced.')
        process_out = False
    if not process_in and not process_out:
        # No files to be generated
        print ('Neither indegrees nor outdegrees were chosen for processing, '
            'exiting.')
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


@format_docstring(formats=', '.join(FORMAT.__args__))
def collate_degrees(
    stoat_folder: Path,
    format: FORMAT,
    output_file: FILE_LIKE,
    base_to_match: str = 'indegree_*',
    col_name: str = 'Indegrees',
    overwrite: bool = False,
) -> None:
    """
    Collates the data from degree files in a given folder into a single
    file.

    Parameters
    ----------
    stoat_folder : Path
        Folder in which to look for degree files
    format : FORMAT
        Format of the degree files, it will also be the one to save
        the collated degrees in, one of:
        {formats}
    output_file : FILE_LIKE
        Path to a file or an IO buffer where the collated degrees will
        be saved
    base_to_match : str, optional
        Prefix of the degree files to be collated,
        by default 'indegree_*'
    col_name : str, optional
        Name of the column in the degree files containing the data,
        by default 'Indegrees'
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    """

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


def calculate_ari(
    series1: pd.Series,
    series2: pd.Series,
    unclassified_label: int = -1,
) -> float:
    """
    Calculates the adjusted Rand index between two different pandas
    Series of annotations.

    Parameters
    ----------
    series1 : pd.Series
        First Series of annotations
    series2 : pd.Series
        Second Series of annotations
    unclassified_label : int, optional
        Label used for unclassified indices which are excluded from the
        calculation, by default -1

    Returns
    -------
    float
        Adjusted Rand index of the two annotations
    """

    clust1 = series1[series1 != unclassified_label]
    clust2 = series2[series2 != unclassified_label]
    overlap = set(clust1.index).intersection(clust2.index)
    if len(overlap) != len(clust1) or len(overlap) != len(clust2):
        print ('Some samples were removed when overlapping the clusters.')
    new_id = sorted(overlap)

    return adjusted_rand_score(series1.loc[new_id], series2.loc[new_id])


def perform_gsea(
    anndata: AnnData,
    gene_set: str,
    validity: str,
    layer: Optional[str] = None,
    cluster_col: str = 'clusters',
    exclude_vals: set = {-1},
    **kwargs,
) -> dict:
    """
    Perform GSEA for every identified cluster in the AnnData object.
    The comparisons are done to all the other clusters. Assumes the data
    is log1p transformed and gene names are used.

    Parameters
    ----------
    anndata : AnnData
        AnnData object containing the data
    gene_set : str
        Name of the gene sets to be used
    validity : str
        Column of the observables with booleans determining which
        indices should be used
    layer : Optional[str], optional
        Layer of the data to be used or None to use the raw data,
        by default None
    cluster_col : str, optional
        Column of the observables corresponding to the clustering
        annotations, by default 'clusters'
    exclude_vals : set, optional
        Set of values in the clustering column to be excluded,
        by default {-1}
    **kwargs
        Keyword arguments will be passed to gseapy.gsea

    Returns
    -------
    dict
        Dictionary linking the clustering annotations to the GSEA
        results for the given cluster
    """

    # Select the valid indices
    anndata_filt = anndata[anndata.obs[validity]]
    # Select all valid values (not excluded)
    valid_vals = sorted(v for v in anndata_filt.obs[cluster_col].unique()
        if v not in exclude_vals)
    # Create helper variables (in/out of cluster)
    for i in valid_vals:
        # 1 or 0
        anndata_filt.obs[f'is_{i}'] = (anndata_filt.obs[cluster_col] == i
            ).astype(int)
    data = anndata_filt.to_df(layer=layer)
    res_all = {}
    # Iterate over all clusters and run GSEA
    for i in valid_vals:
        in_cluster = anndata_filt.obs[f'is_{i}'].copy()
        # Make sure the first value is 1 so that comparison is 1 vs 0
        in_cluster.sort_values(ascending=False, inplace=True)
        res_all[i] = gp.gsea(
            # row -> genes, column -> samples - hence the transpose
            data=data.reindex(in_cluster.index).T,
            gene_sets=gene_set,
            cls=in_cluster,
            **kwargs,
        )

    return res_all


def perform_deg_analysis(
    anndata: AnnData,
    validity: str,
    layer: Optional[str] = None,
    cluster_col: str = 'clusters',
    exclude_vals: set = {-1},
    **kwargs
) -> AnnData:
    """
    Perform differentially expressed gene anaalysis for every identified
    cluster in the AnnData object. The comparisons are done to all the
    other clusters. Assumes the data is log1p transformed and gene names
    are used.

    Parameters
    ----------
    anndata : AnnData
        AnnData object containing the data
    validity : str
        Column of the observables with booleans determining which
        indices should be used
    layer : Optional[str], optional
        Layer of the data to be used or None to use the raw data,
        by default None
    cluster_col : str, optional
        Column of the observables corresponding to the clustering
        annotations, by default 'clusters'
    exclude_vals : set, optional
        Set of values in the clustering column to be excluded,
        by default {-1}
    **kwargs
        Keyword arguments will be passed to sc.tl.rank_genes_groups

    Returns
    -------
    AnnData
        AnnData object containing the DEG analysis results
    """

    # Select the valid indices (both validity columns and non-excluded values)
    anndata_filt = anndata[anndata.obs[validity] &
        ~anndata.obs[cluster_col].isin(exclude_vals)]
    deg_adata = sc.tl.rank_genes_groups(anndata_filt, groupby=cluster_col,
        layer=layer, method='wilcoxon', tie_correct=True, copy=True, **kwargs)

    return deg_adata