### Imports and definitions ###
import os
import glob

import pandas as pd
import numpy as np
import scanpy as sc
import gseapy as gp
import matplotlib.pyplot as plt

from stoat.stoat import Stoat
from stoat.clustering import *
from stoat.stoat_analysis import StoatAnalysis

from typing import Union

PATH = Union[str, os.PathLike]
FILE_LIKE = Union[str, bytes, os.PathLike]

### Functions ###
def analyse_fully(
    stoat_obj: Stoat,
    stoat_folder: PATH,
    indegree_file: Optional[FILE_LIKE],
    extension: str,
    output_folder: PATH
) -> None:    
    # Runs the entire pipeline
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)    

    describe_expression(stoat_obj)

    if indegree_file is None:
        if len(glob.glob(os.path.join(stoat_folder, 'indegree_*'))) == 0:
            calculate_indegrees(stoat_folder, extension)
        indegree_file = os.path.join(output_folder, 
            f'final_indegree.{extension}')
        collate_indegrees(stoat_folder, extension, indegree_file)

    # Clustering on expression - unfiltered/filtered
    # Clustering on indegree - unfiltered/filtered


def prepare_stoat_object(
    prior_dir: PATH,
    data_dir: PATH
) -> Stoat: 
    # Loads a STOAT object
    stoat_obj = Stoat(
        motif_prior=prior_dir + 'tf_prior_fixed.tsv', 
        ppi_prior=prior_dir + 'ppi_prior.tsv',
        computing='gpu',
        output_extension='feather')

    stoat_obj.load_expression_raw(
        matrix_path=data_dir + 'matrix.mtx',
        barcodes_path=data_dir + 'barcodes.tsv',
        features_path=data_dir + 'features.tsv')
    
    positions_file = glob.glob(data_dir + 'tissue_positions*.csv')[0]
    stoat_obj.load_spatial(positions_file)
    
    stoat_obj.ensure_compatibility()
    stoat_obj.remove_nan()
    stoat_obj.drop_deprecated()

    return stoat_obj


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
    gene_coverage[-1] = expr_df.loc[spatial_df['isTissue']].apply(
        cov_lambda, axis=1)
    stoat_obj.filter_genes()
    success = stoat_obj.expression.loc[spatial_df['isTissue']]
    gene_coverage[0] = success.apply(cov_lambda, axis=1)
    for i in range(1, 4):
        stoat_obj.average_expression(neighbours=i)
        avg_success = stoat_obj.avg_expression.loc[spatial_df['isTissue']]
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


def calculate_indegrees(
    stoat_folder: PATH,
    extension: str,
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
    df: pd.DataFrame,
    filename: FILE_LIKE,
    extension: str,
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
    stoat_folder: PATH,
    extension: str,
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


def generate_anndata(
    df: pd.DataFrame,
    validity: pd.Series,
    classes: pd.Series,
    ens_to_name: pd.Series,
) -> sc.AnnData:    
    # Create an AnnData object that can be passed for further analysis
    df_f = df.loc[validity]
    df_scaled = df_f.subtract(df_f.mean(axis=0), axis=1).divide(
        df_f.std(axis=0), axis=1).fillna(0)
    anndata = sc.AnnData(df_scaled.copy(), df_scaled.index.to_frame(
        name='clusters'), df_scaled.columns.to_frame(name='gene_ids'))
    anndata.obs['clusters'] = classes.astype(str)
    sc.pp.log1p(anndata)
    anndata.var['gene_names'] = anndata.var['gene_ids'].apply(
        lambda x: ens_to_name[x])
    anndata.var_names = anndata.var['gene_names']

    return anndata


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