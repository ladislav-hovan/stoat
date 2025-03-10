### Imports ###
import hdbscan

import pandas as pd
import scanpy as sc
import numpy as np

import matplotlib.pyplot as plt

from math import ceil

from typing import Optional, Mapping, Iterable, Any, Tuple

from stoat.plotting import plot_spot_classification

### Functions ###
def normalise_data(
    df: pd.DataFrame,
    spatial: pd.DataFrame,
    validity: str = 'Acceptable',
    normalise: bool = True,
    normalise_genes: bool = True,
) -> pd.DataFrame:
    # Subset only the valid spots
    df_f = df.loc[spatial[validity]]
    if normalise:
    # Normalise either each gene or each spot
        if normalise_genes:
            df_scaled = df_f.subtract(df_f.mean(axis=0), axis=1).divide(
                df_f.std(axis=0), axis=1).fillna(0)
        else:
            df_scaled = df_f.subtract(df_f.mean(axis=1), axis=0).divide(
                df_f.std(axis=1), axis=0).fillna(0)
    else:
        df_scaled = df_f

    return df_scaled


def create_anndata_with_pc(
    df_scaled: pd.DataFrame,
    n_variable: int = 2000,
) -> sc.AnnData:
    # Create an AnnData object from the DataFrame
    adata = sc.AnnData(df_scaled.copy(), df_scaled.index.to_frame(
        name='clusters'), df_scaled.columns.to_frame(name='gene_ids'))
    # Clustering preprocessing steps
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_variable)
    sc.pp.pca(adata)

    return adata


def cluster_leiden(
    adata: sc.AnnData,
    clustering_opt: Mapping[Any, Any] = {},
) -> pd.Series:
    # Performs the Leiden clustering on the provided AnnData object with 
    # principal components
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters", **clustering_opt)
    # Retrieve the annotated classes
    classes = adata.obs['clusters'].astype(int).copy()

    return classes


def cluster_hdbscan(
    adata: sc.AnnData,
    clustering_opt: Mapping[Any, Any] = {},
) -> pd.Series:
    clusterer = hdbscan.HDBSCAN(**clustering_opt)
    clusterer.fit(adata.obsm['X_pca'])
    # Retrieve the annotated classes with a proper index
    classes = pd.Series(clusterer.labels_, index=adata.obs.index)
    # Rename the classes to follow ordering from most to least common
    counts = classes.value_counts()
    if -1 in counts:
        counts.drop(-1, inplace=True)
    renaming = {name: pos for pos,name in enumerate(counts.index)}
    classes.replace(renaming, inplace=True)

    return classes


def change_class_annotation(
    classes: pd.Series,
    spatial: pd.DataFrame,
    exclude_extra: bool = False,
    max_classes: int = 20,
):
    
    # Index classes in the same way as the spatial DataFrame, fill in missing
    classes = classes.reindex(spatial.index, fill_value=-1)
    # Count the number of actual classes (not -1)
    n_classes = classes.nunique() - (-1 in classes.values)
    # Classification for up to a given number of classes
    # In case there's more exclude the extra or group into one class
    if n_classes < max_classes:
        classes_mod = classes
        ordering = [i for i in range(n_classes)]
    else:
        if exclude_extra:
            # Exclude the classes above 19 (group into missing)
            classes_mod = classes.apply(
                lambda x: x if x <= max_classes - 1 else -1)
            ordering = [i for i in range(max_classes)]
        else:
            # Group the extra classes into one called 
            classes_mod = classes.apply(
                lambda x: x if int(x) < max_classes - 1 else 
                f'{max_classes - 1}+') 
            ordering = ([i for i in range(max_classes - 1)] + 
                [f'{max_classes - 1}+'])
    
    return (classes_mod, ordering, n_classes)


def determine_cluster_labels(
    df: pd.DataFrame,
    spatial: pd.DataFrame,
    validity: str = 'Acceptable',
    normalise: bool = True,
    normalise_genes: bool = True,
    clustering: str = 'Leiden',
    clustering_opt: Mapping[Any, Any] = {},
    n_variable: int = 2000,
    exclude_extra: bool = False,
    max_classes: int = 20,
) -> Tuple[pd.Series, list, int]:
    # Determines the clusters in the data and returns the labels to be 
    # used for plotting
    # Subset and scale data
    df_scaled = normalise_data(df, spatial, validity, normalise, 
        normalise_genes)
    # Convert to AnnData which contains the principal components
    adata = create_anndata_with_pc(df_scaled, n_variable)
    # Run the desired clustering algorithm
    if clustering == 'Leiden':
        classes = cluster_leiden(adata, clustering_opt)
    elif clustering == 'HDBScan':
        classes = cluster_hdbscan(adata, clustering_opt)
    else:
        print (f'Clustering type {clustering} not implemented, ending')
        print ('Implemented types: Leiden, HDBScan')
        return

    return change_class_annotation(classes, spatial, exclude_extra, 
        max_classes)


def plot_clusters(
    spatial: pd.DataFrame,
    classes: pd.Series,
    n_classes: int,
    ordering: Iterable[str],
    validity: str = 'isTissue',
    ax: Optional[plt.Axes] = None,
    plotting_opt: Mapping[Any, Any] = {},
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    # Plots the results of the clustering based on the spatial pandas
    # DataFrame and the cluster labels
    # Select which spots to display, rest is gray
    spatial['Acceptable'] = (spatial[validity] & (classes != -1))
    plotting_opt_final = dict(colourmap='tab20', validity='Acceptable', 
        n_classes=20, legend=False, ordering=ordering, 
        title=f'{n_classes} classes total')
    plotting_opt_final.update(plotting_opt)
    if ax is None:
        # Create a new Figure and Axes
        fig,ax = plot_spot_classification(spatial, classes=classes, 
            **plotting_opt_final)
        return fig,ax
    else:
        # Use the provided Axes
        plot_spot_classification(spatial, classes=classes, ax=ax,
            **plotting_opt_final)


def cluster_spots(
    df: pd.DataFrame,
    spatial: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    validity: str = 'isTissue',
    normalise: bool = True,
    normalise_genes: bool = True,
    clustering: str = 'Leiden',
    clustering_opt: Mapping[Any, Any] = {},
    n_variable: int = 2000,
    exclude_extra: bool = False,
    plotting_opt: Mapping[Any, Any] = {},
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    # Clusters the spots based on the provided pandas DataFrame with 
    # data and a spatial pandas DataFrame
    classes, ordering, n_classes = determine_cluster_labels(
        df, spatial, validity, normalise, normalise_genes, clustering, 
        clustering_opt, n_variable, exclude_extra)
    
    return plot_clusters(spatial, classes, n_classes, ordering, validity, ax, 
        plotting_opt)
        
    
def compare_clusterings(
    df: pd.DataFrame, 
    spatial: pd.DataFrame, 
    arg_list: Iterable[dict], 
    plots_per_row: int = 2,
) -> Tuple[plt.Figure, plt.Axes]:
    # Plots a meta figure displaying the clustering using different 
    # options
    # Figure out how many rows and columns are actually needed
    n = len(arg_list)
    width = min(n, plots_per_row)
    length = ceil(n / plots_per_row)
    # Make a figure with subplots of that size
    fig,ax = plt.subplots(length, width, figsize=(width*8, length*8), 
        tight_layout=True)
    # Make sure Axes are a 2D array to simplify indexing
    ax = np.reshape(ax, (length, width))
    for pos,arg in enumerate(arg_list):
        # Figure out the x and y coordinate on the canvas
        x = pos // plots_per_row
        y = pos % plots_per_row
        # Plot the clustering with the given arguments on those Axes
        cluster_spots(df, spatial, ax=ax[x][y], **arg)
        
    return fig,ax