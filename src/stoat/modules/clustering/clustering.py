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
import hdbscan

import pandas as pd
import scanpy as sc

from anndata import AnnData
from typing import Any, Mapping, Optional, Tuple

from stoat.config import CLUSTERING
# from stoat.modules.utils import create_sparse_dataframe

### Functions ###
def normalise_data(
    spatial_table: AnnData,
    layer: Optional[str] = None,
    validity: str = 'valid',
    output: str = 'normalised',
    normalise: bool = True,
    normalise_genes: bool = True,
) -> AnnData:

    # Subset only the valid spots
    st_f = spatial_table[spatial_table.obs[validity].astype(bool)].copy()
    if normalise:
    # Normalise either each gene or each spot
        # TODO: Find a way to keep sparse if possible
        # df = create_sparse_dataframe(st_f, layer=layer)
        df = st_f.to_df(layer=layer)
        if normalise_genes:
            st_f.layers[output] = df.subtract(df.mean(axis=0), axis=1).divide(
                df.std(axis=0), axis=1).fillna(0)
        else:
            st_f.layers[output] = df.subtract(df.mean(axis=1), axis=0).divide(
                df.std(axis=1), axis=0).fillna(0)
    else:
        st_f.layers[output] = st_f.X

    return st_f


def calculate_principal_components(
    spatial_table: AnnData,
    layer: Optional[str] = None,
    n_variable: int = 2000,
) -> sc.AnnData:

    sc.pp.highly_variable_genes(
        spatial_table,
        layer=layer,
        flavor="seurat",
        n_top_genes=n_variable,
    )
    sc.pp.pca(
        spatial_table,
        layer=layer,
    )


def cluster_leiden(
    spatial_table: AnnData,
    key_added: str = 'clusters',
    **kwargs,
) -> pd.Series:

    # Performs the Leiden clustering on the provided AnnData object with
    # principal components
    sc.pp.neighbors(
        spatial_table,
        use_rep='X_pca',
    )
    sc.tl.leiden(
        spatial_table,
        key_added=key_added,
        flavor='igraph',
        n_iterations=2,
        **kwargs,
    )
    spatial_table.obs[key_added] = spatial_table.obs[key_added].astype(int)


def cluster_hdbscan(
    spatial_table: AnnData,
    key_added: str = 'clusters',
    **kwargs,
) -> pd.Series:

    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(spatial_table.obsm['X_pca'])
    # Retrieve the annotated classes with a proper index
    classes = pd.Series(clusterer.labels_, index=spatial_table.obs.index)
    # Rename the classes to follow ordering from most to least common
    counts = classes.value_counts()
    if -1 in counts:
        counts.drop(-1, inplace=True)
    renaming = {name: pos for pos,name in enumerate(counts.index)}
    classes.replace(renaming, inplace=True)
    spatial_table.obs[key_added] = classes


def change_class_annotation(
    spatial_table: AnnData,
    classes: pd.Series,
    key_added: str = 'clusters',
    exclude_extra: bool = False,
    max_classes: int = 20,
) -> Tuple[list, int]:

    # Index classes in the same way as the spatial DataFrame, fill in missing
    classes = classes.reindex(spatial_table.obs.index, fill_value=-1)
    # Count the number of actual classes (not -1)
    n_classes = classes.nunique() - (-1 in classes.values)
    # Classification for up to a given number of classes
    # In case there's more exclude the extra or group into one class
    if n_classes < max_classes:
        spatial_table.obs[key_added] = classes
        ordering = [i for i in range(n_classes)]
    else:
        if exclude_extra:
            # Exclude the classes above X (group into missing)
            spatial_table.obs[key_added] = classes.apply(
                lambda x: x if x <= max_classes - 1 else -1)
            ordering = [i for i in range(max_classes)]
        else:
            # Group the extra classes into one called X+
            spatial_table.obs[key_added] = classes.apply(
                lambda x: x if int(x) < max_classes - 1 else
                f'{max_classes - 1}+')
            ordering = ([i for i in range(max_classes - 1)] +
                [f'{max_classes - 1}+'])

    return (ordering, n_classes)


def determine_cluster_labels(
    spatial_table: AnnData,
    layer: Optional[str] = None,
    validity: str = 'valid',
    key_added: str = 'clusters',
    normalise: bool = True,
    normalise_genes: bool = True,
    clustering: CLUSTERING = 'leiden',
    clustering_opts: Mapping[Any, Any] = {},
    n_variable: int = 2000,
    exclude_extra: bool = False,
    max_classes: int = 20,
) -> Tuple[list, int]:

    # Determines the clusters in the data and returns the labels to be
    # used for plotting
    # Subset and scale data
    scaled_table = normalise_data(
        spatial_table,
        layer=layer,
        validity=validity,
        output='normalised',
        normalise=normalise,
        normalise_genes=normalise_genes,
    )
    # Add principal components to the AnnData object
    calculate_principal_components(
        scaled_table,
        layer='normalised',
        n_variable=n_variable,
    )
    # Run the desired clustering algorithm
    if clustering == 'leiden':
        cluster_leiden(
            scaled_table,
            key_added=key_added,
            **clustering_opts,
        )
    elif clustering == 'hdbscan':
        cluster_hdbscan(
            scaled_table,
            key_added=key_added,
            **clustering_opts,
        )
    else:
        print (f'Clustering type {clustering} is not implemented.')
        print (f'Implemented types: {", ".join(CLUSTERING.__args__)}')
        return
    # This modifies spatial_table, so only ordering and n_classes are returned
    return change_class_annotation(
        spatial_table,
        classes=scaled_table.obs[key_added],
        key_added=key_added,
        exclude_extra=exclude_extra,
        max_classes=max_classes,
    )


# def plot_clusters(
#     spatial: pd.DataFrame,
#     classes: pd.Series,
#     n_classes: int,
#     ordering: Iterable[str],
#     validity: str = 'in_tissue',
#     ax: Optional[plt.Axes] = None,
#     plotting_opt: Mapping[Any, Any] = {},
# ) -> Optional[Tuple[plt.Figure, plt.Axes]]:

#     # Plots the results of the clustering based on the spatial pandas
#     # DataFrame and the cluster labels
#     # Select which spots to display, rest is gray
#     spatial['Acceptable'] = (spatial[validity] & (classes != -1))
#     plotting_opt_final = dict(colourmap='tab20', validity='Acceptable',
#         n_classes=20, legend=False, ordering=ordering,
#         title=f'{n_classes} classes total')
#     plotting_opt_final.update(plotting_opt)
#     if ax is None:
#         # Create a new Figure and Axes
#         fig,ax = plot_spot_classification(spatial, classes=classes,
#             **plotting_opt_final)
#         return fig,ax
#     else:
#         # Use the provided Axes
#         plot_spot_classification(spatial, classes=classes, ax=ax,
#             **plotting_opt_final)


# def cluster_spots(
#     df: pd.DataFrame,
#     spatial: pd.DataFrame,
#     ax: Optional[plt.Axes] = None,
#     validity: str = 'in_tissue',
#     normalise: bool = True,
#     normalise_genes: bool = True,
#     clustering: CLUSTERING = 'leiden',
#     clustering_opt: Mapping[Any, Any] = {},
#     n_variable: int = 2000,
#     exclude_extra: bool = False,
#     plotting_opt: Mapping[Any, Any] = {},
# ) -> Optional[Tuple[plt.Figure, plt.Axes]]:

#     # Clusters the spots based on the provided pandas DataFrame with
#     # data and a spatial pandas DataFrame
#     classes, ordering, n_classes = determine_cluster_labels(
#         df, spatial, validity, normalise, normalise_genes, clustering,
#         clustering_opt, n_variable, exclude_extra)

#     return plot_clusters(spatial, classes, n_classes, ordering, validity, ax,
#         plotting_opt)


# def compare_clusterings(
#     df: pd.DataFrame,
#     spatial: pd.DataFrame,
#     arg_list: Iterable[dict],
#     plots_per_row: int = 2,
# ) -> Tuple[plt.Figure, plt.Axes]:

#     # Plots a meta figure displaying the clustering using different
#     # options
#     # Figure out how many rows and columns are actually needed
#     n = len(arg_list)
#     width = min(n, plots_per_row)
#     length = ceil(n / plots_per_row)
#     # Make a figure with subplots of that size
#     fig,ax = plt.subplots(length, width, figsize=(width*8, length*8),
#         tight_layout=True)
#     # Make sure Axes are a 2D array to simplify indexing
#     ax = np.reshape(ax, (length, width))
#     for pos,arg in enumerate(arg_list):
#         # Figure out the x and y coordinate on the canvas
#         x = pos // plots_per_row
#         y = pos % plots_per_row
#         # Plot the clustering with the given arguments on those Axes
#         cluster_spots(df, spatial, ax=ax[x][y], **arg)

#     return fig,ax