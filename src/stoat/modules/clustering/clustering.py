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
import random
import torch
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
# Some deprecated imports of input functions in SpaGCN
with warnings.catch_warnings():
    warnings.filterwarnings('ignore',
        'Importing read_.* from `anndata` is deprecated')
    import SpaGCN as spg

from anndata import AnnData
from typing import Any, Mapping, Optional, Tuple

from stoat.config import CLUSTERING

### Functions ###
def subset_and_normalise_data(
    spatial_table: AnnData,
    layer: Optional[str] = None,
    validity: str = 'valid',
    output: str = 'normalised',
    normalise: bool = True,
    normalise_genes: bool = True,
) -> AnnData:
    """
    Creates a new AnnData object with a subset of the data provided,
    also optionally normalises the data. A new layer is created to hold
    the data.

    Parameters
    ----------
    spatial_table : AnnData
        AnnData object containing the data
    layer : Optional[str], optional
        Name of the newly created layer, by default 'normalised'
    validity : str, optional
        Column indicating which indices are valid, by default 'valid'
    output : str, optional
        New layer to be created for normalised data,
        by default 'normalised'
    normalise : bool, optional
        Whether to normalise the data, by default True
    normalise_genes : bool, optional
        Whether to normalise the genes, if False then spots are
        normalised instead, only used if normalise is True,
        by default True

    Returns
    -------
    AnnData
        AnnData object containing the valid subset of the data,
        optionally also normalised
    """

    # Subset only the valid spots
    st_f = spatial_table[spatial_table.obs[validity].astype(bool)].copy()
    if normalise:
    # Normalise either each gene or each spot
        # TODO: Find a way to keep sparse if possible
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


def order_by_prevalence(
    labels: pd.Series,
    unclassified_label: Any = -1,
) -> pd.Series:

    counts = labels[labels != unclassified_label].value_counts()
    mapping = {}
    for pos,k in enumerate(counts.keys()):
        mapping[k] = pos

    return labels.map(mapping)


def cluster_leiden(
    spatial_table: AnnData,
    key_added: str = 'clusters',
    **kwargs,
) -> None:

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
    spatial_table.obs[key_added] = order_by_prevalence(
        spatial_table.obs[key_added])


def cluster_hdbscan(
    spatial_table: AnnData,
    key_added: str = 'clusters',
    **kwargs,
) -> None:

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


def cluster_spagcn(
    spatial_table: AnnData,
    key_added: str = 'clusters',
    p: float = 0.5,
    r_seed: Optional[int] = None,
    t_seed: Optional[int] = None,
    n_seed: Optional[int] = None,
    **kwargs,
) -> None:

    # Calculate the adjacency matrix
    adj_mat = spg.calculate_adj_matrix(
        x=spatial_table.obs['array_row'],
        y=spatial_table.obs['array_col'],
        x_pixel=spatial_table.obsm['spatial'][:,1],
        y_pixel=spatial_table.obsm['spatial'][:,0],
        histology=False,
    )
    # Find a suitable value of l given the matrix and p
    l = spg.search_l(p, adj_mat, start=0.01, end=1000, tol=0.01, max_run=100)
    scaled_ad = AnnData(spatial_table.layers['normalised'])
    clf = spg.SpaGCN()
    clf.set_l(l)
    # Set the different random seeds
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    # Actually run the model
    clf.train(scaled_ad, adj_mat, init_spa=True, init='louvain', **kwargs)
    y_pred,_ = clf.predict()
    spatial_table.obs[key_added] = y_pred
    spatial_table.obs[key_added] = order_by_prevalence(
        spatial_table.obs[key_added])


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
    scaled_table = subset_and_normalise_data(
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
    elif clustering == 'spagcn':
        cluster_spagcn(
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