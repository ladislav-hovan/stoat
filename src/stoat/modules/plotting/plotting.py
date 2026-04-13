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

# This file contains the implementation of plotting functions for STOAT
# They can be called directly and certain STOAT functions call them

### Imports and settings ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from anndata import AnnData
from math import ceil
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.ticker import MaxNLocator
from typing import (Any, Callable, Iterable, Mapping, Optional, Sequence,
    Tuple, Union)

from stoat.config import COL_TO_TITLE, DIMENSIONS
from stoat.modules.utils import get_layer

### Functions ###
def plot_violin(
    data: pd.Series,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots a violinplot from a pandas Series. This function is intended
    for use primarily with the plot_qc_metrics() method of STOAT.

    Parameters
    ----------
    data : pd.Series
        Series containing the data
    title : Optional[str], optional
        Title for the plot or None for no title, by default None
    ax : Optional[plt.Axes], optional
        Axes to plot on or None to generate new ones, by default None
    **kwargs
        Keyword arguments will be passed to sns.violinplot

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    ax = sns.violinplot(
        data,
        ax=ax,
        **kwargs,
    )
    # Assumes that the data starts at 0 (as most QC metrics do)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid(axis='y')
    # Remove x ticks (only one violin, not needed)
    ax.set_xticks([])
    ax.set_ylabel('Value')
    if title is not None:
        ax.set_title(title)

    return ax


def plot_qc_metrics(
    observables: pd.DataFrame,
    figsize: Tuple[float, float] = (18, 6),
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Plots the QC metrics for spots from a provided DataFrame: the number
    of genes, the total count, and the mitochondrial gene percentage.
    The distributions are plotted using violin plots.

    Parameters
    ----------
    observables : pd.DataFrame
        DataFrame containing the QC metrics
    figsize : Tuple[float, float], optional
        Dimensions of the plot, by default (18, 6)

    Returns
    -------
    Tuple[plt.Figure, np.ndarray[plt.Axes]]
        Figure and array of Axes of the resulting plot
    """

    fig,ax = plt.subplots(1, 3, figsize=figsize, tight_layout=True)
    # Plot the desired QC metrics: number of genes, total counts, MT gene %
    for i,(col, title) in enumerate(COL_TO_TITLE.items()):
        plot_violin(
            observables[col],
            title=title,
            ax=ax[i],
        )

    return (fig, ax)


def plot_joint_qc_metrics(
    observables: pd.DataFrame,
    figsize: Tuple[float, float] = (6, 6),
) -> plt.Axes:
    """
    Plots the QC metrics for spots from a provided DataFrame in a joint
    plot: number of genes versus total count, coloured by mitochondrial
    gene percentage.

    Parameters
    ----------
    observables : pd.DataFrame
        DataFrame containing the QC metrics
    figsize : Tuple[float, float], optional
        Dimensions of the plot, by default (6, 6)

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    _,ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)

    y_col,x_col,hue_col = COL_TO_TITLE.keys()

    sns.scatterplot(
        observables,
        x=x_col,
        y=y_col,
        hue=hue_col,
        alpha=0.8,
        ax=ax,
    )
    ax.set_xlabel(COL_TO_TITLE[x_col])
    ax.set_ylabel(COL_TO_TITLE[y_col])
    ax.legend(title=COL_TO_TITLE[hue_col])

    return ax


def process_colour_variable(
    spatial_table: AnnData,
    layer: Optional[str] = None,
    colour: Optional[str] = None,
) -> Optional[str]:
    """
    Interprets the colour variable by first attempting to match it
    to the variable names or observables, then trying to interpret it
    as a function to be called on the sparse matrix.

    Parameters
    ----------
    spatial_table : AnnData
        AnnData object containing the spatial information
    layer : Optional[str], optional
        Layer of the spatial data to be used or None to use the
        raw data, by default None
    colour : Optional[str], optional
        Colour variable specification, None means no colour,
        by default None

    Returns
    -------
    Optional[str]
        Colour variable name in the variable names or observables, or
        None if it was passed as such or couldn't be interpreted
    """

    data = get_layer(spatial_table, layer)
    if (colour not in spatial_table.var_names and
        colour not in spatial_table.obs.columns and
        colour is not None):
        # If not a column, try to interpret as a function to be called
        # on the sparse matrix
        if hasattr(data, colour):
            # Call the function and save the results in the observables
            fn = getattr(data, colour)
            colour = '_temp_colour_col'
            spatial_table.obs[colour] = fn(axis=1)
        else:
            print (f'Cannot recognise the color variable {colour}, '
                'switching it to None.')
            colour = None

    return colour


def plot_spot_expression(
    spatial_table: AnnData,
    layer: Optional[str] = None,
    validity: str = 'in_tissue',
    colour_from: Optional[str] = None,
    colourmap: str = 'Greens',
    label: Optional[str] = None,
    title: Optional[str] = None,
    hide_overflow: bool = True,
    overflow_threshold: float = 0.01,
    overflow_colour: Any = 'navy',
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots the map of spots for the spatial expression data. It can
    colour the spots based on an additional supplied gene name.

    Parameters
    ----------
    spatial_table : AnnData
        AnnData object containing the spatial information
    layer : Optional[str], optional
        Layer of the spatial data to be used or None to use the
        raw data, by default None
    validity : str, optional
        Column name in the observables to be used to determine validity,
        by default 'in_tissue'
    colour_from : str, optional
        Name of the gene that the colouring will be based on, or that
        of a function to be called on the sparse matrix (e.g. sum),
        or None to colour all valid cells the same colour,
        by default None
    colourmap : str, optional
        Name of the matplotlib colourmap to use, by default 'Greens'
    label : str, optional
        Label for the colourbar or None for no label, by default None
    title : str, optional
        Title for the figure or None for no title, by default None
    hide_overflow : bool, optional
        Whether to restrict the range to the bottom (1-X) proportion
        of values and colour the top X proportion with a different
        colour, X being the overflow_threshold, by default True
    overflow_threshold : float, optional
        Proportion of top values that should be coloured differently,
        only used if hide_overflow is True, by default 0.01
    overflow_colour : Any, optional
        Colour to be used for overflowing spots, only used if
        hide_overflow is True, by default 'navy'
    ax : plt.Axes, optional
        Axes to plot on or None to generate new ones, by default None

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    # Create a colourmap and assign colours
    colour_col = process_colour_variable(
        spatial_table,
        layer=layer,
        colour=colour_from,
    )
    if colour_col is not None:
        # Use a specific gene or a summary function
        if colour_col in spatial_table.var_names:
            colour_vals = spatial_table[:,colour_col].to_df(
                layer=layer)[colour_col]
        else:
            colour_vals = spatial_table.obs[colour_col]
        cmap,norm,colours = generate_cmap_and_colours(
            values=colour_vals,
            colourmap=colourmap,
            cm_limits=None,
            unclassified_label=None,
            hide_overflow=hide_overflow,
            overflow_threshold=overflow_threshold,
            overflow_colour=overflow_colour,
        )
    else:
        # All spots get the same colour
        cmap = plt.colormaps[colourmap]
        colours = pd.Series(1, index=spatial_table.obs.index)
    # Create the basic hexagonal plot
    ax = plot_hexagons(
        spatial_table,
        validity=spatial_table.obs[validity],
        colours=colours,
        title=title,
        ax=ax,
    )
    ax_height = ax.get_window_extent().height
    if colour_col is not None:
        # Add a colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(
            sm,
            ax=ax,
            fraction=0.1,
            shrink=0.5,
            pad=0.02,
        )
        cb.ax.tick_params(labelsize=(ax_height / 60))
        cb.set_label(label, size=(ax_height / 40))

    return ax


def plot_spot_classification(
    spatial_table: AnnData,
    classes: pd.Series,
    validity: str = 'in_tissue',
    colourmap: str = 'tab20',
    unclassified_label: Any = -1,
    legend: bool = True,
    labels: Optional[Mapping] = None,
    title: Optional[str] = None,
    ordering: Optional[Iterable[str]] = None,
    n_classes: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots the map of spots for the spatial expression data coloured by
    their classification.

    Parameters
    ----------
    spatial_table : AnnData
        AnnData object containing the spatial information
    classes : pd.Series
        A series of class names for the spots
    validity : str, optional
        Column name in the observables to be used to determine validity,
        by default 'in_tissue'
    colourmap : str, optional
        Name of the matplotlib colourmap to use, by default 'tab20'
    unclassified_label: Any, optional
        Label indicating the spot is not classified, by default -1
    legend : bool, optional
        Whether to plot the legend, by default True
    labels : Mapping, optional
        Mapping of class names provided in classes to names to be used
        for the legend or None to keep the names, by default None
    title : str, optional
        Title for the figure or None for no title, by default None
    ordering : Iterable[str], optional
        Ordering of the classes or None to order by the overall
        share descending, by default None
    n_classes: int, optional
        Number of classes to be considered for colour generation,
        useful to make plots with different number of actual classes
        consistent, None means the number of actual classes will be
        used, by default None
    ax : plt.Axes, optional
        Axes to plot on or None to generate new ones, by default None

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    # Use only classes present in valid spots
    classes_valid = classes.loc[spatial_table.obs[validity].astype(bool)]
    if ordering is None:
        # Order by frequency
        classes_list = list(classes_valid.value_counts().index)
    else:
        # Preserve the provided ordering
        classes_list = [i for i in ordering if i in classes_valid.unique()]
    # If the classes are not integers, define a new series
    if classes.dtype == 'int64':
        classes_int = classes_valid
    else:
        classes_int = classes_valid.apply(lambda x: classes_list.index(x))
    # Create a colourmap and assign colours
    if n_classes is None:
        n_classes = len(classes_list)
    cmap,norm,colours = generate_cmap_and_colours(
        values=classes_int,
        colourmap=colourmap,
        cm_limits=(-0.5, n_classes-0.5),
        unclassified_label=unclassified_label,
        hide_overflow=False,
    )
    # Create the basic hexagonal plot
    ax = plot_hexagons(
        spatial_table,
        validity=spatial_table.obs[validity],
        colours=colours,
        title=title,
        ax=ax,
    )
    # Add the legend if required
    if legend:
        # Choose one spot per category
        col_name = classes_int.name
        sample_points = classes_int.reset_index().groupby(col_name).min()
        # Create a dummy for labels if none were provided
        if labels is None:
            labels = {x: x for x in classes_list}
        # Replot these chosen spots with a proper label
        for i in range(len(labels)):
            spot_index = sample_points.loc[i, 'index']
            spot_spatial = spatial_table.obs.loc[spot_index]
            x,y = convert_coordinates(
                spot_spatial['array_row'],
                spot_spatial['array_col'],
            )
            hex_spot = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=2/3,
                orientation=np.radians(120),
                facecolor=cmap(norm(i)),
                edgecolor='gray',
                label=labels[classes_list[i]],
            )
            ax.add_patch(hex_spot)
        # Create the legend
        # TODO: Remove magic numbers to make it scalable
        ax.legend(
            fontsize=16,
            loc='upper left',
            bbox_to_anchor=(0, 0),
            handlelength=0.7,
        )

    return ax


def add_circle(
    x: float,
    y: float,
    ax: Optional[plt.Axes] = None,
    radius: float = 1.5,
    colour: Any = 'C3',
    label: Optional[str] = None,
    fontsize: int = 25,
) -> None:
    """
    Creates a circle at coordinates obtained after transformation of
    the provided ones.

    Parameters
    ----------
    x : float
        X coordinate to be transformed
    y : float
        Y coordinate to be transformed
    ax : plt.Axes, optional
        Axes where to plot the circle or None to get the current Axes,
        by default None
    radius : float, optional
        Radius of the circle, by default 1.5
    colour : Any, optional
        Colour of the circle, by default 'C3'
    label : Optional[str], optional
        Label inside the circle or None for no label, by default None
    fontsize : float, optional
        Font size for the label, by default 25
    """

    # Get current Axes if none were specified
    if ax is None:
        ax = plt.gca()
    # Calculate the transformed coordinates for plotting
    coords = convert_coordinates(x, y)
    # Add the circle
    ax.add_patch(Circle(coords, radius=radius, color=colour, lw=3, fill=False))
    # Add the label if required
    if label is not None:
        ax.text(*coords, label, color=colour, ha='center', va='center',
            size=fontsize)


def convert_coordinates(
    x: np.array,
    y: np.array
) -> Tuple[np.array, np.array]:
    """
    Converts the x and y indices of the spatial array to coordinates
    corresponding to the centres of hexagons in a hexagonal plot.

    Parameters
    ----------
    x : np.array
        Array of x indices
    y : np.array
        Array of y indices

    Returns
    -------
    Tuple[np.array, np.array]
        Tuple containing the converted x and y coordinates
    """

    # Vertical cartesian coordinates
    new_y = -x
    # Horizontal cartesian coordinates
    new_x = (2 * np.sin(np.radians(60)) / 3) * y

    return (new_x, new_y)


def generate_cmap_and_colours(
    values: pd.Series,
    colourmap: str,
    cm_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
    unclassified_label: Any = -1,
    hide_overflow: bool = True,
    overflow_threshold: float = 0.01,
    overflow_colour: Any = 'navy',
    unclassified_colour: Any = 'darkgray',
) -> Tuple[Colormap, Normalize, pd.Series]:
    """
    Generates the colourmap, the normalisation function and the
    normalised series of colour values.

    Parameters
    ----------
    values : pd.Series
        Numerical values to be used for colour assignment
    colourmap : str
        Name of the matplotlib colourmap to use
    cm_limits : Tuple[Optional[float], Optional[float]], optional
        Upper and lower limits of the colourmap, inferred from
        the data if None, by default None
    unclassified_label: Any, optional
        Label indicating the spot is not classified, by default -1
    hide_overflow : bool, optional
        Whether to restrict the range to the bottom (1-X) proportion
        of values and colour the top X proportion with a different
        colour, X being the overflow_threshold, by default True
    overflow_threshold : float, optional
        Proportion of top values that should be coloured differently,
        only used if hide_overflow is True, by default 0.01
    overflow_colour : Any, optional
        Colour to be used for overflowing spots, only used if
        hide_overflow is True, by default 'navy'
    unclassified_colour : Any, optional
        Colour to be assigned to unclassified spots,
        by default 'darkgray'

    Returns
    -------
    Tuple[Colormap, Normalize, pd.Series]
        Tuple containing the colourmap, the normalisation function
        and the normalised colour Series
    """

    # Make a copy of the built-in colourmap to modify
    cmap = plt.colormaps[colourmap].copy()
    # Adjust colourmap limits
    if cm_limits is None:
        cm_limits = (None, None)
    vmin_value,vmax_value = cm_limits
    if vmin_value is None:
        vmin_value = min(values)
    if vmax_value is None:
        vmax_value = max(values)
    # Potentially set an overflow value
    if hide_overflow:
        # Create a colourmap with an overflow value for the top ones
        cmap.set_over(overflow_colour)
        vmax_value = sorted(values)[int((1 - overflow_threshold) *
            len(values))]
    norm = Normalize(vmin=vmin_value, vmax=vmax_value)
    colours = values.apply(lambda x: unclassified_colour
        if x == unclassified_label else cmap(norm(x)))

    return (cmap, norm, colours)


def plot_hexagons(
    spatial_table: AnnData,
    validity: pd.Series,
    colours: pd.Series,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 16),
    edge_colour: Any = 'gray',
    invalid_colour: Any = 'gray',
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plots the map of spots for the spatial expression data as hexagons.
    Validity and colours are based on the provided series.

    Parameters
    ----------
    spatial_table : AnnData
        AnnData object containing the spatial information
    validity : pd.Series
        Series of booleans determining whether the spots are to be
        considered valid
    colours : pd.Series
        Series of floats corresponding to the colours in the colourmap
    colourmap : Colormap
        Matplotlib colourmap to be used
    title : str, optional
        Title for the figure or None for no title, by default None
    figsize : Tuple[float, float], optional
        Size of the Figure if new Axes are being generated,
        by default (16, 16)
    edge_colour : Any, optional
        Colour of the hexagon edges, by default 'gray'
    invalid_colour : Any, optional
        Colour of the invalid spots, by default 'gray'
    ax : plt.Axes, optional
        Axes to plot on or None to generate new ones, by default None

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    # Cartesian coordinates
    hcoord, vcoord = convert_coordinates(
        spatial_table.obs['array_row'],
        spatial_table.obs['array_col'],
    )
    # Create a figure
    if ax is None:
        _,ax = plt.subplots(1, figsize=figsize, tight_layout=True)
    ax.set_aspect('equal')
    ax.set_axis_off()
    # Create a DataFrame to ensure the Series align by index
    plot_df = pd.DataFrame({'x': hcoord, 'y': vcoord, 'c': colours,
        'v': validity})
    # Add coloured hexagons to the plot
    for ind in plot_df.index:
        x,y,c,v = plot_df.loc[ind]
        if not v:
            # Invalid hexagons are grey
            face_colour = invalid_colour
        else:
            face_colour = c
        hex_spot = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=2/3,
            orientation=np.radians(120),
            facecolor=face_colour,
            edgecolor=edge_colour,
        )
        ax.add_patch(hex_spot)
    # Adjust the limits
    ax.set_xlim(min(hcoord) - 1, max(hcoord) + 1)
    ax.set_ylim(min(vcoord) - 1, max(vcoord) + 1)
    if title is not None:
        # Add a figure title
        ax.set_title(title, size=ax.get_window_extent().height / 40)

    return ax


def distribute_plots(
    p_function: Callable,
    n_plots: int,
    n_cols: int,
    n_lines: int = 10,
    height_per_line: float = 0.3,
    overhead: float = 0.0,
    width_per_col: float = 3.0,
    fig: Optional[plt.Figure] = None,
    p_options: Optional[Sequence[Mapping[Any, Any]]] = None,
) -> Tuple[plt.Figure, Union[plt.Axes, np.array]]:
    """
    Distributes multiple plots generated by a single function into
    a regular array. It is intended to be used for plotting data with
    multiple rows per plot, such as top differentially expressed genes
    or bubble plots.

    Parameters
    ----------
    p_function : Callable
        Function to be used for plotting the individual plots
    n_plots : int
        Total number of plots
    n_cols : int
        Number of columns to distribute the plots into
    n_lines : int, optional
        Number of lines in a single plot, by default 10
    height_per_line : float, optional
        Height of a subplot for every line, by default 0.3
    overhead : float, optional
        Extra height of a subplot independent of the number of lines,
        by default 0.0
    width_per_col : float, optional
        Width of a subplot, by default 3.0
    fig : Optional[plt.Figure], optional
        Figure to be used or None to create a new one, by default None
    p_options : Optional[Sequence[Mapping[Any, Any]]], optional
        Sequence of Mappings of plotting options for every subplot
        or None to not provide any, by default None

    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, np.array]]
        Figure and Axes or array of Axes of the resulting plot
    """

    # Guard against no provided options
    if p_options is None:
        print ('No plotting options were provided, assuming none are '
            'required - all the plots will be the same.')
        p_options = [{} * n_plots]
    # Check number of options is correct
    elif len(p_options) != n_plots:
        raise ValueError('The numbers of plots and plotting options '
            'do not match.')
    # Calculate the required number of rows
    n_rows = ceil(n_plots / n_cols)
    # Generate a new Figure if needed, with properly computed size
    if fig is None:
        fig,ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                n_cols * width_per_col,
                n_rows * (n_lines + overhead) * height_per_line,
            ),
            tight_layout=True,
        )
    else:
        ax = fig.subplots(n_rows, n_cols)
    # Plot using the options
    for i in range(n_plots):
        if n_plots == 1:
            ax_i = ax
        else:
            ax_i = ax[i // n_cols][i % n_cols]
        p_function(**p_options[i], ax=ax_i)
    # Hide the possible extra axes from the plot
    for i in range(n_plots, n_rows * n_cols):
        ax_i = ax[i // n_cols][i % n_cols]
        ax_i.set_axis_off()

    return (fig, ax)


def plot_deg_data_single(
    data: dict,
    n_genes: int = 10,
    max_score: float = 50,
    score_spacing: float = 10,
    cmap: str = 'tab20',
    cluster_id: int = 0,
    max_clusters: int = 20,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots the most differentially expressed genes for a single cluster.

    Parameters
    ----------
    data : dict
        Dictionary with the data about the genes, in a format provided
        by scanpy's rank_genes_groups() function
    n_genes : int, optional
        Number of top genes to show, by default 10
    max_score : float, optional
        Maximum range of the x axis with score, by default 50
    score_spacing : float, optional
        Spacing of the score ticks, by default 10
    cmap : str, optional
        Colourmap for colour assignment, by default 'tab20'
    cluster_id : int, optional
        ID of the cluster, by default 0
    max_clusters : int, optional
        Maximum number of clusters to be considered for colour
        generation, useful to make plots with different
        number of clusters consistent, should probably correspond to
        the number of colours on the colourmap, by default 20
    ax : Optional[plt.Axes], optional
        Axes to plot on or None to generate new ones, by default None

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    # Create appropriately sized Axes if required
    dims = DIMENSIONS.loc['deg']
    if ax is None:
        _,ax = plt.subplots(figsize=(dims['width_per_col'],
            (dims['overhead'] + n_genes) * dims['height_per_line']))
    # Adjust the axes' ticks and labels
    ax.set_xlim(0, max_score)
    ax.set_ylim(-n_genes, 1)
    ax.grid(False)
    ax.set_xticks([i for i in range(0, max_score + 1, score_spacing)])
    ax.set_yticks([])  # No yticks
    # Remove the top and right spines altogether
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Get the colourmap for colour assignment
    cm = plt.colormaps[cmap]
    # In the dictionary, the labels are strings, not integers
    label = str(cluster_id)
    # Plot all the bars
    ax.barh([-i for i in range(n_genes)],
        data['scores'][label][:n_genes] - 1,
        color=cm(cluster_id / max_clusters), align='center')
    # Add the labels
    for pos,(n,s) in enumerate(zip(data['names'][label][:n_genes],
        data['scores'][label][:n_genes])):
        ax.text(s, -pos, n, ha='left', va='center')

    return ax


def plot_deg_data(
    data: dict,
    n_genes: int = 10,
    n_cols: int = 4,
    max_score: float = 50,
    score_spacing: float = 10,
    cmap: str = 'tab20',
    max_clusters: int = 20,
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, Union[plt.Axes, np.array]]:
    """
    Plots the most differentially expressed genes for all the clusters
    in the data.

    Parameters
    ----------
    data : dict
        Dictionary with the data about the genes, in a format provided
        by scanpy's rank_genes_groups() function
    n_genes : int, optional
        Number of top genes to show, by default 10
    n_cols : int, optional
        Number of columns to distribute the plots into, by default 4
    max_score : float, optional
        Maximum range of the x axis with score, by default 50
    score_spacing : float, optional
        Spacing of the score ticks, by default 10
    cmap : str, optional
        Colourmap for colour assignment, by default 'tab20'
    max_clusters : int, optional
        Maximum number of clusters to be considered for colour
        generation, useful to make plots with different
        number of clusters consistent, should probably correspond to
        the number of colours on the colourmap, by default 20
    fig : Optional[plt.Figure], optional
        Figure to be used or None to create a new one, by default None

    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, np.array]]
        Figure and Axes or array of Axes of the resulting plot
    """

    # Retrieve the dimensions from the configuration
    dims = DIMENSIONS.loc['deg']
    n_plots = len(data['scores'][0])
    # Options shared between plots
    common_opts = {
        'max_score': max_score,
        'score_spacing': score_spacing,
        'cmap': cmap,
        'max_clusters': max_clusters,
        'data': data,
        'n_genes': n_genes,
    }
    p_options = [common_opts.copy() for _ in range(n_plots)]
    # The only plot specific option is the cluster ID
    for i in range(n_plots):
        p_options[i]['cluster_id'] = i
    # Call the master function for distributed plots
    fig,ax = distribute_plots(
        p_function=plot_deg_data_single,
        n_plots=n_plots,
        n_cols=n_cols,
        n_lines=n_genes,
        height_per_line=dims['height_per_line'],
        overhead=dims['overhead'],
        width_per_col=dims['width_per_col'],
        fig=fig,
        p_options=p_options,
    )

    return (fig, ax)


def plot_deg_heatmap(
    data: pd.DataFrame,
    figsize: Tuple[int, int] = (6,5),
    percentile: Tuple[int, int] = (2,98),
    title: str = '',
    cmap: str = 'viridis',
    n_cluster_spots: Optional[int] = None,
    cluster_colour: Any = 'red',
    background_colour: Any = 'lightgrey',
    show_every: int = 1,
) -> plt.Axes:

    df = data.iloc[::-1]

    vmin = np.percentile(df, percentile[0])
    vmax = np.percentile(df, percentile[1])
    norm = Normalize(vmin=vmin, vmax=vmax)

    _,ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(df.values, rasterized=True, norm=norm, cmap=cmap)
    ax.set_title(title, size=18)

    if n_cluster_spots is not None:
        ax.plot([0, n_cluster_spots], [-1,-1], color=cluster_colour, lw=3)
        ax.plot([n_cluster_spots, len(df.columns)], [-1,-1],
            color=background_colour, lw=3)

    ax.set_yticks([i+0.5 for i in range(0, len(df), show_every)],
        df.index[::show_every])
    ax.yaxis.set_tick_params('major', left=False)

    cb = plt.colorbar(mappable=pcm, ax=ax, shrink=0.5, aspect=10)
    cb.ax.yaxis.set_tick_params(
        color='white', direction='in', left=True, right=True,
    )
    cb_locator = MaxNLocator(nbins=5, integer=True)
    cb.locator = cb_locator
    cb.update_ticks()
    cb.ax.set_title('N. E.', loc='left', fontweight='bold')
    for spine in cb.ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)

    return ax


def plot_gsea_dotplot(
    df: pd.DataFrame,
    column: str = 'FDR q-val',
    n_terms: int = 10,
    threshold: float = 0.05,
    x: str = 'NES',
    y: str = 'Term',
    title: str = '',
    cmap: str = 'viridis_r',
    dot_scale: float = 5.0,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4, 6),
) -> plt.Axes:
    # TODO: Fix this whole thing, figure out why we had to comment so much out

    # df = df.loc[df[column] <= threshold]
    # if len(df) == 0:
    #     msg = f'No enriched terms with {column} <= {threshold}'
    #     if ax is None:
    #         raise ValueError(msg)
    #     else:
    #         ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14,
    #             transform=ax.transAxes)
    #         ax.set_axis_off()
    #         return

    colnd = {'Adjusted P-value': 'FDR', 'P-value': 'Pval', 'NOM p-val': 'Pval',
        'FDR q-val': 'FDR'}
    if column in colnd:
        df = df.sort_values(by=column)
        df[column] = df[column].replace(0, None).bfill()
        df['p_inv'] = np.log10(1 / df[column].astype(float))
        colname = 'p_inv'
        cbar_title = r'$\log_{10} \frac{1}{ ' + colnd[column] + ' }$'
    else:
        colname = column
        cbar_title = column

    # df = df.sort_values(by=colname).tail(n_terms)
    df = df.head(n_terms)[::-1]

    if df.columns.isin(['Overlap', 'Tag %']).any():
        ol = df.columns[df.columns.isin(['Overlap', 'Tag %'])]
        temp = df[ol].squeeze(axis=1).str.split('/', expand=True).astype(int)
        df['Hits_ratio'] = temp.iloc[:, 0] / temp.iloc[:, 1]
    else:
        df['Hits_ratio'] = 1.0

    df['area'] = (df['Hits_ratio'] * dot_scale *
        plt.rcParams['lines.markersize']).pow(2)

    if ax is None:
        _,ax = plt.subplots(figsize=figsize)
    fig = ax.get_figure()

    colmap = df[colname].astype(int)
    vmin = np.percentile(colmap, 2)
    vmax = np.percentile(colmap, 98)

    sc = ax.scatter(
        x=x,
        y=y,
        data=df,
        s='area',
        edgecolors='none',
        c=colname,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
    )
    ax.set_xlabel(x, fontsize=14, fontweight='bold')
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_axisbelow(True)
    ax.grid(axis='y', zorder=-1)
    ax.margins(x=0.25)
    ax.set_ylim(-1, len(df))

    # handles, labels = sc.legend_elements(
    #     prop='sizes',
    #     num=3,
    #     fmt='{x:.2f}',
    #     color='gray',
    #     func=lambda s: (np.sqrt(s) / plt.rcParams['lines.markersize'] /
    #         dot_scale),
    # )
    # ax.legend(
    #     handles,
    #     labels,
    #     title='% Genes\nin set',
    #     bbox_to_anchor=(1.02, 0.9),
    #     loc='upper left',
    #     frameon=False,
    #     labelspacing=2,
    # )
    ax.set_title(title, fontsize=20, fontweight='bold')

    # cbar = fig.colorbar(
    #     sc,
    #     shrink=0.25,
    #     aspect=10,
    #     anchor=(0.0, 0.2),
    #     location='right',
    # )
    # cbar.ax.yaxis.set_tick_params(
    #     color='white', direction='in', left=True, right=True
    # )
    # cbar.ax.set_title(cbar_title, loc='left', fontweight='bold')
    # for _, spine in cbar.ax.spines.items():
    #     spine.set_visible(False)

    return ax


def plot_gsea_dotplots(
    data: Mapping[int, pd.DataFrame],
    n_cols: int = 3,
    column: str = 'FDR q-val',
    n_terms: int = 10,
    threshold: float = 0.05,
    x: str = 'NES',
    y: str = 'Term',
    cmap: str = 'viridis_r',
    dot_scale: float = 5.0,
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, Union[plt.Axes, np.array]]:

    dims = DIMENSIONS.loc['gsea']
    n_plots = len(data)
    common_opts = {'column': column, 'threshold': threshold,
        'cmap': cmap, 'x': x, 'y': y, 'dot_scale': dot_scale,
        'n_terms': n_terms}
    p_options = [common_opts.copy() for _ in range(n_plots)]
    for pos,(k,v) in enumerate(data.items()):
        p_options[pos]['df'] = v
        p_options[pos]['title'] = f'Cluster {k}'

    fig,ax = distribute_plots(
        p_function=plot_gsea_dotplot,
        n_plots=n_plots,
        n_cols=n_cols,
        n_lines=n_terms,
        height_per_line=dims['height_per_line'],
        overhead=dims['overhead'],
        width_per_col=dims['width_per_col'],
        fig=fig,
        p_options=p_options,
    )

    return (fig, ax)


def plot_cluster_matching_single(
    matching: pd.Series,
    cluster_id: int,
    colours: Mapping[int, Any],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots a pie chart showing how a given cluster maps onto clusters
    in a different clustering.

    Parameters
    ----------
    matching : pd.Series
        Series containing the cluster IDs and counts in the second
        clustering
    cluster_id : int
        Cluster ID of the current cluster in the first clustering
    colours : Mapping[int, Any]
        Mapping of cluster IDs to colours, assumed same in both
        clusterings
    ax : Optional[plt.Axes], optional
        Axes to plot on or None to generate new ones, by default None

    Returns
    -------
    plt.Axes
        Axes of the resulting plot
    """

    # Create appropriately sized Axes if required
    dims = DIMENSIONS.loc['match']
    if ax is None:
        _,ax = plt.subplots(figsize=(dims['width_per_col'], dims['overhead']))
    # Create the pie chart
    ax.pie(
        matching.values,
        colors=[colours[k] for k in matching.index],
    )
    # Set the title with background colour
    ax.set_title(
        f'Cluster {cluster_id}',
        weight='bold',
        color='white',
        backgroundcolor=colours[cluster_id],
    )

    return ax


def plot_cluster_matching(
    first: pd.Series,
    second: pd.Series,
    n_cols: int = 4,
    cmap: str = 'tab20',
    max_clusters: int = 20,
    legend: bool = True,
    unclassified_label: Any = -1,
    unclassified_colour: str = 'darkgray',
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, Union[plt.Axes, np.array]]:
    """
    Plots an array of pie charts showing the correspondence between
    different clusterings.

    Parameters
    ----------
    first : pd.Series
        _description_
    second : pd.Series
        _description_
    n_cols : int, optional
        _description_, by default 4
    cmap : str, optional
        _description_, by default 'tab20'
    max_clusters : int, optional
        _description_, by default 20
    legend : bool, optional
        _description_, by default True
    unclassified_label : Any, optional
        _description_, by default -1
    unclassified_colour : str, optional,
        _description_, by default 'darkgray
    fig : Optional[plt.Figure], optional
        _description_, by default None

    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, np.array]]
        Figure and Axes or array of Axes of the resulting plot
    """

    # Merge them into one DataFrame
    comp = pd.DataFrame([first.rename('first'), second.rename('second')]).T
    # Determine the matching clusters
    matching = comp.groupby('first').value_counts()
    # Figure out the unique labels in the two Series
    first_labels = comp['first'].unique()
    second_labels = comp['second'].unique()
    # Figure out the higher number of clusters to be used
    n_clusters_1 = len(first_labels) - int(unclassified_label in first_labels)
    n_clusters_2 = (len(second_labels) -
        int(unclassified_label in second_labels))
    n_clusters = max(n_clusters_1, n_clusters_2)
    # Figure out the colours for the clusters from the colourmap
    cm = plt.colormaps[cmap]
    colours = {i: cm(i / max_clusters) for i in range(max_clusters)}
    colours[unclassified_label] = unclassified_colour
    # Set plotting options for the individual plots
    p_options = [{'colours': colours} for _ in range(n_clusters_1)]
    for i in range(n_clusters_1):
        p_options[i]['matching'] = matching.loc[i]
        p_options[i]['cluster_id'] = i
    # Distribute the plots
    dims = DIMENSIONS.loc['match']
    fig,ax = distribute_plots(
        plot_cluster_matching_single,
        n_plots=n_clusters_1,
        n_cols=n_cols,
        n_lines=0,
        height_per_line=dims['height_per_line'],
        overhead=dims['overhead'],
        width_per_col=dims['width_per_col'],
        fig=fig,
        p_options=p_options,
    )
    # Add legend if required
    if legend:
        custom_lines = (
            [plt.Line2D([0], [0], color=cm(i / max_clusters), lw=8)
            for i in range(n_clusters)] +
            [plt.Line2D([0], [0], color='grey', lw=8)]
        )
        fig.legend(
            custom_lines,
            [f'Cluster {i}' for i in range(n_clusters)] + ['Not in cluster'],
            bbox_to_anchor=(1, 1),
            loc='upper left',
            handlelength=0.7,
        )

    return (fig, ax)