# This file contains the implementation of plotting functions for STOAT
# They can be called directly and certain STOAT functions call them

### Imports and settings ###
import pandas as pd
import numpy as np

from typing import Optional, Tuple, Mapping, Union, Callable, Iterable

import matplotlib.pyplot as plt

from matplotlib.patches import RegularPolygon, Circle
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap, Normalize

# plt.rcParams['text.usetex'] = True

### Functions ###
def plot_spot_expression(
    spatial: pd.DataFrame,
    expression: pd.DataFrame,
    validity: str = 'Success',
    colour_from: Optional[Union[str, Callable]] = None,
    colourmap: str = 'Greens',
    label: Optional[str] = None,
    title: Optional[str] = None,
    hide_overflow: bool = True,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the map of spots for the spatial expression data. It can 
    colour the spots based on an additional supplied gene name.

    Indices within the two dataframes should match.

    Parameters
    ----------
    spatial : pd.DataFrame
        The dataframe containing spatial information about the spots
    expression : pd.DataFrame
        The dataframe containing the expression levels for the spots
    validity : str, optional
        The column name in the spatial dataframe to be used to determine
        validity, by default 'Success'
    colour_from : Union[str, Callable], optional
        The name of the gene that the colouring will be based on, or a 
        function to be applied to every spot (for example sum), or None 
        to colour all valid cells the same colour, by default None
    colourmap : str, optional
        The name of the matplotlib colourmap to use, by default 
        'Greens'
    label : str, optional
        The label for the colourbar or None for no label, by default
        None
    title : str, optional
        A title for the figure or None for no title, by default None
    hide_overflow : bool, optional
        Whether to restrict the range to the bottom 99% of values and
        colour the top 1% with a different colour, by default True
    ax : plt.Axes, optional
        The axes to plot on or None to generate new ones, by default 
        None

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects of the resulting plot,
        only returned if ax was not provided
    """

    # Create a colourmap and assign colours
    if colour_from is not None:
        # Use a specific gene or a summary function
        if type(colour_from) == str:
            colour_vals = expression[colour_from]
        else:  # TODO: Explicitly check callable
            colour_vals = expression.apply(colour_from, axis=1)
        cmap, norm, colours = generate_cmap_and_colours(colour_vals, colourmap,
            None, hide_overflow)
    else:
        # All spots get the same colour
        cmap = get_cmap(colourmap)
        colours = pd.Series(1, index=spatial.index)

    # Create the basic hexagonal plot
    ax_create = ax is None
    output = plot_hexagons(spatial, spatial[validity], colours, cmap, title,
        ax)
    if ax_create:
        fig, ax = output

    if colour_from is not None:
        # Add a colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.1, shrink=0.5, pad=0.02)
        cb.ax.tick_params(labelsize=20)
        cb.set_label(label, size=30)

    if ax_create:
        return fig, ax


def plot_spot_classification(
    spatial: pd.DataFrame,
    classes: pd.Series,
    validity: str = 'Success',
    colourmap: str = 'Set2',
    legend: bool = True,
    labels: Optional[Mapping] = None,
    title: Optional[str] = None,
    ordering: Optional[Iterable[str]] = None,
    n_classes: Optional[int] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the map of spots coloured by their classification.

    Indices within the series and the dataframe should match.

    Parameters
    ----------
    spatial : pd.DataFrame
        The dataframe containing spatial information about the spots
    classes : pd.Series
        A series of class names for the spots
    validity : str, optional
        The column name in the spatial dataframe to be used to determine
        validity ('Valid' or 'Success'), by default 'Success'
    colourmap : str, optional
        The name of the matplotlib colourmap to use, by default 'Set2'
    legend : bool, optional
        Whether to plot the legend, by default True
    labels : Mapping, optional
        A mapping of class names provided in classes to names to be used
        for the legend or None to keep the names, by default None
    title : str, optional
        A title for the figure or None for no title, by default None
    ordering : Iterable[str], optional
        An ordering of the classes or None to order by the overall
        share, by default None
    n_classes: int, optional
        The number of classes to be considered for colour generation,
        useful to make plots with different number of actual classes
        consistent, None means the number of actual classes will be
        used, by default None
    ax : plt.Axes, optional
        The axes to plot on or None to generate new ones, by default 
        None

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects of the resulting plot,
        only returned if ax was not provided
    """

    # Use only classes present in valid spots
    classes_valid = classes.loc[spatial[validity]]
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
    cmap, norm, colours = generate_cmap_and_colours(classes_int, colourmap,
        cm_limits=(-0.5, n_classes-0.5), hide_overflow=False)

    # Create the basic hexagonal plot
    ax_create = ax is None
    output = plot_hexagons(spatial, spatial[validity], colours, cmap, title,
        ax)
    if ax_create:
        fig, ax = output

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
            spot_spatial = spatial.loc[spot_index]
            x,y = convert_coordinates(spot_spatial['xInd'], 
                spot_spatial['yInd'])
            hex_spot = RegularPolygon((x,y), numVertices=6, radius=2/3,
                orientation=np.radians(120), facecolor=cmap(norm(i)), 
                edgecolor='gray', label=labels[classes_list[i]])
            ax.add_patch(hex_spot)
        # Create the legend
        ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(0, 0), 
            handlelength=0.7)

    if ax_create:
        return fig, ax


def add_circle(
    x: float,
    y: float,
    ax: Optional[plt.Axes] = None,
    radius: float = 1.5,
    colour: str = 'C3',
    label: Optional[str] = None,
    fontsize: int = 25
) -> None:
    """
    Creates a circle at coordinates obtained after transformation of 
    the provided ones.

    Parameters
    ----------
    x : float
        The x coordinate to be transformed
    y : float
        The y coordinate to be transformed
    ax : plt.Axes, optional
        The Axes object where to plot the circle or None to get current
        Axes, by default None
    radius : float, optional
        The radius of the circle, by default 1.5
    colour : str, optional
        The colour of the circle, by default 'C3'
    label : Optional[str], optional
        The label inside the circle or None for no label, by default 
        None
    fontsize : float, optional
        The font size for the label, by default 25
    """


    if ax is None:
        ax = plt.gca()

    coords = convert_coordinates(x, y)

    ax.add_patch(Circle(coords, radius=radius, color=colour, lw=3, fill=False))

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
        An array of x indices
    y : np.array
        An array of y indices

    Returns
    -------
    Tuple[np.array, np.array]
        A tuple containing the converted x and y coordinates
    """

    # Vertical cartesian coordinates
    new_y = -x
    # Horizontal cartesian coordinates
    new_x = (2 * np.sin(np.radians(60)) / 3) * y

    return new_x, new_y


def generate_cmap_and_colours(
    values: pd.Series,
    colourmap: str,
    cm_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
    hide_overflow: bool = True
) -> Tuple[Colormap, Normalize, pd.Series]:
    """
    Generates the colourmap, the normalisation function and the 
    normalised series of colour values.

    Parameters
    ----------
    values : pd.Series
        The numerical values to be used for colour assignment
    colourmap : str
        The name of the matplotlib colourmap to use
    cm_limits : Tuple[Optional[float], Optional[float]], optional
        The upper and lower limits of the colourmap, inferred from
        the data if None, by default None
    hide_overflow : bool, optional
        Whether to restrict the range to the bottom 99% of values and
        colour the top 1% with a different colour, overwrites the upper
        limit if provided, by default True

    Returns
    -------
    Tuple[Colormap, Normalize, pd.Series]
        A tuple containing the colourmap, the normalisation function
        and the normalised colour series
    """

    cmap = get_cmap(colourmap).copy()

    if cm_limits is None:
        cm_limits = (None, None)
    vmin_value, vmax_value = cm_limits
    if vmin_value is None:
        vmin_value = min(values)
    if vmax_value is None:
        vmax_value = max(values)
        
    if hide_overflow:
        # Create a colourmap with an overflow value for the top 1%
        cmap.set_over('navy')
        # TODO: Dispose of the magic numbers
        vmax_value = sorted(values)[int(0.99 * len(values))]
    norm = Normalize(vmin=vmin_value, vmax=vmax_value)
    colours = values.apply(lambda x: norm(x))

    return cmap, norm, colours


def plot_hexagons(
    spatial: pd.DataFrame,
    validity: pd.Series,
    colours: pd.Series,
    colourmap: Colormap,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the map of spots for the spatial expression data as hexagons.
    Validity and colours are based on the provided series. 

    Indices within the series and the dataframe should match.

    Parameters
    ----------
    spatial : pd.DataFrame
        The dataframe containing spatial information about the spots
    validity : pd.Series
        A series of booleans determining whether the spots are to be
        considered valid
    colours : pd.Series
        A series of floats corresponding to the colours in the colourmap
    colourmap : Colormap
        The matplotlib colourmap to be used
    title : str, optional
        A title for the figure or None for no title, by default None
    ax : plt.Axes, optional
        The axes to plot on or None to generate new ones, by default 
        None

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The matplotlib Figure and Axes objects of the resulting plot,
        only returned if ax was not provided
    """

    # Cartesian coordinates
    hcoord, vcoord = convert_coordinates(spatial['xInd'], spatial['yInd'])

    # Create a figure
    ax_create = ax is None
    if ax_create:
        fig, ax = plt.subplots(1, figsize=(16, 16), tight_layout=True)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Create a DataFrame to ensure the Series align by index 
    plot_df = pd.DataFrame({'x': hcoord, 'y': vcoord, 'c': colours, 
        'v': validity})

    # Add coloured hexagons to the plot
    for ind in plot_df.index:
        x, y, c, v = plot_df.loc[ind]
        if not v:
            # Invalid hexagons are grey
            facecolor = 'gray'
        else:
            facecolor = colourmap(c)
        hex_spot = RegularPolygon((x,y), numVertices=6, radius=2/3,
            orientation=np.radians(120), facecolor=facecolor, 
            edgecolor='gray')
        ax.add_patch(hex_spot)

    # Adjust the limits
    ax.set_xlim(min(hcoord)-1, max(hcoord)+1)
    ax.set_ylim(min(vcoord)-1, max(vcoord)+1)

    if title is not None:
        # Add a figure title
        ax.set_title(title, size=ax.get_window_extent().height / 40)

    if ax_create:
        return fig, ax