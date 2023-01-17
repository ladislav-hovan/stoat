# This file contains the implementation of plotting functions for STOAT
# They can be called directly and certain STOAT functions call them

### Imports and settings ###
import pandas as pd
import numpy as np

from typing import Optional, Tuple, Mapping

import matplotlib.pyplot as plt

from matplotlib.patches import RegularPolygon
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap, Normalize

# plt.rcParams['text.usetex'] = True

### Functions ###
def plot_spot_expression(
    spatial: pd.DataFrame,
    expression: pd.DataFrame,
    validity: str = 'Success',
    colour_from: Optional[str] = None,
    colourmap: str = 'Greens',
    label: Optional[str] = None,
    title: Optional[str] = None,
    hide_overflow: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    
    """

    # Create a colourmap and assign colours
    if colour_from is not None:
        # Sum all the genes or use a specific one
        if colour_from == 'sum_all':
            colour_vals = expression.sum(axis=1)
        else:
            colour_vals = expression[colour_from]
        cmap, norm, colours = generate_cmap_and_colours(colour_vals, colourmap,
            hide_overflow)
    else:
        # All spots get the same colour
        cmap = get_cmap(colourmap)
        colours = pd.Series(1, index=spatial.index)

    # Create the basic hexagonal plot
    fig, ax = plot_hexagons(spatial, spatial[validity], colours, cmap, title)

    if colour_from is not None:
        # Add a colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.1, shrink=0.5, pad=0.02)
        cb.ax.tick_params(labelsize=20)
        cb.set_label(label, size=30)
    
    fig.show()

    return fig, ax


def plot_spot_classification(
    spatial: pd.DataFrame,
    classes: pd.Series,
    validity: str = 'Success',
    colourmap: str = 'Set2',
    legend: bool = True,
    labels: Optional[Mapping] = None,
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:


    # If the classes are not integers, define a new series
    classes_list = list(classes.unique())
    if classes.dtype == 'int64':
        classes_int = classes
    else:
        classes_int = classes.apply(lambda x: classes_list.index(x))

    # Create a colourmap and assign colours
    cmap, norm, colours = generate_cmap_and_colours(classes_int, colourmap,
        hide_overflow=False)

    # Create the basic hexagonal plot
    fig, ax = plot_hexagons(spatial, spatial[validity], colours, cmap, title)

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
            hex = RegularPolygon((x,y), numVertices=6, radius=2/3,
                orientation=np.radians(120), facecolor=cmap(norm(i)), 
                edgecolor='gray', label=labels[classes_list[i]])
            ax.add_patch(hex)
        # Create the legend
        ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(0, 0), 
            handlelength=0.7)

    fig.show()

    return fig, ax


def convert_coordinates(
    x: np.array,
    y: np.array
) -> Tuple[np.array, np.array]:


    # Vertical cartesian coordinates
    new_y = -x
    # Horizontal cartesian coordinates
    new_x = (2 * np.sin(np.radians(60)) / 3) * y

    return new_x, new_y


def generate_cmap_and_colours(
    values: pd.Series,
    colourmap: str,
    hide_overflow = True
) -> Tuple[Colormap, Normalize, pd.Series]:


    cmap = get_cmap(colourmap).copy()
    if hide_overflow:
        # Create a colourmap with an overflow value for the top 1%
        cmap.set_over('navy')
        vmax_value = sorted(values)[int(0.99 * len(values))]
        norm = Normalize(vmin=min(values), vmax=vmax_value)
    else:
        norm = Normalize(vmin=min(values), vmax=max(values))
    colours = values.apply(lambda x: norm(x))

    return cmap, norm, colours


def plot_hexagons(
    spatial: pd.DataFrame,
    validity: pd.Series,
    colours: pd.Series,
    colourmap: Colormap,
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """_summary_

    Parameters
    ----------
    spatial : pd.DataFrame
        _description_
    validity : pd.Series
        _description_
    colours : pd.Series
        _description_
    colourmap : Colormap
        _description_
    title : Optional[str], optional
        _description_, by default None

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        _description_
    """

    # Cartesian coordinates
    hcoord, vcoord = convert_coordinates(spatial['xInd'], spatial['yInd'])

    # Create a figure
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
        hex = RegularPolygon((x,y), numVertices=6, radius=2/3,
            orientation=np.radians(120), facecolor=facecolor, 
            edgecolor='gray')
        ax.add_patch(hex)

    # Adjust the limits
    ax.set_xlim(min(hcoord)-1, max(hcoord)+1)
    ax.set_ylim(min(vcoord)-1, max(vcoord)+1)

    if title is not None:
        # Add a figure title
        ax.set_title(title, size=40)
    
    fig.show()

    return fig, ax