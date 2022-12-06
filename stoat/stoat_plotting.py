# This file contains the implementation of plotting functions for STOAT
# They can be called directly and certain STOAT functions call them

import pandas as pd
import numpy as np

from typing import Optional, Tuple

import matplotlib.pyplot as plt

from matplotlib.patches import RegularPolygon
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


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

    # Vertical cartesian coordinates
    vcoord = -spatial['xInd']
    # Horizontal cartesian coordinates
    hcoord = (2 * np.sin(np.radians(60)) / 3) * spatial['yInd']

    # Create a figure
    fig,ax = plt.subplots(1, figsize=(16, 16), tight_layout=True)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Create a colourmap
    cmap = get_cmap(colourmap).copy()
    if colour_from is not None:
        # Sum all the genes or use a specific one
        if colour_from == 'sum_all':
            colour_vals = expression.sum(axis=1)
        else:
            colour_vals = plot_df[colour_from]
        if hide_overflow:
            # Create a colourmap with an overflow value for the top 1%
            cmap.set_over('navy')
            vmax_value = sorted(colour_vals)[int(0.99 * len(colour_vals))]
            norm = Normalize(vmin=min(colour_vals), vmax=vmax_value)
        else:
            norm = Normalize(vmin=min(colour_vals), vmax=max(colour_vals))
        colours = colour_vals.apply(lambda x: norm(x))
    else:
        # All spots get the same colour
        colours = pd.Series(1, index=hcoord.index)

    # Pick the validity Series
    valid = spatial[validity]

    # Create a DataFrame to ensure the Series align by index 
    plot_df = pd.DataFrame({'x': hcoord, 'y': vcoord, 'c': colours, 
        'v': valid})

    # Add coloured hexagons to the plot
    for ind in plot_df.index:
        x, y, c, v = plot_df.loc[ind]
        if not v:
            # Invalid hexagons are grey
            facecolor = 'gray'
        else:
            facecolor = cmap(c)
        hex = RegularPolygon((x,y), numVertices=6, radius=2/3,
            orientation=np.radians(120), facecolor=facecolor, 
            edgecolor='gray')
        ax.add_patch(hex)

    # Adjust the limits
    ax.set_xlim(min(hcoord)-1, max(hcoord)+1)
    ax.set_ylim(min(vcoord)-1, max(vcoord)+1)

    if colour_from is not None:
        # Add a colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.1, shrink=0.5, pad=0.02)
        cb.ax.tick_params(labelsize=20)
        cb.set_label(label, size=30)

    if title is not None:
        # Add a figure title
        ax.set_title(title, size=40)
    
    fig.show()

    return fig, ax


def hexagonal_plot():
    pass

    # TODO: Create a classification plotting too and make this a common
    # function used by both