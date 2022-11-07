### Imports and settings ###
from typing import Optional, Union, Iterable

import pandas as pd
import numpy as np

from netZooPy.panda.panda import Panda

import matplotlib.pyplot as plt

from matplotlib.patches import RegularPolygon
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

# plt.rcParams['text.usetex'] = True

### Class definition ###
class Stoat:

    ### Initialisation ###
    def __init__(
        self,
        motif_prior: Optional[str] = None,
        ppi_prior: Optional[str] = None,
        panda_obj: Optional[Panda] = None,
        computing: str = 'cpu',
        output_dir: str = 'output/',
        auto_calculate: bool = False
    ) -> None:


        # TODO: Implement network methods other than PANDA (PUMA, DRAGON?)

        # Input variables
        self.motif_prior = motif_prior
        self.ppi_prior = ppi_prior
        self.panda_obj = panda_obj
        self.computing = computing
        self.output_dir = output_dir

        # Variables generated by the class
        self.expression = None
        self.spatial = None
        # TODO: Implement a dictionary that will allow specification

        # Run a sanity check on the provided values
        self.check_input()

        # Get the complete PANDA network and expression matrix from panda_obj
        # Then release memory
        if panda_obj is not None:
            self.panda_network = self.panda_obj.panda_network
            if not hasattr(self.panda_obj, 'expression_matrix'):
                print ('The PANDA object does not contain the expression ' +
                       'matrix, expression data need to be provided ' +
                       'from a different source')
            else:
                self.expression = pd.DataFrame(
                    self.panda_obj.expression_matrix).T
                self.expression.index = self.panda_obj.expression_samples
                self.expression.columns = self.panda_obj.genes
            del self.panda_obj

        # TODO: Implement automatic workflow after class creation (with
        # possibility of turning it off)
        if auto_calculate:
            pass


    def check_input(
        self
    ) -> None:
        """
        _summary_

        Raises
        ------
        NotImplementedError
            _description_
        """

        if self.computing not in ['cpu', 'gpu']:
            raise NotImplementedError('Computing type not supported: ' + 
                '{}'.format(self.computing))


    ### Dataset loading ###
    def load_expression_raw(
        self, 
        matrix_path: str,
        barcodes_path: str,
        features_path: str
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        matrix_path : str
            _description_
        barcodes_path : str
            _description_
        features_path : str
            _description_
        """

        # Load raw count matrix first
        df = pd.read_csv(matrix_path, names=['Feature ID', 'Barcode ID', 
            'Count'], skiprows=2, sep=' ')
        df.drop(0, inplace=True)  # First row contains total numbers of 
                                  # features/barcodes/matrix points
        df['Feature ID'] = df['Feature ID'].astype(int)
        df['Barcode ID'] = df['Barcode ID'].astype(int)
        df2 = df.pivot(values='Count', index='Barcode ID', 
            columns='Feature ID')  # Convert to matrix format
        
        # Load barcodes for index and feature names for columns
        barcodes = np.loadtxt(barcodes_path, dtype=str, delimiter='\t')
        # Shift because barcodes are 1-indexed
        bar_dict = {ind+1: val for ind,val in enumerate(barcodes)}  
        df2.index = [bar_dict[i] for i in df2.index]
        features = np.loadtxt(features_path, dtype=str, delimiter='\t')
        # Shift because features are 1-indexed
        feat_dict = {ind+1: val[1] for ind,val in enumerate(features)}
        df2.columns = [feat_dict[i] for i in df2.columns]

        self.expression = df2


    def load_expression(
        self, 
        expression_path: str,
        sep: str = '\t',
        index_col: int = 0,
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        expression_path : str
            _description_
        sep : str, optional
            _description_, by default '\t'
        index_col : int, optional
            _description_, by default 0
        """
        
        # Load the pandas DataFrame
        df = pd.read_csv(expression_path, sep=sep, index_col=index_col)

        self.expression = df


    def load_spatial(
        self,
        spatial_path: str
    ) -> None:
        """
        _summary_

        Parameters
        ----------
        spatial_path : str
            _description_
        """

        # Load spatial data
        coords = pd.read_csv(spatial_path, index_col=0, names=['Success', 
            'xInd', 'yInd', 'xPos', 'yPos'])
        coords['Success'] = coords['Success'].astype(bool)
        coords[['xInd', 'yInd']] = coords[['xInd', 'yInd']].astype(int)
        coords[['xPos', 'yPos']] = coords[['xPos', 'yPos']].astype(float)

        self.spatial = coords


    ### Input modification ###
    def remove_nan(
        self,
        method: str = 'fill_zero'
    ) -> None:


        if method == 'fill_zero':
            self.expression.fillna(0, inplace=True)
        elif method == 'fill_random':
            # TODO: Implement
            pass
        else:
            raise NotImplementedError('Unrecognised NaN removal method: ' + 
                '{}'.format(method))


    ### Data plotting ###


    ### Network calculation ###
    def calculate_panda(
        self
    ) -> None:
        """
        Runs the calculation of the full PANDA network (for all samples)
        using the stored values of motif prior, PPI prior, and gene 
        expression data. Stores the resulting network in 
        self.panda_network.
        """
        
        # Run the PANDA calculation using the provided priors
        panda_obj = Panda(self.expression.T, self.motif_prior, self.ppi_prior,
                          computing=self.computing)

        self.panda_network = panda_obj.panda_network
        del panda_obj


    def calculate(
        self,
        spot_barcodes: Union[str, Iterable[str], None] = None,
        save_panda: bool = False
    ) -> None:


        if not hasattr(self, 'panda_network'):
            print ('The full PANDA network has not been provided or ' +
                'calculated, calculating it now')
            self.calculate_panda()

        if spot_barcodes is None:
            # Calculate a STOAT network for every spot
            barcodes = self.expression.index
        elif type(spot_barcodes) == str:
            # Load the barcodes from a file
            f = open(spot_barcodes, 'r')
            lines = f.readlines()
            barcodes = [i.split('\n')[0] for i in lines]
            f.close()
        else:  
            # An Iterable of barcodes
            barcodes = spot_barcodes

        panda_input = self.expression.T

        n_spots = len(panda_input.columns)

        for bc in barcodes:
            print ('Calculating the STOAT network for spot {}'.format(bc))

            panda_obj = Panda(panda_input.drop(bc, axis=1), self.motif_prior, 
                self.ppi_prior, computing=self.computing)

            if save_panda:
                panda_outfile = (self.output_dir + 'panda_' + 
                    '{}.txt'.format(bc))
                print ('Saving the intermediate PANDA network to ' + 
                    '{}'.format(panda_outfile))
                panda_obj.save_panda_results(panda_outfile)
            
            panda_net = panda_obj.panda_network

            stoat_net = n_spots * (self.panda_network - panda_net) + panda_net

            stoat_outfile = (self.output_dir + 'stoat_' + '{}.txt'.format(bc))
            print ('Saving the STOAT network to {}'.format(stoat_outfile))
            stoat_net.to_csv(stoat_outfile, sep='\t')


def plot_spots(x_coords, y_coords, color_from=None, colormap='Greens', label=None, title=None, valid=None):
    """
    This function plots the spots for spatial transcriptomics data and optionally also their validity. 
    It can colour the spots based on an additional supplied series from the dataframe.
    
    Arguments:
        x_coords:
            A series from a dataframe containing the x coordinates of the spots.
        y_coords:
            A series from a dataframe containing the y coordinates of the spots.
        color_from (default None):
            A series from a dataframe that controls the colour of the spots, if None the colour is uniform.
        colormap (default 'Greens'):
            The colourmap to be used for colouring the spots.
        label (default None):
            The label of the colourbar.
        title (default None):
            The title of the plot.
        valid (default None):
            A series from a dataframe that determines the validity of the spots, if None all are valid.
            
    Returns:
        None
    """

    # Horizontal cartesian coords
    vcoord = [-c for c in x_coords]

    # Vertical cartersian coords
    hcoord = [2. * np.sin(np.radians(60)) * c /3. for c in y_coords]

    fig, ax = plt.subplots(1, figsize=(16, 16), tight_layout=True)
    ax.set_aspect('equal')

    cmap = get_cmap(colormap).copy()
    cmap.set_over('navy')
    if color_from is not None:
        norm = Normalize(vmin=min(color_from), vmax=sorted(color_from)[int(0.99*len(color_from))])
        colors = [norm(i) for i in color_from]
    else:
        colors = np.ones_like(hcoord)

    if valid is None:
        valid = np.ones_like(hcoord)

    # Add some coloured hexagons
    for x, y, c, v in zip(hcoord, vcoord, colors, valid):
        if not v:
            facecolor = 'gray'
        else:
            facecolor = cmap(c)
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3, 
                             orientation=np.radians(120), facecolor=facecolor,
                             alpha=1, edgecolor='gray')
        ax.add_patch(hex)

    ax.set_xlim(min(hcoord)-1, max(hcoord)+1)
    ax.set_ylim(min(vcoord)-1, max(vcoord)+1)
    ax.set_axis_off()
    if color_from is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.1, shrink=0.5, pad=0.02)
        cb.ax.tick_params(labelsize=20)
        cb.set_label(label, size=30)
    if title is not None:
        ax.set_title(title, size=40)
# End of plot_spots
        
def average_dataset(df, expr_columns):
    """
    This function replaces the expression values for each point with the average of the values 
    of the point and its closest neighbours. By default, it will only return the values for 
    points which have the highest possible number of neighbours (6, 7 including the point itself).
    """
    
    pass
# End of average_dataset

def convert_dataset(df):
    """
    This function converts a dataset into the format expected by PANDA.
    """
    
    pass
# End of convert_dataset

def ensure_compatibility(df, motif_prior, ppi_prior, features, annotations):
    """
    This function ensures that the expression data and the priors match as much of their genes
    as possible, while removing the ones that do not match.
    """
    
    pass
# End of ensure_compatibility