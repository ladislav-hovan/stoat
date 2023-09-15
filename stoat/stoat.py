### Imports ###
from typing import Optional, Union, Iterable, Tuple, Literal, List
import typing

import pandas as pd
import numpy as np

import os.path

from io import BytesIO

from netZooPy.panda.panda import Panda

from biomart import BiomartServer

from stoat.plotting import *
from stoat.functions import *

COMPUTING_TYPE = Literal['cpu', 'gpu']
EXTENSION = Literal['tsv', 'feather', 'parquet']

ENSEMBL_URL = 'http://www.ensembl.org/biomart/'

### Class definition ###
class Stoat:

    ### Initialisation ###
    def __init__(
        self,
        motif_prior: Optional[str] = None,
        ppi_prior: Optional[str] = None,
        panda_obj: Optional[Panda] = None,
        computing: COMPUTING_TYPE = 'cpu',
        output_dir: str = 'output/',
        output_extension: EXTENSION = 'tsv',
        auto_calculate: bool = False
    ):


        # TODO: Implement network methods other than PANDA (PUMA, DRAGON?)

        # Input variables
        self.motif_prior = motif_prior
        self.ppi_prior = ppi_prior
        self.panda_obj = panda_obj
        self.computing = computing
        self.output_dir = output_dir
        self.extension = output_extension

        # Variables generated by the class
        self.expression = None
        self.avg_expression = None
        self.spatial = None
        self.features = None

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
                self.avg_expression = self.expression.copy()
            del self.panda_obj

        # TODO: Implement automatic workflow after class creation (with the
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

        COMPUTING_ALLOWED: List[COMPUTING_TYPE] = list(
            typing.get_args(COMPUTING_TYPE))
        if self.computing not in COMPUTING_ALLOWED:
            raise NotImplementedError('Computing type not supported: '
                f'{self.computing}, use one of {COMPUTING_ALLOWED}')

        EXTENSION_ALLOWED: List[EXTENSION] = list(typing.get_args(EXTENSION))
        if self.extension not in EXTENSION_ALLOWED:
            raise NotImplementedError('Output extension not supported: '
                f'{self.extension}, use one of {EXTENSION_ALLOWED}')


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
            'Count'], comment='%', sep=' ')
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
        # Fill in possible missing barcodes with NaNs
        df2 = df2.reindex(index=barcodes)
        features = np.loadtxt(features_path, dtype=str, delimiter='\t')
        # Shift because features are 1-indexed
        feat_dict = {ind+1: val[0] for ind,val in enumerate(features)}
        df2.columns = [feat_dict[i] for i in df2.columns]
        # No need to fill missing features, they are just excluded

        self.expression = df2
        self.avg_expression = df2.copy()
        self.features = pd.DataFrame(features, 
            columns=['Ensemble', 'Name', 'Type'])

        if ((self.spatial is not None) and 
            (len(self.expression) != len(self.spatial))):
            print ('The lengths of the spatial and expression data do not ' + 
                'match, this may cause problems')


    def load_expression(
        self, 
        expression_path: Union[str, pd.DataFrame],
        sep: str = '\t',
        index_col: int = 0,
    ) -> None:
        
        
        if type(expression_path) == str:
            # Load the DataFrame
            self.expression = pd.read_csv(expression_path, sep=sep, 
                index_col=index_col)
            self.avg_expression = self.expression.copy()
        else:
            # Assign the passed DataFrame
            self.expression = expression_path
            self.avg_expression = expression_path.copy()

        if self.spatial is not None and (len(self.expression) != 
            len(self.spatial)):
            print ('The lengths of the spatial and expression data do not ' + 
                'match, this may cause problems')


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

        # Check if there is a header
        f = open(spatial_path, 'r')
        first_line = f.readline()
        # If there is not a header, the last character on the first line 
        # should be a digit (part of pixel position in full resolution image)
        # The last character is newline, so second last is what counts
        if first_line[-2].isdigit():
            header_row = None
        else:
            header_row = 0
        # Load spatial data
        coords = pd.read_csv(spatial_path, index_col=0, names=['Success', 
            'xInd', 'yInd', 'xPos', 'yPos'], header=header_row)
        # Adjust typing
        coords['Success'] = coords['Success'].astype(bool)
        coords[['xInd', 'yInd']] = coords[['xInd', 'yInd']].astype(int)
        coords[['xPos', 'yPos']] = coords[['xPos', 'yPos']].astype(float)
        # Create scaled coordinates which correlate well with actual distance
        # and can be used to define neighbours
        coords['xIndSc'] = coords['xInd'] / (4/3)**0.5
        coords['yIndSc'] = coords['yInd'] / 2
        # Column for data validity, relevant for averaging later
        coords['Valid'] = coords['Success']

        self.spatial = coords

        if self.expression is not None and (len(self.expression) != 
            len(self.spatial)):
            print ('The lengths of the spatial and expression data do not ' + 
                'match, this may cause problems')


    ### Input modification ###
    def remove_nan(
        self,
        method: str = 'fill_zero',
        random_seed: Optional[int] = None,
        st_dev: float = 1e-6
    ) -> None:


        if method == 'fill_zero':
            self.expression.fillna(0, inplace=True)
        elif method == 'fill_random':
            from numpy.random import default_rng
            # Initialise the PRNG
            rng = default_rng(random_seed)
            # Create a DataFrame with the same index/columns and random values
            fill_vals = np.abs(rng.normal(0, st_dev, self.expression.shape))
            fill_df = pd.DataFrame(fill_vals, index=self.expression.index,
                columns=self.expression.columns)
            # Update the expression values (only the NaNs)
            self.expression.update(fill_df, overwrite=False)
        else:
            raise NotImplementedError('Unrecognised NaN removal method: ' + 
                f'{method}\nOptions are: fill_zero, fill_random')
        # Replace the average expression after this function has been called
        self.avg_expression = self.expression.copy()


    def drop_deprecated(
        self
    ) -> None:
        

        deprecated = [i for i in self.expression.columns if 
            i[:11] == 'DEPRECATED_']
        if len(deprecated) > 0:
            print (f'Dropping {len(deprecated)} deprecated columns')
            self.expression.drop(columns=deprecated, inplace=True)
            self.avg_expression.drop(columns=deprecated, inplace=True)
        else:
            print ('No deprecated columns found')


    def filter_genes(
        self,
        drop_non_protein_coding: bool = True
    ) -> None:
        

        if drop_non_protein_coding:
            server = BiomartServer(ENSEMBL_URL)
            ensembl = server.datasets['hsapiens_gene_ensembl']
            to_retrieve = ['ensembl_gene_id', 'gene_biotype']
            response = ensembl.search({'attributes': to_retrieve})
            ens_to_type = pd.read_csv(BytesIO(response.content), sep='\t', 
                names=to_retrieve).set_index('ensembl_gene_id')
            to_drop = [i for i in self.expression.columns 
                if i not in ens_to_type.index
                or ens_to_type.loc[i]['gene_biotype'] != 'protein_coding']
            self.expression.drop(columns=to_drop, inplace=True)
            self.avg_expression.drop(columns=to_drop, inplace=True)


    def ensure_compatibility(
        self,
        features: Optional[str] = None,
        annotations: Optional[str] = None,
    ) -> None:


        if features is not None:
            self.features = pd.read_csv(features, sep='\t', header=None,
                names=['Ensemble', 'Name', 'Type'])
        if annotations is not None:
            ann_df = pd.read_csv(annotations, sep='\t')

        if type(self.motif_prior) == str:
            # Names are chosen for compatibility with PANDA later
            motif_df = pd.read_csv(self.motif_prior, sep='\t', header=None,
                names=['source', 'target', 'weight'])
            prior_genes = set(motif_df['target'])
        else:
            raise RuntimeError('There is no way to obtain gene names in the ' + 
                'motif prior, provide a path to the prior during object ' +
                'creation')

        if self.expression is not None:
            expr_genes = set(self.expression.columns)
        else:
            raise RuntimeError('There is no way to obtain gene names in the ' + 
                'expression data, load it from a saved frame or raw counts ' +
                'or provide a PANDA object during object creation')

        print (f'The prior contains {len(prior_genes)} genes')
        print (f'The expression contains {len(expr_genes)} genes')
        print ('These two sets have '
            f'{len(prior_genes.intersection(expr_genes))}'
            ' gene names in common')

        if self.features is None and annotations is None:
            print ('No features or annotations have been provided')
            print ('Matching will be done based purely on current names')

        to_drop = list(expr_genes - prior_genes)
        self.expression.drop(to_drop, axis=1, inplace=True)
        self.avg_expression.drop(to_drop, axis=1, inplace=True)
        expr_genes = set(self.expression.columns)
        motif_df = motif_df[motif_df['target'].isin(expr_genes)]
        self.motif_prior = motif_df


    def normalise_library_size(
        self
    ) -> None:
        

        size_factors = self.expression.sum(axis=1)
        size_factors /= size_factors.mean()
        self.expression = self.expression.divide(size_factors, axis=0)
        self.avg_expression = self.avg_expression.divide(size_factors, axis=0)
        self.size_factors = size_factors


    def average_expression(
        self,
        neighbours: int = 1,
        distance: float = None,
        max_invalid: int = 0,
        edges_invalid: bool = True,
        kernel: str = 'uniform',
        sigma: float = 0.5
    ) -> None:


        if self.spatial is None:
            raise RuntimeError('The averaging requires spatial information, '
                'but none has been provided')

        if distance is None:
            # Use the nearest neighbours for averaging, defined using the
            # scaled coordinates
            self.spatial['Neighbours'] = self.spatial.apply(lambda row: 
                self.spatial[((self.spatial['xIndSc'] - row['xIndSc'])**2 + 
                (self.spatial['yIndSc'] - row['yIndSc'])**2)**0.5 < 
                neighbours + 0.5].index, axis=1)
        else:
            # Use the actual distance to define neighbours
            self.spatial['Neighbours'] = self.spatial.apply(lambda row: 
                self.spatial[((self.spatial['xPos'] - row['xPos'])**2 + 
                (self.spatial['yPos'] - row['yPos'])**2)**0.5 < 
                distance].index, axis=1)

        # Count the number of neighbours: total and (in)valid
        self.spatial['NumNeigh'] = self.spatial['Neighbours'].apply(len)
        self.spatial['ValNeighbours'] = self.spatial['Neighbours'].apply(
            lambda x: [i for i in x if self.spatial.loc[i]['Success']])
        self.spatial['NumValNeigh'] = self.spatial['ValNeighbours'].apply(len)
        self.spatial['NumInvNeigh'] = (self.spatial['NumNeigh'] -
            self.spatial['NumValNeigh'])

        # Set the validity column based on the invalid neighbours
        # Also invalidate if the spot itself was invalid
        self.spatial['Valid'] = self.spatial['Success'] & (
            self.spatial['NumInvNeigh'] <= max_invalid)
        # Exclude edges too
        if edges_invalid:
            if distance is not None:
                # Consider only the maximum number of neighbours as not edge
                max_neigh = self.spatial['NumNeigh'].max()
            else:
                # Calculate the precise amount of neighbours expected
                max_neigh = 1
                for layer in range(1, neighbours + 1):
                    max_neigh += 6 * layer
            self.spatial['Valid'] = self.spatial['Valid'] & (max_neigh == 
                self.spatial['NumNeigh'])

        # All of the kernels exclude points where the measurements were
        # invalid (i.e. 'Success' is False)
        if kernel == 'uniform':
            # The contribution of every cell to the average is independent of
            # the distance from the central cell
            sum_exp = self.expression.apply(lambda row: 
                self.expression.loc[self.spatial.loc[row.name]
                ['ValNeighbours']].sum(), axis=1)
            self.avg_expression = sum_exp.apply(lambda x: 
                x / self.spatial['NumValNeigh'])
            self.avg_expression.fillna(0, inplace=True)
        elif kernel == 'gaussian':
            # The contribution is based on the distance from the central cell
            # and decreases proportionally to exp(-r**2)
            # Define a gaussian distribution with a provided sigma
            calculate_gaussian_fixed = lambda r: calculate_gaussian(r, sigma)
            # Provide the function as an input to distance weighting template
            compute_weighted_sum = lambda row: weight_by_distance(row, 
                self.spatial, self.expression, calculate_gaussian_fixed)
            sum_exp = self.expression.apply(compute_weighted_sum, axis=1)
            self.avg_expression = sum_exp.apply(lambda row: row.divide(
                get_distance_to_neighbours(row.name, self.spatial).apply(
                calculate_gaussian_fixed).sum(), axis=0), axis=1)
            self.avg_expression.fillna(0, inplace=True)
        else:
            raise NotImplementedError(f'Unrecognised kernel: {kernel}'
                '\nOptions are: uniform, gaussian')


    ### Data plotting ###
    def plot_spot_expression(
        self,
        averaged: bool = False,
        colour_from: Optional[str] = None,
        colourmap: str = 'Greens',
        label: Optional[str] = None,
        title: Optional[str] = None,
        hide_overflow: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the map of spots for the spatial expression data. It can 
        colour the spots based on an additional supplied gene name.

        Parameters
        ----------
        averaged : bool, optional
            Whether to use the averaged expression data instead of
            the original, by default False
        colour_from : Union[str, Callable], optional
            The name of the gene that the colouring will be based on, 
            or a function to be applied to every spot (for example sum), 
            or None to colour all valid cells the same colour, 
            by default None
        colourmap : str, optional
            The name of the matplotlib colourmap to use, by default 
            'Greens'
        label : str, optional
            The label for the colourbar or None for no label, by default
            None
        title : str, optional
            A title for the figure or None for no title, by default None
        hide_overflow : bool, optional
            Whether to restrict the range to the bottom 99% of values 
            and colour the top 1% with a different colour, by default 
            True

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The matplotlib Figure and Axes objects of the resulting plot
        """

        # Choose the correct validity column based on averaging
        if averaged:
            expr_df = self.avg_expression
            validity = 'Valid'
        else:
            expr_df = self.expression
            validity = 'Success'

        # Call the corresponding plotting function, get new figure and axes
        return plot_spot_expression(self.spatial, expr_df, validity, 
            colour_from, colourmap, label, title, hide_overflow, ax=None)


    ### Network calculation ###
    def calculate_panda(
        self
    ) -> None:
        """
        Runs the calculation of the full PANDA network (for all samples)
        using the stored values of motif prior, PPI prior, and averaged
        gene expression data. Stores the resulting network in 
        self.panda_network.
        """
        
        # Run the PANDA calculation using the provided priors
        panda_obj = Panda(self.avg_expression.loc[self.spatial['Valid']].T, 
            self.motif_prior, self.ppi_prior, computing=self.computing)

        self.panda_network = panda_obj.panda_network
        del panda_obj


    def get_full_name(
        self,
        base_filename: str
    ) -> str:
        

        return f'{base_filename}.{self.extension}'


    def save_dataframe(
        self,
        df: Union[pd.DataFrame, pd.Series],
        base_filename: str
    ) -> None:
        

        if self.extension == 'tsv':
            df.to_csv(f'{base_filename}.tsv', sep='\t')
        elif self.extension == 'feather':
            # Resetting the index will convert to DataFrame
            df.reset_index().to_feather(f'{base_filename}.feather')
        elif self.extension == 'parquet':
            if type(df) == pd.DataFrame:
                df.to_parquet(f'{base_filename}.parquet')
            else:
                df.to_frame().to_parquet(f'{base_filename}.parquet')
        # Options not listed here should not be possible (checked during
        # class creation)


    def calculate(
        self,
        spot_barcodes: Union[str, Iterable[str], None] = None,
        save_panda: bool = False,
        save_degrees: bool = False,
        overwrite_old = True
    ) -> None:


        if not hasattr(self, 'panda_network'):
            print ('The full PANDA network has not been provided or ' +
                'calculated, calculating it now')
            self.calculate_panda()

        # Create output directory if nonexistent
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if spot_barcodes is None:
            # Calculate a STOAT network for every valid spot
            barcodes = self.spatial[self.spatial['Valid']].index
        elif type(spot_barcodes) == str:
            # Load the barcodes from a file
            f = open(spot_barcodes, 'r')
            lines = f.readlines()
            barcodes = [i.split('\n')[0] for i in lines]
            f.close()
        else:  
            # An Iterable of barcodes
            barcodes = spot_barcodes

        panda_input = self.avg_expression.loc[self.spatial['Valid']].T

        n_spots = len(panda_input.columns)

        for bc in barcodes:
            # Names of output files (base, without extension)
            panda_outfile = (self.output_dir + f'panda_{bc}')
            stoat_outfile = (self.output_dir + f'stoat_{bc}')

            # Check if we're overwriting
            if not overwrite_old and (
                os.path.exists(self.get_full_name(stoat_outfile)) or 
                (save_panda and 
                os.path.exists(self.get_full_name(panda_outfile)))):
                print (f'Skipping spot {bc} because the STOAT or ' 
                    'PANDA file already exists in the target directory')
                continue

            print (f'Calculating the STOAT network for spot {bc}')

            # PANDA network with the current spot missing
            panda_obj = Panda(panda_input.drop(bc, axis=1), self.motif_prior, 
                self.ppi_prior, computing=self.computing)

            panda_net = panda_obj.panda_network

            if save_panda:
                print ('Saving the intermediate PANDA network to', 
                    self.get_full_name(panda_outfile))
                self.save_dataframe(panda_net, panda_outfile)
            
            # Equation for deriving the spot-specific network
            stoat_net = n_spots * (self.panda_network - panda_net) + panda_net

            print (f'Saving the STOAT network to',
                self.get_full_name(stoat_outfile))
            self.save_dataframe(stoat_net, stoat_outfile)

            if save_degrees:
                # Names of output files
                in_outfile = (self.output_dir + f'indegree_{bc}')
                out_outfile = (self.output_dir + f'outdegree_{bc}')

                print ('Saving the indegrees to',
                    self.get_full_name(in_outfile))
                self.save_dataframe(stoat_net.sum().rename('Indegrees'), 
                    in_outfile)
                print ('Saving the outdegrees to',
                    self.get_full_name(out_outfile))
                self.save_dataframe(stoat_net.sum(axis=1).rename('Outdegrees'), 
                    out_outfile)