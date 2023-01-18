### Imports ###
import pandas as pd

from typing import Callable

from math import exp

### Functions intended for averaging ###
def calculate_gaussian(
    r: float, 
    sigma: float
) -> float:
    """
    Calculates the value of the Gaussian PDF with standard deviation
    sigma at distance r from the mean.

    Parameters
    ----------
    r : float
        The distance from the mean
    sigma : float
        The standard deviation of the Gaussian distribution

    Returns
    -------
    float
        The value of the Gaussian PDF at distance r
    """

    # Normalisation is irrelevant because of the finite discretised 
    # scope, it will be done based on the sum of contributing parts
    return exp(-r**2/(2*sigma**2))


def weight_by_distance(
    row: pd.Series, 
    spatial: pd.DataFrame, 
    expression: pd.DataFrame, 
    function: Callable[[float], float]
) -> pd.Series:
    """
    Calculates the weighted sum of the expression of the neighbouring
    spots, the weights are calculated based on distance and the provided
    function.

    This function is meant to be applied to the expression dataframe.

    Parameters
    ----------
    row : pd.Series
        A row from the expression dataframe (expression values for a
        single spot)
    spatial : pd.DataFrame
        The dataframe containing spatial information about the spots
    expression : pd.DataFrame
        The dataframe containing the expression levels for the spots
    function : Callable[[float], float]
        A function that converts distance into weight

    Returns
    -------
    pd.Series
        A weighted sum of expression values from the row
    """

    neigh_ind = spatial.loc[row.name]['Neighbours']
    weighted_expr = expression.loc[neigh_ind].multiply(
        get_distance_to_neighbours(row.name, spatial).apply(function), axis=0)
    weighted_sum = weighted_expr.sum()

    return weighted_sum


def get_distance_to_neighbours(
    spotname: str, 
    spatial: pd.DataFrame, 
) -> pd.Series:
    """
    Calculates the distances to the neighbours of the given spot.

    Parameters
    ----------
    spotname : str
        The name of the spot which is used for indexing
    spatial : pd.DataFrame
        The dataframe containing spatial information about the spots

    Returns
    -------
    pd.Series
        The distances to the neighbours of the given spot
    """

    neigh_ind = spatial.loc[spotname]['Neighbours']
    distance = ((spatial.loc[neigh_ind]['xIndSc'] - 
        spatial.loc[spotname]['xIndSc'])**2 +
        (spatial.loc[neigh_ind]['yIndSc'] - 
        spatial.loc[spotname]['yIndSc'])**2)**0.5

    return distance