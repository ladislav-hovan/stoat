### Imports ###
from math import exp

### Functions intended for averaging ###
def calculate_gaussian(r, sigma):


    # Normalisation is irrelevant because of the finite discretised 
    # scope, it will be done based on the sum of contributing parts
    return exp(-r**2/(2*sigma**2))


def weight_by_distance(row, spatial, expression, function):


    neigh_ind = spatial.loc[row.name]['Neighbours']
    weighted_expr = expression.loc[neigh_ind].multiply(
        get_distance_to_neighbours(row, spatial).apply(function), axis=0)
    weighted_sum = weighted_expr.sum()

    return weighted_sum


def get_distance_to_neighbours(row, spatial):


    neigh_ind = spatial.loc[row.name]['Neighbours']
    distance = ((spatial.loc[neigh_ind]['xIndSc'] - 
        spatial.loc[row.name]['xIndSc'])**2 +
        (spatial.loc[neigh_ind]['yIndSc'] - 
        spatial.loc[row.name]['yIndSc'])**2)**0.5

    return distance