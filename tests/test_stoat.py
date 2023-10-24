import sys

sys.path.append('../')

from stoat.stoat import Stoat

import pandas as pd
import numpy as np

import pandas.testing as pt

#def setup_stoat_obj():
#
#    # Loads expression and spatial into a trial object
#    stoat_obj = Stoat(motif_prior='../../input/priors/new/tf_prior_fixed.tsv', 
#        ppi_prior='../../input/priors/new/ppi_prior.tsv',
#        output_dir='output/',
#        computing='gpu',
#        output_extension='feather')
#    
#    data_path = '../../input/data/xavier/frozen/STNR10A/'
#    stoat_obj.load_expression_raw(
#        matrix_path=data_path + 'matrix.mtx',
#        barcodes_path=data_path + 'barcodes.tsv',
#        features_path=data_path + 'features.tsv'
#    )
#    
#    stoat_obj.load_spatial(data_path + 'tissue_positions_list.csv')
#    stoat_obj.ensure_compatibility()
#    stoat_obj.remove_nan()
#
#    return stoat_obj


def test_normalise_library_size():
    stoat_obj = Stoat()
    stoat_obj.expression = pd.DataFrame([
        [1, 2, 3],
        [2, 2, 1],
        [0, 0, 1]
        ])
    stoat_obj.avg_expression = stoat_obj.expression.copy()
    stoat_obj.normalise_library_size()
    pt.assert_series_equal(stoat_obj.size_factors, pd.Series([1.5, 1.25, 0.25]))
    pt.assert_frame_equal(stoat_obj.expression, pd.DataFrame([
        [1/1.5, 2/1.5, 3/1.5],
        [2/1.25, 2/1.25, 1/1.25],
        [0, 0, 4]
    ]))

