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
import pytest

import pandas as pd

from stoat import Stoat

### Fixtures ###
@pytest.fixture
def trial_stoat_obj():
    stoat_obj = Stoat()
    stoat_obj.load_visium_dataset(
        'tests/stoat/',
        
    )

    return stoat_obj
#    # Loads expression and spatial into a trial object
#    stoat_obj = Stoat(motif_prior='../../input/priors/new/tf_prior_fixed.tsv', 
#        ppi_prior='../../input/priors/new/ppi_prior.tsv',
#        output_dir='output/',
#        computing='gpu',
#        output_extension='feather')
   
#    data_path = '../../input/data/xavier/frozen/STNR10A/'
#    stoat_obj.load_expression_raw(
#        matrix_path=data_path + 'matrix.mtx',
#        barcodes_path=data_path + 'barcodes.tsv',
#        features_path=data_path + 'features.tsv'
#    )
   
#    stoat_obj.load_spatial(data_path + 'tissue_positions_list.csv')
#    stoat_obj.ensure_compatibility()
#    stoat_obj.remove_nan()

#    return stoat_obj

### Unit tests ###
# Trivial tests
def test_basic_functionality():
    stoat_obj = Stoat()

    assert type(stoat_obj) == Stoat