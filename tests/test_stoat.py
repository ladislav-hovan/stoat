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
import glob
import os
import pytest

from stoat import Stoat

### Fixtures ###
@pytest.fixture
def trial_stoat_obj():
    stoat_obj = Stoat()
    stoat_obj.load_zarr('tests/stoat/test_stoat.zarr')

    return stoat_obj

### Unit tests ###

### Class unit tests ###

### Integration tests ###
# Dataset loading
def test_object_loading(trial_stoat_obj):
    assert type(trial_stoat_obj) == Stoat
    # It is a Visium dataset
    assert trial_stoat_obj.coord_type == 'grid'
    assert trial_stoat_obj.n_neighs == 6
    # Check dimensionality
    assert trial_stoat_obj.spatial[trial_stoat_obj.table].shape == (338, 1219)

# Short full workflow
def test_workflow(trial_stoat_obj, tmp_path):
    trial_stoat_obj.filter_genes(drop_deprecated=True, min_counts=1)
    trial_stoat_obj.filter_spots(min_counts=1)
    trial_stoat_obj.average_expression(kernel='gaussian', sigma=0.4)
    trial_stoat_obj.assign_regions(layer='averaged')
    trial_stoat_obj.calculate_networks(
        save_dir=tmp_path,
        motif_prior=None,
        ppi_prior=None,
        computing='cpu',
    )

    assert len(glob.glob('stoat_*.feather', root_dir=tmp_path)) == 23