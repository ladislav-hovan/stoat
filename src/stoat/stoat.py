# Copyright (C) 2025 Ladislav Hovan <ladislav.hovan@ncmbm.uio.no>
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
import pandas as pd

from functools import wraps
from pathlib import Path
from scanpy.preprocessing import (calculate_qc_metrics, filter_cells,
    filter_genes, normalize_total)
from spatialdata import SpatialData
from spatialdata_io import visium, visium_hd
from typing import Optional, Union

from stoat.config import *
from stoat.modules.expression_smoother import ExpressionSmoother
from stoat.modules.network_calculator import NetworkCalculator
from stoat.modules.plotting import *
from stoat.modules.region_assigner import RegionAssigner
from stoat.modules.utils import weigh_by_distance

### Class definition ###
class Stoat:
    ### Initialisation ###
    def __init__(
        self,
    ) -> None:

        # SpatialData object to be managed by the class
        self._spatial = None
        self.table = None

    ### Properties ###
    @property
    def spatial(
        self,
    ) -> SpatialData:

        if self._spatial is None:
            raise ValueError('No SpatialData object has been loaded yet, '
                'please load one prior to using downstream functions.')

        return self._spatial


    @spatial.setter
    def spatial(
        self,
        value: SpatialData,
    ) -> None:

        self._spatial = value

    ### Methods ###
    # TODO: Adjust the function signature (return None)
    # and documentation (format, type hints)
    @wraps(visium)
    def load_visium_dataset(
        self,
        *args,
        **kwargs,
    ) -> None:

        self.spatial = visium(*args, **kwargs)
        self.table = 'table'
        self.n_neighs = 6


    @wraps(visium_hd)
    def load_visium_hd_dataset(
        self,
        *args,
        **kwargs,
    ) -> None:

        self.spatial = visium_hd(*args, **kwargs)
        self.table = max(self.spatial.tables.keys())
        if len(self.spatial.tables) > 1:
            print (f'Multiple tables were detected, using {self.table}.')
        self.n_neighs = 4


    @wraps(filter_genes)
    def filter_genes(
        self,
        drop_deprecated: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:

        if drop_deprecated:
            deprecated = self.spatial[self.table].var_names.str.startswith(
                'DEPRECATED_')
            if sum(deprecated) > 0:
                print (f'Dropping {sum(deprecated)} deprecated genes.')
                self.spatial[self.table]._inplace_subset_var(~deprecated)
            else:
                print ('No deprecated genes found.')

        if args or kwargs:
            filter_genes(data=self.spatial[self.table], *args, **kwargs)


    @wraps(filter_cells)
    def filter_spots(
        self,
        mt_pct_threshold: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:

        if mt_pct_threshold is not None:
            st = self.spatial[self.table]
            st.var['mt'] = st.var_names.str.startswith('MT-')
            calculate_qc_metrics(st, qc_vars=['mt'], inplace=True, log1p=False)
            filter = st.obs['pct_counts_mt'] <= mt_pct_threshold
            self.spatial[self.table]._inplace_subset_obs(filter)

        if args or kwargs:
            filter_cells(data=self.spatial[self.table], *args, **kwargs)


    @wraps(normalize_total)
    def normalise_library_size(
        self,
        *args,
        **kwargs,
    ) -> None:

        normalize_total(adata=self.spatial[self.table], *args, **kwargs)


    # TODO: Implement plotting functions (fast via SpatialData)


    def average_expression(
        self,
        n_rings: int = 1,
        max_invalid: int = 0,
        edges_invalid: bool = True,
        avg_function: Callable = weigh_by_distance,
        *args,
        **kwargs,
    ) -> None:

        smoother = ExpressionSmoother(
            spatial_table=self.spatial[self.table],
            n_rings=n_rings,
            n_neighs=self.n_neighs,
        )
        if edges_invalid:
            smoother.filter_edges()
        smoother.enforce_max_invalid(max_invalid=max_invalid)
        smoother.smooth_expression(avg_function=avg_function, *args, **kwargs)


    def assign_regions(
        self,
        mapping: Optional[pd.Series] = None,
    ) -> None:

        assigner = RegionAssigner(spatial_table=self.spatial[self.table])
        assigner.assign_regions(mapping=mapping)
        assigner.collapse_expression()


    def calculate_networks(
        self,
        save_dir: Path = './',
        motif_prior: Optional[Union[Path, pd.DataFrame]] = None,
        ppi_prior: Optional[Union[Path, pd.DataFrame]] = None,
        log1p_transform: bool = True,
        extension: EXTENSION = 'feather',
        regions: Union[Path, Iterable[str], None] = None,
        save_network: bool = False,
        save_degrees: bool = False,
        overwrite_old: bool = True,
        *args,
        **kwargs,
    ) -> None:

        calculator = NetworkCalculator(
            spatial_table=self.spatial[self.table],
            motif_prior=motif_prior,
            ppi_prior=ppi_prior,
        )
        calculator.ensure_compatibility()
        if log1p_transform:
            calculator.log1p_transform()
        calculator.calculate_basis(*args, **kwargs)
        calculator.calculate(
            save_dir=save_dir,
            extension=extension,
            regions=regions,
            save_network=save_network,
            save_degrees=save_degrees,
            overwrite_old=overwrite_old,
            *args,
            **kwargs,
        )