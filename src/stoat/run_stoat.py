#!/usr/bin/env python

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

# This is supposed to become a general purpose script for running
# STOAT from the command line
# Not fully implemented yet

import numpy as np
import pandas as pd

from stoat.stoat import Stoat

from netZooPy.panda.panda import Panda

from stoat_settings import *

import cupy as cp

gpu_devices = [cp.cuda.Device(i) for i in range(4)]

# Preprocessing (in the future, via stoat functions)
expr_df = pd.read_csv(expression_file, sep='\t', index_col=0)
expr_df.fillna(0, inplace=True)

print (expr_df.head())

# Expression file and motifs already preprocessed here
#expr_df = pd.read_csv(expression_file, sep='\t', index_col=0)

# stoat_obj = Stoat()

if barcode_file is not None:
    f = open(barcode_file, 'r')
    lines = f.readlines()
    barcodes = [i.split('\n')[0] for i in lines]
    f.close()
    #n_spots = len(barcodes)
else:
    # Use all spots, not implemented yet
    pass

# LIONESS implementation from Marieke (for Matlab)
# LocNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha);
# PredNet = NumConditions * (AgNet - LocNet) + LocNet;

with gpu_devices[3]:
    for bc in barcodes:
        print ('Generating network for barcode:', bc)
        panda_obj = Panda(expr_df.T.drop(bc, axis=1), motif_prior, ppi_prior, save_tmp=True,
                        save_memory=False, remove_missing=False, keep_expression_matrix=False, computing='gpu')
        panda_obj.save_panda_results(output_dir + 'panda_{}.txt'.format(bc))
