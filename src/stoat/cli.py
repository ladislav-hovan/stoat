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

### Imports and settings ###
import cupy as cp

from argparse import (ArgumentDefaultsHelpFormatter, ArgumentParser,
    RawDescriptionHelpFormatter)

from .stoat import Stoat

### Class definition ###
class CustomFormatter(
    ArgumentDefaultsHelpFormatter,
    RawDescriptionHelpFormatter,
):
    """
    Combines the properties of these two formatters in order to display
    description as formatted while displaying default values for the
    parameters.
    """

    pass

### Functions ###
def cli(
) -> None:
    """
    Command line interface to the STOAT class. Execute with the --help
    option for more details.
    """

    DESCRIPTION = """
    STOAT  Copyright (C) 2025  Ladislav Hovan  <ladislav.hovan@ncmbm.uio.no>
    This program comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it under certain conditions.
    Please refer to the GPL-3.0 license for more details.

    STOAT - Spatial TranscriptOmics to Assess Transcriptional regulation.
    Generates spatially resolved gene regulatory networks, by default using PANDA.
    The underlying approach is similar to 
    that used by LIONESS."""
    EPILOG = 'Code available on: https://github.com/ladislav-hovan/stoat'

    parser = ArgumentParser(formatter_class=CustomFormatter,
        description=DESCRIPTION, epilog=EPILOG)

    parser.add_argument('-id', '--gpu-id', dest='gpu_id',
        help='ID of GPU to use, if not provided GPU will not be used',
        default=None, metavar='ID')
    parser.add_argument('-dt', '--data-type', dest='data_type',
        help='type of the dataset (visium or visium_hd)',
        default='visium')

    args = parser.parse_args()

    stoat_obj = Stoat()
    
    # TODO: Add preprocessing steps
    if args.gpu_id is not None:
        with cp.cuda.Device(args.gpu_id):
            # TODO: Add options to calculate_networks
            stoat_obj.calculate_networks(computing='gpu')
    else:
        stoat_obj.calculate_networks()