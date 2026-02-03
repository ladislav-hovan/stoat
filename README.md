[![main](https://github.com/ladislav-hovan/stoat/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/ladislav-hovan/stoat/actions/workflows/test.yaml)
[![devel](https://github.com/ladislav-hovan/stoat/actions/workflows/test.yaml/badge.svg?branch=devel)](https://github.com/ladislav-hovan/stoat/actions/workflows/test.yaml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


# STOAT - Spatial TranscriptOmics to Assess Transcriptional regulation
The STOAT package generates spatially resolved gene regulatory networks
from spatial transcriptomics data.


## Table of Contents
- [STOAT - Spatial TranscriptOmics to Assess Transcriptional regulation](#stoat---spatial-transcriptomics-to-assess-transcriptional-regulation)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Features](#features)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Project Status](#project-status)
  - [Room for Improvement](#room-for-improvement)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)
  - [License](#license)


## General Information
This repository contains the STOAT package, which allows the generation 
of spatially resolved gene regulatory networks based on the provided 
spatial transcriptomics data and two different prior networks: the prior 
gene regulatory network and the protein-protein interaction network.


## Features
The features already available are:
- Generation of spatially resolved gene regulatory networks
- Downstream analysis tools


## Setup
The requirements are provided in a `requirements.txt` file.


## Usage
A simple workflow would be as follows:

``` python
# Import the class definition
from stoat.stoat import Stoat
# Create the STOAT object
stoat_obj = Stoat()
# Load the 10x Visium data
stoat_obj.load_visium_dataset(
    data_dir='visium_data/,
    dataset_id='my_experiment',
    counts_file='raw_feature_bc_matrix.h5',
    tissue_positions_file='spatial/tissue_positions.csv',
    scalefactors_file='spatial/scalefactors_json.json',
)
# Optional filtering steps
stoat_obj.filter_genes(drop_deprecated=True, min_counts=1)
stoat_obj.filter_spots(min_counts=5000, mt_pct_threshold=5)
# Average the expression over nearest neighbours
stoat_obj.average_expression(kernel='gaussian', sigma=0.4)
# Assign the spots to regions - by default each spot would be a region
# This setting would use clustering on averaged expression
stoat_obj.assign_regions(from_expression=True, layer='averaged')
# Calculate the region-specific networks
stoat_obj.calculate_networks(
    save_dir='networks/',
    motif_prior='motif_prior.tsv',
    ppi_prior='ppi_prior.tsv',
    save_degrees=True,
)
```


## Project Status
The project is: _in progress_.


## Room for Improvement
Room for improvement:
- Add more and better tests
- Add more downstream analysis

To do:
- Automatic flow, addition of a command line script


## Acknowledgements
Many thanks to the members of the 
[Kuijjer group](https://www.kuijjerlab.org/) 
at NCMM for their feedback and support.

This README is based on a template made by 
[@flynerdpl](https://www.flynerd.pl/).


## Contact
Created by Ladislav Hovan (ladislav.hovan@ncmbm.uio.no).
Feel free to contact me!


## License
This project is open source and available under the 
[GNU General Public License v3](LICENSE).
