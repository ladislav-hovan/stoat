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
# Create the STOAT object with desired settings
stoat_obj = Stoat(motif_prior='tf_prior.tsv', 
    ppi_prior='ppi_prior.tsv',
    output_dir='output/',
    computing='cpu'
)
# Load the gene expression data
stoat_obj.load_expression('expression.tsv')
# Load the position data
stoat_obj.load_spatial('tissue_positions.csv')
# Make sure the gene expression data and priors match
stoat_obj.ensure_compatibility()
# Filter out deprecated genes
stoat_obj.filter_genes()
# Replace the NaN reads with zero
stoat_obj.remove_nan()
# Average the expression over nearest neighbours
stoat_obj.average_expression(kernel='gaussian')
# Calculate the consensus PANDA network for all the spots
stoat_obj.calculate_panda()
# Calculate the spot-specific networks
stoat_obj.calculate(spot_barcodes=None, save_panda=False, 
    save_degrees=True, overwrite_old=False)
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
Created by Ladislav Hovan (ladislav.hovan@ncmm.uio.no).
Feel free to contact me!


## License
This project is open source and available under the 
[GNU General Public License v3](LICENSE).
