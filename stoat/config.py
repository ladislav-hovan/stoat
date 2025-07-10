### Imports ###
import pandas as pd

from typing import Literal

### Definitions ###
# Typing literals
COMPUTING_TYPE = Literal['cpu', 'gpu']
EXTENSION = Literal['tsv', 'feather', 'parquet']
# URLs
ENSEMBL_URL = 'http://www.ensembl.org/biomart/'
# Plotting defaults
DIMENSIONS = pd.DataFrame({
    'type': ['deg', 'gsea'],
    'overhead': [2.5, 2.5],
    'width_per_col': [3, 8],
    'height_per_line': [0.3, 0.5],
}).set_index('type')