"""
Analyse MASCARA data for a single star.
"""
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from pathlib import Path

from mvs import io

# Get data folder from command line
data_folder = Path(argv[1])
data_filenames = sorted(data_folder.glob("*.hdf5"))

# Get ASCC of star from command line
ascc = argv[2]

# Read data
star = io.create_star_from_hdf5_files(ascc, data_filenames)
