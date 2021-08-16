"""
Analyse MASCARA data for a single star.
"""
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from pathlib import Path
from astropy.timeseries import LombScargle as GLS
from operator import truediv
from functools import partial

from mvs import io

# Get data folder from command line
data_folder = Path(argv[1])
data_filenames = sorted(data_folder.glob("*.hdf5"))

# Get ASCC of star from command line
ascc = argv[2]

# Read data
star, data = io.read_all_data_for_one_star(data_filenames, ascc)

# Calculate GLS
gls = GLS(data["HJD"], data["mag0"], dy=data["emag0"])

frequencies = np.logspace(-2, 2, 10000)
power = gls.power(frequencies)

# Plot GLS
invert = partial(truediv, 1)

plt.figure(figsize=(10,3))
plt.plot(frequencies, power, c='k', lw=1)
plt.xlabel("Frequency [d$^{-1}$]")
plt.ylabel("Power")
plt.xscale("log")
plt.yscale("log")
plt.xlim(frequencies.min(), frequencies.max())
plt.ylim(1e-5, 1.01)
secax = plt.gca().secondary_xaxis("top", functions=(invert, invert))
secax.set_xlabel("Period [d]")
plt.grid(ls="--")
plt.title(f"ASCC {ascc}")
plt.show()
plt.close()
