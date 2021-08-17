"""
Analyse MASCARA data for a single star.
"""
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from pathlib import Path
from astropy.timeseries import LombScargle as GLS
from PyAstronomy.pyasl import foldAt
from scipy.signal import find_peaks

from mvs import io, plot

# Get data folder from command line
data_folder = Path(argv[1])
data_filenames = sorted(data_folder.glob("*.hdf5"))

# Get ASCC of star from command line
ascc = argv[2]

# Read data
star, data = io.read_all_data_for_one_star(data_filenames, ascc)

# Calculate GLS
gls = GLS(data["HJD"], data["mag0"], dy=data["emag0"])

frequencies = np.concatenate([np.logspace(-2, -1, 2000, endpoint=False), np.logspace(-1, 2, 50000)])
periods = 1 / frequencies
power = gls.power(frequencies)

# Plot GLS
plot.plot_GLS(frequencies, power, title=f"Original Lomb-Scargle periodogram for ASCC {ascc}")

# Find peaks in GLS
peaks, peak_properties = find_peaks(power, height=1e-2, distance=30)
peak_frequencies = frequencies[peaks]
peak_heights = peak_properties["peak_heights"]

# Plot GLS with peaks indicated
plot.plot_GLS(frequencies, power, peaks=(peak_frequencies, peak_heights), title=f"Original Lomb-Scargle periodogram for ASCC {ascc}, with peaks indicated")

# Find strongest peaks
peak_order = np.argsort(peak_heights)[::-1]
peaks_sorted = peaks[peak_order]
peak_frequencies_sorted = peak_frequencies[peak_order]
peak_heights_sorted = peak_heights[peak_order]

main_frequency = peak_frequencies_sorted[0]
main_period = 1/main_frequency

# Phase-fold
phase = foldAt(data["HJD"], main_period)

# Running average

# Phase plot
plt.figure(figsize=(4,3))
plt.errorbar(phase, data["mag0"], yerr=data["emag0"], color="k", fmt="o", markersize=3)
plt.xlim(-0.01, 1.01)
plt.ylim(1.05*data["mag0"].max(), 1.05*data["mag0"].min())
plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.grid(ls="--")
plt.title(f"ASCC {ascc}")
plt.show()
plt.close()
