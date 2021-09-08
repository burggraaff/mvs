"""
Analyse MASCARA data for a single star.
"""
import numpy as np
from matplotlib import pyplot as plt, patheffects
from sys import argv
from pathlib import Path
from astropy.timeseries import LombScargle as GLS
from astropy import table
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
nr_bins = 151
binwidth = 0.025
# First, sort everything by phase
phase_indices = np.argsort(phase)
phase_sorted = phase[phase_indices].data
time_sorted = data["HJD"][phase_indices].data
mag_sorted = data["mag0"][phase_indices].data
uncertainty_sorted = data["emag0"][phase_indices].data

# Then, pad the start and end so the phase loops around (-0.01 is the same as 0.99)
# Find the right padding
pad_left_index = np.searchsorted(phase_sorted, 1 - binwidth)
pad_right_index = np.searchsorted(phase_sorted, binwidth)
slice_left, slice_right = np.s_[pad_left_index:], np.s_[:pad_right_index]
phase_left = phase_sorted[slice_left] - 1
phase_right = phase_sorted[slice_right] + 1

# Pad the data
phase_padded = np.concatenate([phase_left, phase_sorted, phase_right])
mag_padded = np.concatenate([mag_sorted[slice_left], mag_sorted, mag_sorted[slice_right]])
uncertainty_padded = np.concatenate([uncertainty_sorted[slice_left], uncertainty_sorted, uncertainty_sorted[slice_right]])

# Then, find the indices that correspond to the phase bins
bin_centers = np.linspace(0, 1, nr_bins)
bin_lower = bin_centers - binwidth
bin_upper = bin_centers + binwidth
lower_indices = np.searchsorted(phase_padded, bin_lower)
upper_indices = np.searchsorted(phase_padded, bin_upper)
slices = [np.s_[lower:upper] for lower, upper in zip(lower_indices, upper_indices)]

# Calculate the running averages
mag_average = np.array([np.average(mag_padded[s], weights=1/uncertainty_padded[s]**2) for s in slices])
# To do: uncertainty on running average
# To do: deal with missing phases
# To do: investigate PyAstronomy.pyasl.binningx0dt

# Find a suitable epoch
phase_at_minimum = bin_centers[np.nanargmax(mag_average)]
phase_at_minimum_index = np.searchsorted(phase_sorted, phase_at_minimum)
time_at_minimum_phase = time_sorted[phase_at_minimum_index]

# Phase plot
plt.figure(figsize=(4,3))
plt.errorbar(phase, data["mag0"], yerr=data["emag0"], color="k", fmt="o", markersize=3, rasterized=True)
plt.plot(bin_centers, mag_average, linewidth=2, color="yellow", path_effects=[patheffects.Stroke(linewidth=4, foreground="black"), patheffects.Normal()], zorder=10)
plt.xlim(0, 1)
plt.ylim(1.05*data["mag0"].max(), 1.05*data["mag0"].min())
plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.grid(ls="--")
plt.title(f"ASCC {ascc}")
plt.savefig("phaseplot.pdf", bbox_inches="tight", dpi=600)
plt.show()
plt.close()
