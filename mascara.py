"""
Analyse MASCARA data for a single star.
"""
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from pathlib import Path

from mvs import io, plot, periodicity
from mvs.periodicity import frequencies_default as frequencies

# Get data folder from command line
data_folder = Path(argv[1])
data_filenames = sorted(data_folder.glob("*.hdf5"))

# Get ASCC of star from command line
ascc = argv[2]

# Read data
star, data = io.read_all_data_for_one_star(data_filenames, ascc)

# Calculate GLS
gls = periodicity.GLS(data["HJD"], data["mag0"], dy=data["emag0"])
power = gls.power(frequencies)

# Plot GLS
plot.plot_GLS(frequencies, power, title=f"Original Lomb-Scargle periodogram for ASCC {ascc}")

# Find peaks in GLS
peak_indices, peak_frequencies, peak_heights = periodicity.gls_find_peaks(frequencies, power)

# Plot GLS with peaks indicated
plot.plot_GLS(frequencies, power, peaks=(peak_frequencies, peak_heights), title=f"Original Lomb-Scargle periodogram for ASCC {ascc}, with peaks indicated")

# Find strongest peaks
main_frequency = peak_frequencies[0]
main_period = periodicity.invert(main_frequency)

# Phase-fold
phase, phase_average, magnitude_average = periodicity.phase_fold(main_period, data["HJD"].data, data["mag0"].data, data["emag0"].data)

# Phase plot
plot.plot_phasecurve(phase, data["mag0"], data["emag0"], running_average=[phase_average, magnitude_average], title=f"ASCC {ascc} - original phase plot", saveto="phaseplot.pdf")
