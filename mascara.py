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
gls = periodicity.GLS(data["BJD"], data["mag0"], dy=data["emag0"])
power = gls.power(frequencies)
plot.plot_GLS(frequencies, power, title=f"Original Lomb-Scargle periodogram for ASCC {ascc}")

# Find peaks in GLS
peak_indices, peak_frequencies, peak_heights = periodicity.gls_find_peaks(frequencies, power)

# Remove harmonics of 1 sidereal day
peak_frequencies, peak_heights = periodicity.remove_harmonics(periodicity.siderealday_frequency, peak_frequencies, peak_heights, remove_main_frequency=True)

# Find strongest peaks
main_frequency = peak_frequencies[0]
main_period = periodicity.invert(main_frequency)

# Phase-fold
phase, phase_average, magnitude_average = periodicity.phase_fold(main_period, data["BJD"].data, data["mag0"].data, data["emag0"].data)

# Phase plot
plot.plot_phasecurve(phase, data["mag0"], data["emag0"], running_average=[phase_average, magnitude_average], title=f"ASCC {ascc} - original phase plot\nPeriod = {main_period:.4f} d", saveto="phaseplot.pdf")

# Detrending
magnitude_detrended = periodicity.detrend(data["BJD"].data, data["mag0"].data, data["emag0"].data, data["camera"].data, main_period)
data.add_column(magnitude_detrended)
plot.LST_curve(data["lst"], data["mag0"], data["magD"], title=f"ASCC {ascc} - Local Sidereal Time trend", saveto="LST.pdf")

# Calculate GLS
gls = periodicity.GLS(data["BJD"], data["magD"], dy=data["emag0"])
power = gls.power(frequencies)

# Plot GLS
plot.plot_GLS(frequencies, power, title=f"Post-detrending Lomb-Scargle periodogram for ASCC {ascc}")
