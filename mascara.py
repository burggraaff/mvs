"""
Analyse MASCARA data for a single star.
"""
import numpy as np
from matplotlib import pyplot as plt, patheffects
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
plt.figure(figsize=(4,3))
plt.errorbar(phase, data["mag0"], yerr=data["emag0"], color="k", fmt="o", markersize=3, rasterized=True)
plt.plot(phase_average, magnitude_average, linewidth=2, color="yellow", path_effects=[patheffects.Stroke(linewidth=4, foreground="black"), patheffects.Normal()], zorder=10)
plt.xlim(0, 1)
plt.ylim(1.05*data["mag0"].max(), 1.05*data["mag0"].min())
plt.xlabel("Phase")
plt.ylabel("Magnitude")
plt.grid(ls="--")
plt.title(f"ASCC {ascc}")
plt.savefig("phaseplot.pdf", bbox_inches="tight", dpi=600)
plt.show()
plt.close()
