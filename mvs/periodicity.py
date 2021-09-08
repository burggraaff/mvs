"""
Functions for investigating periodicity.
"""
from operator import truediv
from functools import partial

import numpy as np
from astropy.timeseries import LombScargle as GLS
from scipy.signal import find_peaks
from PyAstronomy.pyasl import foldAt

# Inversion - useful for converting between periods and frequencies
invert = partial(truediv, 1.)

# Default frequencies/periods for GLS
frequencies_default = np.concatenate([np.logspace(-2, -1, 2000, endpoint=False), np.logspace(-1, 2, 50000)])
periods_default = invert(frequencies_default)


def gls_find_peaks(frequencies, power, **kwargs):
    """
    Use scipy.signal.find_peaks to find peaks in the GLS power spectrum, then
    return their locations and properties.
    """
    # Find peaks
    peak_indices, peak_properties = find_peaks(power, height=1e-2, distance=30, **kwargs)

    # Extract peak properties
    peak_frequencies = frequencies[peak_indices]
    peak_heights = peak_properties["peak_heights"]

    # Sort by peak height
    peak_order = np.argsort(peak_heights)[::-1]
    peak_indices_sorted = peak_indices[peak_order]
    peak_frequencies_sorted = peak_frequencies[peak_order]
    peak_heights_sorted = peak_heights[peak_order]

    return peak_indices_sorted, peak_frequencies_sorted, peak_heights_sorted


def phase_fold(period, time, magnitude, magnitude_uncertainty=None, nr_bins=151, binwidth=0.025):
    """
    Phase-fold the data and compute a running average.
    """
    if magnitude_uncertainty is None:
        magnitude_uncertainty = np.zeros_like(magnitude)

    # Phase-fold
    phase = foldAt(time, period)

    # Running average
    # First, sort everything by phase
    phase_indices = np.argsort(phase)
    phase_sorted = phase[phase_indices]
    time_sorted = time[phase_indices]
    mag_sorted = magnitude[phase_indices]
    uncertainty_sorted = magnitude_uncertainty[phase_indices]

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

    return phase, bin_centers, mag_average
