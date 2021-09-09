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

# Constants
siderealday = 23.9344699/24.  # Days
siderealday_frequency = invert(siderealday)  # Per day
lunar_period = 29.5  # Days
lunar_frequency = invert(lunar_period)  # Per day

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


def remove_harmonics(main_frequency, peak_frequencies, *peak_properties, threshold=1e-2, remove_main_frequency=False):
    """
    Remove harmonics of the main frequency from the given peaks.
    Any number of additional peak properties (e.g. heights) can also be passed
    to filter these at the same time.

    max_alias specifies up to which alias we should filter, e.g. if max_alias=10
    then we filter up to 10*F

    remove_main_frequency determines whether we also remove the main frequency
    itself, for instance for detrending purposes.
    """
    # Calculate the ratio of these frequencies with the main frequency and their remainders
    # to see if these are close to integer ratios
    harmonic_ratios = peak_frequencies / main_frequency
    harmonic_remainders = harmonic_ratios % 1

    # Find near-integer ratios
    harmonic_indices = np.where((harmonic_remainders < threshold) | (1 - harmonic_remainders < threshold))[0]

    # If we want to keep the main frequency, find its corresponding peaks and don't remove those
    if not remove_main_frequency:
        main_indices = np.where((1 - threshold < harmonic_ratios) & (harmonic_ratios < 1 + threshold))[0]
        harmonic_indices = np.array([ind for ind in harmonic_indices if ind not in main_indices])

    # The indices to keep are simply the inverse of the list of indices to remove
    safe_indices = ~np.in1d(range(peak_frequencies.shape[0]), harmonic_indices)

    # Finally, filter the peak frequencies and properties
    safe_frequencies = peak_frequencies[safe_indices]
    safe_properties = [peak_property[safe_indices] for peak_property in peak_properties]

    return safe_frequencies, *safe_properties


def sort_data(data_main, *data_other):
    """
    Sort an arbitrary number of data arrays/columns according to the first one.
    """
    # Check that all lengths are the same
    assert all(len(data) == len(data_main) for data in data_other), "Difference in length between main sorting array and data."

    # Indices that sort everything
    sorting_indices = np.argsort(data_main)

    # Sort everything
    data_main_sorted = data_main[sorting_indices]
    data_other_sorted = [data[sorting_indices] for data in data_other]

    return data_main_sorted, *data_other_sorted


def running_average(phase_sorted, magnitude_sorted, magnitude_uncertainty_sorted=None, bin_min=0, bin_max=1, nr_bins=151, binwidth=0.025, averaging_function=np.average):
    """
    Calculate a running average.
    """
    if magnitude_uncertainty_sorted is None:
        magnitude_uncertainty_sorted = np.zeros_like(magnitude_sorted)

    # Pad the start and end so the phase loops around (-0.01 is the same as 0.99)
    # Find the right padding
    pad_left_index = np.searchsorted(phase_sorted, 1 - binwidth)
    pad_right_index = np.searchsorted(phase_sorted, binwidth)
    slice_left, slice_right = np.s_[pad_left_index:], np.s_[:pad_right_index]
    phase_left = phase_sorted[slice_left] - 1
    phase_right = phase_sorted[slice_right] + 1

    # Pad the data
    phase_padded = np.concatenate([phase_left, phase_sorted, phase_right])
    mag_padded = np.concatenate([magnitude_sorted[slice_left], magnitude_sorted, magnitude_sorted[slice_right]])
    uncertainty_padded = np.concatenate([magnitude_uncertainty_sorted[slice_left], magnitude_uncertainty_sorted, magnitude_uncertainty_sorted[slice_right]])

    # Find the indices that correspond to the phase bins
    bin_centers = np.linspace(bin_min, bin_max, nr_bins, endpoint=False)
    bin_lower = bin_centers - binwidth
    bin_upper = bin_centers + binwidth
    lower_indices = np.searchsorted(phase_padded, bin_lower)
    upper_indices = np.searchsorted(phase_padded, bin_upper)
    slices = [np.s_[lower:upper] for lower, upper in zip(lower_indices, upper_indices)]

    # Calculate the running averages
    magnitude_average = np.array([averaging_function(mag_padded[s], weights=1/uncertainty_padded[s]**2) for s in slices])
    # To do: uncertainty on running average
    # To do: deal with missing phases
    # To do: investigate PyAstronomy.pyasl.binningx0dt

    return bin_centers, magnitude_average


def phase_fold(period, time, magnitude, magnitude_uncertainty=None, nr_bins=151, binwidth=0.025):
    """
    Phase-fold the data and compute a running average.
    """
    if magnitude_uncertainty is None:
        magnitude_uncertainty = np.zeros_like(magnitude)

    # Phase-fold
    phase = foldAt(time, period)

    # Sort everything by phase
    phase_sorted, time_sorted, mag_sorted, uncertainty_sorted = sort_data(phase, time, magnitude, magnitude_uncertainty)

    # Calculate the running average
    phase_average, magnitude_average = running_average(phase_sorted, mag_sorted, uncertainty_sorted)

    # Find a suitable epoch
    phase_at_minimum = phase_average[np.nanargmax(magnitude_average)]
    phase_at_minimum_index = np.searchsorted(phase_sorted, phase_at_minimum)
    time_at_minimum_phase = time_sorted[phase_at_minimum_index]

    # Adjust the calculated phase to the epoch of the minimum
    phase = (phase - phase_at_minimum) % 1
    phase_average = (phase_average - phase_at_minimum) % 1
    phase_average, magnitude_average = sort_data(phase_average, magnitude_average)

    return phase, phase_average, magnitude_average
