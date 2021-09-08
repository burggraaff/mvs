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
