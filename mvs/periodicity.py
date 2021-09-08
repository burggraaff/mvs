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
