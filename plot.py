import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection

def make_ylim(y, mag = True, zoom = False):
    """
    Generate ylim that neatly fit the data.

    If `zoom` is False, force all data to fit in the plot -- maybe consider just using plt.axis("tight")
    If `zoom` is True, most but not necessarily all data will fit in the plot.

    Parameters
    ----------
    y:
        data that will be plotted.
    mag: bool, optional
        if False, return (lower, upper); if True, return (upper, lower).
        default: True
    zoom: bool, optional
        if True, do not force all data to be in the plot.
        default: False

    Returns
    -------
    result: tuple
        (ymin, ymax)
    """
    if zoom:
        perc_low = np.nanpercentile(y, 1)
        perc_high = np.nanpercentile(y, 99)
        amp = perc_high - perc_low
        ymin = perc_low - 0.1 * amp
        ymax = perc_high + 0.1 * amp
    else:
        mean = 0.5*(np.nanmax(y) + np.nanmin(y))
        amp = 0.5*(np.nanmax(y) - np.nanmin(y))
        ymin = mean - 1.1 * amp
        ymax = mean + 1.1 * amp
    if mag:
        ylim = (ymax, ymin)
    else:
        ylim = (ymin, ymax)
    return ylim