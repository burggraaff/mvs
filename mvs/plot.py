from operator import truediv
from functools import partial

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection

from spectacle.plot import _saveshow

from .misc import cameras

symbol_map = {'N': '^', 'S': 'v', 'E': 'D', 'W': 's', 'C': 'd', "?": 'o'}

invert = partial(truediv, 1)

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


def plot_GLS(frequencies, power, title="Lomb-Scargle periodogram", peaks=None, saveto=None):
    """
    Plot a single Lomb-Scargle periodogram.
    """
    # Create figure
    plt.figure(figsize=(10,3))

    # Plot the GLS
    plt.plot(frequencies, power, c='k', lw=1)

    # Plot peaks if they are given
    if peaks:
        peak_frequencies, peak_heights = peaks  # Unpack tuple/list
        for freq, height in zip(peak_frequencies, peak_heights):
            plt.annotate("", xy=(freq, height*1.3), xytext=(freq, height*1.6), arrowprops=dict(arrowstyle="->"))

    # Axes settings
    plt.xlabel("Frequency [d$^{-1}$]")
    plt.ylabel("Power")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(frequencies.min(), frequencies.max())
    plt.ylim(1e-5, 1.01)

    # Add a second x axis to the top: period in days
    secax = plt.gca().secondary_xaxis("top", functions=(invert, invert))
    secax.set_xlabel("Period [d]")

    # Plot settings
    plt.grid(ls="--")
    plt.title(title)

    # Save if saveto is given, otherwise just show
    _saveshow(saveto)


def _points(ax, x, y, **kwargs):
    """
    Make a simple plot of x, y data points in a given axes object

    Parameters
    ----------
    ax: matplotlib.axes._axes.Axes
        Axes object to plot into
    x: array-like
        Horizontal axis data (e.g. time, phase)
    y: array-like
        Vertical axis data (e.g. magnitude)
    **kwargs:
        Keyword arguments for ax.scatter
    """
    s_kw = {"s": 25, "marker": 'o', "label": "Data", "zorder": 1, "cmap": plt.cm.jet, "c": "black"}
    s_kw.update(kwargs)
    ax.scatter(x, y, **s_kw)

def _errorbars(ax, x, y, xerr = None, yerr = None, **kwargs):
    """
    Make a simple plot of x, y error bars in a given axes object

    Parameters
    ----------
    ax: matplotlib.axes._axes.Axes
        Axes object to plot into
    x: array-like
        Horizontal axis data (e.g. time, phase)
    y: array-like
        Vertical axis data ()
    xerr: array-like, optional
        Horizontal axis errors
        Default: None
    yerr: array-like, optional
        Vertical axis errors
        Default: None
    **kwargs:
        Keyword arguments for ax.errorbar
    """
    if xerr is None and yerr is None:
        return
    e_kw = {"fmt": "none", "ecolor": "black", "zorder": 0}
    e_kw.update(kwargs)
    ax.errorbar(x, y, xerr = xerr, yerr = yerr, **e_kw)

def _points_cameras(ax, x, y, cam, c="black", **kwargs):
    """
    Make a simple plot of x, y data points in a given axes object

    Parameters
    ----------
    ax: matplotlib.axes._axes.Axes
        Axes object to plot into
    x: array-like
        Horizontal axis data (e.g. time, phase)
    y: array-like
        Vertical axis data (e.g. magnitude)
    cam: array-like
        Which camera each data point is from
    c: str *or* array-like, optional
        Which colour to plot the data points in
        If str, a single colour
        If array-like, a colour map is used (see _points)
        Default: "black"
    **kwargs:
        Keyword arguments for _points
    """
    cameras_present = np.unique(cam)
    for C in cameras_present:
        marker = symbol_map[C]
        indices = np.where(cam == C)[0]
        x_ = x[indices]
        y_ = y[indices]
        c_ = c if isinstance(c, str) else c[indices]
        _points(x_, y_, ax, marker = marker, c = c_, **kwargs)

def _trendline(x, y, ax, **kwargs):
    """
    Make a simple plot of a trend line in a given axes object
    This is primarily a helper function, you probably don't want to use it on its own.
    Use _trendline_errors for plotting the error bars

    Parameters
    ----------
    x: array-like
        Horizontal axis data (e.g. time, phase)
    y: array-like
        Vertical axis data ()
    ax: matplotlib.axes._axes.Axes
        Axes object to plot into
    **kwargs:
        Keyword arguments for ax.plot
    """
    l_kw = {"lw": 4, "color": "red", "zorder": 2, "path_effects": [path_effects.Stroke(linewidth=6, foreground="black"), path_effects.Normal()], "label": "Trend line"}
    l_kw.update(kwargs)
    ax.plot(x, y, **l_kw)

def time_data(t, y, xerr = None, yerr = None, saveto = None, color = "black", marker = "o", mag = True, trendline = None):
    """
    Plot data (mag/flux) over time
    """
    pass

def lst_data(l, y, lerr = None, yerr = None, saveto = None, color = "black", marker = "o", mag = True):
    """
    Plot data (mag/flux) over LST
    """
    pass

def phase_data(ph, y, pherr = None, yerr = None, saveto = None, color = "black", marker = "o", mag = True, trendline = None):
    """
    Plot phase-folded data (mag/flux)
    """
