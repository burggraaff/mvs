import numpy as np

from matplotlib import pyplot as plt, patheffects

from spectacle.plot import _saveshow
from spectacle import symmetric_percentiles

from .periodicity import invert

def make_ylim(y, mag=True, percentile=0.5):
    """
    Generate ylim that neatly fit the data.

    Parameters
    ----------
    y:
        data that will be plotted.
    mag: bool, optional
        if False, return (lower, upper); if True, return (upper, lower).
        default: True

    Returns
    -------
    result: tuple
        (ymin, ymax)
    """
    ymin, ymax = symmetric_percentiles(y, percent=percentile)
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


def plot_phasecurve(phase, magnitude, magnitude_uncertainty=None, running_average=None, symbols="o", title="Phase plot", saveto=None):
    """
    Plot a phase curve with data (scatter/errorbar) and running average (line).
    running_average should be like [x, y].
    """
    # Create figure
    plt.figure(figsize=(4,3))

    # Scatter plot for the data
    if magnitude_uncertainty is None:
        magnitude_uncertainty = np.zeros_like(magnitude)
    plt.errorbar(phase, magnitude, yerr=magnitude_uncertainty, color="k", marker=symbols, markersize=3, linestyle="None", rasterized=True)

    # Line plot for the running average
    if running_average is not None:
        plt.plot(*running_average, linewidth=2, color="yellow", path_effects=[patheffects.Stroke(linewidth=4, foreground="black"), patheffects.Normal()], zorder=10)

    # Plot settings
    plt.xlim(0, 1)
    plt.ylim(make_ylim(magnitude))
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    plt.grid(ls="--")
    plt.title(title)

    # Save/show plot
    _saveshow(saveto, dpi=600)
