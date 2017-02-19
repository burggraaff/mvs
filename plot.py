import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection

from .tables import convert_gls_array_to_table

def save_show(saveto = None, fig = None, **kwargs):
    if saveto is None:
        if fig is None:
            plt.show(**kwargs)
        else:
            fig.show(**kwargs)
    else:
        if fig is None:
            plt.savefig(saveto, **kwargs)
        else:
            fig.savefig(saveto, **kwargs)

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

def lspplot(pLSP, bestp = None, retax = False, axkw = {}, fkw = {}, lkw = {}):
    """
    Plot a Lomb-Scargle Periodogram on linear and logarithmic scales.

    Parameters
    ----------
    pLSP: astropy.table.Table
        table with period and LSP power
    bestp: float, optional
        the best period in the LSP
    retax: bool, optional
        whether or not to return the axes objects
    axkw: dict, optional
        **kwargs for both the axes objects
    fkw: dict, optional
        **kwargs for the main figure
    lkw: dict, optional
        **kwargs for the plotted lines

    Returns
    -------
    fig: matplotlib.figure.Figure
        the created figure
    axs: list of matplotlib.axes.AxesSubplot, optional
        the axes objects of plots themselves
    """
    fkwA = {"title": "Lomb-Scargle Periodogram"}
    combinekwargs(fkwA, fkw)

    axkwA = {"xlim": (pLSP["p"].min(), pLSP["p"].max()), "xscale": "log", "ylabel": "Power"}
    combinekwargs(axkwA, axkw)

    lkwA = {"color": "black", "lw": '1', "label": "Lomb-Scargle Periodogram"}
    combinekwargs(lkwA, lkw)

    fig, axs = plt.subplots(nrows = 2, sharex = True, figsize=figsize, subplot_kw = axkwA)

    axs[0].tick_params(axis='x', labelbottom="off")
    axs[1].tick_params(axis='x', which="both", direction="out", length=10, top="off")
    axs[1].set_xlabel("Period (days)")

    for ax in axs:
        ax.plot(pLSP["p"], pLSP["Power"], **lkwA)
        if bestp is not None:
            ax.axvspan(0.9*bestp, 1.1*bestp, color="r", alpha=0.5, zorder=0)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        ax.grid(axis='x', which="major", color=lkwA["color"], ls="--")
        ax.grid(axis='y', which="major", color=lkwA["color"], ls="-.")

    axs[0].set_title(fkwA["title"])

    axs[1].set_yscale("log")

    maxx = pLSP["Power"].max()
    axs[0].set_ylim(0, 1.03*maxx)
    axs[1].set_ylim(10.**-5., 1.)

    fig.tight_layout()

    return returnwhich([fig, axs], [True, retax])

def gls_one(GLS, y_linlog = "log", x_linlog = "log", highlight = None, title = "Generalised Lomb-Scargle Periodogram", grid = True, axes_kwargs = {}, line_kwargs = {}, **figure_kwargs):
    f_kw = {"tight_layout": True, "figsize": (20, 15)}
    f_kw.update(figure_kwargs)

    x_linlog = "linear" if x_linlog == "lin" else "log"
    y_linlog = "linear" if y_linlog == "lin" else "log"
    default_ylim = (0, 1) if y_linlog == "linear" else (1e-5, 1)
    ax_kw = {"xlabel": "Period (days)", "ylabel": "Power", "title": title, "xscale": x_linlog, "yscale": y_linlog, "ylim": default_ylim}
    ax_kw.update(axes_kwargs)

    l_kw  = {"color": "black", "lw": '1'}
    l_kw.update(line_kwargs)

    fig, ax = plt.subplots(subplot_kw = ax_kw, **f_kw)
    ax.tick_params(axis = "x", which = "both", direction = "out", length = 10, top = "off")
    ax.plot(GLS["period"], GLS["power"], **l_kw)

    if highlight is not None:
        ax.axvspan(0.9 * highlight, 1.1 * highlight, color = "red", alpha = 0.5, zorder = 0)
    if grid:
        ax.grid(axis='x', which="major", color="black", ls="-.")
        ax.grid(axis='y', which="major", color="black", ls="-.")

    return fig, ax

def gls_both(GLS):
    pass

def gls(GLS, y_linlog = "log", saveto = None, save_kwargs = {}, **kwargs):
    y_linlog = y_linlog.lower()
    assert y_linlog in ("lin", "log", "both"), "mvs.plot.gls expects a value of `lin`, `log`, or `both` for the parameter `y_linlog`; instead got `{0}`".format(y_linlog)
    GLS_ = convert_gls_array_to_table(GLS)
    if y_linlog == "both":
        fig, axs = gls_both(GLS_, **kwargs)
    else:
        fig, ax = gls_one(GLS_, y_linlog = y_linlog, **kwargs)
    save_show(saveto, fig, **save_kwargs)