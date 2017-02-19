import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection

from .tables import convert_gls_array_to_table

def save_show(saveto = None, fig = None, **kwargs):
    """
    Save a plot or show it

    Parameters
    ----------
    saveto: str, optional
        Path to save to. If None, show the figure instead.
    fig: matplotlib.figure.Figure
        Figure to save/show. If None, use API instead (plt.show rather than fig.show)
    **kwargs:
        keyword arguments for show/savefig
    """
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

def gls_one(GLS, y_linlog = "log", x_linlog = "log", frequency = False, highlight = None, title = "Generalised Lomb-Scargle Periodogram", grid = True, axes_kwargs = {}, line_kwargs = {}, **figure_kwargs):
    """
    Plot a given GLS

    Parameters
    ----------
    GLS: astropy.table.table.Table
        GLS to plot. It is assumed to have "period" and "power" columns.
        If you want to plot a GLS in a different format, please use the mvs.plot.gls wrapper function
    y_linlog: str, optional
        If "log", plot on a logarithmic y-scale. If "lin", linear.
    x_linlog: str, optional
        If "log", plot on a logarithmic x-scale. If "lin", linear.
    frequency: boolean, optional
        If False, use period as x-axis. If True, use frequency.
        Default: False
    highlight: float, optional
        Highlight a certain area with a red square.
        Default: None
    title: str, optional
        Title for the plot
        Default: "Generalised Lomb-Scargle Periodogram"
    grid: boolean, optional
        Whether or not to add a grid to the plot
        Default: True
    axes_kwargs: dict, optional
        Keyword arguments for the axes object.
        e.g. xlabel, ylabel, xlim, ylim, ...
        Default: {}
    line_kwargs: dict, optional
        Keyword arguments for the line object.
        e.g. ls, lw, color, ...
        Default: {}
    **figure_kwargs:
        Keyword arguments for the figure object.
        e.g. tight_layout, figsize, ...

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure containing the plot
    ax: matplotlib.axes._axes.Axes
        Axes object containing the plot
    """
    f_kw = {"tight_layout": True, "figsize": (20, 15)}
    f_kw.update(figure_kwargs)

    x_linlog = "linear" if x_linlog == "lin" else "log"
    y_linlog = "linear" if y_linlog == "lin" else "log"
    default_xlabel = "Frequency (days$^{-1}$)" if frequency else "Period (days)"
    default_ylim = (0, 1) if y_linlog == "linear" else (1e-5, 1)
    default_xlim = (GLS["period"].min(), GLS["period"].max())
    ax_kw = {"xlabel": default_xlabel, "ylabel": "Power", "title": title, "xscale": x_linlog, "yscale": y_linlog, "ylim": default_ylim, "xlim": default_xlim}
    ax_kw.update(axes_kwargs)

    l_kw  = {"color": "black", "lw": '1'}
    l_kw.update(line_kwargs)

    fig, ax = plt.subplots(subplot_kw = ax_kw, **f_kw)
    ax.tick_params(axis = "x", which = "both", direction = "out", length = 10, top = "off")

    period_or_frequency = GLS["frequency"] if frequency else GLS["period"]
    power = GLS["power"]

    ax.plot(period_or_frequency, power, **l_kw)

    if highlight is not None:
        ax.axvspan(0.9 * highlight, 1.1 * highlight, color = "red", alpha = 0.5, zorder = 0)
    if grid:
        ax.grid(axis='x', which="major", color="black", ls="-.")
        ax.grid(axis='y', which="major", color="black", ls="-.")

    return fig, ax

def gls_both(GLS, x_linlog = "log", frequency = False, highlight = None, title = "Generalised Lomb-Scargle Periodogram", grid = True, ylims=((0, 1), (1e-5, 1)), axes_kwargs = {}, line_kwargs = {}, **figure_kwargs):
    """
    Plot a given GLS with linear *and* logarithmic y axes

    Parameters
    ----------
    GLS: astropy.table.table.Table
        GLS to plot. It is assumed to have "period" and "power" columns.
        If you want to plot a GLS in a different format, please use the mvs.plot.gls wrapper function
    x_linlog: str, optional
        If "log", plot on a logarithmic x-scale. If "lin", linear.
    frequency: boolean, optional
        If False, use period as x-axis. If True, use frequency.
        Default: False
    highlight: float, optional
        Highlight a certain area with a red square.
        Default: None
    title: str, optional
        Title for the plot
        Default: "Generalised Lomb-Scargle Periodogram"
    grid: boolean, optional
        Whether or not to add a grid to the plot
        Default: True
    ylims: tuple, optional
        Limits on the vertical axes of the plots
        ((lin_min, lin_max), (log_min, log_max))
        Default: ((0, 1), (1e-5, 1))
    axes_kwargs: dict, optional
        Keyword arguments for the axes object.
        e.g. xlabel, ylabel, xlim, ylim, ...
        Default: {}
    line_kwargs: dict, optional
        Keyword arguments for the line object.
        e.g. ls, lw, color, ...
        Default: {}
    **figure_kwargs:
        Keyword arguments for the figure object.
        e.g. tight_layout, figsize, ...

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure containing the plots
    axs: list of matplotlib.axes._axes.Axes
        Axes objects containing the plots
    """
    f_kw = {"tight_layout": True, "figsize": (20, 15), "sharex": True}
    f_kw.update(figure_kwargs)

    l_kw  = {"color": "black", "lw": "1"}
    l_kw.update(line_kwargs)

    default_xlim = (GLS["period"].min(), GLS["period"].max())
    default_xlabel = "Frequency (days$^{-1}$)" if frequency else "Period (days)"
    ax_kw = {"xlabel": default_xlabel, "ylabel": "Power", "xscale": x_linlog, "xlim": default_xlim}
    ax_kw.update(axes_kwargs)
    xlabel = ax_kw.pop("xlabel") # only want this on the lower plot

    fig, axs = plt.subplots(2, subplot_kw = ax_kw, **f_kw)
    axs[0].tick_params(axis='x', labelbottom="off") # no labels on horizontal axis for top plot
    axs[1].tick_params(axis='x', which="both", direction="out", length=10, top="off")
    axs[1].set_xlabel(xlabel)

    period_or_frequency = GLS["frequency"] if frequency else GLS["period"]
    power = GLS["power"]

    for ax in axs:
        ax.plot(period_or_frequency, power, **l_kw)
        if highlight is not None:
            ax.axvspan(0.9 * highlight, 1.1 * highlight, color = "red", alpha = 0.5, zorder = 0)
        if grid:
            ax.grid(axis='x', which="major", color="black", ls="-.")
            ax.grid(axis='y', which="major", color="black", ls="-.")

    axs[0].set_title(title)
    axs[1].set_yscale("log")
    axs[0].set_ylim(ylims[0])
    axs[1].set_ylim(ylims[1])

    return fig, axs

def gls(GLS, mode = "log", frequency = False, saveto = None, save_kwargs = {}, return_fig_axs = False, **kwargs):
    """
    Wrapper for plotting GLS

    Parameters
    ----------
    GLS: astropy.table.table.Table or np.ndarray
        GLS. If table, assumed to have "period" and "power" columns.
        If numpy array, it is converted to an astropy table.
    mode: str, optional
        If "lin", make one plot with a linear y-axis.
        If "log", make one plot with a logarithmic y-axis.
        If "both", make a plot that has both of the above.
        Default: "log"
    frequency: boolean, optional
        If False, use period as x-axis. If True, use frequency.
        Default: False
    saveto: str, optional
        Where to save the figure. Figure is shown if None.
        Default: None
    save_kwargs: dict, optional
        Keyword arguments for savefig/show.
        Default: {}
    return_fig_axs: boolean, optional
        Whether or not to return the figure and axes objects.
        If False, the figure is closed.
        Default: False
    **kwargs:
        Keyword arguments for mvs.plot.gls_both or mvs.plot.gls_one

    Returns
    -------
    Both only if return_fig_axs = True:

    fig: matplotlib.figure.Figure
        Figure containing the plots
    axs: (list of) matplotlib.axes._axes.Axes
        Axes object(s) containing the plot(s)
    """
    mode = mode.lower()
    assert mode in ("lin", "log", "both"), "mvs.plot.gls expects a value of `lin`, `log`, or `both` for the parameter `mode`; instead got `{0}`".format(mode)
    GLS_ = convert_gls_array_to_table(GLS)
    if frequency:
        if "frequency" not in GLS_.keys():
            GLS.add_column(data = 1./GLS_["period"], name = "frequency")
        GLS_.sort("frequency")
    else:
        GLS_.sort("period")
    if mode == "both":
        fig, ax = gls_both(GLS_, frequency = frequency, **kwargs)
    else:
        fig, ax = gls_one(GLS_, y_linlog = mode, frequency = frequency, **kwargs)
    save_show(saveto, fig, **save_kwargs)
    if return_fig_axs:
        return fig, ax