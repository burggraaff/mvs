import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection

from .tables import convert_gls_array_to_table
from .misc import cameras

symbol_map = {'N': '^', 'S': 'v', 'E': 'D', 'W': 's', 'C': 'd', "?": 'o'}

def _save_show(saveto = None, fig = None, **kwargs):
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

def gls_one(GLS, y_linlog = "log", x_linlog = "log", frequency = False, highlight = None, title = "Generalised Lomb-Scargle Periodogram", grid = True, figure_kwargs = {}, line_kwargs = {}, **axes_kwargs):
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
    figure_kwargs: dict, optional
        Keyword arguments for the figure object.
        e.g. tight_layout, figsize, ...
    line_kwargs: dict, optional
        Keyword arguments for the line object.
        e.g. ls, lw, color, ...
        Default: {}
    **axes_kwargs:
        Keyword arguments for the axes object.
        e.g. xlabel, ylabel, xlim, ylim, ...
        Default: {}

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure containing the plot
    ax: matplotlib.axes._axes.Axes
        Axes object containing the plot
    """
    f_kw = {"tight_layout": True, "figsize": (20, 15)}
    f_kw.update(figure_kwargs)

    period_or_frequency = GLS["frequency"] if frequency else GLS["period"]
    power = GLS["power"]

    x_linlog = "linear" if x_linlog == "lin" else "log"
    y_linlog = "linear" if y_linlog == "lin" else "log"
    default_xlabel = "Frequency (days$^{-1}$)" if frequency else "Period (days)"
    default_ylim = (0, 1) if y_linlog == "linear" else (1e-5, 1)
    default_xlim = (period_or_frequency.min(), period_or_frequency.max())
    ax_kw = {"xlabel": default_xlabel, "ylabel": "Power", "title": title, "xscale": x_linlog, "yscale": y_linlog, "ylim": default_ylim, "xlim": default_xlim}
    ax_kw.update(axes_kwargs)

    l_kw  = {"color": "black", "lw": '1'}
    l_kw.update(line_kwargs)

    fig, ax = plt.subplots(subplot_kw = ax_kw, **f_kw)
    ax.tick_params(axis = "x", which = "both", direction = "out", length = 10, top = "off")

    ax.plot(period_or_frequency, power, **l_kw)

    if highlight is not None:
        ax.axvspan(0.9 * highlight, 1.1 * highlight, color = "red", alpha = 0.5, zorder = 0)
    if grid:
        ax.grid(axis='x', which="major", color="black", ls="-.")
        ax.grid(axis='y', which="major", color="black", ls="-.")

    return fig, ax

def gls_both(GLS, x_linlog = "log", frequency = False, highlight = None, title = "Generalised Lomb-Scargle Periodogram", grid = True, ylims=((0, 1), (1e-5, 1)), figure_kwargs = {}, line_kwargs = {}, **axes_kwargs):
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
    figure_kwargs: dict, optional
        Keyword arguments for the figure object.
        e.g. tight_layout, figsize, ...
    line_kwargs: dict, optional
        Keyword arguments for the line object.
        e.g. ls, lw, color, ...
        Default: {}
    **axes_kwargs:
        Keyword arguments for the axes objects.
        e.g. xlabel, ylabel, xlim, ylim, ...
        Default: {}

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

    period_or_frequency = GLS["frequency"] if frequency else GLS["period"]
    power = GLS["power"]

    default_xlim = (period_or_frequency.min(), period_or_frequency.max())
    default_xlabel = "Frequency (days$^{-1}$)" if frequency else "Period (days)"
    ax_kw = {"xlabel": default_xlabel, "ylabel": "Power", "xscale": x_linlog, "xlim": default_xlim}
    ax_kw.update(axes_kwargs)
    xlabel = ax_kw.pop("xlabel") # only want this on the lower plot

    fig, axs = plt.subplots(2, subplot_kw = ax_kw, **f_kw)
    axs[0].tick_params(axis='x', labelbottom="off") # no labels on horizontal axis for top plot
    axs[1].tick_params(axis='x', which="both", direction="out", length=10, top="off")
    axs[1].set_xlabel(xlabel)

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
    _save_show(saveto, fig, **save_kwargs)
    if return_fig_axs:
        return fig, ax

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