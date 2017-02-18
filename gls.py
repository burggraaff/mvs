import numpy as np
from .constants import second, minute
from .mapping import map_single
from . import mapping
from astropy import table

default_periods = np.concatenate((np.logspace(np.log10(11 * minute), 1.0, 40000, endpoint=False), np.logspace(1.0, 2.0, 2000)))

def oneperiod(p, t, y, yerr):
    """
    Calculate the GLS strength for one period

    Parameters
    ----------
    p: float
        period to evaluate
    t: array-like
        horizontal axis (time) of data
    y: array-like
        vertical axis (magnitude) of data
    yerr: array-like
        errors on y

    Returns
    -------
    power: float
        GLS power for given data at given period
    """
    omega = 2. * np.pi / (second * p) #angular frequencies
    w = 1./(yerr**2.) * 1./(np.sum(1./(yerr**2.))) #weights

    w = w.mean()
    Y = np.sum(w*y)

    t_omega = t * second * omega
    cosOt = np.cos(t_omega)
    sinOt = np.sin(t_omega)

    C = np.sum(w * cosOt)
    S = np.sum(w * sinOt)

    YY = np.sum(w * y**2.) - Y * Y
    YC = np.sum(w * y * cosOt) - Y * C
    YS = np.sum(w * y * sinOt) - Y * S
    ChatC = np.sum(w * cosOt**2.)
    CC = ChatC - C * C
    SS = 1 - ChatC - S * S
    CS = np.sum(w * cosOt * sinOt) - C * S

    D = CC * SS - CS**2.

    power = 1./(YY * D) * (SS * YC**2. + CC * YS**2. - 2. * CS * YC * YS)

    return power

def gls_multi(t, y, yerr, p = default_periods, min_period = 11 * minute, max_period = 100., astable = True):
    assert len(t) == len(y) == len(yerr), "mvs.gls.gls_multi: lengths of t, y and yerr are not equal: {0}, {1}, {2}".format(len(t), len(y), len(yerr))
    if type(p) is list:
        p = np.array(p)
    p_ = p[np.where(p > int(2.*min(np.diff(t))))] #no need to check periods shorter than the interval between data
    p_ = p_[np.where(p_ >= min_period)]
    p_ = p_[np.where(p_ <= max_period)]
    power = map_single(oneperiod, p_, t, y, yerr)
    if astable:
        pPf = table.Table(data=[p_, power, 1./p_], names=["p", "Power", "f"])
    else:
        pPf = np.hstack((p_, power, 1./p_))
    return pPf