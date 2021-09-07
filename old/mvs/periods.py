from .constants import second, minute
from .mapping import map_single
from .tables import convert_gls_array_to_table

import numpy as np
from astropy import table
from warnings import warn

default_periods = np.concatenate((np.logspace(np.log10(11 * minute), 1.0, 40000, endpoint=False), np.logspace(1.0, 2.0, 2000)))

def alias_of(p1, p2, margin = 0.02):
    """
    Check if two periods are aliases, i.e. 1/p1 = 1/p2 + k with k an integer

    Parameters
    ----------
    p1, p2: float
        periods to check
    margin: float
        how close must they be to be considered aliases
        default: 0.02

    Returns
    -------
    True if they are close to aliases, False if not
    """
    f1 = 1./p1 ; f2 = 1./p2
    diff = max(f1, f2) - min(f1, f2)
    diffmod = diff%1
    return (0.95 < diff < 6 and (-margin < diffmod < margin or diffmod > 1. - margin))

def harmonic_of(p1, p2, margin = 0.03, limit = 6):
    """
    Check if two periods are harmonics, i.e. p1 = k*p2 or p1 = p2/k with k an integer

    Parameters
    ----------
    p1, p2: float
        periods to check
    margin: float
        how close must they be to be considered harmonics
        default: 0.03
    limit: int
        up to what number harmonic should be tested
        default: 6

    Returns
    -------
    True if they are close to harmonics, False if not
    """
    ratio = max(p1, p2) / min(p1, p2)
    return any(i - margin < ratio < i + margin for i in range(1, limit))


def find_strongest_unique_period(GLS, *periods_to_avoid):
    """
    Find the strongest period in a GLS that is neither an alias nor a harmonic of the given periods_to_avoid.

    Parameters
    ----------
    GLS: array-like
        table with periods and power.
        if an astropy.table.table.Table instance, assume it has keys "period" and "power".
        else, try to convert it to that format.
    periods_to_avoid: float(s)
        periods whose aliases should be avoided
        1. and 29.5 are recommended

    Returns
    -------
    period: float
        period with the strongest signal that is not an alias of the given periods_to_avoid
    frequency: float
        frequency corresponding to the strongest period
    power: float
        GLS power at the strongest period
    index: int
        index in the GLS array of the strongest period
    """
    GLS_ = convert_gls_array_to_table(GLS)
    period = 0.
    power = 0.
    index = 0.
    i = 0
    found = False
    while not found:
        i += 1
        index = GLS_["power"].argmax()
        period = GLS_["period"][index]
        power = GLS["power"][index]
        ws = np.where(GLS_["power"] < GLS_["power"][index]*0.1)[0]
        exclude0 = ws[ws <= index]
        exclude1 = ws[ws >= index]
        if len(exclude0) > 0 and len(exclude1) > 0:
            exclude = (exclude0[-1], exclude1[0])
        elif len(exclude0) > 0:
            exclude = (exclude0[-1], None)
        elif len(exclude1) > 0:
            exclude = (0, exclude1[0])
        else:
            raise ValueError("Cannot find secondary peak")
        GLS_["power"][exclude[0]:exclude[1]] = 0
#        if (not checkalias(val) and not any([0.99*ex < val < 1.01*ex for ex in periods_to_avoid])) or i > 100:
        if not any(alias_of(period, a) or harmonic_of(period, a) for a in periods_to_avoid):
            found = True
        if i > 200 and not found: # maximum number of iterations
            found = True
            warn("mvs.periods.find_strongest_unique_period: reached maximum number of iterations; returning value found so far")
    frequency = 1./period
    return period, frequency, power, index

def gls_singleperiod(p, t, y, yerr):
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
