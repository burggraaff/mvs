from .constants import second, minute
from .mapping import map_single

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

def convert_gls_array_to_table(gls_arr):
    """
    Helper function for find_N_strongest_periods and find_strongest_non_alias_period.
    Makes a non-table GLS into a nicer format

    Parameters
    ----------
    gls_arr: array-like
        array to convert to a table
        if this is a table already, a copy is returned

    Returns
    -------
    gls_table: astropy.table.table.Table
        table with the contents of the GLS array
    """
    gls_table = gls_arr.copy()
    if isinstance(gls_table, table.Table):
        pass # do nothing
    elif isinstance(gls_table, np.ndarray):
        if gls_table.shape[0] in (2, 3) and gls_table.shape[1] > 3:
            # assume it needs to be transposed
            gls_table = gls_table.T
        if gls_table.shape[1] == 2:
            gls_table = table.Table(gls_table, names=["period", "power"])
        elif gls_table.shape[1] == 3:
            gls_table = table.Table(gls_table, names=["period", "power", "f"])
    else:
        raise NotImplementedError("mvs.periods.find_N_strongest_periods can handle astropy tables and numpy arrays, not objects of type {0}".format(type(gls_arr)))
    return gls_table

def find_N_strongest_periods(GLS, n = 1, astable = True):
    """
    Find the N periods with the strongest signals in a given GLS, preventing multiple detections of the same peak.

    Parameters
    ----------
    GLS: array-like
        table with periods and power.
        if an astropy.table.table.Table instance, assume it has keys "p" and "Power".
        else, try to convert it to that format.
    n: int, optional
        number of periods to find.
        default: 1
    astable: bool, optional
        return all results in one astropy table
        default: True
    Returns
    -------
    if astable:
        period_table: astropy.table.table.Table
        table with strongest periods, with associated frequencies, powers and indices in GLS
    else:
        periods: np.ndarray
            periods with the strongest signals in the GLS
        frequencies: np.ndarray
            frequencies corresponding to those signals
        powers: np.ndarray
            GLS power values for those periods
        indices: np.ndarray
            indices of those periods in the input GLS
    """
    n = int(n)
    assert n >= 1, "mvs.periods.find_N_strongest_periods: n must have a value of at least 1; instead got {0}".format(n)
    GLS_ = convert_gls_array_to_table(GLS)
    periods = np.zeros(n)
    powers = periods.copy()
    indices = np.zeros(n, dtype=int)
    for i in range(n):
        here = int(GLS_["power"].argmax())
        indices[i] = here
        periods[i] = GLS_["period"][here]
        powers[i] = GLS_["power"][here]
        ws = np.where(GLS_["power"] < GLS_["power"][here]*0.1)[0]
        exclude0 = ws[np.where(ws <= here)]
        exclude1 = ws[np.where(ws >= here)]
        if len(exclude0) > 0 and len(exclude1) > 0:
            exclude = (exclude0[-1], exclude1[0])
        elif len(exclude0) > 0:
            exclude = (exclude0[-1], None)
        elif len(exclude1) > 0:
            exclude = (0, exclude1[0])
        else:
            warn("Could not find all {0} periods requested; instead returning top {1}".format(n, i))
            periods = periods[:i+1]
            powers = powers[:i+1]
            indices = indices[:i+1]
            break
        GLS_["power"][exclude[0]:exclude[1]] = 0
    frequencies = 1./periods
    if astable:
        period_table = table.Table(data = [periods, frequencies, powers, indices], names = ["period", "frequency", "power", "index"])
        return period_table
    else:
        return periods, frequencies, powers, indices

def find_strongest_unique_period(GLS, *periods_to_avoid):
    """
    Find the strongest period in a GLS that is neither an alias nor a harmonic of the given periods_to_avoid.

    Parameters
    ----------
    GLS: array-like
        table with periods and power.
        if an astropy.table.table.Table instance, assume it has keys "p" and "Power".
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
    """
    Calculate the GLS for multiple periods
    This function uses multiprocessing if available

    Parameters
    ----------
    t: array-like
        horizontal axis (time) of data
    y: array-like
        vertical axis (magnitude) of data
    yerr: array-like
        errors on y
    p: array-like, optional
        periods to evaluate
        default: mvs.periods.default_periods
    min_period: float, optional
        minimum period for evaluation (e.g. because of Nyquist)
        default: 11 minutes (approximate Nyquist limit of MASCARA long cadence data)
    max_period: float, optional
        maximum period for evaluation (e.g. because of limited data)
        default: 100 days (appropriate for using 400 days of MASCARA data)
    astable: bool, optional
        if True, return as an astropy table; if False, as a numpy array
        default: True

    Returns
    -------
    pPf: astropy.table.table.Table *or* np.ndarray
        table/array with periods, powers and frequencies
    """
    assert len(t) == len(y) == len(yerr), "mvs.periods.gls_multi: lengths of t, y and yerr are not equal: {0}, {1}, {2}".format(len(t), len(y), len(yerr))
    if type(p) is list:
        p = np.array(p)
    p_ = p[np.where(p > 2.*min(np.diff(t)))] #no need to check periods shorter than the interval between data
    p_ = p_[np.where(p_ >= min_period)]
    p_ = p_[np.where(p_ <= max_period)]
    power = map_single(oneperiod, p_, t, y, yerr)
    if astable:
        pPf = table.Table(data=[p_, power, 1./p_], names=["period", "power", "frequency"])
    else:
        pPf = np.vstack((p_, power, 1./p_)).T
    return pPf

def gls_zoom(t, y, yerr, GLS, nrperiods=250, min_period = 11 * minute, max_period = 100., limit = 15):
    """
    Calculate the GLS for periods around the strongest periods in a given GLS, to prevent resolution problems.

    Parameters
    ----------
    t: array-like
        horizontal axis (time) of data
    y: array-like
        vertical axis (magnitude) of data
    yerr: array-like
        errors on y
    GLS: astropy.table.table.Table or np.ndarray
        table/array with previous GLS results
    nrperiods: int
        number of periods to check around previously found peaks and their aliases/harmonics
        default: 250
    min_period: float, optional
        minimum period for evaluation (e.g. because of Nyquist)
        default: 11 minutes (approximate Nyquist limit of MASCARA long cadence data)
    max_period: float, optional
        maximum period for evaluation (e.g. because of limited data)
        default: 100 days (appropriate for using 400 days of MASCARA data)
    limit: int, optional
        how many periods should be used
        default: 15

    Returns
    -------
    GLS: astropy.table.table.Table or np.ndarray
        input GLS updated with extra values
        same format as input
    """
    # get the strongest periods that are not aliases/duplicates of each other
    # n.b. we do not remove harmonics on purpose
    p_best = find_N_strongest_periods(GLS, 200, astable = False)[0]
    # remove duplicates:
    p_best = [p for j,p in enumerate(p_best) if not any(0.98 < p/p_ < 1.02 for p_ in p_best[:j])]
    # remove aliases:
    p_best = [p for j,p in enumerate(p_best) if not any(alias_of(p, p_) for p_ in p_best[:j])]
    p_best = p_best[:limit]

    test_aliases = [[1./(1./p + k) for k in range(-3, 4)] for p in p_best]
    test_harmonics = [[m * p for m in (2., 3., 4., 1./2., 1./3., 1./4.)] for p in p_best]
    test_periods = sorted([p for sub in test_aliases + test_harmonics for p in sub if p > min_period and p < max_period])

    perranges = [np.linspace(0.975*p, 1.125*p, nrperiods) for p in test_periods]
    perrange = np.concatenate(perranges)
    perrange = perrange[perrange > min_period]
    perrange = perrange[perrange < max_period]

    astable = isinstance(GLS, table.Table)

    GLS_extra = gls_multi(t, y, yerr, p = perrange, min_period = min_period, max_period = max_period, astable = astable)

    if astable:
        GLS_combined = table.vstack((GLS, GLS_extra))
        GLS_combined.sort("period")
    else:
        GLS_combined = np.vstack((GLS, GLS_extra))
        GLS_combined = np.sort(GLS_combined, axis = 0)
    return GLS_combined

def gls_full(t, y, yerr, periods_initial = default_periods, min_period = 11 * minute, max_period = 100., astable = True, limit = 15, nrperiods=250):
    gls_initial = gls_multi(t, y, yerr, p = periods_initial, min_period = min_period, max_period = max_period, astable = astable)
    gls_second  = gls_zoom(t, y, yerr, gls_initial, min_period = min_period, max_period = max_period, nrperiods = nrperiods, limit = limit)
    if astable:
        gls_second = table.unique(gls_second, "period")
    else:
        warn("mvs.periods.gls_full: cannot check for duplicate periods in GLS if not using astropy tables")
    return gls_second