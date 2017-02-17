"""
MASCARA variable star project
Olivier Burggraaff

functions and constants
"""

### IMPORTS

# SYSTEM
import sys
sys.path.append("/disks/strw1/burggraaff/msc/fc/") # import modules from that folder
import os
os.chdir("/disks/strw1/burggraaff/msc/Mascara/") # force script to run in this folder
#os.nice(1 - os.nice(0))
import subprocess
import parmap
import multiprocessing as mp
import time as timing
from glob import glob
import shutil

# SUBMODULES
#import old

# READ / WRITE
import h5py
from astropy.io.ascii import read
from scipy.misc import imread

# PLOTTING
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
matplotlib.rcParams.update({"font.size": 30})

# OTHER
import numpy as np
from astropy import table as T
from PyAstronomy import pyasl
from mechanize import Browser
import wget
import scipy.stats

# CONSTANTS

#time units expressed in minutes
#u = {"minutes": 1., "hours": 60., "days": 1440., "weeks": 10080.}
u = {"sec": 1./86400., "min": 1./1440., "hr": 1./24., "d": 1., "w": 7.}

siderealday = 23.9344699*u["hr"]

#the periods we will be testing for - in minutes
#periods = np.concatenate((np.arange(320.*u["sec"], 2.*u["hr"], 1.*u["sec"]), np.arange(2.*u["hr"], 12.*u["hr"], 10.*u["sec"]), np.arange(12.*u["hr"], 72.*u["hr"], 30.*u["sec"]), np.arange(3.*u["d"], 10.*u["d"], 1.*u["min"]), np.arange(10.*u["d"], 100.*u["d"], 1.*u["hr"]))) #, np.arange(0.566*u["days"], 0.568*u["days"], 0.0001)
minper = 11.*u["min"]
maxper = 100.
periods = np.concatenate((np.logspace(np.log10(minper), 1.0, 40000, endpoint=False), np.logspace(1.0, 2.0, 2000)))

#the different types of variables for Kepler
vartype_kepler = {"PER": "PER", "HADS": "DSCT", "RRAB": "RRAB", "RRC": "RRC", "CEP": "DCEP", "EW": "EW", "EB": "EB", "EA": "EA", "MIRA": "M", "QPER": "QPER", "APER": "CST"}

#fraction of best period to search in for second iteration
iter2span = (0.8, 1.2)

figsize = (20, 15)

f_res = "/disks/strw1/burggraaff/msc/Mascara/1yr"

crit = 10. # arcsec ; two objects from different catalogues are considered the same if their coords are within this limit

cmap = plt.get_cmap("jet")

ASCCtable = read("ASCC_names.dat", format="fixed_width")
ASCCtable["RA"] *= 15.

quarters = ["5Q1", "5Q2", "5Q3", "5Q4", "6Q1"]
starsindata = map(int, np.unique(np.concatenate([h5py.File("data/red0_vmag_201"+Z+"LP"+camera+".hdf5", 'r')["header"]["ascc"][:] for Z in quarters for camera in ("N", "S", "E", "W", "C") if Z+camera != "6Q1N"])))
starsindata.sort()

nightlim = (57054., 57485.)

startdate = 2400000

#poolnr = 12
poolnr = max([4, mp.cpu_count()-1])

symbols = ("^", "v", "o", "s", "*")
cameras = ("North", "South", "East", "West", "Central")

writeformat = "ascii.fixed_width"

chi2str = r"$\chi^2$"

# FUNCTIONS

def wrap_fluxplot(RES, t, which="", **kwargs):
    try:
        flxplt, flxax = fluxplot(t["jdmid"], t["mag"], yerr=t["emag"], sym=t["sym"], colours=t["night"], retax=True, **kwargs)
        ylim = flxax.get_ylim()
        flxax.legend(loc="best")
    except:
        raise ValueError("Could not make flux plot")

    try:
        flxplt.savefig(RES+"_data"+which+".png")
    except:
        raise IOError("Could not save flux plot")
    else:
        plt.close(flxplt)
        del flxplt
        return ylim

def wrap_gls(RES, t, which="", **kwargs):
    try:
        pLSP = GLS(t["jdmid"], t["mag"], t["emag"], **kwargs)
    except Exception as e:
        raise ValueError("Could not calculate GLS periodogram", e)

    try:
        pLSP.write(RES+which+".LSP", format=writeformat)
    except:
        raise IOError("Could not write GLS periodogram")
    else:
        return pLSP

def wrap_lspplot(RES, pLSP, which="", **kwargs):
    try:
        lspplt = lspplot(pLSP, **kwargs)
    except:
        raise ValueError("Could not plot GLS periodogram")

    try:
        lspplt.savefig(RES+"_lsp"+which+".png")
    except:
        raise IOError("Could not write GLS periodogram plot")
    else:
        plt.close(lspplt)
        del lspplt

def wrap_phaseplot(RES, t, which="", **kwargs):
    try:
        phsplt = phaseplot(t, ykey="mag", yerr=t["emag"], colours=t["night"], sym=t["sym"], **kwargs)
    except:
        raise ValueError("Could not make phase plot")

    try:
        phsplt.savefig(RES+"_phase"+which+".pdf")
    except Exception as e:
        print e
        raise IOError("Could not save phase plot")
    else:
        plt.close(phsplt)
        del phsplt

def wrap_LSTplot(RES, t, *args, **kwargs):
    try:
        LSTplt = LSTplot(t, *args, **kwargs)
    except:
        raise ValueError("Could not make LST plot")

    try:
        LSTplt.savefig(RES+"_LST.pdf", dpi=300)
    except:
        raise IOError("Could not save LST plot")
    else:
        plt.close(LSTplt)
        del LSTplt

def wrap_detrend(t, *args, **kwargs):
    try:
        mo_amp = detrend(t, *args, **kwargs)
    except:
        raise ValueError("Could not detrend data")

    try:
        t.remove_column("mag")
        t.rename_column("magD", "mag")
    except:
        raise ValueError("Could not change column names")
    else:
        return mo_amp

def wrap_bestperiods(RES, pLSP, which=""):
    try:
        temp = bestperiodN(pLSP, 200)
    except:
        raise ValueError("Could not find strongest periods")

    try:
        bestpers = T.Table(data=(temp[0], temp[1], 1./temp[0]), names=("period", "power", "frequency"))
    except ZeroDivisionError:
        raise ZeroDivisionError("Attempted to divide by zero in table of strongest periods")
    except:
        raise ValueError("Could not make table of strongest periods")

    try:
        bestpers.write(RES+"_periods"+which+".txt", format=writeformat)
    except:
        raise IOError("Could not write table of best periods to file")

def wrap_chi2(RES, t, p_best, zeroat, *args, **kwargs):
    try:
        RMStables, doubled, interval_found = periodRMS(t, p_best, zeroat)
    except:
        raise ValueError("Could not do chi^2 minimalisation")

    try:
        if interval_found:
            lowest, best, highest = p_interval(RMStables[doubled], len(t))
        else:
            lowest = highest = -1
            best = p_best
    except:
        raise ValueError("Could not determine errors on period")

    try:
        RMSplt = RMSplot(RMStables, doubled, interval_found, p_orig=p_best, nrpoints=len(t), *args, **kwargs)
    except:
        raise ValueError("Could not plot chi^2")

    try:
        RMSplt.savefig(RES+"_RMSplot.png")
    except:
        raise IOError("Could not save chi^2 plot")
    else:
        plt.close(RMSplt)
        del RMSplt

    try:
        RMStable = T.vstack(RMStables)
        RMStable.sort("p")
    except:
        raise ValueError("Could not make combined chi^2 table")

    try:
        RMStable.write(RES+".chi2", format=writeformat)
    except:
        raise IOError("Could not save chi^2 table to file")
    else:
        if lowest == 0. or highest == 100. or lowest == best or highest == best:
            best = p_best
            lowest = -1
            highest = -1
        return lowest, best, highest

def wrap_datatrend(RES, t, *args, **kwargs):
    try:
        fdpplt = plotdataphase(t, *args, sym=t["sym"], **kwargs)
    except:
        raise ValueError("Could not make plot of data with trend")

    try:
        fdpplt.savefig(RES+"_data_phase.png")
    except:
        raise IOError("Could not save plot of data with trend")
    else:
        plt.close(fdpplt)
        del fdpplt

def wrap_write(RES, t, binned):
    try:
        if "residual" not in t.keys():
            t.add_column(T.Column(data=residual(t, binned, key1="mag", key2="mag"), name="residual"))
    except:
        raise ValueError("Could not find/calculate residuals")

    try:
        t = t["sym", "night", "lst", "jdmid", "mag", "emag", "residual", "phase"]
        t.rename_column("jdmid", "hjd")
    except:
        raise KeyError("Could not find all necessary keys for writing table")

    try:
        t.write(RES+".dat", format=writeformat)
    except:
        raise IOError("Could not write table to file")

    try:
        binned.write(RES+".bins", format=writeformat)
    except:
        raise IOError("Could not write binned light curve to file")

def wrap_fetch(RES, row, p):
    try:
        restable, flagged = fetch_close(row, p)
    except:
        raise ValueError("Could not find close variables")

    try:
        restable.write(RES+"_close_variables.dat", format=writeformat)
    except:
        raise IOError("Could not write close variables to file")

    try:
        fetch_SDSS(row, RES)
    except:
        raise ValueError("Could not fetch SDSS finding chart")

    return flagged, restable

def wrap_jktplot(RES, *args, **kwargs):
    try:
        jktplt = jktplot(*args, **kwargs)
    except:
        raise ValueError("Could not make JKTEBOP plot")

    try:
        jktplt.savefig(RES+"_jkt.png")
    except:
        raise IOError("Could not write JKTEBOP plot to file")
    else:
        plt.close(jktplt)
        del jktplt

def wrap_info(RES, t, row, restable, low, high, ID, zeroat, bins):
    try:
        info = restable["Name", "Coords", "Period"][0:1] #:1 to make it a table instead of a row
        info.add_column(T.Column(name="p_low", data=[low]))
        info.add_column(T.Column(name="p_high", data=[high]))
        info.add_column(T.Column(name="ASCC", data=[int(ID)]))
        info.add_column(T.Column(name="\#pt", data=[len(t)]))
        info.add_column(T.Column(name="t_0", data=[zeroat]))
        info.add_column(T.Column(name="amp", data=[np.nanmax(bins["mag"]) - np.nanmin(bins["mag"])]))
    except:
        raise ValueError("Could not make info table")

    try:
        info.write(RES+".info", format=writeformat)
    except:
        raise IOError("Could not write info table to file")

def order(x):
    return -np.floor(np.log10(x))

def period_with_error(low, best, high, latex=False):
    lowdiff = best - low
    highdiff = high - best

    if low <= 0. or high <= 0. or highdiff > best:
        if latex:
            return r"$"+format(best, 'f')[:4]+r":$"
        else:
            return format(best, 'f')[:4]+":"

    loworder =  int(np.floor(np.log10( lowdiff)))
    highorder = int(np.floor(np.log10(highdiff)))

    lowestorder = -min(loworder, highorder)
    p_round = round(best,     lowestorder)
    l_round = round(lowdiff,  lowestorder)
    h_round = round(highdiff, lowestorder)

    percstr = "%0."+str(lowestorder)+"f"
    p_str = percstr % p_round
    l_str = percstr % l_round
    h_str = percstr % h_round

    if l_str == h_str:
        if latex:
            return r"$"+p_str+r" \pm "+l_str+r"$"
        else:
            return p_str+" +- "+l_str
    else:
        if latex:
            return r"$"+p_str+r"^{+"+h_str+r"}_{-"+l_str+"}$"
        else:
            return p_str+" +"+h_str+" -"+l_str

def v(ID):
    if type(ID) is not str:
        ID = str(ID)
    for camera, symbol in zip(("N", "S", "E", "W", "C"), symbols):
        for Z in quarters:
            try:
                l = h5py.File("data/red0_vmag_201"+Z+"LP"+camera+".hdf5", 'r')
            except IOError:
                continue
            try:
                V = l["header"]["vmag"][np.where(l["header"]["ascc"][:] == ID)[0][0]]
                return V
            except IndexError:
                continue
            finally:
                l.close()
    #if not returned yet
    raise ValueError("Could not find V magnitude of star "+ID)

def make_ylim(y, zoom=False):
    if not zoom:
        mean = 0.5*(np.nanmax(y) + np.nanmin(y))
        amp = 0.5*(np.nanmax(y) - np.nanmin(y))
        ylim = (mean + 1.1*amp, mean - 1.1*amp)
    else:
        perc_low = np.nanpercentile(y, 1)
        perc_high = np.nanpercentile(y, 99)
        amp = perc_high - perc_low
        ylim = (perc_high + 0.1 * amp, perc_low - 0.1 * amp)

    return ylim

def splitlist(l, argv):
    """
    INPUT 1: STEP (HOW MANY PROCESSES)
    INPUT 2: START (WHICH PROCESS IS THIS)
    """

    #l_ = l.copy()
    l_ = l[:] # copy

    justone = False

    start = 0
    step = 1

    if len(argv) == 2: # do just one star
        justone = True
        start = 1000

        print "Only star", argv[1]

    elif len(argv) == 3: # do a subset of stars
        step = int(argv[1])
        start = int(argv[2])

        print "Process", start, "/", step

    if justone:
        try:
            l_ = l[l["ASCC"] == int(argv[1])]
        except IndexError:
            l_ = [int(argv[1])]
    else:
        l_ = l[start::step]

    return l_, start

def MAD(residuals, *args): #*args deals with errors that need to be passed to chi2 but not MAD
    return np.median(abs(residuals)) # mean already subtracted

def chi2(residuals, errors):
    return np.sum((residuals / errors)**2.0) # mean already subtracted

# wortel van het verschil in chi2 is sigma afwijking

def chi2_red(residuals, errors, dof=1):
    return chi2(residuals, errors) / (len(residuals) - dof)

def deltachi2_sigma(nrpoints, param=1, sigma=3):
    dof = nrpoints - param
    conf_int = scipy.stats.chi2.cdf(sigma**2., 1.)
    sigma_errordiff = scipy.stats.chi2.ppf(conf_int, dof)
    return sigma_errordiff

def p_interval(chi2table, nrpoints, **kwargs):
    try:
        best = chi2table["chi2"].argmin()
    except KeyError:
        key = chi2table.keys().remove("p")[0]
        best = chi2table[key].argmin()

    deltachi2 = chi2table["chi2"] - chi2table["chi2"][best]

    sigma_errordiff = deltachi2_sigma(nrpoints, **kwargs)
    within_range = np.where(deltachi2 <= sigma_errordiff)[0]
    lowest = within_range[0]
    highest = within_range[-1]

    return chi2table["p"][[lowest, best, highest]] #, chi2table["p"][best], chi2table["p"][highest]

def ascctable(ASCC):
    ascc = int(ASCC)
    try:
        ind = np.where(ASCCtable["ASCC"] == ascc)[0][0]
    except IndexError:
        raise IndexError("Could not find "+str(ASCC)+" in ASCC table")
    return ASCCtable[ind]

def readdata(ASCC):
    ASCC = str(ASCC)
    ts = []
    len_total = 0
    for camera, symbol in zip(("N", "S", "E", "W", "C"), symbols):
        for Z in quarters:
            try:
                l = h5py.File("data/red0_vmag_201"+Z+"LP"+camera+".hdf5", 'r')
            except IOError:
                continue
            try:
                alldata = l["data"][ASCC]
            except KeyError:
                l.close()
                continue
            len_total += len(alldata["jdmid"])
#            if len(alldata["jdmid"]) < 250:
#                # if there is insufficient (lst) coverage to accurately
#                # calibrate this camera with the others
#                continue
            t = T.Table()
            for key in ("jdmid", "mag0", "emag0", "nobs", "lst"):
                t.add_column(T.Column(alldata[key], name=key))

            l.close()
            t.rename_column("mag0", "mag")
            t.rename_column("emag0", "emag")
            t.remove_rows(np.where(t["emag"] == 0.0)[0])
            t.remove_rows(np.where(t["nobs"] < 50))
            if len(t) == 0: # if no data is left over, go to the next file
                continue
            t["emag"] /= np.sqrt(t["nobs"])
            t.remove_column("nobs")
            t["jdmid"] -= startdate
            try:
                info = ascctable(ASCC)
                t["jdmid"] = map(lambda jd: pyasl.helio_jd(float(jd), info["RA"], info["Dec"]), t["jdmid"])
            except IndexError:
                t["jdmid"] = map(float, t["jdmid"])
            t = t[-np.isnan(t['mag'])]
#            if len(t) < nobs_cutoff:
#                print ""
#                continue
#            if len(t) < 250:
#                continue
            t.add_column(T.Column(whatnight(t["jdmid"]), name="night"))
            #if len(badnight) > 0:
            #    t.remove_rows(np.concatenate([np.where(t["night"] == i)[0] for i in badnight]))

            t.add_column(T.Column(data=[symbol]*len(t), name="sym", dtype=str))

            ts.append(t)
    if len(ts) == 0:
        raise ValueError("No data found for star ASCC "+ASCC)
    t = T.vstack(ts)
    for symbol in symbols:
        where_this = np.where(t["sym"] == symbol)[0]
        if len(where_this) < 250: # remove if only a few from this camera
        # because you cannot properly detrend if there's only a small nr of points
            t.remove_rows(where_this)
    t.sort("jdmid")
    if len(t) < 250:
        raise ValueError("Light curve only contains "+str(len(t))+" points")
    print "original length:", len_total
    return t

def returnwhich(retvals, bools):
    """
    For a long list of possible returns, choose which to return.

    Parameters
    ----------
    toret: array_like
        array containing the objects or values that may be returned.
    bools: array_like
        array containing booleans telling which objects/values should be returned.


    Returns
    -------
    toreturn: array_like
        array containing the objects chosen to be returned, in the same order as the input.


    Raises
    ------
    IOError
        when the inputs do not have the same length
        (yes I know that's not what IOError is for).
    ValueError
        if "bools" are all False, i.e. nothing should be returned.

    Examples
    --------
    >>> returnwhich([1, fig, (2, 6), "hi"], [False, True, True, False])
        [fig, (2, 6)]

    >>> returnwhich([1, 2, 3], [False, False])
        IOError("CANNOT RETURN")

    >>> returnwhich([1, 2], [False, False])
        ValueError("NOTHING TO RETURN")
    """
    if len(retvals) != len(bools):
        raise IOError("CANNOT RETURN")
    toreturn = [r for r, b in zip(retvals, bools) if b]
    if len(toreturn) > 1:
        return toreturn
    if len(toreturn) == 1:
        return toreturn[0]
    if len(toreturn) == 0:
        raise ValueError("NOTHING TO RETURN")

def combinekwargs(main, extra):
    """
    Combines two dictionaries into one.

    Parameters
    ----------
    main: dict
        the dictionary that should contain the end result.
    extra: dict
        extra values to be added to main, or keys of main that should be changed.

    Examples
    --------
    >>> def_kwargs = {"ls": "--", "title": "a plot", "lw": 3}
    >>> combinekwargs(def_kwargs, {"lw": 5})
    >>> def_kwargs
        {"ls": "--", "title": "a plot", "lw": 5}


    >>> def_kwargs = {"ls": "--", "title": "a plot", "lw": 3}
    >>> combinekwargs(def_kwargs, {"ylim": (0, 1)})
    >>> def_kwargs
        {"ls": "--", "title": "a plot", "lw": 3, "ylim": (0, 1)}

    >>> def_kwargs = {"ls": "--", "title": "a plot", "lw": 3}
    >>> combinekwargs(def_kwargs, {})
    >>> def_kwargs
        {"ls": "--", "title": "a plot", "lw": 3}
    """

    for kw in extra:
        main[kw] = extra[kw]

def vartype(vt, catalogue="gcvs"):
    """
    Makes a string for a variable type of a star more readable.

    Parameters
    ----------
    vt: str
        the variable type to be pulled apart.
    catalogue: {"gcvs", "kepler"}
        which catalogue to use (GCVS or Kepler).

    Returns
    -------
    result: str
        reformatted string containing the variable type.

    Examples
    --------
    >>> vartype("EW")
        "EW"

    >>> vartype("EW/ELL")
        "EW / ELL"

    >>> vartype("DSCT:+EW")
        "DSCT [uncertain] + EW"

    >>> vartype("HADS/MIRA", catalogue="kepler")
        "DSCT / M"
    """
    a = (catalogue.lower() == "kepler")

    varsplit = vt.split("/")
    var = [[]]*len(varsplit)
    for j,v in enumerate(varsplit):
        v = v.strip()
        if "+" in v:
            newv = v.split("+")
            for k, nv in enumerate(newv):
                nv = nv.strip()
                if ":" in nv:
                    #newv[k] = f[nv[:-1]]+" [uncertain]"
                    if a:
                        newv[k] = vartype_kepler[nv[:-1]]+" [uncertain]"
                    else:
                        newv[k] = nv[:-1]+" [uncertain]"
                else:
                    #newv[k] = f[nv]
                    if a:
                        newv[k] = vartype_kepler[nv]
                    else:
                        newv[k] = nv
            var[j] = reduce(lambda x,y: x+" + "+y, newv)
        else:
            if ":" in v:
                if a:
                    var[j] = vartype_kepler[v[:-1]]+" [uncertain]"
                else:
                    var[j] = v[:-1]+" [uncertain]"
                #var[j] = f[v[:-1]]+" [uncertain]"
            else:
                #var[j] = f[v]
                if a:
                    var[j] = vartype_kepler[v]
                else:
                    var[j] = v

    result = reduce(lambda x, y: x+" / "+y, var)
    return result

def whatnight(time, margin=0.5):
    """
    Determine on what observing night a particular measurement was.

    *Note*: strange behaviour when input `time' is not sorted.

    Parameters
    ----------
    time: array_like
        times (*in days*) of data measurements.
    margin: float, optional
        how many days should be between successive data points for them to be
        considered to be on separate nights.

    Returns
    -------
    nights: list
        what observing nights the data points were on.

    Examples
    --------
    >>> whatnight([0.1, 0.2, 0.3, 1.2])
        [0, 0, 0, 1]
    >>> whatnight([0.1, 0.2, 1.2, 1.9])
        [0, 0, 1, 2]
    >>> whatnight([0.1, 0.2, 0.3, 1.2, 0.4])
        [0, 0, 0, 1, 1]
    """
    nights = [[int(round(np.mean(z), 0))]*len(z) for z in np.array_split(time, np.where(np.diff(time) > margin)[0]+1)]
    nights = [n for N in nights for n in N] # flatten list of lists to just a list
    return nights

def glsmath(p_, t, y, yerr):
    omega = 2. * np.pi / (u["sec"] * p_) #angular frequencies
    w = 1./(yerr**2.) * 1./(np.sum(1./(yerr**2.))) #weights

    #w = np.tile(w.mean(), w.shape) # for testing
    w = w.mean()
    Y = np.sum(w*y)

    #t_omega = np.meshgrid(t*u["sec"], omega)[0] * omega[:, np.newaxis]

    t_omega = t*u["sec"]*omega
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

def gls(t, y, yerr, p = periods):
    if len(t) != len(y):
        raise ValueError("t and y do not have the same shape.")

    if type(p) is list:
        p = np.array(p)

    p_ = p[np.where(p > int(2.*min(np.diff(t))))] #no need to check periods shorter than the interval between data
    p_ = p_[np.where(p_ >= minper)]
    p_ = p_[np.where(p_ <= maxper)]
    pool = mp.Pool(poolnr)
    power = parmap.map(glsmath, p_, t, y, yerr, pool=pool)
    pool.close()
    pool.join()

    pLSP = T.Table(data=[p_, power, 1./p_], names=["p", "Power", "f"])
    return pLSP

def gls2(t, y, yerr, pLSP, bestp = None, bestp2 = None, nrperiods=7500, norm=True, **kwargs):
    """
    Perform the second iteration of the Lomb-Scargle Periodogram.

    Parameters
    ----------
    t: array_like
        data times (when taken), in **minutes**.
    y: array_like
        data values.
    pLSP: astropy.table.Table
        original LSP, containing original periods and powers.
    bestp: float, optional
        the *original* best period (calculated if not given).
    2ndbestp: float, optional
        the *original* second best period (calculated if not given).
    nrperiods: int, optional
        the number of extra periods around the best and second best to calculate on.
    norm: bool, optional
        whether or not to normalise (divide by variance of data)
    **kwargs:
        additional keyword arguments for lsp()

    Returns
    -------
    pLSP3: astropy.table.Table
        table with period and LSP power, merger of original and new
    """

    if bestp is not None and bestp2 is None:
#        bestp2 = bestperiodN(pLSP, 2)[0][1]
        bestp2 = bestperiodA(pLSP, bestp)[0][1]
    elif bestp is None and bestp2 is not None:
#        bestp = bestperiodN(pLSP, 1)[0][0]
        bestp = bestperiodA(pLSP)[0]
    elif bestp is None and bestp2 is None:
#        bestp2 = bestperiodN(pLSP, 2)[0][1]
        bestp = bestperiodA(pLSP)[0]
        bestp2 = bestperiodA(pLSP, bestp)[0]


    highestper = max([bestp, bestp2])
    lowestper = min([bestp, bestp2])
    if iter2span[1] * lowestper < iter2span[0] * highestper:
        p2 = np.linspace(iter2span[0] * bestp, iter2span[1] * bestp, nrperiods, endpoint=False)
        p3 = np.linspace(iter2span[0] * bestp2, iter2span[1] * bestp2, nrperiods, endpoint=False)
        p2 = np.unique(np.concatenate((p2, p3)))
    else: #if there is overlap
        p2 = np.linspace(iter2span[0]*lowestper, iter2span[1]*highestper, 2*nrperiods, endpoint=False)
    pLSP2 = gls(t, y, yerr, p = p2, **kwargs)
    newbest = bestperiodA(pLSP2)[0]
    pLSP3 = gls(t, y, yerr, p = np.linspace(0.99*newbest, 1.01*newbest, nrperiods*2.))
    pLSP4 = T.vstack((pLSP, pLSP2, pLSP3))
    del pLSP, pLSP2, pLSP3
    pLSP4.sort("p")

    return pLSP4

def gls3(t, y, yerr, pLSP, nrperiods=1000):

    # get the 20 strongest periods that are not aliases/duplicates of 1 day/each other
    p_best = bestperiodN(pLSP, 200)[0]
    p_best = [p for p in p_best if not checkalias(p)]
    #remove duplicates:
    p_best = [p for j,p in enumerate(p_best) if not any(0.98 < p/p_ < 1.02 for p_ in p_best[:j])]
    #remove aliases:
#    p_best = [p for j,p in enumerate(p_best) if not any(-0.02 < (1./p - 1./p_)%1 < 0.02 or (1./p - 1./p_)%1 > 0.98 for p_ in p_best[:j])]
    p_best = [p for j,p in enumerate(p_best) if not any(alias_of(p, p_) for p_ in p_best[:j])]

    p_best = p_best[:10]

    totest = [[1./(1./p + k) for k in range(-3, 4)] for p in p_best]
    totest2 = [[m * p for m in (2., 3., 4., 1./2., 1./3., 1./4.)] for p in p_best]
    totest = sorted([p for sub in totest+totest2 for p in sub if p > minper])

    perranges = [np.linspace(0.975*p, 1.125*p, nrperiods/5) for p in totest]
    perrange = [p for sub in perranges for p in sub]

    newpLSP = gls(t, y, yerr, p = perrange)

    pLSP = T.vstack((pLSP, newpLSP))
    del newpLSP
    pLSP.sort("p")

    return pLSP


def GLS(t, y, yerr, k1 = {}, k2 = {}):
    return T.unique(gls3(t, y, yerr, gls(t, y, yerr, **k1), **k2), "p")

def bestperiodN(pLSP, n = 1):
    """
    Find the strongest N periods in an LSP.
    """
    if n < 1:
        return [], [], []

    pLSP_ = pLSP.copy() # we'll be editing it so it's easier to make a copy
    val = np.zeros(n)
    ind = np.zeros(n, dtype=int)
    pows = val.copy()
    for i in range(n):
        ind[i] = int(pLSP_["Power"].argmax())
        val[i] = pLSP_["p"][ind[i]]
        pows[i] = pLSP_["Power"][ind[i]]
        ws = np.where(pLSP_["Power"] < pLSP_["Power"][ind[i]]*0.1)[0]
        exclude0 = ws[np.where(ws <= ind[i])]
        exclude1 = ws[np.where(ws >= ind[i])]
        if len(exclude0) > 0 and len(exclude1) > 0:
            exclude = (exclude0[-1], exclude1[0])
        elif len(exclude0) > 0:
            exclude = (exclude0[-1], None)
        elif len(exclude1) > 0:
            exclude = (0, exclude1[0])
        else:
            print "@@@ Could not find all", n, "peaks in periodogram @@@"
            return val, pows, ind
        pLSP_["Power"][exclude[0]:exclude[1]] = 0


    return val, pows, ind

def bestperiodA(pLSP, *notthese):
    """
    Find the strongest non-alias period in an LSP.
    """

    pLSP_ = pLSP.copy() # we'll be editing it so it's easier to make a copy
    val = 0
    ind = 0
    found = False
    i = 0
    while not found:
        i += 1
        ind = pLSP_["Power"].argmax()
        val = pLSP_["p"][ind]
        ws = np.where(pLSP_["Power"] < pLSP_["Power"][ind]*0.1)[0]
        exclude0 = ws[ws <= ind]
        exclude1 = ws[ws >= ind]
        if len(exclude0) > 0 and len(exclude1) > 0:
            exclude = (exclude0[-1], exclude1[0])
        elif len(exclude0) > 0:
            exclude = (exclude0[-1], None)
        elif len(exclude1) > 0:
            exclude = (0, exclude1[0])
        else:
            raise ValueError("Cannot find secondary peak")
        pLSP_["Power"][exclude[0]:exclude[1]] = 0

        if (not checkalias(val) and not any([0.99*ex < val < 1.01*ex for ex in notthese])) or i > 100:
            found = True

    return val, ind, i

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

def phase(t, p, *args, **kwargs):
    if "residual" in t.keys():
        t.remove_column("residual")
    bins = phase2(t, p, *args, **kwargs)
    bins = bins[bins["bin"] >= 0.]
    zeroat = t["jdmid"][abs(t["phase"]-bins["bin"][bins["mag"].argmax()]).argmin()]
    bins = phase2(t, p, *args, zeroat=zeroat, **kwargs)
    return bins, zeroat


def phase2(t, p, tkey="jdmid", ykey="mag", yerr=True, zeroat=None, retzeroat=False, tobin=True, bs=0.025, bs2=0.01):
    """
    Phase-wrap data (and make a plot).

    Parameters
    ----------
    t: astropy.table.Table
        ASCII table with data
    p: float
        period, in minutes.
    tkey: str, optional
        what key in t the time data is at
    ykey: bool, optional
        whether or not the table includes y error data
    yerrkey: str, optional
        what key in t the y error data is at
    zeroat: float, optional
        what time should correspond to 0 phase.
    retzeroat: bool, optional
        whether or not to return the value of zeroat.
    tobin: bool, optional
        whether or not to bin the data

    Returns
    -------
    binneddata: astropy.table.Table, optional
        the phased data, binned in phase space.
    zeroat: float, optional
        what time corresponds to 0 phase.
    """

#    time = t[tkey] # DAYS
#    yval = t[ykey]
#    yerrkey = "emag"
#    if yerr:
#        yerrval = t[yerrkey]

    if zeroat is None:
        zeroat = t[tkey][t[ykey].argmax()]

    phase = ((t[tkey]-zeroat)%p)/p
    if "phase" not in t.keys():
        t.add_column(T.Column(data=phase, name="phase"))
    else:
        t["phase"] = phase

    if not tobin:
        binneddata = None

#    yvalrepeat = np.concatenate((yval,yval,yval))
#    if yerr:
#        yerrrepeat = np.concatenate((yerrval,yerrval,yerrval))
#    else:
#        yerrrepeat = None
#    phaserepeat = np.concatenate((phase-1.,phase,phase+1.))

    if tobin:
        N = 151.
        binwidth = bs
        bin_centres = np.linspace(0., 1., N)
        means, errs = zip(*[binstuff(t, bin_centre, binwidth, ykey = ykey) for bin_centre in bin_centres])
        binneddata = T.Table(data = [bin_centres, means, errs], names = ["bin", ykey, "emag"])

    return returnwhich([binneddata, zeroat], [tobin, retzeroat])

def binstuff(t, bincentre, binwidth, pkey="phase", ykey="mag"):
    low = bincentre - binwidth/2.
    high = bincentre + binwidth/2.
    which = np.where(np.logical_and(low <= t[pkey], t[pkey] <= high))[0]
    if low <= 0:
        which_ = np.where(t[pkey] >= 1. + low)[0]
        which = np.concatenate((which, which_))
    if high >= 1:
        which_ = np.where(t[pkey] <= high - 1.)[0]
        which = np.concatenate((which, which_))

    if len(which) == 0:
        return np.nan, np.nan

    which = np.unique(which)

    try:
        w = t["emag"][which]**-2.
    except KeyError:
        w = np.ones_like(t[ykey][which])

    mean = np.average(t[ykey][which], weights = w)
    std = np.sqrt(np.average((t[ykey][which] - mean)**2., weights = w))
    err = std / np.sqrt(len(which))

    return mean, err

def fluxplot(t, y, yerr=None, colours=None, sym=None, fig=None, retfig=True, ax=None, retax=False, axkw={}, skw={}):
    """
    Plot flux over time.

    Parameters
    ----------
    t: array_like
        time of measurements.
    y: array_like
        data values.
    yerr: array_like, optional
        errors in data values.
    colours: array_like, optional
        values to colour datapoints by (if wanted)
    fig: matplotlib.figure.Figure, optional
        figure to plot things in - if None, a new one is created
    ax: matplotlib.axes.AxesSubplot, optional
        axes to plot things in - if None, a new one is created
    retax: bool, optional
        whether or not to return the axes object
    axkw: dict, optional
        **kwargs for the axes object
    skw: dict, optional
        **kwargs for the scatter plot

    Returns
    -------
    fig: matplotlib.figure.Figure, optional
        the created figure.
    ax: matplotlib.axes.AxesSubplot, optional
        the axes object containing the plot.
    axkwA: dict, optional
        **kwargs used for ax, merger of input and defaults.

    """

    skwA = {"s": 25, "marker": 'o', "label": "Data", "zorder": 2, "edgecolors": "none", "vmin": nightlim[0], "vmax": nightlim[1]}
    if colours is None:
        skwA["color"] = "black"
    combinekwargs(skwA, skw)

    axkwA = {"title": "Flux plot", "xlabel": "2,400,000+ Heliocentric Julian date (HJD)", "ylabel": "$\Delta$Magnitude", "xlim": nightlim}
    new = False
    if fig is None:
        fig = plt.figure(figsize=figsize)
        new = True
    if ax is None:
        if not "xlim" in axkwA:
            xdiff = t.max() - t.min()
            axkwA["xlim"] = (t.min() - xdiff*0.05, t.max()+xdiff*0.05)
    combinekwargs(axkwA, axkw)
    if ax is None:
        ax = fig.add_subplot(111, **axkwA)
        ax.ticklabel_format(style="plain", useOffset=False)
        new = True

    if not "ylim" in axkwA:
        ax.set_ylim(make_ylim(y))
        #mean = 0.5*(np.nanmax(y) + np.nanmin(y))
        #amp = 0.5*(np.nanmax(y) - np.nanmin(y))
        #ax.set_ylim(mean + 1.1*amp, mean - 1.1*amp)

    if colours is None:
        if sym is not None:
            skwA.pop("marker", None)
            for s in symbols:
                wh = np.where(sym == s)[0]
                if len(wh) == 0:
                    continue
                ax.scatter(t[wh], y[wh], marker=s, rasterized=True, **skwA)
        else:
            ax.scatter(t, y, rasterized=True, **skwA)
        if yerr is not None:
            ax.errorbar(t, y, yerr=yerr, fmt="none", color=skwA["color"], ecolor=skwA["color"], rasterized=True, zorder=skwA["zorder"]-1)
    else:
        skwA.pop("c", None) #remove c keyword if needed
        if sym is not None:
            skwA.pop("marker", None)
            for s, label in zip(symbols, cameras):
                wh = np.where(sym == s)[0]
                if len(wh) == 0:
                    continue
                if skwA["label"] is not None:
                    skwA["label"] = label
                ax.scatter(t[wh], y[wh], c=colours[wh], marker=s, cmap=cmap, rasterized=True, **skwA)
        else:
            ax.scatter(t, y, c=colours, cmap=cmap, rasterized=True, **skwA)
        if yerr is not None:
            ax.errorbar(t, y, yerr=yerr, fmt="none", color="black", ecolor="black", rasterized=True, zorder=skwA["zorder"]-1)

    if new:
        fig.tight_layout()

    try:
        return returnwhich([fig, ax], [retfig, retax])
    except ValueError: #if neither fig nor ax should be returned
        pass

def phaseplot(t, pkey="phase", ykey="mag", yerr=None, sym=None, colours=None, binned=None, binpkey="bin", binerrkey="emag", retax=False, axkw = {}, skw = {}, lkw={}):
    axkwA = {"title": "Phase plot", "xlim": (0,2), "ylabel": "Magnitude"}
    combinekwargs(axkwA, axkw)

    skwA = {"zorder": 2}
    combinekwargs(skwA, skw)

    lkwA = {"lw": 4, "color": "red", "zorder": 3, "path_effects": [path_effects.Stroke(linewidth=6, foreground="black"), path_effects.Normal()], "label": "Binned ("+str(len(binned))+" bins)"}
    combinekwargs(lkwA, lkw)

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3, 1)}, figsize=figsize)
    fluxplot(t[pkey], t[ykey], yerr = yerr, retfig=False, fig=fig, ax=axs[0], skw=skwA, colours=colours, sym=sym)
    skwA["label"] = None
    fluxplot(t[pkey]+1., t[ykey], yerr = yerr, fig=fig, ax=axs[0], retfig=False, axkw=axkwA, skw=skwA, colours=colours, sym=sym)
    axs[0].set(**axkwA)

    for ax in axs:
        ax.axvline(1, lw=2, ls="--", c='0.5', zorder=0)
        ax.axvline(0.5, lw=1, ls="--", c='0.5', zorder=0)
        ax.axvline(1.5, lw=1, ls="--", c='0.5', zorder=0)

    binned_ = binned.copy()
    binned_[binpkey] += 1
    binned_ = T.vstack([binned, binned_])
    axs[0].plot(binned_[binpkey], binned_[ykey], **lkwA)
    if binerrkey is not None:
        axs[0].plot(binned_[binpkey], binned_[ykey]-binned_[binerrkey], lw=1, c=lkwA["color"], zorder=lkwA["zorder"], label="Error")
        axs[0].plot(binned_[binpkey], binned_[ykey]+binned_[binerrkey], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    for i in (np.nanmin(binned[ykey]), np.nanmax(binned[ykey])):
        axs[0].axhline(i, c='k', ls="-.", zorder=lkwA["zorder"]+1)

    res = residual(t, binned, key1=ykey, key2=ykey) if not "residual" in t.keys() else t["residual"]

    fluxplot(t[pkey], res, yerr=yerr, fig=fig, ax=axs[1], retfig=False, skw=skwA, colours=colours, sym=sym)
    fluxplot(t[pkey]+1., res, yerr=yerr, fig=fig, ax=axs[1], retfig=False, skw=skwA, colours=colours, sym=sym)
    axs[1].set_xlabel("Phase")
    axs[1].set_ylabel("Residual")
    axs[1].set_title("Median absolute deviation: "+str(MAD(res)))
    axs[1].axhline(0, **lkwA)
    if binerrkey is not None:
        axs[1].plot(binned_[binpkey], -binned_[binerrkey], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])
        axs[1].plot(binned_[binpkey], binned_[binerrkey], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

#    axs[0].legend(loc="best")
    fig.tight_layout()

    return returnwhich([fig, axs], [True, retax])

def LSTplot(t, p, binned, zeroat, sym=None, retax=False, axkw={}, skw={}, lkw={}):

    axkwA = {"title": "Local Sidereal Time trend plot", "xlabel": "Local Sidereal Time (hours)"}
    combinekwargs(axkwA, axkw)

    skwA = {"marker": "o"}
    combinekwargs(skwA, skw)

    lkwA = {}
    combinekwargs(lkwA, lkw)

    if sym is None:
        sym = symbols[np.argmax([len(t[t["sym"] == s]) for s in symbols])]

    if "marker" not in skwA:
        skwA["marker"] = sym

    tnew = t.copy()
    tnew = tnew[tnew["sym"] == sym]
    tnew.sort("lst")
    tnew2 = tnew.copy()
    tnew2["lst"] += 24.
    tnew = T.vstack((tnew, tnew2))
    del tnew2

    tnew["residual"] = residual(tnew, binned, key1="mag", key2="mag")

    if "xlim" not in axkwA:
        lstdiff = np.diff(tnew["lst"])
        where_gap = lstdiff.argmax()
        left = tnew["lst"][where_gap]
        right = tnew["lst"][where_gap+1]
        axkwA["xlim"] = (right-0.5, left+24.5)

    lstplt, lstax = fluxplot(tnew["lst"], tnew["residual"], yerr=tnew["emag"], axkw=axkwA, skw=skwA, retax=True)

    lstax.set_xticks(range(0, 24*3, 1))
    lstax.set_xticklabels(3*range(0, 24, 1))
    lstax.set_xlim(axkwA["xlim"])
    lstax.grid(True)

    lstplt.tight_layout()

    return returnwhich([lstplt, lstax], [True, retax])

def plotdataphase(t, bins, p, zeroat, ykey="mag", tkey="jdmid", retax=False, skw={}, axkw={}, lkw={}, sym=None):
    axkwA = {"title": "Magnitude over time", "ylabel": "Magnitude", "xlim": nightlim}
    combinekwargs(axkwA, axkw)

    skwA = {"zorder": 3}
    combinekwargs(skwA, skw)

    lkwA = {"lw": 1, "color": "black", "zorder": 2}
    combinekwargs(lkwA, lkw)

    binstep = bins["bin"][1] - bins["bin"][0]
    t0 = zeroat + (axkwA["xlim"][0] - zeroat)//p * p - 2.*p # zero before window
    t1 = zeroat + (axkwA["xlim"][1] - zeroat)//p * p + 4.*p # zero after window
    nr_p = int((t1 - t0)/p)
    prange = np.arange(0, nr_p, binstep)
    timerange = [t0 + phase * p for phase in prange]
    bins_rep = T.vstack([bins[:-1] for i in range(nr_p)]) # :-1 so 0/1 isn't repeated

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3, 1)}, figsize=figsize)
    axs[0].set(**axkwA)
    fluxplot(t[tkey], t[ykey], yerr = t["e"+ykey], retfig=False, fig=fig, ax=axs[0], axkw=axkwA, skw=skwA, colours=t["night"], sym=sym)
    axs[0].plot(timerange, bins_rep[ykey], **lkwA)
    axs[0].legend(loc="best")

    if "residual" in t.keys():
        fluxplot(t[tkey], t["residual"], yerr=t["e"+ykey], fig=fig, ax=axs[1], retfig=False, skw=skwA, colours=t["night"], sym=sym)
        axs[1].axhline(0, **lkwA)
        axs[1].set_xlabel("Adjusted Julian date")
        axs[1].set_ylabel("Residual")

    fig.tight_layout()
    return returnwhich([fig, axs], [True, retax])

def residual(t, binned, binkey="bin", key1="magD", key2="mag"):
    binstep2 = (binned[binkey][1] - binned[binkey][0])/2.
    residuals = np.zeros_like(t[key1])
    for b in binned:
        if not 0. <= b[binkey] <= 1.:
            continue
        closest = np.where(np.logical_and(t["phase"] >= b[binkey] - binstep2, t["phase"] <= b[binkey] + binstep2))[0]
        residuals[closest] = t[key1][closest] - b[key2]
    return residuals

def detrend(t, prebinned, p_orig = 1., tkey="jdmid", ykey="mag", yerrkey="emag", iters=100, ASCC="", makeplots=False):
    binned = prebinned.copy()
    t.add_column(T.Column(data=np.zeros_like(t[ykey]), name="residual"))
    t.add_column(T.Column(data=np.copy(t[ykey]), name=ykey+"D"))

    for s in symbols:
        wh = np.where(t["sym"] == s)[0]
        if len(wh) == 0:
            continue
#        print s,
        for i, toremove in enumerate(np.concatenate(([29.5], siderealday / np.arange(1., iters+1.)))):
            bs = 0.025 if toremove != 29.5 else 0.15
            t["residual"][wh] = residual(t[wh], binned, key1=ykey+"D", key2=ykey)

            binned_r = phase2(t, toremove, tkey=tkey, ykey="residual", yerr=True, tobin=True, bs=bs)

            if toremove == 29.5:
                mo_amp = np.nanmax(binned_r["residual"]) - np.nanmin(binned_r["residual"])

            zz = residual(t[wh], binned_r, key1="magD", key2="residual")
            t["magD"][wh] = zz.data

            binned = phase2(t, p_orig, tkey=tkey, ykey="magD", yerr=True, tobin=True)

            binned.rename_column("magD", "mag")
            if np.nanmax(binned_r["residual"]) - np.nanmin(binned_r["residual"]) < 0.001:
                break

    return mo_amp

def tryperiod(period, t, zeroat):
    bins = phase2(t, period, ykey="mag", tobin=True, zeroat=zeroat)
    c = chi2(residual(t, bins, key1="mag", key2="mag"), t["emag"])
    return c

def looptryperiod(periodstotry, t, zeroat):
    pool = mp.Pool(poolnr)
    allrms = parmap.map(tryperiod, periodstotry, t, zeroat, pool=pool)
    pool.close()
    pool.join()

    return T.Table(data=[periodstotry, allrms], names=["p", "chi2"], dtype=[float, float])

def periodRMS(t, p_orig, zeroat, nr=1000, width=(0.995, 1.005)):
    doubleperiod = False

    RMStables = [None] * 2

    deltachi2 = deltachi2_sigma(len(t))

    for j in [1, 2]:
        perrange = j*np.linspace(width[0]*p_orig, width[1]*p_orig, nr)
        RMStables[j-1] = looptryperiod(perrange, t, zeroat)

    minchi2_orig = RMStables[0]["chi2"].min()
    minchi2_doubled = RMStables[1]["chi2"].min()
    if minchi2_doubled + deltachi2 <= minchi2_orig:
        doubleperiod = True # double the period if it is 3 sigma better

    minchi2 = minchi2_doubled if doubleperiod else minchi2_orig

    RMStable_errors = RMStables[doubleperiod]
    low, best, high = p_interval(RMStable_errors, len(t))

    if low > RMStable_errors["p"][0] and high < RMStable_errors["p"][-1]: # if uncertainty area fits in tested range,
        # improve accuracy
        intervalfound = True
        newperrange = np.linspace(1.5*low - 0.5*best, 1.5*high - 0.5*best, nr/10)
        RMStable_extra = looptryperiod(newperrange, t, zeroat)
        RMStable_errors = T.vstack((RMStable_extra, RMStable_errors))
        RMStable_errors.sort("p")

    low, best, high = p_interval(RMStable_errors, len(t))
    minchi2 = RMStable_errors["chi2"].min()

    origp = 2*p_orig if doubleperiod else p_orig

    i = 0 # counter to kill while loop after too many iterations
    while RMStable_errors["chi2"][0] < minchi2 + deltachi2 and i < 10: # if lower edge of uncertainty area outside tested range
        # in this case our accuracy is already bad anyway so we do not have to test new periods with amazing resolution
        i += 1
        oldbest = best
        low, best, high = p_interval(RMStable_errors, len(t))
        if best != oldbest:
            if 0.95 < best/origp < 1.05: # if it has changed a little, keep going
                i = 0
            else: # if the best period changes too much, say we cannot find a proper confidence interval
                i = 1000
                interval_found = False
                break
        newlow = max(0, -best + 2*low) # twice as far away
        if newlow == 0:
            break # if we are at 0
        newperrange = np.linspace(newlow, low, nr/10, endpoint=False)
        RMStable_extra = looptryperiod(newperrange, t, zeroat)
        RMStable_errors = T.vstack((RMStable_extra, RMStable_errors))
        RMStable_errors.sort("p")

    if i < 10:
        i = 0 # reset and do the high end
        interval_found = True
    else:
        interval_found = False # if no lower bound was found, don't bother with an upper one
             # else: pass" instead of nothing to make this clearer
    while RMStable_errors["chi2"][-1] < minchi2 + deltachi2 and i < 10: # if higher edge of uncertainty area outside tested range
        i += 1
        oldbest = best
        low, best, high = p_interval(RMStable_errors, len(t))
        if best != oldbest:
            if 0.95 < best/origp < 1.05: # if it has changed a little, keep going
                i = 0
            else: # if the best period changes too much, say we cannot find a proper confidence interval
                i = 1000
                interval_found = False
                break
        newhigh = min(100, -best + 2*high) # twice as far away
        if RMStable_errors["p"][-1] == 100.:
            break
        newperrange = np.linspace(high, newhigh, nr/10, endpoint=False)
        RMStable_extra = looptryperiod(newperrange, t, zeroat)
        RMStable_errors = T.vstack((RMStable_errors, RMStable_extra))
        RMStable_errors.sort("p")
    if i < 10:
        interval_found = True
    else:
        interval_found = False
    if doubleperiod:
        RMStables = [RMStables[0], RMStable_errors]
    else:
        RMStables = [RMStable_errors, RMStables[1]]

    return RMStables, doubleperiod, interval_found

def RMSplot(RMStables, doubled, interval_found, p_orig=0, nrpoints=2, sigma=3, retaxs=False, axkw={}):
#        axs[j-1].plot(RMStable[j-1]["p"], RMStable[j-1][mode], lw=3, c='r', label=label)
#        axs[j-1].axvline(p_orig*j, lw=2, ls="--", c='b', label="Original period")
#        axs[j-1].axvline(bestp[j-1], lw=2, ls="--", c='g', label="New best period")
#        axs[j-1].set_xlim(RMStable[j-1]["p"][0], RMStable[j-1]["p"][-1])
#        axs[j-1].set_xlabel("Period (days)")
#        axs[j-1].get_xaxis().get_major_formatter().set_useOffset(False)
#        for tick in axs[j-1].get_xticklabels():
#            tick.set_rotation(45)

    yfmt = ScalarFormatter(useMathText=True, useOffset=False)
    yfmt.set_powerlimits((0, 0))
    xfmt = ScalarFormatter(useMathText=True, useOffset=False)

    low, bestp, high = p_interval(RMStables[doubled], nrpoints, sigma=sigma)

    grkw = {"width_ratios": (1, 3) if doubled else (3, 1)}
    RMSplt, axs = plt.subplots(ncols=2, sharey=True, figsize=figsize, gridspec_kw=grkw)

    for ax, table in zip(axs, RMStables):
        ax.plot(table["p"], table["chi2"], lw=3, c="r", label=chi2str)
        ax.axvline(p_orig, lw=3, ls="--", c="g", label="Original period") #should be only in one but meh
        ax.axhline(RMStables[doubled]["chi2"].min(), lw=1, ls="--", c="r")
        ax.set_xlim(table["p"][[0, -1]]) # make it show only periods we actually tested
        ax.get_xaxis().set_major_formatter(xfmt)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    ax = axs[doubled]
    ax.set_xlabel("Period (days)")
    ax.axvline(bestp, lw=3, ls="--", c="k", label="Best period")
    if interval_found:
        ax.axvspan(low, high, color="0.5", alpha=0.5, label="Area of "+str(sigma)+r"$\sigma$ uncertainty")
    else:
        axkw["title"] += r"; 3$\sigma$ confidence interval could not be determined"
    ax.legend(loc="best")
    ax.set(**axkw)
#    ax.set_title(chi2str+" for various periods for ASCC "+ASCC)

    axs[0].set_ylabel(chi2str)
    axs[0].get_yaxis().set_major_formatter(yfmt)
    RMSplt.tight_layout()

    return returnwhich([RMSplt, axs], [True, retaxs])


def finderrors(t, p_best_rms, zeroat, nr=1000, width=(0.9999, 1.0001), sigma=3, ASCC="", axkw={}, RES=None):
    if type(ASCC) is not str:
        ASCC = str(ASCC)

    fig, ax = plt.subplots(figsize=figsize)
    perrange = np.linspace(width[0]*p_best_rms, width[1]*p_best_rms, nr)
    pool = mp.Pool(poolnr)
    allchi2 = parmap.map(tryperiod, perrange, t, zeroat, chi2, pool=pool)
    pool.close()
    pool.join()

    chi2table = T.Table(data=[perrange, allchi2], names=["p", "chi2"], dtype=[float, float])

    lowest, best, highest, sigma_errordiff = p_interval(chi2table, len(t))

    if RES is not None:
        ax.axvline(p_best_rms, lw=2, ls="--", c='g', label="Best period (old)")
        ax.axvline(best, lw=2, ls="--", c='r', label="Best period (new)")
        ax.axvspan(lowest, highest, color='0.5', alpha=0.5, label="Area of uncertainty ("+str(sigma)+r"$\sigma$)")
        ax.axhline(chi2table["chi2"].min() + sigma_errordiff, ls="--", c="k", lw=1, label=r"$\Delta \chi^2$ = "+"{0:.0g}".format(sigma_errordiff))

        ax.plot(chi2table["p"], chi2table["chi2"], lw=3, c="k", label=r"$\chi^2$")
        ax.set_xlim(chi2table["p"][0], chi2table["p"][-1])
        ax.set_xlabel("Period (days)")

        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        ax.set_ylabel(r"$\chi^2$")
        ax.set_title(r"$\chi^2$ error determination for star ASCC "+ASCC)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(RES+"_chi2.png")

    return lowest, best, highest, chi2table


def makenote(*args):
    return "".join(["A" if checkalias(a) else "+" for a in args])

def mu(p):
    return str(p.__format__(".3f"))+" d"

def scatter_colour(x, y, c, fkw={}, akw={}, retax=False, skw={}, retscat=False, cbkw={}, retcb=False, cbl=True, cblkw={}):
    """
    Make a scatter plot with colour-coded dots.

    Parameters
    ----------
    x: array_like
        x axis
    y: array_like
        y axis
    c: array_like
        colour axis

    fkw: dict, optional
        **kwargs for the figure
    akw: dict, optional
        **kwargs for the axes object
    retax: bool, optional
        whether or not to return the axes
    skw: dict, optional
        **kwargs for the scatter plot
    retscat: bool, optional
        whether or not to return the scatter object
    cbkw: dict, optional
        **kwargs for the colourbar
    retcb: bool, optional
        whether or not to return the colourbar
    cbl: bool, optional
        whether or not to draw lines on the colourbar
    cblkw: dict, optional:
        **kwargs for lines on colourbar
    black: bool, optional
        make plot white-on-black (True) or black-on-white (False)

    Returns
    -------
    fig: matplotlib.figure.Figure
        the figure containing the plot
    ax: matplotlib.axes.AxesSubplot, optional
        the axes containing the plot
    s: matplotlib.collections.PathCollection, optional
        the scatter plot
    cb: matplotlib.colorbar.Colorbar, optional
        the colourbar
    """

    figkw = {"figsize": figsize}
    combinekwargs(figkw, fkw)

    axkw = {"xlabel": "x", "ylabel": "y"}
    combinekwargs(axkw, akw)

    scatkw = {"s": 25, "vmin": 0, "vmax": 1}
    combinekwargs(scatkw, skw)

    colbkw = {"label": "colourbar"}
    combinekwargs(colbkw, cbkw)

    if cbl:
        colblinkw = {"colors": "black", "lw": 3, "linestyles": "--"}
        combinekwargs(colblinkw, cblkw)


    fig = plt.figure(**figkw)
    ax = fig.add_subplot(111, **axkw)
    if akw["xticks"] != []:
        ax.tick_params(axis="x", bottom="on", labelbottom="on", top="on", labeltop="on")
    ax.tick_params(axis="y", left="on", labelleft="on", right="on", labelright="on")
    s = ax.scatter(x, y, c=c, cmap=cmap, **scatkw)
    cb = fig.colorbar(s, **colbkw)
    if cbl:
        z = float(s.get_clim()[1])
        cb.ax.hlines([c.min()/z, c.max()/z], 0, 1, **colblinkw)
    ax.grid(True, color="black")

    fig.tight_layout(pad=0.5)

    return returnwhich([fig, ax, s, cb], [True, retax, retscat, retcb])

def RAdegtoHMS(RAdeg, hms=False):
    """
    Convert a value of Right Ascension from degrees to hours-minutes-seconds

    Parameters
    ---------
    RAdeg: float
        RA in degrees
    hms: bool, optional
        if True, return letters between units; else spaces

    Returns
    -------
    RAHMS: str
        RA in hours, minutes, seconds
    """

    h = RAdeg/15
    m = 60*(h - int(h))
    s = 60*(m - int(m))

    strh = str(int(h))
    strm = str(int(m))
    strs = str(int(round(s)))

    if h < 10:
        strh = "0"+strh
    if m < 10:
        strm = "0"+strm
    if round(s) < 10:
        strs = "0"+strs

    RAHMS = strh + " " + strm + " " + strs
    if hms:
        RAHMS = strh + "h" + strm + "m" + strs + "s"

    return RAHMS

def DecdegtoDMS(Decdeg, dms=False):
    """
    Convert a value of Declination from degrees to degrees-arcminutes-arcseconds

    Parameters
    ---------
    Decdeg: float
        Declination in degrees
    dms: bool, optional
        if True, return letters between units; else spaces

    Returns
    -------
    DecDMS: str
        Declination in degrees, arcminutes, arcseconds
    """
    d = Decdeg
    m = 60*(d - int(d))
    s = 60*(m - int(m))

    if m < 0:
        m = -m
    if s < 0:
        s = -s

    strd = str(int(d))
    strm = str(int(m))
    strs = str(int(round(s)))

    if 0 <= d < 10:
        strd = "+0"+strd
    elif d >= 10:
        strd = "+"+strd
    elif -10 < d < 0:
        strd = "-0"+strd[1:]
    if strd == "-0":
        strd = "-00"
    if m < 10:
        strm = "0"+strm
    if round(s) < 10:
        strs = "0"+strs

    DecDMS = strd + " " + strm + " " + strs
    if dms:
        DecDMS = strd + "d" + strm + "m" + strs + "s"

    return DecDMS

def VSX_to_deg(VSX):
    """
    Convert a set of coordinates from the VSX to degrees

    Parameters
    ----------
    VSX: str
        VSX-style RA, Dec coordinates

    Returns
    -------
    ra: float
        right ascension in degrees
    dec: float
        declination in degrees

    Raises
    ------
    ValueError:
        If there is no + or - to indicate the sign of the Declination
    """
    if "+" in VSX:
        ra, dec = VSX.split("+")
        sign = 1
    elif "-" in VSX:
        ra, dec = VSX.split("-")
        sign = -1
    else:
        ra = reduce(lambda x,y: x+y, VSX.split()[:3])
        dec = reduce(lambda x,y: x+y, VSX.split()[3:])
        sign = 1

    ra = (float(ra[:2]) + float(ra[3:5])/60. + float(ra[6:11])/3600.)/24. * 360.
    dec = float(dec[:2]) + float(dec[3:5])/60. + float(dec[6:10])/3600.
    dec *= sign

    return ra, dec

def alias_of(p1, p2):
    """
    Is p1 an alias of p2?
    """
    f1 = 1./p1 ; f2 = 1./p2
    diff = max(f1, f2) - min(f1, f2)
    diffmod = diff%1
    return (0.95 < diff < 6 and (-0.02 < diffmod < 0.02 or diffmod > 0.98))

def harmonic_of(p1, p2):
    """
    Is p1 a harmonic of p2?
    (integer ratio)
    """
    ratio = max(p1, p2) / min(p1, p2)
    return any(0.975*i < ratio < 1.025*i for i in range(2, 6))

def checkalias(p):
    if 0.95 <= p <= 1.08:
        return True
    if 27. <= p <= 32. or 13.75 <= p <= 15.:
        return True
    for i in range(2,8):
        if 0.98/i <= p <= 1.03/i:
            return True
    for i in range(8,20):
        if 0.99/i <= p <= 1.015/i:
            return True
    return False


def fetch_close(info, p):
    flagged = False
    RA = info["RA"]
    Dec = info["Dec"]
    coord = str(RA)+" "
    if Dec > 0:
        coord += "+"
    coord += str(Dec)
    br = Browser()
    br.set_handle_robots(False)
    br.open("https://www.aavso.org/vsx/index.php?view=search.top&ql=1")
    br.select_form(nr=0)
    br.form["fieldsize"] = "40"
    br.form["format"] = ["d",]
    br.form["targetcenter"] = coord
    br.submit()
    br.select_form(nr=0)
    results = br.submit()
    l = results.readlines()
    l2 = [a.split(",") for a in l[1:]]
    l2 = [[a.strip('"') for a in line] for line in l2]
    l2 = [[a[:-2].strip('"') if a[-2:] == "\r\n" else a for a in line] for line in l2]
    restable = T.Table(data=np.array(l2), names=["Name", "AUID", "Coords", "Const", "Type", "Period", "Mag"], dtype=["S20", "S20", "S25", "S3", "S10", "S15", "S15"])
    dists = [60.*pyasl.getAngDist(RA, Dec, *VSX_to_deg(c)) for c in restable["Coords"]] # arcmin
    restable.add_column(T.Column(data=dists, name="Dist"))
    restable = restable[("Name", "Coords", "Dist", "Type", "Mag", "Period")]
    restable.add_row(vals=[info["Name"], RAdegtoHMS(RA)+".00 "+DecdegtoDMS(Dec)+".00", "0.", "--", str(info["BVmin"]), str(p)])
    restable.sort("Dist")
    for row in restable[1:]:
        try:
            per = float(row["Period"])
        except:
            continue
        if 0.975 <= per / p <= 1.025 or 0.975 <= 2.*per / p <= 1.025 or 0.975 <= 0.5*per / p <= 1.025:
            flagged = True
            break

    br.close()

    return restable, flagged

def fetch_SDSS(info, RES):
    RA = info["RA"]
    Dec = info["Dec"]
    URL = "http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx?ra="+str(RA)+"&dec="+str(Dec)+"&scale=1&width=3600&height=3600&opt=XIG"
    path = RES+"_field.jpeg"
    if os.path.exists(path):
        subprocess.call(["rm -f "+path], stdout=open(os.devnull, 'w'), shell=True)
        # otherwise you get file (1), file (2) etc from wget
    wget.download(URL, out=path, bar=None)
    plt.figure(figsize=(20,20))
    plt.imshow(imread(path), zorder=0)
#    for r in [21.*60., 6.*60.]:
#        plt.gca().add_artist(plt.Circle((1200, 1200), radius=r, ls="dashed", color="r", lw=3, zorder=1, fill=True, alpha=0.1))
#    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
#    p.set_array(np.array(colors))
#    ax.add_collection(p)
    pk = {"zorder": 1, "alpha": 0.1, "lw": 3}
    patches = PatchCollection([Wedge((1800, 1800), 21.*60., 0, 360, width=(21.-6.)*60., color="red", **pk), Circle((1800, 1800), radius=2.5*60., color="blue", **pk)], alpha = pk["alpha"])
    plt.gca().add_collection(patches)
    plt.title("Neighbourhood of star ASCC "+str(info["ASCC"])+" ("+info["Name"]+") ; 1x1 degree box from 2MASS ; inner and outer aperture shown")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(RES+"_ann.png", bbox_inches='tight')
    plt.close()
    subprocess.call(["rm -f "+path], stdout=open(os.devnull, 'w'), shell=True)

def remove_files(*args):
    print ""
    for f in args:
        if os.path.exists(f):
            print "Removing", f
            os.remove(f)

def jkt(t, RES, period, magmin, magmax, zeroat, ASCC):
    jktfolder = "../jkt/"
    datafile = ASCC+".jkt"
    prmfile  = ASCC+".prm"
    prmout   = ASCC+"_out.prm"
    result   = ASCC+".result"
    model    = ASCC+".model"
    t["jdmid", "mag", "emag"].write(datafile, format="ascii.no_header")
    Fratio = 10.**((magmin - magmax)/-2.5)
    ratio = str(0.8*np.sqrt(1. - 1./Fratio))

    f = open(jktfolder+"default_3", "r")
    lines = f.readlines()
    for i in range(13, 23):
        lines[i] = lines[i][6:]
    prepend = ["3 5", "0.8 "+ratio, "90 -1", "10.0 0", "0 0", "1 0", "lin lin", "0.5 0.5", "0 0", "0 0", "0.0 "+str(magmin), str(period), str(zeroat), " 1 1", " 1 0", " 0 0", " 0 0", " 1 0", " 0 0", " 0 0", " 1 1", " 0 1", " 1 1", datafile, prmout, result, model]
    for i, pre in enumerate(prepend):
        lines[i] = pre + lines[i]

    newf = open(prmfile, "w")
    parameterstring = reduce(lambda a, b: a + b, lines)
    newf.write(parameterstring)
    newf.close()
    f.close()

    remove_files(prmout, result, model)
#    for f in (prmout, result, model):
#        zf = f
#        if os.path.exists(zf):
#            os.remove(zf) # Delete files so JKTEBOP can open them

    os.system(jktfolder+"jktebop "+prmfile) # Run JKTEBOP

#    #If limb darkening fit went wrong:
#    f_out = open(prmout, "r")
#    f_out_str = f_out.read()
#    f_out.close()
#    if "## Warning: the total limb darkening at the limb of star " in f_out_str:
#        lines[18] = " 0 1"+lines[18][4:]
#        lines[19] = " 0 1"+lines[19][4:]
##        prepend[18] = " 0 0"
##        for i, pre in enumerate(prepend):
##            lines[i] = pre + lines[i]
#        remove_files(prmout, result, model)
#        newf = open(prmfile, "w")
#        parameterstring = reduce(lambda a, b: a + b, lines)
#        newf.write(parameterstring)
#        newf.close()
#        os.system(jktfolder+"jktebop "+prmfile) # Run JKTEBOP

    model_phase = read(model)
    model_data  = read(result)

    model_phase.rename_column("PHASE", "bin")
    model_phase.rename_column("MAGNITUDE", "mag")

    model_data.rename_column("TIME", "jdmid")
    model_data.rename_column("MAGNITUDE", "mag")
    model_data.rename_column("ERROR", "emag")
    model_data.rename_column("PHASE", "phase")

    remove_files(datafile, prmfile, model, result)
#    for f in (datafile, prmfile, model, result):
#        if os.path.exists(f):
#            os.remove(f)

    model_phase.write(RES+".model", format="ascii.fixed_width")
    model_data.write(RES+".result", format="ascii.fixed_width")

    shutil.move(prmout, RES+"_out.prm")

    return model_phase, model_data

def jktplot(t, model_phase, model_data, retfig = True, retax = False, axkw = {}, lkw = {}):

    axkwA = {}
    combinekwargs(axkwA, axkw)

    lkwA = {"color": "blue", "label": "JKTEBOP model"}
    combinekwargs(lkwA, lkw)

    fig, ax = phaseplot(model_data, binned=model_phase, yerr=t["emag"], colours=t["night"], sym=t["sym"], binerrkey=None, retax=True, axkw=axkwA, lkw=lkwA)
    return returnwhich([fig, ax], [retfig, retax])

def lspplot_RRLyr():
    fkwA = {"title": "Generalised Lomb-Scargle Periodogram (GLS) for ASCC 425414 / RR Lyr"}
    lkwA = {"color": "black", "lw": 1, "label": "Lomb-Scargle Periodogram", "rasterized": True}
    matplotlib.rcParams.update({"font.size": 20})

    pLSP = read("/disks/strw1/burggraaff/msc/Mascara/1yr/425414_D.LSP", format="fixed_width")
    plt.figure(figsize=(20, 7))
    plt.plot(pLSP["p"], pLSP["Power"], **lkwA)
    plt.gca().tick_params(axis='x', which="both", direction="out", length=10, top="off")
    plt.gca().set_xlabel("Period (days)")
    plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.gca().grid(axis='x', which="major", color=lkwA["color"], ls="--")
    plt.gca().grid(axis='y', which="major", color=lkwA["color"], ls="-.")
    plt.title(fkwA["title"])
    plt.ylim(10.**-5., 1.)
    plt.xlim(pLSP["p"].min(), pLSP["p"].max())
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Period (d)")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.savefig("/disks/strw1/burggraaff/paper/RRLyr_LSP.pdf", dpi=300)
    matplotlib.rcParams.update({"font.size": 30})

def phsplt_RRLyr(sigma=3):
    t = read("/disks/strw1/burggraaff/msc/Mascara/1yr/425414.dat", format="fixed_width")
    binned = read("/disks/strw1/burggraaff/msc/Mascara/1yr/425414.bins", format="fixed_width")
    axkwA = {"title": "Phase plot of ASCC 425414 (RR Lyrae)", "xlim": (0,1), "ylabel": "$\Delta$Magnitude", "ylim": make_ylim(t["mag"], zoom=True)}
    skwA = {"zorder": 2}
    lkwA = {"lw": 4, "color": "red", "zorder": 3, "path_effects": [path_effects.Stroke(linewidth=6, foreground="black"), path_effects.Normal()], "label": "Binned"}

    mu, std = scipy.stats.norm.fit(t["residual"])
    t = t[np.abs(t["residual"]) <= sigma * std]

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3, 1)}, figsize=figsize)
    fluxplot(t["phase"], t["mag"], yerr = t["emag"], retfig=False, fig=fig, ax=axs[0], skw=skwA)
    axs[0].set(**axkwA)

    axs[0].plot(binned["bin"], binned["mag"], **lkwA)
    axs[0].plot(binned["bin"], binned["mag"]-binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"], label="Error in\nbinned curve")
    axs[0].plot(binned["bin"], binned["mag"]+binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    res = residual(t, binned, key1="mag", key2="mag") if not "residual" in t.keys() else t["residual"]

    fluxplot(t["phase"], res, yerr=t["emag"], fig=fig, ax=axs[1], retfig=False, skw=skwA)
    axs[1].set_xlabel("Phase")
    axs[1].set_ylabel("Residual")
    axs[1].axhline(0, **lkwA)
    axs[1].plot(binned["bin"], -binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])
    axs[1].plot(binned["bin"],  binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    axs[0].grid(True)
    axs[1].grid(True)

    fig.tight_layout()

    fig.savefig("/disks/strw1/burggraaff/paper/RRLyr_phase.pdf", dpi=300)

def phsplt_67033(t_, binned_):
    axkwA = {"title": "Phase plot of ASCC 67033 / TYC 4027-631-1", "xlim": (0,1), "ylabel": "$\Delta$Magnitude", "ylim": make_ylim(t_["mag"], zoom=True)}
    skwA = {"zorder": 2}
    lkwA = {"lw": 4, "color": "red", "zorder": 3, "path_effects": [path_effects.Stroke(linewidth=6, foreground="black"), path_effects.Normal()], "label": "Binned"}

    t = t_.copy()
    binned = binned_.copy()

    t["phase"][t["phase"] >= 0.5] -= 1.
    t["phase"] += 0.5
    binned = binned[binned["bin"] >= 0]
    binned = binned[binned["bin"] <= 1]
    binned["bin"][binned["bin"] >= 0.5] -= 1.
    binned["bin"] += 0.5
    binned.sort("bin")

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3, 1)}, figsize=figsize)
    fluxplot(t["phase"], t["mag"], yerr = t["emag"], retfig=False, fig=fig, ax=axs[0], skw=skwA, colours=t["night"], sym=t["sym"])
    axs[0].set(**axkwA)

    axs[0].plot(binned["bin"], binned["mag"], **lkwA)
    axs[0].plot(binned["bin"], binned["mag"]-binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"], label="Error in\nbinned curve")
    axs[0].plot(binned["bin"], binned["mag"]+binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    res = residual(t, binned, key1="mag", key2="mag") if not "residual" in t.keys() else t["residual"]

    fluxplot(t["phase"], res, yerr=t["emag"], fig=fig, ax=axs[1], retfig=False, skw=skwA, colours=t["night"], sym=t["sym"])
    axs[1].set_xlabel("Phase")
    axs[1].set_ylabel("Residual")
#    axs[1].set_title("Median absolute deviation: "+str(MAD(res)))
    axs[1].axhline(0, **lkwA)
    axs[1].plot(binned["bin"], -binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])
    axs[1].plot(binned["bin"],  binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    axs[0].legend(loc="best")
    axs[0].grid(True)
    axs[1].grid(True)

    fig.tight_layout()

    fig.savefig("/disks/strw1/burggraaff/paper/67033_phase.pdf")

def cep(x=0.07):
    # 3.72815745834 |     3.7280415435 |    3.72820269339
    skwA = {"zorder": 2}
    lkwA = {"lw": 4, "color": "red", "zorder": 3, "path_effects": [path_effects.Stroke(linewidth=6, foreground="black"), path_effects.Normal()], "label": "Binned"}
    p_best= 3.72815745834
    t_cep = read("1yr/571833.dat", format="fixed_width")
    bins_cep = read("1yr/571833.bins", format="fixed_width")
    t_fpo = readdata(571737)
    bins_fpo, zeroat = phase(t_fpo, p_best, tobin=True)
    mo_amp = wrap_detrend(t_fpo, bins_fpo, p_best) ; del mo_amp
    bins_fpo = phase2(t_fpo, p_best, tobin=True, zeroat=57083.407516 - x*p_best)
    fpo_amp = bins_fpo["mag"].max() - bins_fpo["mag"].min()

    print "starting plot"
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (1, 1)}, figsize=figsize)
    fluxplot(t_cep["phase"], t_cep["mag"], yerr = t_cep["emag"], retfig=False, fig=fig, ax=axs[0], skw=skwA)
    fluxplot(t_fpo["phase"], t_fpo["mag"], yerr = t_fpo["emag"], retfig=False, fig=fig, ax=axs[1], skw=skwA)

    for a, t, b, n in zip(axs, (t_cep, t_fpo), (bins_cep, bins_fpo), ("ASCC 571833", "ASCC 571737")):
        a.set_xlim(0, 1)
        a.set_ylim(make_ylim(t["mag"], zoom=True))
        a.grid(True)
        a.set_ylabel("$\Delta$Magnitude")
        a.set_title(n)
        a.plot(b["bin"], b["mag"], **lkwA)

    axs[1].set_xlabel("Phase")

    fig.tight_layout()
    fig.savefig("/disks/strw1/burggraaff/paper/fpos.pdf", dpi=300)
    plt.close(fig)

    return fpo_amp, t_cep, t_fpo

def Oconnell():
    t = read("1yr/307947.dat", format="fixed_width")
    binned = read("1yr/307947.bins", format="fixed_width")
    axkwA = {"title": "Phase plot of ASCC 307947 (V376 And)", "xlim": (0,1), "ylabel": "$\Delta$Magnitude", "ylim": make_ylim(t["mag"], zoom=True)}
    skwA = {"zorder": 2}
    lkwA = {"lw": 4, "color": "red", "zorder": 3, "path_effects": [path_effects.Stroke(linewidth=6, foreground="black"), path_effects.Normal()], "label": "Binned"}

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3, 1)}, figsize=figsize)
    fluxplot(t["phase"], t["mag"], yerr = t["emag"], retfig=False, fig=fig, ax=axs[0], skw=skwA)
    axs[0].set(**axkwA)

    axs[0].plot(binned["bin"], binned["mag"], **lkwA)
    axs[0].plot(binned["bin"], binned["mag"]-binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"], label="Error in\nbinned curve")
    axs[0].plot(binned["bin"], binned["mag"]+binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    res = residual(t, binned, key1="mag", key2="mag") if not "residual" in t.keys() else t["residual"]

    fluxplot(t["phase"], res, yerr=t["emag"], fig=fig, ax=axs[1], retfig=False, skw=skwA)
    axs[1].set_xlabel("Phase")
    axs[1].set_ylabel("Residual")
    axs[1].axhline(0, **lkwA)
    axs[1].plot(binned["bin"], -binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])
    axs[1].plot(binned["bin"],  binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    axs[0].grid(True)
    axs[1].grid(True)

    fig.tight_layout()
    fig.savefig("/disks/strw1/burggraaff/paper/V376And.pdf", dpi=300)
    plt.close("all")

def Hertz():
    t = read("1yr/848497.dat", format="fixed_width")
    binned = read("1yr/848497.bins", format="fixed_width")
    axkwA = {"title": "Phase plot of ASCC 848497 (W Gem)", "xlim": (0,1), "ylabel": "$\Delta$Magnitude", "ylim": make_ylim(t["mag"], zoom=True)}
    skwA = {"zorder": 2}
    lkwA = {"lw": 4, "color": "red", "zorder": 3, "path_effects": [path_effects.Stroke(linewidth=6, foreground="black"), path_effects.Normal()], "label": "Binned"}

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (3, 1)}, figsize=figsize)
    fluxplot(t["phase"], t["mag"], yerr = t["emag"], retfig=False, fig=fig, ax=axs[0], skw=skwA)
    axs[0].set(**axkwA)

    axs[0].plot(binned["bin"], binned["mag"], **lkwA)
    axs[0].plot(binned["bin"], binned["mag"]-binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"], label="Error in\nbinned curve")
    axs[0].plot(binned["bin"], binned["mag"]+binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    res = residual(t, binned, key1="mag", key2="mag") if not "residual" in t.keys() else t["residual"]

    fluxplot(t["phase"], res, yerr=t["emag"], fig=fig, ax=axs[1], retfig=False, skw=skwA)
    axs[1].set_xlabel("Phase")
    axs[1].set_ylabel("Residual")
    axs[1].axhline(0, **lkwA)
    axs[1].plot(binned["bin"], -binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])
    axs[1].plot(binned["bin"],  binned["emag"], lw=1, c=lkwA["color"], zorder=lkwA["zorder"])

    axs[0].grid(True)
    axs[1].grid(True)

    fig.tight_layout()
    fig.savefig("/disks/strw1/burggraaff/paper/Hertzsprung_progression.pdf", dpi=300)
    plt.close("all")