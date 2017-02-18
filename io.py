from warnings import warn
import h5py
from astropy.io.ascii import read
from astropy import table
from scipy.misc import imread
import numpy as np
try:
    from pyastronomy.pyasl import helio_jd
    HJD = True
except ImportError:
    warn("Could not import pyastronomy -- will use regular Julian Dates rather than Heliocentric", ImportWarning)
    HJD = False

cameras = ("N", "W", "E", "S", "C")

def read_hdf5_to_table_for_one_star(filename, ASCC, keys, force = False):
    assert len(keys), "mvs.io.read_hdf5_to_table_for_one_star: did not receive any keys to parse"
    # assert the keys are in the table
    try:
        l = h5py.File(filename, 'r')
    except IOError:
        if force:
            warn("mvs.io.read_hdf5_to_table_for_one_star: could not open file `{0}`".format(filename))
            return None
        else:
            raise IOError("mvs.io.read_hdf5_to_table_for_one_star: could not open file `{0}`".format(filename))
    try:
        data = l["data"][ASCC]
    except KeyError:
        l.close()
        if force:
            return None
        else:
            raise ValueError("mvs.io.read_hdf5_to_table_for_one_star: star ASCC {0} was not found in file `{1}`".format(ASCC, filename))
    t = table.Table()
    for key in keys:
        col = table.Column(data[key], name = key)
        t.add_column(col)
    l.close()
    return t

def read_multiple_hdf5s_for_one_star(ASCC, force = True, keys = ("jdmid", "mag0", "emag0", "nobs", "lst"), time = "jdmid", mag = "mag0", emag = "emag0", add_cameraname = True, min_nr_points = 250, min_nobs = 50, *filenames):
    # camera name ?
    assert time in keys, "mvs.io.read_multiple_hdf5s_for_one_star: key for time (`{0}`) not in keys: {1}".format(time, keys)
    assert mag in keys, "mvs.io.read_multiple_hdf5s_for_one_star: key for mag (`{0}`) not in keys: {1}".format(mag, keys)
    table_tuples = [("a", read_hdf5_to_table_for_one_star(filename, ASCC, keys, force = force)) for filename in filenames]
    table_tuples = [tup for tup in table_tuples if tup[1] is not None]
    assert len(table_tuples), "mvs.io.read_multiple_hdf5s_for_one_star: No data for star ASCC {0} found".format(ASCC)
    for letter, t in table_tuples:
        letter_col = table.Column([letter] * len(t), name = "camera", dtype = str)
        t.add_column(letter_col)
    only_tables = zip(*table_tuples)[1]
    full_table = table.vstack(only_tables)
    if "nobs" in keys:
        below_min = np.where(full_table["nobs"] < min_nobs)[0]
        full_table.remove_rows(below_min)
    if emag in keys:
        not_positive = np.where(full_table[emag] <= 0.)[0]
        full_table.remove_rows(not_positive)
        if "nobs" in keys:
            full_table[emag] /= np.sqrt(full_table["nobs"])
    for c in cameras:
        # if not enough data from this camera, you cannot detrend properly
        which = np.where(full_table["camera"] == c)[0]
        if len(which) < min_nr_points:
            full_table.remove_rows(which)
    if HJD:
        t[time] = helio_jd(t[time])
    full_table.sort(time)
    return full_table

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