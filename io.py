from warnings import warn
from .misc import Star
import h5py
from astropy.io.ascii import read
from astropy import table
from scipy.misc import imread
import numpy as np
try:
    from PyAstronomy.pyasl import helio_jd
    HJD = True
except ImportError:
    warn("Could not import pyastronomy -- will use regular Julian Dates rather than Heliocentric", ImportWarning)
    HJD = False

cameras = ("N", "W", "E", "S", "C")

def create_star_from_hdf5_files(ASCC, filenames, force = False):
    ASCC = str(ASCC)
    for f in filenames:
        l = h5py.File(f, 'r')
        header = l["header"]
        try:
            index = np.where(header["ascc"][:] == ASCC)[0][0]
            ra = header["ra"][index]
            dec = header["dec"][index]
            spectype = header["spectype"][index]
            B = header["bmag"][index]
            V = header["vmag"][index]
            star = Star(ASCC, ra, dec, spectype, B, V)
            break
        except IndexError:
            if force:
                continue
            else:
                raise ValueError("Could not find the star ASCC {0} in the file {1}. Consider re-running with `force = True`.".format(ASCC, filenames))
        finally:
            l.close()
    try:
        return star
    except NameError:
        raise ValueError("Could not find the star ASCC {0} in any of the given files: {1}".format(ASCC, filenames))

def which_camera(filename):
    try:
        letter = filename.split(".")[0].split("LP")[1]
    except:
        letter = "?"
    if letter not in ("N", "S", "E", "W", "C"):
        letter = "?"
    return letter

def read_filenames_from_text_file(textfilename):
    with open(textfilename, 'r') as f:
        filenames = f.readlines()
        filenames = [F.strip() for F in filenames]
    return filenames

def read_hdf5_to_table_for_one_star(filename, ASCC, keys = ("jdmid", "mag0", "emag0", "nobs", "lst"), force = False):
    assert len(keys), "mvs.io.read_hdf5_to_table_for_one_star: did not receive any keys to parse"
    # assert the keys are in the table
    ASCC = str(ASCC)
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

def read_all_data_for_one_star(filenames, ASCC, force = True, keys = ("jdmid", "mag0", "emag0", "nobs", "lst"), time = "jdmid", mag = "mag0", emag = "emag0", add_cameraname = True, min_nr_points = 250, min_nobs = 50):
    assert time in keys, "mvs.io.read_all_data_for_one_star: key for time (`{0}`) not in keys: {1}".format(time, keys)
    assert mag in keys, "mvs.io.read_all_data_for_one_star: key for mag (`{0}`) not in keys: {1}".format(mag, keys)

    ASCC = str(ASCC)
    star = create_star_from_hdf5_files(ASCC, filenames, force = force)

    table_tuples = [(which_camera(filename), read_hdf5_to_table_for_one_star(filename, ASCC, keys, force = force)) for filename in filenames]
    table_tuples = [tup for tup in table_tuples if tup[1] is not None]
    assert len(table_tuples), "mvs.io.read_all_data_for_one_star: No data for star ASCC {0} found".format(ASCC)

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
    if any(full_table[time] > 2.4e6):
        full_table[time] -= 2.4e6
    if HJD:
        full_table[time] = [helio_jd(jd, star.ra, star.dec) for jd in full_table[time]]
        full_table.rename_column(time, "HJD")
        time = "HJD"

    full_table.sort(time)
    return star, full_table