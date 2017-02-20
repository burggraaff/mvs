from warnings import warn
from .misc import Star, cameras
import h5py
from astropy.io.ascii import read
from astropy import table
from scipy.misc import imread
import numpy as np
try:
    from PyAstronomy.pyasl import helio_jd
    HJD = True
except ImportError:
    warn("Could not import PyAstronomy -- will use regular Julian Dates rather than Heliocentric", ImportWarning)
    HJD = False

def create_star_from_hdf5_files(ASCC, filenames, force = True):
    """
    Create an mvs.misc.Star object for a given star from given hdf5 files

    Parameters
    ----------
    ASCC:
        ASCC code of the star
    filenames: array-like
        HDF5 filenames to search for the given star in
    force: boolean, optional
        If True, ignore files that do not contain this star.
        If False, raise an Exception in those cases.
        Default: True

    Returns
    -------
    star: mvs.misc.Star
        Object with stellar properties

    Raises
    ------
    ValueError:
        If the star cannot be found in a single (`force = False`) or any (`force = True`) of the given files
    """
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
    """
    Find out which camera a given filename was taken by (MASCARA).
    N.B. this assumes the filename to be of the format *LPx.*

    Parameters
    ----------
    filename: str
        HDF5 filename

    Returns
    -------
    letter: str
        First letter (from NSWEC) of the camera
        "?" if the camera letter could not be determined
    """
    try:
        letter = filename.split(".")[0].split("LP")[1]
    except:
        letter = "?"
    if letter not in cameras:
        letter = "?"
    return letter

def read_filenames_from_text_file(textfilename):
    """
    Read a text file containing paths into a list of filenames

    Parameters
    ----------
    textfilename: str
        Path to the text file

    Returns
    -------
    filenames: list
        List of filenames as strings
    """
    with open(textfilename, 'r') as f:
        filenames = f.readlines()
        filenames = [F.strip() for F in filenames]
    return filenames

def read_hdf5_to_table_for_one_star(filename, ASCC, keys = ("jdmid", "mag0", "emag0", "nobs", "lst"), force = False):
    """
    Read a single HDF5 file into an astropy table for a single star

    Parameters
    ----------
    filename: str
        HDF5 filename
    ASCC:
        ASCC code of the star to read data for
    keys: array-like, optional
        Keys from the table to read in
        Default: ("jdmid", "mag0", "emag0", "nobs", "lst")
    force: boolean, optional
        If False, raise an Exception when the file cannot be opened or the star is not in that file.
        If True, only warn in those cases (recommended when loading multiple files)
        Default: False

    Returns
    -------
    t: astropy.table.table.Table
        Table with data from filename for star ASCC
    """
    assert len(keys), "mvs.io.read_hdf5_to_table_for_one_star: did not receive any keys to parse"
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
    assert all(key in data.keys() for key in keys), "mvs.io.read_hdf5_to_table_for_one_star: some of the requested keys are not in the HDF5 file.\nYou wanted {0}\nThe file has {1}".format(keys, data.keys())
    t = table.Table()
    for key in keys:
        col = table.Column(data[key], name = key)
        t.add_column(col)
    l.close()
    return t

def read_all_data_for_one_star(filenames, ASCC, force = True, keys = ("jdmid", "mag0", "emag0", "nobs", "lst"), time = "jdmid", mag = "mag0", emag = "emag0", add_cameraname = True, min_nr_points = 250, min_nobs = 50):
    """
    Read data from a list of filenames for a given ASCC code

    Parameters
    ----------
    filenames: array-like
        List of HDF5 files to read
    ASCC:
        ASCC code of the star you want data for
    force: boolean, optional
        If True, ignore files that do not exist or do not contain the star. If False, raise an Exception in those cases.
        Default: True
    keys: array-like, optional
        Keys from the HDF5 files to retrieve
        Default: ("jdmid", "mag0", "emag0", "nobs", "lst")
    time: str, optional
        Which key corresponds to the horizontal (time) axis
        Default: "jdmid"
    mag: str, optional
        Which key corresponds to the vertical (magnitude) axis
        Default: "mag0"
    emag: str, optional
        Which key corresponds to the vertical error (emagnitude) axis
        Default: "emag0"
    add_cameraname: boolean, optional
        If True, add a column with the name of the camera each file is from using mvs.io.which_camera
    min_nr_points: int, optional
        Minimum number of points required from each camera.
        If there are fewer points from a certain camera, detrending may not work properly.
        Data from that camera is then discarded.
        Default: 250
    min_nobs: int, optional
        Minimum number of short cadence data points each long cadence point should be binned from.
        All others are discarded.
        Default: 50

    Returns
    -------
    s: mvs.misc.Star
        Star object with information about the star (name, coords, ...)
    t: astropy.table.table.Table
        Table with the given keys for the given star from the given filenames
    """
    assert time in keys, "mvs.io.read_all_data_for_one_star: key for time (`{0}`) not in keys: {1}".format(time, keys)
    assert mag in keys, "mvs.io.read_all_data_for_one_star: key for mag (`{0}`) not in keys: {1}".format(mag, keys)

    ASCC = str(ASCC)
    star = create_star_from_hdf5_files(ASCC, filenames, force = force)

    table_tuples = [(which_camera(filename), read_hdf5_to_table_for_one_star(filename, ASCC, keys, force = force)) for filename in filenames]
    table_tuples = [tup for tup in table_tuples if tup[1] is not None]
    assert len(table_tuples), "mvs.io.read_all_data_for_one_star: No data for star ASCC {0} found".format(ASCC)

    if add_cameraname:
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
    if any(full_table[time] > 2.4e6): # reduced JD
        full_table[time] -= 2.4e6
    if HJD:
        full_table[time] = [helio_jd(jd, star.ra, star.dec) for jd in full_table[time]]
        full_table.rename_column(time, "HJD")
        time = "HJD"

    full_table.sort(time)
    return star, full_table

def write_data_table(data_table, filename, format="ascii.fixed_width", **kwargs):
    """
    Write a data table to file
    This is a simple wrapper for Table.write, please see its documentation for more information.

    Parameters
    ----------
    data_table: astropy.table.table.Table
        Table to write
    filename: str
        Filename to write to
    format: str
        Format to write in
        Default: "ascii.fixed_width"
    **kwargs:
        extra keyword arguments for Table.write
    """
    data_table.write(filename, format = format, **kwargs)

read_data_table = read