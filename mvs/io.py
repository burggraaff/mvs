from .misc import Star
import h5py
from astropy.io.ascii import read
from astropy import table, time, coordinates
import numpy as np

# Keys to retrieve from HDF5 files
hdf5keys = ("jdmid", "mag0", "emag0", "nobs", "lst", "lstseq")


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
        letter = filename.stem.split(".")[0].split("LP")[1]
    except:
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


def read_hdf5_to_table_for_one_star(filename, ASCC, force=False):
    """
    Read a single HDF5 file into an astropy table for a single star

    Parameters
    ----------
    filename: str
        HDF5 filename
    ASCC:
        ASCC code of the star to read data for
    force: boolean, optional
        If True, raise an Exception when the file cannot be opened or the star is not in that file.
        If False, skip ahead and return None
        Default: False

    Returns
    -------
    data_table: astropy.table.table.Table
        Table with data from filename for star ASCC
    """
    # Convert the ASCC number to a string, because that's what h5py expects
    ASCC = str(ASCC)

    # Read the HDF5 file with h5py
    with h5py.File(filename, "r") as file:
        # Try to access the data
        try:
            data = file["data"][ASCC]

        # If this star was not found, assume it is not present in this data set
        except KeyError:
            data_table = None

            # Raise an error if trying to force data from this file
            if force:
                raise ValueError(f"Star ASCC {ASCC} was not found in file `{filename}`")

        # If this star was found, extract the data into a table
        else:
            assert all(key in data.keys() for key in hdf5keys), f"Some of the requested keys are not in the HDF5 file.\nYou wanted {hdf5keys}\nThe file has {data.keys()}"

            # Manually put each column into an AstroPy table
            data_table = table.Table()
            for key in hdf5keys:
                col = table.Column(data[key], name=key)
                data_table.add_column(col)

    return data_table


def read_all_data_for_one_star(filenames, ASCC, force=False, min_nr_points=250, min_nobs=50):
    """
    Read data from a list of filenames for a given ASCC code

    Parameters
    ----------
    filenames: array-like
        List of HDF5 files to read
    ASCC:
        ASCC code of the star you want data for
    force: boolean, optional
        If True, raise an error if files do not contain this star.
        If False, skip those files.
        Default: False
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
    star: mvs.misc.Star
        Star object with information about the star (name, coords, ...)
    full_table: astropy.table.table.Table
        Table with the given keys for the given star from the given filenames
    """
    # Convert the ASCC number to a string, because that's what h5py expects
    ASCC = str(ASCC)

    # Create a Star object from the data
    star = Star.from_hdf5_files(ASCC, filenames, force=force)

    # Read the data
    data_tables = [read_hdf5_to_table_for_one_star(filename, ASCC, force=force) for filename in filenames]
    camera_names = [which_camera(filename) for filename in filenames]

    # Remove empty tables
    data_tables, camera_names = zip(*[(data, camera) for data, camera in zip(data_tables, camera_names) if data is not None])

    # Check whether data were found at all
    assert len(data_tables), f"No data for star ASCC {ASCC} found"

    # Add a column for camera name
    for data, camera in zip(data_tables, camera_names):
        camera_column = table.Column([camera] * len(data), name="camera", dtype="S3")
        data.add_column(camera_column)

    # Combine the data tables into one
    full_table = table.vstack(data_tables)

    # Remove data with insufficient nobs
    not_enough_nobs = np.where(full_table["nobs"] < min_nobs)[0]
    full_table.remove_rows(not_enough_nobs)

    # Remove data with negative uncertainties
    # These come from errors in the data pipeline
    negative_uncertainties = np.where(full_table["emag0"] <= 0.)[0]
    full_table.remove_rows(negative_uncertainties)

    # Normalise the uncertainties by the number of observations
    full_table["emag0"] /= np.sqrt(full_table["nobs"])

    # Remove cameras with insufficient data to detrend
    cameras_used = np.unique(full_table["camera"])
    for camera in cameras_used:
        # Find the data from this camera
        data_from_camera = np.where(full_table["camera"] == camera)[0]

        # If there are too few, remove them all
        if len(data_from_camera) < min_nr_points:
            full_table.remove_rows(data_from_camera)

    # Convert Julian Dates (JD) to heliocentric (HJD)
    # Temporary: use La Palma coordinates - in the future, get from files
    lapalma = coordinates.EarthLocation.from_geodetic(lat=28.763611, lon=-17.894722, height=2396)
    times = time.Time(full_table["jdmid"], format="jd", location=lapalma)
    full_table["jdmid"] = star.convert_JD_to_heliocentric(times)
    full_table.rename_column("jdmid", "BJD")
    full_table.sort("BJD")

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
