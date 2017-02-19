from astropy.table import Table
from numpy import ndarray

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
    if isinstance(gls_table, Table):
        pass # do nothing
    elif isinstance(gls_table, ndarray):
        if gls_table.shape[0] in (2, 3) and gls_table.shape[1] > 3:
            # assume it needs to be transposed
            gls_table = gls_table.T
        if gls_table.shape[1] == 2:
            gls_table = Table(gls_table, names=["period", "power"])
        elif gls_table.shape[1] == 3:
            gls_table = Table(gls_table, names=["period", "power", "f"])
    else:
        raise NotImplementedError("mvs.periods.find_N_strongest_periods can handle astropy tables and numpy arrays, not objects of type {0}".format(type(gls_arr)))
    return gls_table