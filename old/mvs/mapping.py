import numpy as np
from warnings import warn

try:
    from parmap import map as parmap, starmap
    parallel = True
except ImportError:
    parallel = False
    warn("Did not find `parmap` package -- will not use multiprocessing!", ImportWarning)

def disable_multiprocessing():
    global parallel
    parallel = False

def enable_multiprocessing():
    global parallel
    parallel = True

def pmap_nopar(func, iterable, *args, **kwargs):
    """
    Non-parallel equivalent of pmap - a simple listcomp:
    [func(i, *args, **kwargs) for i in iterable]
    Consider using numpy instead to speed things up.

    Parameters
    ----------
    func:
        function to use.
    iterable:
        object to iterate over.
    *args:
        extra arguments for func.
    **kwargs:
        extra keyword arguments for func.

    Returns
    -------
    res:
        result of evaluating func over iterable.
        np.ndarray if iterable is np.ndarray.
    """

    res = [func(i, *args, **kwargs) for i in iterable]
    if isinstance(iterable, np.ndarray):
        res = np.array(res)
    return res

if parallel:
    def pmap_par(func, iterable, *args, **kwargs):
        """
        Small wrapper for parmap.map.

        Parameters
        ----------
        func:
            function to use.
        iterable:
            object to iterate over.
        *args:
            extra arguments for func.
        **kwargs:
            extra keyword arguments for func _or_ for parmap.map.

        Returns
        -------
        res:
            result of evaluating func over iterable.
            np.ndarray if iterable is np.ndarray.
        """
        res = parmap(func, iterable, *args, **kwargs)
        if isinstance(iterable, np.ndarray):
            res = np.array(res)
        return res

def map_single(func, iterable, *args, **kwargs):
    """
    Map a function over one iterable, using multiprocessing if possible.
    If multiprocessing is not available (cannot import `parmap` module) or disabled (`par = False`), equivalent to a list comprehension.
    Consider also looking at the documentation of mvs.mapping.pmap_par and mvs.mapping.pmap_nopar.

    Parameters
    ----------
    func:
        function to use.
    iterable:
        object to iterate over.
    par: bool, optional
        use multiprocessing (if possible)
        default: True
    *args:
        extra arguments for func.
    **kwargs:
        extra keyword arguments for func _or_ parmap.map if using multiprocessing.

    Returns
    -------
    res:
        result of calling pmap_par or pmap_nopar with given parameters:
        res = pmap_x(func, iterable, *args, **kwargs)
    """
    if parallel:
        res = pmap_par  (func, iterable, *args, **kwargs)
    else:
        res = pmap_nopar(func, iterable, *args, **kwargs)
    return res

def smap_nopar(func, iterables, *args, **kwargs):
    return NotImplemented

if parallel:
    def smap_par(func, iterables, *args, **kwargs):
        res = starmap(func, iterables, *args, **kwargs)
        if all(isinstance(i, np.ndarray) for i in np.ndarray):
            res = np.array(res)
        return res

def map_multi(func, iterables, par = True, *args, **kwargs):
    if par and parallel:
        return smap_par  (func, iterables, *args, **kwargs)
    else:
        return smap_nopar(func, iterables, *args, **kwargs)