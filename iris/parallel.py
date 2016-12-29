
from collections.abc import Sized
import ctypes
from functools import partial
from itertools import zip_longest
import multiprocessing as mp
import threading
import numpy as n

def chunked(iterable, chunksize = 1):
    """
    Generator yielding multiple iterables of length 'chunksize'.

    Parameters
    ----------
    iterable : iterable
        Must have a __len__ attribute.
    chunksize : int, optional
    """
    #TODO: find a way to make it work with generators?
    length = len(iterable)
    for ndx in range(0, length, chunksize):
        yield iterable[ndx:min(ndx + chunksize, length)]

def parallel_map(func, iterable, args = tuple(), kwargs = {}, processes = None):
    """
    Parallel application of a function with keyword arguments.

    Parameters
    ----------
    func : callable

    iterable : iterable
    
    args : tuple

    kwargs : dictionary, optional

    processes : int or None, optional
    """
    if not isinstance(iterable, Sized):
        iterable = tuple(iterable)
    
    with mp.Pool(processes) as pool:
        # Best chunking is largest possible chunking
        chunksize = max(1, int(len(iterable)/pool._processes))
        
        map_func = pool.map
        if args:
            map_func = pool.starmap
            iterable = ((i,) + args for i in iterable)

        return map_func(func = partial(func, **kwargs), 
                        iterable = iterable, 
                        chunksize = chunksize)

def parallel_sum(func, iterable, args = tuple(), kwargs = {}, processes = None):
    """
    Parallel sum that accumulates the result.

    Parameters
    ----------
    func : callable
        Callable of the form func(iterable, *args, **kwargs) that returns a single value.
        func must be callable on iterables of any length.
    iterable : iterable
    
    args : tuple

    kwargs : dictionary, optional

    processes : int or None, optional
    """
    if not isinstance(iterable, Sized):
        iterable = tuple(iterable)
    
    results = list()
    with mp.Pool(processes) as pool:
        chunksize = max(1, int(len(iterable)/pool._processes))
        for batch in chunked(iterable = iterable, chunksize = chunksize):
            # batch has to be embedded into a tuple because pool.apply unpacks arguments
            results.append(pool.apply_async(func, args = (batch,) + args, kwds = kwargs))
        
        return sum((r.get() for r in results))