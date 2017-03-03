from collections.abc import Sized
from functools import partial
import multiprocessing as mp
import numpy as n

class cached_property(object):
    """
    Decorator that minimizes computations of class attributes by caching
    the attribute values if it ever accessed. Attrbutes are calculated once.
    
    This descriptor should be used for computationally-costly attributes that
    don't change much.
    """
    _missing = object()
    
    def __init__(self, attribute, name = None):      
        self.attribute = attribute
        self.__name__ = name or attribute.__name__
    
    def __get__(self, instance, owner = None):
        if instance is None:
            return self
        value = instance.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.attribute(instance)
            instance.__dict__[self.__name__] = value
        return value

def chunked(iterable, chunksize = 1):
    """
    Generator yielding multiple iterables of length 'chunksize'.

    Parameters
    ----------
    iterable : iterable
        Must have a __len__ attribute.
    chunksize : int, optional
    """
    length = len(iterable)
    for ndx in range(0, length, chunksize):
        yield iterable[ndx:min(ndx + chunksize, length)]

def pmap(func, iterable, args = tuple(), kwargs = {}, processes = None):
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