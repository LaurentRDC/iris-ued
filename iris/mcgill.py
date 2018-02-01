# -*- coding: utf-8 -*-
"""
Siwick Research Group RawDataset class as an example use
of AbstractRawDataset
"""
from contextlib import suppress
from glob import iglob
from os import listdir
from os.path import isdir, isfile, join
from re import search, sub

import numpy as np
from cached_property import cached_property
from skimage.io import imread

from npstreams import average

from . import AbstractRawDataset


def parse_tagfile(path):
    """ Parse a tagfile.txt from a raw dataset into a dictionary of values """
    metadata = dict()
    with open(path) as f:
        for line in f:
            key, value = sub('\s+', '', line).split('=') # \s+ means all white space , including 'unicode' white space
            try:
                value = float(value.strip('s'))             # exposure values have units of seconds
            except ValueError:
                value = None                                # value might be 'BLANK'
            metadata[key.lower()] = value
    return metadata

class McGillRawDataset(AbstractRawDataset):
    """
    Raw dataset from the Siwick Research Group Diffractometer

    Parameters
    ----------
    source : str
        Raw data directory
    metadata : dict, optional
        Experimental parameters and metadata
    
    Raises
    ------
    ValueError : if the source directory does not exist.
    """

    def __init__(self, source, metadata = dict()):
        if not isdir(source):
            raise ValueError('{} does not point to an existing directory'.format(source))
        super().__init__(source)

        # Populate experimental parameters
        # from a metadata file called 'tagfile.txt'
        _metadata = parse_tagfile(join(self.source, 'tagfile.txt'))
        self.fluence = _metadata.get('fluence') or 0
        self.resolution = (2048, 2048)
        self.current = _metadata.get('current') or 0
        self.exposure = _metadata.get('exposure') or 0
        self.energy = _metadata.get('energy') or 90
        
        # Determine acquisition date
        # If directory name doesn't match the time pattern, the
        # acquisition date will be the default value
        with suppress(AttributeError):
            self.acquisition_date = search('(\d+[.])+', self.source).group()[:-1]      #Last [:-1] removes a '.' at the end

        # To determine the scans and time-points, we need a list of all files
        image_list = [f for f in listdir(self.source) 
                      if isfile(join(self.source, f)) 
                      and f.endswith(('.tif', '.tiff'))]

        # Determine the number of scans
        # by listing all possible files
        scans = [search('[n][s][c][a][n][.](\d+)', f).group() for f in image_list if 'nscan' in f]
        self.scans = tuple({int(string.strip('nscan.')) for string in scans})

        # Determine the time-points by listing all possible files
        time_data = [search('[+-]\d+[.]\d+', f).group() for f in image_list if 'timedelay' in f]
        time_list =  list(set(time_data))     #Conversion to set then back to list to remove repeated values
        time_list.sort(key = float)
        self.time_points = tuple(map(float, time_list))

    @cached_property
    def background(self):
        """ Laser background """
        backgrounds = map(imread, iglob(join(self.source, 'background.*.pumpon.tif')))
        return average(backgrounds)

    def raw_data(self, timedelay, scan = 1, bgr = True, **kwargs): 
        """
        Returns an array of the image at a timedelay and scan. Pump-on background 
        is removed from the pattern before being returned.
        
        Parameters
        ----------
        timedelay : float
            Time-delay in picoseconds.
        scan : int, optional
            Scan number. 
        bgr : bool, optional
            If True (default), laser background is removed before being returned. 
        
        Returns
        -------
        arr : ndarray, shape (N,M)
        
        Raises
        ------
        ImageNotFoundError
            Filename is not associated with an image/does not exist.
        """ 
        #Template filename looks like:
        #    'data.timedelay.+1.00.nscan.04.pumpon.tif'
        sign = '' if float(timedelay) < 0 else '+'
        str_time = sign + '{0:.2f}'.format(float(timedelay))
        filename = 'data.timedelay.' + str_time + '.nscan.' + str(int(scan)).zfill(2) + '.pumpon.tif'

        im = imread(join(self.source, filename)).astype(np.float)
        if bgr:
            im -= self.background
            im[im < 0] = 0
        
        return im
