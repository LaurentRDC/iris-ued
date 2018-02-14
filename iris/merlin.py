# -*- coding: utf-8 -*-
"""
Interface to datasets acquired by the Siwick group on the Merlin system.
"""
from glob import glob, iglob
from os import listdir
from os.path import basename, isdir, join

import numpy as np
from cached_property import cached_property

from npstreams import mean, pmap

from . import AbstractRawDataset
from skued import diffread

class MerlinRawDataset(AbstractRawDataset):
    """
    Raw Datasets taken with the Merlin Quad readout system using a MediPix 3Rx detector.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Time-points are given by names of subfolders in ``self.source``
        folders = [name for name in listdir(self.source) if isdir(join(self.source, name))]
        self.time_points    = sorted([float(name) for name in folders if name != 'probe_off'])

        # Filenames have the form: 
        #           'pumpon_21.000ps_5.mib
        pumpon_filenames = glob(join(self.source, '{:.3f}'.format(self.time_points[0]), 'pumpon_*ps_*'))
        scans_str = [basename(name).split('_')[-1].split('.')[0] for name in pumpon_filenames]

        self.fluence        = 15
        self.pixel_width    = 55
        self.exposure       = 1e-3
        self.energy         = 90
        self.resolution = diffread(pumpon_filenames[0]).shape

        self.scans          = sorted(map(int, scans_str))

    @cached_property
    def background(self):
        """ Average of all ``probe off`` pictures. """
        images = iglob(join(self.source, 'probe_off', 'probe_off*'))
        return mean(map(diffread, images))

    def raw_data(self, timedelay, scan = 1, bgr = True, **kwargs):
        """
        Returns an array of the image at a timedelay and scan.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        bgr : bool, optional
            If True (default), laser background is removed before being returned.
        
        Returns
        -------
        arr : `~numpy.ndarray`
        """
        time_str = '{:.3f}'.format(timedelay)
        fname = join(self.source, time_str, 'pumpon_{}ps_{}.tif'.format(time_str, scan))
        im = mibread(fname)

        if bgr:
            im = np.subtract(im, self.background)
            im[np.greater(self.background, im)] = 0
        
        return im