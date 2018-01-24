# -*- coding: utf-8 -*-
"""
Siwick Research Group RawDataset class as an example use
of AbstractRawDataset
"""
from glob import glob, iglob
from os.path import join, basename, isdir
from os import listdir
from cached_property import cached_property
from npstreams import mean

from ..merlin_images import mibread

from . import AbstractRawDataset

class MerlinRawDataset(AbstractRawDataset):
    """
    Raw Datasets taken with the Merlin Quad readout system using a MediPix 3Rx detector.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fluence        = 15
        self.resolution     = (256, 256)
        self.pixel_width    = 55
        self.exposure       = 1e-3
        self.energy         = 90

        # Time-points are given by names of subfolders in ``self.source``
        folders = [name for name in listdir(self.source) if isdir(join(self.source, name))]
        self.time_points    = sorted([float(name) for name in folders if name != 'probe_off'])

        # Filenames have the form: 
        #           'pumpon_21.000ps_5.mib
        pumpon_filenames = glob(join(self.source, '{:.3f}'.format(self.time_points[0]), 'pumpon_*ps_*.mib'))
        scans_str = [basename(name).split('_')[-1].split('.')[0] for name in pumpon_filenames]

        self.scans          = sorted(map(int, scans_str))

    @cached_property
    def probe_off(self):
        """ Average of all ``probe off`` pictures. """
        images = iglob(join(self.source, 'probe_off', 'probe_off*.mib'))
        return mean(map(mibread, images))

    def raw_data(self, timedelay, scan = 1, *args, **kwargs):
        """
        Returns an array of the image at a timedelay and scan.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        """
        time_str = '{:.3f}'.format(timedelay)
        fname = join(self.source, time_str, 'pumpon_{}ps_{}.mib'.format(time_str, scan))
        pumpon = mibread(fname)
        sub = pumpon - self.probe_off
        sub[self.probe_off > pumpon] = 0
        sub[sub > 4096] = 0 # dead pixels
        return sub