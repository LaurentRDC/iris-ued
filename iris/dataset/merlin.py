# -*- coding: utf-8 -*-
"""
Siwick Research Group RawDataset class as an example use
of AbstractRawDataset
"""
from glob import glob
from os.path import join, basename
from cached_property import cached_property
from ..merlin import mibread

from . import AbstractRawDataset

class MerlinRawDataset(AbstractRawDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pumpon_filenames = glob(join(self.source, 'pumpon_*ps.mib'))
        times_str = [basename(name).split('ps')[0][7:] for name in pumpon_filenames]

        self.time_points = sorted(map(float, times_str))
        self.fluence = 15

    @cached_property
    def probe_off(self):
        return mibread(join(self.source, 'probe_off.mib'))

    def raw_data(self, timedelay, *args, **kwargs):
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
        pumpon = mibread(join(self.source, 'pumpon_{:.3f}ps.mib'.format(timedelay)))
        sub = pumpon - self.probe_off
        sub[self.probe_off > pumpon] = 0
        sub[sub > 4096] = 0 # dead pixels
        return sub
