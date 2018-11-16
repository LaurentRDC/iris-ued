# -*- coding: utf-8 -*-
"""
Siwick Research Group raw dataset class as an example use
of AbstractRawDataset
"""
from configparser import ConfigParser
from contextlib import suppress
from functools import lru_cache
from glob import iglob
from os import listdir
from os.path import isdir, isfile, join
from re import search, sub

import numpy as np

from npstreams import average
from skued import diffread

from . import AbstractRawDataset, check_raw_bounds


class McGillRawDataset(AbstractRawDataset):
    def __init__(self, source, *args, **kwargs):
        if not isdir(source):
            raise ValueError(
                "{} does not point to an existing directory".format(source)
            )

        metadata_dict = self.parse_metadata(join(source, "metadata.cfg"))
        super().__init__(source, metadata_dict)

    def parse_metadata(self, fname):
        """ 
        Translate metadata from experiment into Iris's metadata format. 
        
        Parameters
        ----------
        fname : str or path-like
            Filename to the config file.
        """
        metadata = dict()

        parser = ConfigParser(inline_comment_prefixes=("#"))
        parser.read(fname)
        exp_params = parser["EXPERIMENTAL PARAMETERS"]

        # Translation is required between metadata formats
        metadata["energy"] = exp_params["electron energy"]
        metadata["acquisition_date"] = exp_params["acquisition date"]
        metadata["fluence"] = exp_params["fluence"]
        metadata["temperature"] = exp_params["temperature"]
        metadata["exposure"] = exp_params["exposure"]
        metadata["notes"] = exp_params["notes"]
        metadata["pump_wavelength"] = exp_params["pump wavelength"]

        metadata["scans"] = list(range(1, int(exp_params["nscans"]) + 1))
        metadata["time_points"] = eval(exp_params["time points"])

        return metadata

    @check_raw_bounds
    def raw_data(self, timedelay, scan=1, **kwargs):
        """
        Returns an array of the image at a timedelay and scan. Dark background is
        always removed.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        kwargs
            Extra keyword arguments are ignored.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        # scan directory looks like 'scan 0132'
        # Note that a glob pattern is required because every diffraction pattern
        # has a timestamp in the filename.
        directory = join(self.source, "scan {:04d}".format(scan))
        try:
            fname = next(
                iglob(join(directory, "pumpon_{:+010.3f}ps_*.tif".format(timedelay)))
            )
        except StopIteration:
            raise IOError(
                "Expected the file for {t}ps and scan {s} to exist, but could not find it.".format(
                    t=timedelay, s=scan
                )
            )

        return diffread(fname)


class LegacyMcGillRawDataset(AbstractRawDataset):
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

    def __init__(self, source, *args, **kwargs):
        if not isdir(source):
            raise ValueError(
                "{} does not point to an existing directory".format(source)
            )
        super().__init__(source)

        # Populate experimental parameters
        # from a metadata file called 'tagfile.txt'
        _metadata = self.parse_tagfile(join(self.source, "tagfile.txt"))
        self.fluence = _metadata.get("fluence") or 0
        self.resolution = (2048, 2048)
        self.current = _metadata.get("current") or 0
        self.exposure = _metadata.get("exposure") or 0
        self.energy = _metadata.get("energy") or 90

        # Determine acquisition date
        # If directory name doesn't match the time pattern, the
        # acquisition date will be the default value
        with suppress(AttributeError):
            self.acquisition_date = search("(\d+[.])+", self.source).group()[
                :-1
            ]  # Last [:-1] removes a '.' at the end

        # To determine the scans and time-points, we need a list of all files
        image_list = [
            f
            for f in listdir(self.source)
            if isfile(join(self.source, f)) and f.endswith((".tif", ".tiff"))
        ]

        # Determine the number of scans
        # by listing all possible files
        scans = [
            search("[n][s][c][a][n][.](\d+)", f).group()
            for f in image_list
            if "nscan" in f
        ]
        self.scans = tuple({int(string.strip("nscan.")) for string in scans})

        # Determine the time-points by listing all possible files
        time_data = [
            search("[+-]\d+[.]\d+", f).group() for f in image_list if "timedelay" in f
        ]
        time_list = list(
            set(time_data)
        )  # Conversion to set then back to list to remove repeated values
        time_list.sort(key=float)
        self.time_points = tuple(map(float, time_list))

    @staticmethod
    def parse_tagfile(path):
        """ Parse a tagfile.txt from a raw dataset into a dictionary of values """
        metadata = dict()
        with open(path) as f:
            for line in f:
                key, value = sub("\s+", "", line).split(
                    "="
                )  # \s+ means all white space , including 'unicode' white space
                try:
                    value = float(
                        value.strip("s")
                    )  # exposure values have units of seconds
                except ValueError:
                    value = None  # value might be 'BLANK'
                metadata[key.lower()] = value
        return metadata

    @property
    @lru_cache(maxsize=1)
    def background(self):
        """ Laser background """
        backgrounds = map(diffread, iglob(join(self.source, "background.*.pumpon.tif")))
        return average(backgrounds)

    @check_raw_bounds
    def raw_data(self, timedelay, scan=1, bgr=True, **kwargs):
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
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        # Template filename looks like:
        #    'data.timedelay.+1.00.nscan.04.pumpon.tif'
        sign = "" if timedelay < 0 else "+"
        str_time = sign + "{0:.2f}".format(timedelay)
        filename = (
            "data.timedelay."
            + str_time
            + ".nscan."
            + str(scan).zfill(2)
            + ".pumpon.tif"
        )

        im = diffread(join(self.source, filename)).astype(np.float)
        if bgr:
            im -= self.background
            im[im < 0] = 0

        return im
