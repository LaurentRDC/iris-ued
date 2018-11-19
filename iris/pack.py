# -*- coding: utf-8 -*-
"""
Pack a RawDataset instance int oa compressed HDF5 file.
"""

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .meta import HDF5ExperimentalParameter, MetaHDF5Dataset, MetaRawDataset
from .raw import AbstractRawDataset, open_raw


def pack(source_fname, packed_fname):
    """
    Pack a raw dataset into a compressed HDF5 archive.

    Parameters
    ----------
    source_fname : path-like
        Path to the original RawDataset instance. Its type will be inferred
        based on the available plugins, as determined by ``open_raw``.
    packed_fname : path-like
        Location of the resulting packed dataset.
    """
    try:
        with open_raw(str(source_fname)) as raw_dset:
            CompactRawDataset.pack(raw_dset, packed_fname)
    except RuntimeError:
        raise RuntimeError("The source format could not be inferred.")


class CompactMeta(MetaHDF5Dataset, MetaRawDataset):
    """ Metaclass combining the features of HDF5 datasets with abstract raw dataset."""

    def __init__(cls, clsname, bases, clsdict):
        # We only want to keep the HDF5 experimental parameters
        # Therefore, we initialize MetaRawDataset, then empty its valid_metadata
        # attribute, which will be re-created by MetaHDF5Dataset
        MetaRawDataset.__init__(cls, clsname, bases, clsdict)
        cls.valid_metadata = set([])

        MetaHDF5Dataset.__init__(cls, clsname, bases, clsdict)


class CompactRawDataset(h5py.File, AbstractRawDataset, metaclass=CompactMeta):
    """
    This class represents a compressed RawDataset.

    Parameters
    ----------
    source : path-like
        Path to the HDF5 file.
    """

    # List of valid metadata below
    # Using the HDF5ExperimentalParameter allows for automatic registering
    # of the parameters as valid.
    # These attributes can be accessed using the usual property access
    date = HDF5ExperimentalParameter("date", str, default="")
    energy = HDF5ExperimentalParameter("energy", float, default=90)  # keV
    pump_wavelength = HDF5ExperimentalParameter(
        "pump_wavelength", int, default=800
    )  # nanometers
    fluence = HDF5ExperimentalParameter("fluence", float, default=0)  # mj / cm**2
    time_zero_shift = HDF5ExperimentalParameter(
        "time_zero_shift", float, default=0
    )  # picoseconds
    temperature = HDF5ExperimentalParameter(
        "temperature", float, default=293
    )  # Kelvins
    exposure = HDF5ExperimentalParameter("exposure", float, default=1)  # seconds
    resolution = HDF5ExperimentalParameter("resolution", tuple, default=(2048, 2048))
    time_points = HDF5ExperimentalParameter(
        "time_points", tuple, default=tuple()
    )  # picoseconds
    scans = HDF5ExperimentalParameter("scans", tuple, default=(1,))
    camera_length = HDF5ExperimentalParameter(
        "camera_length", float, default=0.23
    )  # meters
    pixel_width = HDF5ExperimentalParameter(
        "pixel_width", float, default=14e-6
    )  # meters
    notes = HDF5ExperimentalParameter("notes", str, default="")

    def __init__(self, source, *args, **kwargs):
        # Default mode is 'r'. We can't put this in the parameters
        # because the class must follow the AbstractRawDataset __init__
        mode = kwargs.pop("mode", "r")
        super().__init__(name=source, mode=mode, **kwargs)

    @property
    def experimental_parameters_group(self):
        return self.require_group(name="/")

    @classmethod
    def pack(cls, source, path, **kwargs):
        """ 
        Create a compact raw dataset from some other raw dataset.
        
        Parameters
        ----------
        source : AbstractRawDataset 
            Any instance of ``AbstractRawDataset``.
        fname : path-like
            Path to the resulting HDF5 file.
        """
        if isinstance(path, Path):
            path = str(path)

        metadata = source.metadata
        timedelays = metadata["time_points"]
        scans = source.scans
        scan_shape = tuple(source.resolution) + (len(timedelays),)
        dtype = source.raw_data(timedelays[0], scans[0]).dtype

        # During testing, it was found that higher compression ratio for gzip
        # did not result in smaller files, but very long compression time.
        dset_kwargs = {
            "maxshape": scan_shape,
            "chunks": True,
            "shuffle": True,
            "fletcher32": True,
            "compression": "gzip",
            "compression_opts": 4,
        }

        # Creating a placeholder for a full scan accelerates encoding
        # Writing large blocks to file is much faster
        scan_buffer = np.empty(shape=scan_shape, dtype=dtype, order="C")

        # By default, fail to create if file exists
        # For tests, however, better to force overwrite
        if "mode" not in kwargs:
            kwargs["mode"] = "w-"

        with cls(path, **kwargs) as archive:

            # Saving of metadata is provided by MetaHDF5Dataset
            # Note that keys not associated with an HDF5ExperimentalParameter
            # descriptor will not be recorded in the file.
            for key, val in metadata.items():
                if key not in cls.valid_metadata:
                    continue
                setattr(archive, key, val)

            # Each scan is saved in its own dataset
            gp = archive.create_group("Scans")

            for scan in tqdm(
                scans,
                desc="Packing: ",
                unit=" scans",
                unit_scale=True,
                leave=True,
                ascii=True,
            ):

                for index, image in enumerate(
                    tqdm(
                        source.iterscan(scan),
                        unit=" time points",
                        unit_scale=True,
                        leave=False,
                        ascii=True,
                        total=len(timedelays),
                    )
                ):
                    scan_buffer[:, :, index] = image

                # TODO: affix time stamps as an attribute on this dataset
                dset = gp.create_dataset(
                    "Scan {:d}".format(scan), data=scan_buffer, **dset_kwargs
                )

        return path

    def raw_data(self, timedelay, scan=1, **kwargs):
        """
        Returns an array of the image at a timedelay and scan.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        kwargs
            Keyword-arguments are ignored.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        timedelay, scan = float(timedelay), int(scan)

        # timedelay_index cannot be cast to int() if np.argwhere returns an empty array
        # catch the corresponding TypeError
        try:
            timedelay_index = int(
                np.argwhere(np.asarray(self.time_points) == timedelay)
            )
        except TypeError:
            raise ValueError("Invalid time-delay")

        dset = self["Scans/Scan {:d}".format(scan)]

        return dset[:, :, timedelay_index]
