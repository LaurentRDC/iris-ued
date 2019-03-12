# -*- coding: utf-8 -*-
"""
Controller object to handle DiffractionDataset objects
"""
import traceback
import warnings
from contextlib import suppress
from functools import wraps
from shutil import copy2
from types import FunctionType

import numpy as np
from PyQt5 import QtCore

from .. import AbstractRawDataset, DiffractionDataset, PowderDiffractionDataset


def error_aware(func):
    """
    Wrap an instance method with a try/except and emit a message.
    Instance must have a signal called 'error_message_signal' which
    will be emitted with the message upon error. 
    """

    @wraps(func)
    def aware_func(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except:
            exc = traceback.format_exc()
            self.error_message_signal.emit(exc)
            warnings.warn(exc, UserWarning)

    return aware_func


class ErrorAware(type(QtCore.QObject)):
    """ Metaclass for error-aware Qt applications. """

    def __new__(meta, classname, bases, class_dict):
        new_class_dict = dict()
        for attr_name, attr in class_dict.items():
            if isinstance(attr, FunctionType):
                attr = error_aware(attr)
            new_class_dict[attr_name] = attr

        return super().__new__(meta, classname, bases, new_class_dict)


def indicate_in_progress(method):
    """ Decorator for IrisController methods that should
    emit the ``operation_in_progress`` signal and automatically
    revert when the operation is finished. """

    @wraps(method)
    def new_method(*args, **kwargs):
        args[0].operation_in_progress.emit(True)
        try:
            return method(*args, **kwargs)
        finally:
            args[0].operation_in_progress.emit(False)

    return new_method


class IrisController(QtCore.QObject, metaclass=ErrorAware):
    """
    Controller behind Iris.
    """

    raw_dataset_loaded_signal = QtCore.pyqtSignal(bool)
    processed_dataset_loaded_signal = QtCore.pyqtSignal(bool)
    powder_dataset_loaded_signal = QtCore.pyqtSignal(bool)

    raw_dataset_metadata = QtCore.pyqtSignal(dict)
    dataset_metadata = QtCore.pyqtSignal(dict)
    powder_dataset_metadata = QtCore.pyqtSignal(dict)

    error_message_signal = QtCore.pyqtSignal(str)
    operation_in_progress = QtCore.pyqtSignal(bool)
    status_message_signal = QtCore.pyqtSignal(str)

    raw_data_signal = QtCore.pyqtSignal(object)
    averaged_data_signal = QtCore.pyqtSignal(object)
    powder_data_signal = QtCore.pyqtSignal(object, object)

    time_series_signal = QtCore.pyqtSignal(object, object)
    powder_time_series_signal = QtCore.pyqtSignal(object, object)

    relative_powder_enable_signal = QtCore.pyqtSignal(bool)
    relative_averaged_enable_signal = QtCore.pyqtSignal(bool)
    powder_bgr_enable_signal = QtCore.pyqtSignal(bool)

    processing_progress_signal = QtCore.pyqtSignal(int)
    powder_promotion_progress = QtCore.pyqtSignal(int)
    angular_average_progress = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = None
        self.raw_dataset = None
        self.dataset = None

        # Internal state for powder background removal. If True, display_powder_data
        # will send background-subtracted data
        self._bgr_powder = False

        # Internal state for display format of averaged and powder data
        self._relative_powder = False
        self._relative_averaged = False
        self._timedelay_index = 0

        # array containers
        # Preallocation is important for arrays that change often. Then,
        # we can take advantage of the out parameter
        # These attributes are not initialized to None since None is a valid
        # out parameter
        # TODO: is it worth it?
        self._averaged_data_container = None
        self._average_time_series_container = None

    @QtCore.pyqtSlot(int, int)
    def display_raw_data(self, timedelay_index, scan):
        """
        Extract raw data from raw dataset.

        Parameters
        ----------
        timedelay_index : int
            Time-delay index.
        scan : int
            Scan number.
        """
        timedelay = self.raw_dataset.time_points[timedelay_index]
        self.raw_data_signal.emit(self.raw_dataset.raw_data(timedelay, scan, bgr=True))
        self.status_message_signal.emit(
            "Displaying data at {:.3f}ps, scan {:d}.".format(timedelay, scan)
        )

    @QtCore.pyqtSlot(int)
    def display_averaged_data(self, timedelay_index):
        """
        Extract processed diffraction pattern.

        Parameters
        ----------
        timedelay_index : int
            Time-delay index.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
            Diffracted intensity
        """
        # Preallocation of full images is important because the whole block cannot be
        # loaded into memory, contrary to powder data
        # Note that the containers have been initialized when the dataset was loaded
        # Therefore, no error checking required.
        self._timedelay_index = timedelay_index
        timedelay = self.dataset.time_points[timedelay_index]
        self._averaged_data_container[:] = self.dataset.diff_data(
            timedelay,
            relative=self._relative_averaged,
            out=self._averaged_data_container,
        )

        self.averaged_data_signal.emit(self._averaged_data_container)
        self.status_message_signal.emit(
            "Displaying data at {:.3f}ps.".format(timedelay)
        )

    @QtCore.pyqtSlot()
    def display_powder_data(self):
        """ Emit a powder data signal with/out background """
        # Preallocation isn't so important for powder data because the whole block
        # is loaded
        self.powder_data_signal.emit(
            self.dataset.scattering_vector,
            self.dataset.powder_data(
                timedelay=None, relative=self._relative_powder, bgr=self._bgr_powder
            ),
        )

    @QtCore.pyqtSlot(bool)
    def powder_background_subtracted(self, enable):
        """
        Toggle background subtraction for polycrystalline data.

        Parameters
        ----------
        enable : bool
        """
        self._bgr_powder = enable
        self.powder_bgr_enable_signal.emit(enable)
        self.display_powder_data()

    @QtCore.pyqtSlot(bool)
    def enable_powder_relative(self, enable):
        """
        Toggle relative scaling for polycrystalline data.

        Parameters
        ----------
        enable : bool
        """
        self._relative_powder = enable
        self.relative_powder_enable_signal.emit(enable)
        self.display_powder_data()

    @QtCore.pyqtSlot(bool)
    def enable_averaged_relative(self, enable):
        """
        Toggle relative scaling for diffracted data.

        Parameters
        ----------
        enable : bool
        """
        self._relative_averaged = enable
        self.relative_averaged_enable_signal.emit(enable)
        self.display_averaged_data(self._timedelay_index)

    @QtCore.pyqtSlot(float, float)
    def powder_time_series(self, qmin, qmax):
        """
        Extract polycrystalline time-series

        Parameters
        ----------
        qmin, qmax : float
            Lower- and upper-bounds of integration.
        """
        time_series = self.dataset.powder_time_series(
            rmin=qmin, rmax=qmax, bgr=self._bgr_powder, units="momentum"
        )
        self.powder_time_series_signal.emit(self.dataset.time_points, time_series)

    @QtCore.pyqtSlot(tuple)
    def time_series(self, rect):
        """" 
        Single-crystal time-series as the integrated diffracted intensity inside a rectangular ROI

        Parameters
        ----------
        rect : pyqtgraph.ROI
            Rectangle ROI defining integration bounds.
        """
        # Remember for updates
        self._rect = rect

        # Note that the containers have been initialized when the dataset was loaded
        # Therefore, no error checking required.
        integrated = self.dataset.time_series(
            rect, out=self._average_time_series_container
        )
        self.time_series_signal.emit(
            self.dataset.time_points, self._average_time_series_container
        )

    @QtCore.pyqtSlot(dict)
    def powder_calq(self, params):
        """ 
        Calibrate the range of q-vector for a polycrystalline data of known structure. 

        Parameters
        ----------
        params : dict
            Parameters are passed to `skued.powder_calq`. 
        """
        self.dataset.powder_calq(**params)
        self.display_powder_data()
        self.status_message_signal.emit("Scattering vector range calibrated.")

    @QtCore.pyqtSlot(dict)
    def update_metadata(self, metadata):
        """ 
        Update metadata attributes in DiffractionDataset 
        
        Parameters
        ----------
        metadata : dict
            Metadata items to update. Metadata items not in ``DiffractionDataset.valid_metadata`` are ignored.
        """
        if self.dataset is None:
            return

        # dataset is read-only at this point
        # We close and reopen with write intentions
        fname = self.dataset.filename
        cls = type(self.dataset)
        self.dataset.close()

        with cls(fname, mode="r+") as dset:
            for key, val in metadata.items():
                if key in dset.valid_metadata:
                    setattr(dset, key, val)

        self.dataset = cls(fname, mode="r")
        self.dataset_metadata.emit(self.dataset.metadata)

        self.status_message_signal.emit("Metadata updated.")

    @QtCore.pyqtSlot(str)
    def set_dataset_notes(self, notes):
        """ 
        Update notes metadata in DiffractionDataset 
        
        Parameters
        ----------
        notes : str
        """
        metadata = self.dataset.metadata
        metadata["notes"] = notes
        self.update_metadata(metadata)

    @QtCore.pyqtSlot(float)
    def set_time_zero_shift(self, shift):
        """ 
        Set the time-zero shift in picoseconds.
        
        Parameters
        ----------
        shift : float
            Time-delay shift in picoseconds.
        """
        if shift == self.dataset.time_zero_shift:
            return

        # dataset is read-only at this point
        # We close and reopen with write intentions
        # Note : no need to open as a PowderDiffractionDataset for this operations
        fname = self.dataset.filename
        cls = type(self.dataset)
        self.dataset.close()

        with DiffractionDataset(fname, mode="r+") as dset:
            dset.shift_time_zero(shift)

        self.dataset = cls(fname, mode="r")
        self.dataset_metadata.emit(self.dataset.metadata)

        # In case of a time-zero shift, diffraction time-series will
        # be impacted
        self.time_series(self._rect)

        # If _powder_relative is True, the shift in time-zero will impact the display
        if self._relative_powder:
            self.display_powder_data()

        self.status_message_signal.emit("Time-zero shifted by {:.3f}ps.".format(shift))

    @QtCore.pyqtSlot(str, object)
    def load_raw_dataset(self, path, cls):
        """
        Load a raw dataset according to an AbstractRawDataset.

        Parameters
        ----------
        path : str
            Absolute path to the dataset.
        cls : AbstractRawDataset subclass
            Class to use during the loading.
        """
        if cls not in AbstractRawDataset.implementations:
            raise ValueError(
                "Expected a proper subclass of AbstractRawDataset, but received {}".format(
                    cls
                )
            )

        if not path:
            return

        self.close_raw_dataset()
        self.raw_dataset = cls(path)
        self.raw_dataset_loaded_signal.emit(True)
        self.raw_dataset_metadata.emit(
            {
                "time_points": self.raw_dataset.time_points,
                "scans": self.raw_dataset.scans,
            }
        )
        self.display_raw_data(timedelay_index=0, scan=min(self.raw_dataset.scans))
        self.status_message_signal.emit(path + " loaded.")

    @QtCore.pyqtSlot()
    def close_raw_dataset(self):
        """ Close raw dataset. """
        self.raw_dataset = None
        self.raw_dataset_loaded_signal.emit(False)
        self.raw_data_signal.emit(None)

        self.status_message_signal.emit("Raw dataset closed.")

    @QtCore.pyqtSlot(str)
    def load_dataset(self, path):
        """ 
        Load dataset, distinguishing between PowderDiffractionDataset and DiffractionDataset.
        
        Parameters
        ----------
        path : str or path-like
            Path to the dataset.
        """
        if not path:  # e.g. path = ''
            return

        self.close_dataset()

        cls = DiffractionDataset
        with DiffractionDataset(path, mode="r") as d:
            if PowderDiffractionDataset._powder_group_name in d:
                cls = PowderDiffractionDataset
                is_powder = True
            else:
                is_powder = False

        self.dataset = cls(path, mode="r")
        self.dataset_metadata.emit(self.dataset.metadata)

        # Initialize containers
        # This *must* be done before data is displayed
        self._averaged_data_container = np.empty(
            shape=self.dataset.resolution,
            dtype=self.dataset.diffraction_group["intensity"].dtype,
        )
        self._average_time_series_container = np.empty(
            shape=self.dataset.time_points.shape, dtype=np.float
        )

        self.processed_dataset_loaded_signal.emit(True)
        self.powder_dataset_loaded_signal.emit(is_powder)

        self.display_averaged_data(timedelay_index=0)
        if is_powder:
            self.display_powder_data()
        else:
            # Re-assert that a processed dataset was loaded
            # hence, switch views to the relevant widgets
            self.processed_dataset_loaded_signal.emit(True)

        self.status_message_signal.emit(path + " loaded.")

    @QtCore.pyqtSlot()
    def close_dataset(self):
        """ Close current DiffractionDataset. """
        with suppress(AttributeError):  # in case self.dataset is None
            self.dataset.close()
        self.dataset = None
        self.processed_dataset_loaded_signal.emit(False)
        self.powder_dataset_loaded_signal.emit(False)

        self.averaged_data_signal.emit(None)
        self.powder_data_signal.emit(None, None)

        self.status_message_signal.emit("Dataset closed.")

    @QtCore.pyqtSlot(str, dict)
    def symmetrize(self, destination, params):
        """ 
        Launches a background thread that copies the currently-loaded dataset, 
        symmetrize the copy, and load the copy. 
        
        Parameters
        ----------
        destination : str or path-like
            Destination of the symmetrized dataset.
        params : dict
            Symmetrization parameters are passed to ``DiffractionDataset.symmetrize``.
        """
        kwargs = {
            "dataset": self.dataset.filename,
            "destination": destination,
            "callback": self.processing_progress_signal.emit,
        }
        kwargs.update(params)

        # TODO:
        #   Since symmetrization will first copy the dataset,
        #   and the dataset is open in read-only mode, do we have to close it?
        self.close_dataset()

        self.worker = WorkThread(function=symmetrize, kwargs=kwargs)
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.done_signal.connect(
            lambda: self.processing_progress_signal.emit(100)
        )
        self.processing_progress_signal.emit(0)
        self.worker.start()

    @QtCore.pyqtSlot(dict)
    def process_raw_dataset(self, info_dict):
        """
        Launch a background thread that reduces raw data.

        Parameters
        ----------
        info_dict : dict
            Data reduction parameters are passed to ``DiffractionDataset.from_raw``
        """
        info_dict.update(
            {"callback": self.processing_progress_signal.emit, "raw": self.raw_dataset}
        )

        self.worker = WorkThread(function=process, kwargs=info_dict)
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.done_signal.connect(
            lambda: self.processing_progress_signal.emit(100)
        )
        self.processing_progress_signal.emit(0)
        self.worker.start()

    @QtCore.pyqtSlot(dict)
    def calculate_azimuthal_averages(self, params):
        """ 
        Promote a DiffractionDataset to a PowderDiffractionDataset 
        
        Parameters
        ----------
        params : dict
            Parameters are passed to ``PowderDiffractionDataset.from_dataset``.
        """
        params.update(
            {
                "filename": self.dataset.filename,
                "callback": self.powder_promotion_progress.emit,
            }
        )
        self.worker = WorkThread(function=calculate_azimuthal_averages, kwargs=params)

        self.close_dataset()

        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.start()

    @QtCore.pyqtSlot(dict)
    def compute_baseline(self, params):
        """ Compute the powder baseline. The dictionary `params` is passed to 
        PowderDiffractionDataset.compute_baseline(), except its key 'callback'. The callable 'callback' 
        is called (no argument) when computation is done. """
        params.update({"fname": self.dataset.filename})

        self.worker = WorkThread(function=compute_powder_baseline, kwargs=params)
        self.dataset.close()
        self.dataset = None

        self.worker.results_signal.connect(self.load_dataset)
        self.worker.done_signal.connect(lambda: self.powder_background_subtracted(True))
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.start()


class WorkThread(QtCore.QThread):
    """
    Object taking care of threading computations. These computations are very specific:
    a function takes in a filename (pointing to a DiffractionDataset), parameters, 
    and returns a filename again.
    
    Signals
    -------
    done_signal
        Emitted when the function evaluation is over.
    in_progress_signal
        Emitted when the function evaluation starts.
    results_signal
        Emitted when the function evaluation is over. Carries the results
        of the computation.
    """

    results_signal = QtCore.pyqtSignal(str)
    done_signal = QtCore.pyqtSignal()
    in_progress_signal = QtCore.pyqtSignal(bool)

    def __init__(self, function, args=tuple(), kwargs=dict()):

        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        self.wait()

    def run(self):
        self.in_progress_signal.emit(True)

        # This is potentially a very-long-running calculation
        result = self.function(*self.args, **self.kwargs)

        self.results_signal.emit(result)
        self.done_signal.emit()
        self.in_progress_signal.emit(False)


def calculate_azimuthal_averages(**kwargs):
    """ Create a PowderDiffractionDataset from a DiffractionDataset. If azimuthal averages
    were already calculated, recalculate them. """
    filename = kwargs.pop("filename")

    # Determine if azimuthal averages have already been computed
    recomputed = False
    with PowderDiffractionDataset(filename) as d:
        if PowderDiffractionDataset._powder_group_name in d:
            # We simply need to recompute the azimuthal averages
            recomputed = True
            d.compute_angular_averages(**kwargs)

    if recomputed:
        return filename

    # If we are here, the dataset neets promotion
    with PowderDiffractionDataset.from_dataset(DiffractionDataset(filename), **kwargs):
        pass

    return filename


def symmetrize(dataset, destination, **kwargs):
    """ 
    Copies a dataset and symmetrize it. Keyword arguments
    are passed to `DiffractionDataset.symmetrize`.
    
    Parameters
    ----------
    dataset : path-like
    destination : path-like
    """
    copy2(dataset, destination)

    with DiffractionDataset(destination, mode="r+") as dset:
        dset.symmetrize(**kwargs)
        fname = dset.filename
    return fname


def compute_powder_baseline(fname, **kwargs):
    """ 
    Compute a powder baseline. Keyword arguments are passed to 
    `PowderDiffractionDataset.compute_baseline` 
    
    Parameters
    ----------
    dataset : path-like
    """
    with PowderDiffractionDataset(fname, mode="r+") as dset:
        dset.compute_baseline(**kwargs)
    return fname


def process(**kwargs):
    """ Process a RawDataset into a DiffractionDataset """
    with DiffractionDataset.from_raw(**kwargs) as dset:
        fname = dset.filename
    return fname
