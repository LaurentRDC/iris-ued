# -*- coding: utf-8 -*-
"""
Controller object to handle DiffractionDataset objects
"""
import logging
import traceback
import warnings
from contextlib import suppress
from functools import wraps
from shutil import copy2
from types import FunctionType

import numpy as np
from PyQt5 import QtCore
from skued import DiskSelection, bragg_peaks

from .. import AbstractRawDataset, DiffractionDataset, PowderDiffractionDataset
from .qlogger import QLogger


def error_aware(func):
    """
    Wrap an instance method with a try/except and emit a message.
    Instance must have a signal called 'error_message_signal' which
    will be emitted with the message upon error.

    Keyboard interrupts are never ignored.
    """

    @wraps(func)
    def aware_func(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception:
            exc = traceback.format_exc()
            self.error_message_signal.emit(exc)
            warnings.warn(exc, UserWarning)

    return aware_func


class ErrorAware(type(QtCore.QObject)):
    """Metaclass for error-aware Qt applications."""

    def __new__(meta, classname, bases, class_dict):
        new_class_dict = dict()
        for attr_name, attr in class_dict.items():
            if isinstance(attr, FunctionType):
                attr = error_aware(attr)
            new_class_dict[attr_name] = attr

        return super().__new__(meta, classname, bases, new_class_dict)


def indicate_in_progress(method):
    """Decorator for IrisController methods that should
    emit the ``operation_in_progress`` signal and automatically
    revert when the operation is finished."""

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
    initially_run_bragg_signal = QtCore.pyqtSignal(bool)

    powder_dataset_loaded_signal = QtCore.pyqtSignal(bool)

    raw_dataset_metadata = QtCore.pyqtSignal(dict)
    dataset_metadata = QtCore.pyqtSignal(dict)
    powder_dataset_metadata = QtCore.pyqtSignal(dict)

    error_message_signal = QtCore.pyqtSignal(str)
    processing_data_signal = QtCore.pyqtSignal(bool)
    operation_in_progress = QtCore.pyqtSignal(bool)
    status_message_signal = QtCore.pyqtSignal(str)

    raw_data_signal = QtCore.pyqtSignal(object)
    averaged_data_signal = QtCore.pyqtSignal(object, bool)
    powder_data_signal = QtCore.pyqtSignal(object, object)
    bragg_peaks_signal = QtCore.pyqtSignal(dict)

    time_series_signal = QtCore.pyqtSignal(object, object)
    powder_time_series_signal = QtCore.pyqtSignal(object, object)

    relative_powder_enable_signal = QtCore.pyqtSignal(bool)
    relative_averaged_enable_signal = QtCore.pyqtSignal(bool)
    powder_bgr_enable_signal = QtCore.pyqtSignal(bool)
    bragg_peak_enable_signal = QtCore.pyqtSignal(bool)
    bz_enable_signal = QtCore.pyqtSignal(bool)

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
        self._enable_bragg = False
        self._enable_bz = False
        self._timedelay_index = 0
        self.initially_run_bragg_signal.emit(False)
        # array containers
        # Preallocation is important for arrays that change often. Then,
        # we can take advantage of the out parameter
        # These attributes are not initialized to None since None is a valid
        # out parameter
        # TODO: is it worth it?
        self._averaged_data_container = None
        self._average_time_series_container = None

        self.logger = QLogger(parent=self)
        self.logger.debug("Controller started.")

        self._file_dialog = None
        self._parent = None

        # A lot of info-level messages should be mirrored between the status
        # bar and the info log
        # Similarly, the error messages from ErrorAware type should be
        # mirrored in the error log
        self.status_message_signal.connect(self.logger.info)
        self.error_message_signal.connect(self.logger.error)

    @QtCore.pyqtProperty(object)
    def file_dialog(self):
        """I'm the '_file_dialog' property."""
        return self._file_dialog

    @file_dialog.setter
    def file_dialog(self, value):
        self._file_dialog = value

    @file_dialog.deleter
    def file_dialog(self):
        del self._file_dialog

    @QtCore.pyqtProperty(object)
    def parent(self):
        """I'm the '_parent' property."""
        return self._parent

    # @QtCore.pyqtSlot()
    @parent.setter
    def parent(self, value):
        self._parent = value

    # @QtCore.pyqtSlot()
    @parent.deleter
    def parent(self):
        del self._parent

    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot(str, object)
    def log(self, message, level=logging.INFO):
        """
        Interface to the iris logger.

        ALL logging should go through the controller.

        Parameters
        ----------
        message : str
            Logging message
        level : logging.Level, optional
            Level of the logging message.
        """
        self.logger.log(message, level)

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
        self.status_message_signal.emit(f"Displaying data at {timedelay:.3f}ps, scan {scan:d}.")

    @QtCore.pyqtSlot(int)
    @QtCore.pyqtSlot(int, bool)
    def display_averaged_data(self, timedelay_index, autocontrast=False):
        """
        Extract processed diffraction pattern.

        Parameters
        ----------
        timedelay_index : int
            Time-delay index.
        autocontrast : bool, optional
            Whether or not the contrast of the image should be adjusted.

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
        if self._relative_averaged:
            # The range of relative values can be outrageous if very large floats
            # are involved. This causes problems with pyqtgraph displays.
            # I doubt relative changes can exceed 50000%
            np.clip(
                self._averaged_data_container,
                a_min=-60000,
                a_max=60000,
                out=self._averaged_data_container,
            )

        self.averaged_data_signal.emit(self._averaged_data_container, autocontrast)
        self.status_message_signal.emit(f"Displaying data at {timedelay:.3f}ps.")

    @QtCore.pyqtSlot()
    def display_powder_data(self):
        """Emit a powder data signal with/out background"""
        # Preallocation isn't so important for powder data because the whole block
        # is loaded
        self.powder_data_signal.emit(
            self.dataset.scattering_vector,
            self.dataset.powder_data(timedelay=None, relative=self._relative_powder, bgr=self._bgr_powder),
        )

    @QtCore.pyqtSlot()
    def display_bragg_peaks(self):
        params = {
            "enable_peaks": self._enable_bragg,
            "peaks": self.bragg_peaks,
            "enable_bz": self._enable_bz,
            "bz": self.bzs,
            "n_vertices": self.bz_vertices,
        }
        self.bragg_peaks_signal.emit(params)

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
    def enable_bragg(self, enable):
        self._enable_bragg = enable
        self.bragg_peak_enable_signal.emit(enable)
        self.display_bragg_peaks()

    @QtCore.pyqtSlot(bool)
    def enable_bz(self, enable):
        self._enable_bz = enable
        self.bz_enable_signal.emit(enable)
        self.display_bragg_peaks()

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
        time_series = self.dataset.powder_time_series(rmin=qmin, rmax=qmax, bgr=self._bgr_powder, units="momentum")
        self.powder_time_series_signal.emit(self.dataset.time_points, time_series)

    @QtCore.pyqtSlot(tuple)
    def time_series(self, rect):
        """ "
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
        integrated = self.dataset.time_series(rect, out=self._average_time_series_container)
        self.time_series_signal.emit(self.dataset.time_points, self._average_time_series_container)

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
    @indicate_in_progress
    def optimize_bragg_peaks(self, params):
        """
        Update Bragg peaks to actual local maxima, as well as identify BZ
        using Voronoi diagramming

        Parameters
        ----------
        peaks : list
            Items returned from Bragg peak dialog
        """
        self.initially_run_bragg_signal.emit(True)
        peaks = params["peaks"]
        symmetry = params["sym"]

        from scipy.spatial import Voronoi

        current_view = self.dataset.diff_eq()
        peaks = np.vstack((self.dataset.center[::-1], peaks))
        new_peaks = np.array(peaks)
        for idx, peak in enumerate(peaks):
            # if idx == 0:
            #     c, r = peaks[idx]
            # else:
            r, c = peaks[idx]
            peak = np.asarray(peak).astype(int)
            disk = DiskSelection(shape=current_view.shape, center=peak[::-1], radius=25)
            r1, r2, c1, c2 = disk.bounding_box
            region = current_view[r1:r2, c1:c2]
            try:
                true_peak_idx_local = np.where(region == region.max())
                true_peak_idx_global = np.where(current_view == region[true_peak_idx_local])
                new_r, new_c = true_peak_idx_global[0][0], true_peak_idx_global[1][0]
            except:
                if idx != 0:
                    print(f"Could not optimize peak {idx}")
                # self.log(f"Could not optimize peak {idx}", level=logging.DEBUG)
                new_r, new_c = r, c

            new_peaks[idx] = np.array((new_r, new_c)).astype(int)
        self.bragg_peaks = new_peaks

        # load BZs
        self.__vor = Voronoi(new_peaks[:, [1, 0]])
        self.__voronoi_regions = []

        for i, point_region in enumerate(self.__vor.point_region):
            region = self.__vor.regions[point_region]
            vr = VoronoiRegion(point_region)
            for r in region:
                vr.add_vertex(r, self.__vor.vertices)
            vr.point_inside = (i, self.__vor.points[i])
            vr.set_center(self.__vor.points[i])
            self.__voronoi_regions.append(vr)

        for r in self.__voronoi_regions:
            if not r.is_inf:
                verts = np.array(r.vertices).reshape(-1, 2)
                COND1 = np.all(np.sqrt(np.sum(verts**2, axis=1)) < int(0.95 * current_view.shape[0]))
                COND2 = verts.shape[0] == symmetry
                if COND1 and COND2:
                    r.add_visible_vertex(verts)
            r.visible_vertices = np.array(r.visible_vertices).reshape(-1, 2)
            if r.visible_vertices.size != 0:
                r.is_visible = True
        self.bzs = self.__voronoi_regions
        self.bz_vertices = symmetry
        # self.display_bragg_peaks(new_peaks)

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

        self.status_message_signal.emit(f"Time-zero shifted by {shift:.3f}ps.")

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
            raise ValueError(f"Expected a proper subclass of AbstractRawDataset, but received {cls}")

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
        """Close raw dataset."""
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

        # First, open the dataset as if it was the base class
        # and perform migration if required
        with DiffractionDataset(path, mode="r"):
            self.logger.debug(f"Checking if {path} requires migration...")
        self.logger.debug(f"Migration check complete.")

        cls = DiffractionDataset
        with DiffractionDataset(path, mode="r") as d:
            if PowderDiffractionDataset._powder_group_name in d:
                cls = PowderDiffractionDataset
                is_powder = True
            else:
                is_powder = False

        # For powder datasets, there might be the creation of placeholder datasets
        # therefore, we open the file and do nothing
        # TODO: stop writing datasets in PowderDiffractionDataset.__init__
        #       since the GUI will only open datasets in read-mode by default
        if cls is PowderDiffractionDataset:
            with cls(path, mode="r+"):
                pass

        self.dataset = cls(path, mode="r+")
        self.dataset_metadata.emit(self.dataset.metadata)

        # Initialize containers
        # This *must* be done before data is displayed
        self._averaged_data_container = np.empty(
            shape=self.dataset.resolution,
            dtype=self.dataset.diffraction_group["intensity"].dtype,
        )
        self._average_time_series_container = np.empty(shape=self.dataset.time_points.shape, dtype=float)

        self.processed_dataset_loaded_signal.emit(True)
        self.powder_dataset_loaded_signal.emit(is_powder)

        self.display_averaged_data(timedelay_index=0, autocontrast=True)
        if is_powder:
            self.display_powder_data()
        else:
            # Re-assert that a processed dataset was loaded
            # hence, switch views to the relevant widgets
            self.processed_dataset_loaded_signal.emit(True)

        self.status_message_signal.emit(path + " loaded.")

    @QtCore.pyqtSlot()
    def close_dataset(self):
        """Close current DiffractionDataset."""
        with suppress(AttributeError):  # in case self.dataset is None
            self.dataset.close()
        self.dataset = None
        self.processed_dataset_loaded_signal.emit(False)
        self.powder_dataset_loaded_signal.emit(False)

        self.averaged_data_signal.emit(None, True)
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
        self.worker.in_progress_signal.connect(self.processing_data_signal)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.done_signal.connect(lambda: self.processing_progress_signal.emit(100))
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
        info_dict.update({"callback": self.processing_progress_signal.emit, "raw": self.raw_dataset})

        self.worker = WorkThread(function=process, kwargs=info_dict)
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.processing_data_signal)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.done_signal.connect(lambda: self.processing_progress_signal.emit(100))
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
        self.worker.in_progress_signal.connect(self.processing_data_signal)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.start()

    @QtCore.pyqtSlot(dict)
    def compute_baseline(self, params):
        """Compute the powder baseline. The dictionary `params` is passed to
        PowderDiffractionDataset.compute_baseline(), except its key 'callback'. The callable 'callback'
        is called (no argument) when computation is done."""
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
    """Create a PowderDiffractionDataset from a DiffractionDataset. If azimuthal averages
    were already calculated, recalculate them."""
    filename = kwargs.pop("filename")

    # Determine if azimuthal averages have already been computed
    recomputed = False
    with PowderDiffractionDataset(filename, "r+") as d:
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
    """Process a RawDataset into a DiffractionDataset"""
    with DiffractionDataset.from_raw(**kwargs) as dset:
        fname = dset.filename
    return fname


class VoronoiRegion:
    """
    This class creates a region of points defined by the vertices of the Voronoi regions of a given pattern.
    It is made to make calculations later easier to manage

    Parameters
    ----------

    region_id: type(int)
        Defines the ID number of this particular Voronoi region


    """

    def __init__(self, region_id):
        self.id = region_id
        self.vertices = []
        self.is_inf = False
        self.point_inside = None
        self.visible_vertices = []
        self.is_visible = False

    def __str__(self):
        text = f"region id={self.id}"
        if self.point_inside:
            point_idx, point = self.point_inside
            text = f"{text}[point:{point}(point_id:{point_idx})]"
        text += ", vertices: "
        if self.is_inf:
            text += "(inf)"
        for v in self.vertices:
            text += f"{v}"
        return text

    def __repr__(self):
        return str(self)

    def add_vertex(self, vertex, vertices):
        if vertex == -1:
            self.is_inf = True
        else:
            point = vertices[vertex]
            self.vertices.append(point)

    def add_visible_vertex(self, vertex):
        self.visible_vertices.append(vertex)

    def set_center(self, center):
        self.center = center
