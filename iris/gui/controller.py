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

def promote_to_powder(**kwargs):
    """ Create a PowderDiffractionDataset from a DiffractionDataset """
    filename = kwargs.pop('filename')
    with PowderDiffractionDataset.from_dataset(DiffractionDataset(filename), **kwargs):
        pass
    return filename

def recompute_angular_average(**kwargs):
    """ Re-compute the angular average of a PowderDiffractionDataset """
    filename = kwargs.pop('filename')
    with PowderDiffractionDataset(filename, mode = 'r+') as dataset:
        dataset.compute_angular_averages(**kwargs)
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

    with DiffractionDataset(destination, 'r+') as dset:
        dset.symmetrize(**kwargs)
        fname = dset.filename
    return fname

def process(**kwargs):
    """ Process a RawDataset into a DiffractionDataset """
    with DiffractionDataset.from_raw(**kwargs) as dset:
        fname = dset.filename
    return fname

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

class IrisController(QtCore.QObject, metaclass = ErrorAware):
    """
    Controller behind Iris.
    """
    raw_dataset_loaded_signal       = QtCore.pyqtSignal(bool)
    processed_dataset_loaded_signal = QtCore.pyqtSignal(bool)
    powder_dataset_loaded_signal    = QtCore.pyqtSignal(bool)

    raw_dataset_metadata            = QtCore.pyqtSignal(dict)
    dataset_metadata                = QtCore.pyqtSignal(dict)
    powder_dataset_metadata         = QtCore.pyqtSignal(dict)

    error_message_signal            = QtCore.pyqtSignal(str)
    operation_in_progress           = QtCore.pyqtSignal(bool)
    status_message_signal           = QtCore.pyqtSignal(str)

    raw_data_signal                 = QtCore.pyqtSignal(object)
    averaged_data_signal            = QtCore.pyqtSignal(object)
    powder_data_signal              = QtCore.pyqtSignal(object, object)

    time_series_signal              = QtCore.pyqtSignal(object, object)
    powder_time_series_signal       = QtCore.pyqtSignal(object, object)

    processing_progress_signal      = QtCore.pyqtSignal(int)
    powder_promotion_progress       = QtCore.pyqtSignal(int)
    angular_average_progress        = QtCore.pyqtSignal(int)

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
        self.raw_data_signal.emit(self.raw_dataset.raw_data(timedelay, scan, bgr = True))
        self.status_message_signal.emit('Displaying data at {:.3f}ps, scan {:d}.'.format(timedelay, scan))
    
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
        # Source of 'cache miss' could be that _average_data_container is None,
        # new dataset loaded has different shape than before, etc.
        self._timedelay_index = timedelay_index
        timedelay = self.dataset.time_points[timedelay_index]
        try:
            self._averaged_data_container[:] = self.dataset.diff_data(timedelay, relative = self._relative_averaged)
        except:
            self._averaged_data_container = self.dataset.diff_data(timedelay, relative = self._relative_averaged)
        self.averaged_data_signal.emit(self._averaged_data_container)
        self.status_message_signal.emit('Displaying data at {:.3f}ps.'.format(timedelay))
        return self._averaged_data_container
    
    @QtCore.pyqtSlot()
    def display_powder_data(self):
        """ Emit a powder data signal with/out background """
        # Preallocation isn't so important for powder data because the whole block
        # is loaded
        self.powder_data_signal.emit(self.dataset.scattering_vector, 
                                     self.dataset.powder_data(timedelay = None, relative = self._relative_powder, bgr = self._bgr_powder))
    
    @QtCore.pyqtSlot(bool)
    def powder_background_subtracted(self, enable):
        """
        Toggle background subtraction for polycrystalline data.

        Parameters
        ----------
        enable : bool
        """
        self._bgr_powder = enable
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
        self.display_averaged_data(self._timedelay_index)
    
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
        kwargs = {'dataset'    : self.dataset.filename,
                  'destination': destination,
                  'callback': self.processing_progress_signal.emit}
        kwargs.update(params)
        self.dataset.close()

        self.worker = WorkThread(function = symmetrize, kwargs = kwargs)
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
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
        info_dict.update({'callback': self.processing_progress_signal.emit, 
                          'raw': self.raw_dataset})

        self.worker = WorkThread(function = process, kwargs = info_dict)
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.done_signal.connect(lambda boolean: self.processing_progress_signal.emit(100))
        self.processing_progress_signal.emit(0)
        self.worker.start()
    
    @QtCore.pyqtSlot(dict)
    def promote_to_powder(self, params):
        """ 
        Promote a DiffractionDataset to a PowderDiffractionDataset 
        
        Parameters
        ----------
        params : dict
            Parameters are passed to ``PowderDiffractionDataset.from_dataset``.
        """
        params.update({'filename':self.dataset.filename, 'callback':self.powder_promotion_progress.emit})
        self.worker = WorkThread(function = promote_to_powder, kwargs = params)
        self.dataset.close()
        self.dataset = None
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.start()
    
    @QtCore.pyqtSlot(dict)
    def recompute_angular_average(self, params):
        """ 
        Compute the angular average of a PowderDiffractionDataset again.
        
        Parameters
        ----------
        params : dict
            Parameters are passed to ``PowderDiffractionDataset.compute_angular_averages``. 
        """
        params.update({'filename':self.dataset.filename, 'callback':self.powder_promotion_progress.emit})
        self.worker = WorkThread(function = recompute_angular_average, kwargs = params)
        self.dataset.close()
        self.dataset = None
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.start()

    @QtCore.pyqtSlot(float, float)
    def powder_time_series(self, qmin, qmax):
        """
        Extract polycrystalline time-series

        Parameters
        ----------
        qmin, qmax : float
            Lower- and upper-bounds of integration.
        """
        time_series = self.dataset.powder_time_series(rmin = qmin, rmax = qmax, bgr = self._bgr_powder, units = 'momentum')
        self.powder_time_series_signal.emit(self.dataset.time_points, time_series)
    
    @QtCore.pyqtSlot(object)
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

        #If coordinate is negative, return 0
        x1 = round(max(0, rect.topLeft().x() ))
        x2 = round(max(0, rect.x() + rect.width() ))
        y1 = round(max(0, rect.topLeft().y() ))
        y2 = round(max(0, rect.y() + rect.height() ))

        integrated = self.dataset.time_series( (x1, x2, y1, y2) )
        self.time_series_signal.emit(self.dataset.time_points, integrated)
    
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
        self.status_message_signal('Scattering vector range calibrated.')

    @QtCore.pyqtSlot(dict)
    def compute_baseline(self, params):
        """ Compute the powder baseline. The dictionary `params` is passed to 
        PowderDiffractionDataset.compute_baseline(), except its key 'callback'. The callable 'callback' 
        is called (no argument) when computation is done. """
        callback = params.pop('callback')
        self.worker = WorkThread(function = self.dataset.compute_baseline, kwargs = params)
        self.worker.done_signal.connect(lambda b: callback())
        self.worker.done_signal.connect(lambda b: self.dataset_metadata.emit(self.dataset.metadata))
        self.worker.done_signal.connect(lambda b: self.display_powder_data())
        self.worker.in_progress_signal.connect(self.operation_in_progress)
        self.worker.start()
    
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

        for key, val in metadata.items():
            setattr(self.dataset, key, val)
        
        self.load_dataset(self.dataset.filename)
        self.status_message_signal.emit('Metadata updated.')

    @QtCore.pyqtSlot(str)
    def set_dataset_notes(self, notes):
        """ 
        Update notes metadata in DiffractionDataset 
        
        Parameters
        ----------
        notes : str
        """
        self.dataset.notes = notes
    
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

        self.dataset.shift_time_zero(shift)
        self.dataset_metadata.emit(self.dataset.metadata)

        # In case of a time-zero shift, diffraction time-series will
        # be impacted
        self.time_series(self._rect)

        # If _powder_relative is True, the shift in time-zero will impact the display
        if self._relative_powder:
            self.display_powder_data()
        
        self.status_message_signal.emit('Time-zero shifted by {:.3f}ps.'.format(shift))
    
    @QtCore.pyqtSlot(str, object)
    @indicate_in_progress
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
            raise ValueError('Expected a proper subclass of AbstractRawDataset, but received {}'.format(cls))
        
        if not path:
            return
        
        self.close_raw_dataset()
        self.raw_dataset = cls(path)
        self.raw_dataset_loaded_signal.emit(True)
        self.raw_dataset_metadata.emit({'time_points': self.raw_dataset.time_points,
                                        'scans': self.raw_dataset.scans})
        self.display_raw_data(timedelay_index = 0, 
                              scan = min(self.raw_dataset.scans))
        self.status_message_signal.emit(path + ' loaded.')

    @QtCore.pyqtSlot()
    @indicate_in_progress
    def close_raw_dataset(self):
        """ Close raw dataset. """
        self.raw_dataset = None
        self.raw_dataset_loaded_signal.emit(False)
        self.raw_data_signal.emit(None)

        self.status_message_signal.emit('Raw dataset closed.')
        
    @QtCore.pyqtSlot(object) # Due to worker.results_signal emitting an object
    @QtCore.pyqtSlot(str)
    @indicate_in_progress
    def load_dataset(self, path):
        """ 
        Load dataset, distinguishing between PowderDiffractionDataset and DiffractionDataset.
        
        Parameters
        ----------
        path : str or path-like
            Path to the dataset.
        """
        if not path: #e.g. path = ''
            return 
        
        self.close_dataset()

        cls = DiffractionDataset
        with DiffractionDataset(path, mode = 'r') as d:
            if PowderDiffractionDataset._powder_group_name in d:
                cls = PowderDiffractionDataset
        
        self.dataset = cls(path, mode = 'r+')
        self.dataset_metadata.emit(self.dataset.metadata)
        self.processed_dataset_loaded_signal.emit(True)
        self.display_averaged_data(timedelay_index = 0)

        if isinstance(self.dataset, PowderDiffractionDataset):
            self.display_powder_data()
            self.powder_dataset_loaded_signal.emit(True)
        
        self.status_message_signal.emit(path + ' loaded.')
    
    @QtCore.pyqtSlot()
    @indicate_in_progress
    def close_dataset(self):
        """ Close current DiffractionDataset. """
        with suppress(AttributeError):  # in case self.dataset is None
            self.dataset.close()
        self.dataset = None
        self.processed_dataset_loaded_signal.emit(False)
        self.powder_dataset_loaded_signal.emit(False)

        self.averaged_data_signal.emit(None)
        self.powder_data_signal.emit(None, None)

        self.status_message_signal.emit('Dataset closed.')

class WorkThread(QtCore.QThread):
    """
    Object taking care of threading computations.
    
    Signals
    -------
    done_signal
        Emitted when the function evaluation is over.
    in_progress_signal
        Emitted when the function evaluation starts.
    results_signal
        Emitted when the function evaluation is over. Carries the results
        of the computation.
        
    Attributes
    ----------
    function : callable
        Function to be called
    args : tuple
        Positional arguments of function
    kwargs : dict
        Keyword arguments of function
    results : object
        Results of the computation
    
    Examples
    --------
    >>> function = lambda x : x ** 10
    >>> result_function = lambda x: print(x)
    >>> worker = WorkThread(function, 2)  # 2 ** 10
    >>> worker.results_signal.connect(result_function)
    >>> worker.start()      # Computation starts only when this method is called
    """
    results_signal = QtCore.pyqtSignal(str)
    done_signal = QtCore.pyqtSignal(bool)
    in_progress_signal = QtCore.pyqtSignal(bool)

    def __init__(self, function, args = tuple(), kwargs = dict()):
        
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
    
    def __del__(self):
        self.wait()
    
    def run(self):
        self.in_progress_signal.emit(True)
        result = self.function(*self.args, **self.kwargs)   
        self.done_signal.emit(True)  
        self.results_signal.emit(result)
