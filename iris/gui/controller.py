"""
Controller behind Iris
"""
import functools
import traceback

from pyqtgraph import QtCore

from ..dataset import DiffractionDataset, PowderDiffractionDataset
from ..processing import process
from ..raw import RawDataset

def error_aware(message):
    """
    Wrap an instance method with a try/except and emit a message.
    Instance must have a signal called 'error_message_signal' which
    will be emitted with the message upon error. 
    """
    def wrap(func):
        @functools.wraps(func)
        def aware_func(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except:
                exc = traceback.format_exc()
                self.error_message_signal.emit(message + '\n\n\n' + exc)
        return aware_func
    return wrap

class IrisController(QtCore.QObject):
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

    raw_data_signal = QtCore.pyqtSignal(object)
    averaged_data_signal = QtCore.pyqtSignal(object)
    powder_data_signal = QtCore.pyqtSignal(object, object, object)

    time_series_signal = QtCore.pyqtSignal(object, object)
    powder_time_series_signal = QtCore.pyqtSignal(object, object)

    processing_progress_signal = QtCore.pyqtSignal(int)
    powder_promotion_progress = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = None
        self.raw_dataset = None
        self.dataset = None

        # Internal state for powder background removal. If True, display_powder_data
        # will send background-subtracted data
        self._bgr_powder = False

        # array containers
        # Preallocation is important for arrays that change often. Then,
        # we can take advantage of the out parameter
        # These attributes are not initialized to None since None is a valid
        # out parameter
        # TODO: is it worth it?
        self._averaged_data_container = False
        self._powder_time_series_container = False

    @error_aware('Raw data could not be displayed.')
    @QtCore.pyqtSlot(int, int)
    def display_raw_data(self, timedelay_index, scan):
        timedelay = self.raw_dataset.time_points[timedelay_index]
        self.raw_data_signal.emit(self.raw_dataset.raw_data(timedelay, scan) - self.raw_dataset.pumpon_background)
    
    @error_aware('Processed data could not be displayed.')
    @QtCore.pyqtSlot(int)
    def display_averaged_data(self, timedelay_index):
        # Preallocation of full images is important because the whole block cannot be
        # loaded into memory, contrary to powder data
        # Source of 'cache miss' could be that _average_data_container is None,
        # new dataset loaded has different shape than before, etc.
        timedelay = self.dataset.corrected_time_points[timedelay_index]
        try:
            self.dataset.averaged_data(timedelay, out = self._averaged_data_container)
        except:
            self._averaged_data_container = self.dataset.averaged_data(timedelay)
        self.averaged_data_signal.emit(self._averaged_data_container)
        return self._averaged_data_container
    
    @error_aware('Powder data could not be displayed.')
    @QtCore.pyqtSlot()
    def display_powder_data(self):
        """ Emit a powder data signal with/out background """
        # Preallocation isn't so important for powder data because the whole block
        # is loaded
        self.powder_data_signal.emit(self.dataset.scattering_length, 
                                     self.dataset.powder_data(timedelay = None, bgr = self._bgr_powder), 
                                     self.dataset.powder_error(timedelay = None))
    
    @QtCore.pyqtSlot(bool)
    def powder_background_subtracted(self, enable):
        self._bgr_powder = enable
        self.display_powder_data()
    
    @error_aware('Raw dataset could not be processed.')
    @QtCore.pyqtSlot(dict)
    def process_raw_dataset(self, info_dict):
        info_dict.update({'callback': self.processing_progress_signal.emit, 'raw': self.raw_dataset})

        self.worker = WorkThread(function = process, kwargs = info_dict)
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.done_signal.connect(lambda boolean: self.processing_progress_signal.emit(100))
        self.processing_progress_signal.emit(0)
        self.worker.start()
    
    @error_aware('')
    @QtCore.pyqtSlot(tuple)
    def promote_to_powder(self, center):
        """ Promote a DiffractionDataset to a PowderDiffractionDataset """
        self.worker = WorkThread(function = promote_to_powder, kwargs = {'center': center, 'filename':self.dataset.filename, 
                                                                         'callback':self.powder_promotion_progress.emit})
        self.dataset.close()
        self.dataset = None
        self.worker.results_signal.connect(self.load_dataset)
        self.worker.start()

    @error_aware('Powder time-series could not be calculated.')
    @QtCore.pyqtSlot(float, float)
    def powder_time_series(self, smin, smax):
        try:
            self.dataset.powder_time_series(smin = smin, smax = smax, bgr = self._bgr_powder, 
                                            out = self._powder_time_series_container)
        except:
            self._powder_time_series_container = self.dataset.powder_time_series(smin = smin, smax = smax, bgr = self._bgr_powder)
        finally:
            self.powder_time_series_signal.emit(self.dataset.time_points, self._powder_time_series_container)
    
    @error_aware('Single-crystal time-series could not be computed.')
    @QtCore.pyqtSlot(object)
    def time_series(self, rect):
        """" 
        Single-crystal time-series as the integrated diffracted intensity inside a rectangular ROI
        """
        #If coordinate is negative, return 0
        x1 = round(max(0, rect.topLeft().x() ))
        x2 = round(max(0, rect.x() + rect.width() ))
        y1 = round(max(0, rect.topLeft().y() ))
        y2 = round(max(0, rect.y() + rect.height() ))

        integrated = self.dataset.time_series( (y1, y2, x1, x2) )
        self.time_series_signal.emit(self.dataset.time_points, integrated)
    
    @error_aware('Powder baseline could not be computed.')
    @QtCore.pyqtSlot(dict)
    def compute_baseline(self, params):
        """ Compute the powder baseline. The dictionary `params` is passed to 
        PowderDiffractionDataset.compute_baseline(), except its key 'callback'. The callable 'callback' 
        is called (no argument) when computation is done. """
        callback = params.pop('callback')
        self.worker = WorkThread(function = self.dataset.compute_baseline, kwargs = params)
        self.worker.done_signal.connect(lambda b: callback())
        self.worker.start()
    
    @error_aware('Dataset notes could not be updated')
    @QtCore.pyqtSlot(str)
    def set_dataset_notes(self, notes):
        self.dataset.notes = notes
    
    @error_aware('Raw dataset could not be loaded.')
    @QtCore.pyqtSlot(str)
    def load_raw_dataset(self, path):
        if not path:
            return

        self.raw_dataset = RawDataset(path)
        self.raw_dataset_loaded_signal.emit(True)
        self.raw_dataset_metadata.emit({'time_points': self.raw_dataset.time_points,
                                        'nscans': self.raw_dataset.nscans})
        self.display_raw_data(timedelay_index = 0, 
                              scan = min(self.raw_dataset.nscans))
        
    @error_aware('Processed dataset could not be loaded. The path might not be valid')
    @QtCore.pyqtSlot(object) # Due to worker.results_signal emitting an object
    @QtCore.pyqtSlot(str)
    def load_dataset(self, path):
        if not path: #e.g. path = ''
            return 

        # Dispatch between DiffractionDataset and PowderDiffractionDataset
        cls = DiffractionDataset        # Most general case
        with DiffractionDataset(path, mode = 'r') as d:
            if d.sample_type == 'powder':
                cls = PowderDiffractionDataset
        
        self.dataset = cls(path, mode = 'r+')
        self.dataset_metadata.emit(self.dataset.metadata)
        self.processed_dataset_loaded_signal.emit(True)
        self.display_averaged_data(timedelay_index = 0)

        if isinstance(self.dataset, PowderDiffractionDataset):
            self.powder_data_signal.emit(self.dataset.scattering_length, 
                                         self.dataset.powder_data(timedelay = None, 
                                                                  bgr = self.dataset.baseline_removed),
                                         self.dataset.powder_error(timedelay = None))
            self.powder_dataset_loaded_signal.emit(True)

def promote_to_powder(filename, center, callback):
    """ Create a PowderDiffractionDataset from a DiffractionDataset """
    with PowderDiffractionDataset(filename, mode = 'r+') as dataset:
        dataset.sample_type = 'powder'
        dataset.compute_angular_averages(center = center, callback = callback)
    return filename

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