"""
Controller behind Iris
"""

import functools
from .pyqtgraph import QtCore
import traceback

from ..dataset import DiffractionDataset, PowderDiffractionDataset
from ..raw import RawDataset
from .utils import WorkThread

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
            except: # TODO: get traceback message and add to message?
                exc = traceback.format_exc()
                self.error_message_signal.emit(message + '\n \n \n' + exc)
        return aware_func
    return wrap

class IrisController(QtCore.QObject):
    """
    Controller behind Iris.

    Slots
    -----
    display_raw_data

    display_averaged_data

    process_raw_dataset

    load_raw_dataset

    load_dataset

    Signals
    -------
    raw_dataset_loaded_signal

    processed_dataset_loaded_signal

    powder_dataset_loaded_signal

    status_message_loaded_signal

    dataset_info_signal

    error_message_signal

    raw_data_signal

    averaged_data_signal

    powder_data_signal
    """

    raw_dataset_loaded_signal = QtCore.pyqtSignal(bool, name = 'raw_dataset_loaded_signal')
    processed_dataset_loaded_signal = QtCore.pyqtSignal(bool, name = 'processed_dataset_loaded_signal')
    powder_dataset_loaded_signal = QtCore.pyqtSignal(bool, name = 'powder_dataset_loaded_signal')

    status_message_signal = QtCore.pyqtSignal(str, name = 'status_message_signal')
    dataset_info_signal = QtCore.pyqtSignal(dict, name = 'dataset_info_signal')
    error_message_signal = QtCore.pyqtSignal(str, name = 'error_message_signal')

    raw_data_signal = QtCore.pyqtSignal(object, name = 'raw_data_signal')
    averaged_data_signal = QtCore.pyqtSignal(object, name = 'averaged_data_signal')
    powder_data_signal = QtCore.pyqtSignal(object, object, object, name = 'powder_data_signal')

    time_series_signal = QtCore.pyqtSignal(object, object)
    powder_time_series_signal = QtCore.pyqtSignal(object)

    processing_progress_signal = QtCore.pyqtSignal(int, name = 'processing_progress_signal')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = None
        self.raw_dataset = None
        self.dataset = None
    
    @error_aware('Raw data could not be displayed.')
    @QtCore.pyqtSlot(float, int)
    def display_raw_data(self, timedelay, scan):
        self.raw_data_signal.emit(self.raw_dataset.raw_data(timedelay, scan) - self.raw_dataset.pumpon_background)
    
    @error_aware('Processed data could not be displayed.')
    @QtCore.pyqtSlot(float)
    def display_averaged_data(self, timedelay):
        self.averaged_data_signal.emit(self.dataset.averaged_data(timedelay))
    
    @error_aware('Raw dataset could not be processed.')
    @QtCore.pyqtSlot(dict)
    def process_raw_dataset(self, info_dict):
        info_dict.update({'callback': self.processing_progress_signal.emit})

        self.worker = WorkThread(function = self.raw_dataset.process, kwargs = info_dict)
        self.worker.results_signal.connect(self.load_dataset)    # self.dataset.process returns a string path
        self.worker.done_signal.connect(lambda boolean: self.processing_progress_signal.emit(100))

        def in_progress(boolean):
            if boolean: self.status_message_signal.emit('Dataset processing in progress.')
            else: self.status_message_signal.emit('Dataset processing done.')
        
        self.worker.in_progress_signal.connect(in_progress)
        self.processing_progress_signal.emit(0)
        self.worker.start()
    
    @error_aware('Single-crystal time-series could not be computed')
    @QtCore.pyqtSlot(object)
    def time_series_from_ROI(self, ROI):
        """" 
        Single-crystal time-series as the integrated diffracted intensity inside a rectangular ROI

        Parameters
        ----------
        ROI: PyQtGraph.ROI object
        """

        rect = ROI.parentBounds().toRect()

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
        self.worker = WorkThread(function = self.dataset.compute_baseline, kwargs = params)
        self.worker.done_signal.connect(lambda boolean: self.update_dataset_info())
        self.worker.done_signal.connect(lambda boolean: self.powder_data_signal.emit(self.dataset.scattering_length, *self.dataset.powder_data_block(bgr = self.dataset.baseline_removed)))
        self.worker.done_signal.connect(lambda boolean: self.status_message_signal.emit('Baseline computed.'))
        self.worker.start()
    
    @error_aware('Raw dataset could not be loaded.')
    @QtCore.pyqtSlot(str)
    def load_raw_dataset(self, path):
        self.raw_dataset = RawDataset(path)
        self.raw_dataset_loaded_signal.emit(True)
        self.display_raw_data(timedelay = min(self.raw_dataset.time_points), 
                              scan = min(self.raw_dataset.nscans))
        
    @error_aware('Processed dataset could not be loaded.')
    @QtCore.pyqtSlot(object) # Due to worker.results_signal emitting an object
    @QtCore.pyqtSlot(str)
    def load_dataset(self, path):
        # Dispatch between DiffractionDataset and PowderDiffractionDataset
        cls = DiffractionDataset        # Most general case
        with DiffractionDataset(path, mode = 'r') as d:
            if d.sample_type == 'powder':
                cls = PowderDiffractionDataset
        
        self.dataset = cls(path, mode = 'r+')
        self.update_dataset_info()
        self.processed_dataset_loaded_signal.emit(True)
        self.display_averaged_data(timedelay = min(map(abs, self.dataset.time_points)))

        if isinstance(self.dataset, PowderDiffractionDataset):
            self.powder_data_signal.emit(self.dataset.scattering_length, *self.dataset.powder_data_block(bgr = self.dataset.baseline_removed))
            self.powder_dataset_loaded_signal.emit(True)
    
    def update_dataset_info(self):
        """
        Update the dataset info and emits the dataset_info_signal
        to update all widgets.
        """
        # Emit dataset information such as fluence, time-points, ...
        info = dict()
        for attr in DiffractionDataset.experimental_parameter_names:
            info[attr] = getattr(self.dataset, attr)
        
        # PowderDiffractionDataset has some specific parameters
        # like wavelet, decomp. level, etc.
        if isinstance(self.dataset, PowderDiffractionDataset):
            for attr in PowderDiffractionDataset.analysis_parameter_names:
                info[attr] = getattr(self.dataset, attr)
        self.dataset_info_signal.emit(info)