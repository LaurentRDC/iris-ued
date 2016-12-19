# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""

from .. import RawDataset, DiffractionDataset, PowderDiffractionDataset

from .widgets import IrisStatusBar, DatasetInfoWidget, ProcessedDataViewer, RawDataViewer, RadavViewer
from .worker import WorkThread

import functools
import multiprocessing
import numpy as n
import os
from os.path import join, dirname
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

image_folder = join(dirname(__file__), 'images')
config_path = join(dirname(__file__), 'config.txt')

def run():
    import sys
    
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(join(image_folder, 'eye.png')))
    gui = Iris()
    
    sys.exit(app.exec_())

def error_aware(message):
    """
    Wrap a with a try/except and emit a message.
    """
    def wrap(func):
        @functools.wraps(func)
        def aware_func(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except: # TODO: get traceback message and add to message?
                self.error_message_signal.emit(message)
                raise
        return aware_func
    return wrap

class IrisController(QtCore.QObject):
    """
    Controller behind Iris' GUI.

    Slots
    -----
    load_powder_dataset

    load_sc_dataset

    """
    raw_dataset_loaded_signal = QtCore.pyqtSignal(bool, name = 'raw_dataset_loaded_signal')
    powder_dataset_loaded_signal = QtCore.pyqtSignal(bool, name = 'powder_dataset_loaded_signal')
    processed_dataset_loaded_signal = QtCore.pyqtSignal(bool, name = 'processed_dataset_loaded_signal')

    status_message_signal = QtCore.pyqtSignal(str, name = 'status_message_signal')
    dataset_info_signal = QtCore.pyqtSignal(dict, name = 'dataset_info_signal')
    error_message_signal = QtCore.pyqtSignal(str, name = 'error_message_signal')

    raw_data_signal = QtCore.pyqtSignal(object, name = 'raw_data_signal')
    averaged_data_signal = QtCore.pyqtSignal(object, name = 'averaged_data_signal')

    processing_progress_signal = QtCore.pyqtSignal(int, name = 'processing_progress_signal')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None
    
    @QtCore.pyqtSlot(float, int)
    def display_raw_data(self, timedelay, scan):
        """ Emit raw data (array) """
        self.raw_data_signal.emit(self.dataset.raw_image(timedelay, scan))
    
    @QtCore.pyqtSlot(float)
    def display_averaged_data(self, timedelay):
        """ Emit averaged data (array) at a certain timedelay """
        self.averaged_data_signal.emit(self.dataset.averaged_data(timedelay))
    
    @QtCore.pyqtSlot(float)
    def display_powder_data(self, timedelay):
        """ Emit an array of all time-delay powder data """
        raise NotImplementedError
    
    @error_aware('Raw dataset could not be processed.')
    @QtCore.pyqtSlot(dict)
    def process_raw_dataset(self, info_dict):
        info_dict['callback'] = self.processing_progress_signal.emit
        worker = WorkThread( function = self.dataset.process, kwargs = info_dict)
        worker.results_signal.connect(self.load_dataset)    # self.dataset.process returns a string path
        worker.done_signal.connect(lambda boolean: self.processing_progress_signal.emit(100))

        def in_progress(boolean):
            if boolean: self.status_message_signal.emit('Dataset processing in progress.')
            else: self.status_message_signal.emit('Dataset processing done.')
        
        worker.in_progress_signal.connect(in_progress)
        worker.start()
    
    @error_aware(message = 'Dataset could not be loaded')
    @QtCore.pyqtSlot(str)
    def load_dataset(self, path):
        """ Determines which type of dataset should be loaded, and opens it. """
        if path.endswith('.hdf5'):
            with DiffractionDataset(path, mode = 'r+') as d:
                sample_type = d.sample_type
            
            if sample_type == 'powder':
                self._load_powder_dataset(path)
            elif self.dataset.sample_type == 'single crystal':
                self._load_processed_dataset(path)
            else:
                self.error_message_signal.emit('Unrecognized sample type: {}'.format(self.dataset.sample_type))

        elif os.path.isdir(path):
            self._load_raw_dataset(path)

        else:
            self.error_message_signal.emit('Unrecognized dataset format (path {})'.format(path))

        # Emit dataset information such as fluence, time-points, ...
        info = dict()
        for attr in DiffractionDataset._exp_parameter_names:
            try:    # RawDataset doesn't have all the attributes...
                info[attr] = getattr(self.dataset, attr)
            except : pass
        self.dataset_info_signal.emit(info)
    
    def _load_raw_dataset(self, path):
        self.dataset = RawDataset(path)
        self.raw_dataset_loaded_signal.emit(True)
        self.processed_dataset_loaded_signal.emit(False)
        self.powder_dataset_loaded_signal.emit(False)

        # Show a picture
        self.display_raw_data(timedelay = min(map(abs, self.dataset.time_points)),
                                scan = min(self.dataset.nscans))
    
    def _load_powder_dataset(self, path):
        self.dataset = PowderDiffractionDataset(path)
        self.powder_dataset_loaded_signal.emit(True)
        self.raw_dataset_loaded_signal.emit(False)
        self.processed_dataset_loaded_signal.emit(True)
        
        # TODO: display powder data
        # Display data as close as possible to time zero
        self.display_averaged_data(timedelay = min(map(abs, self.dataset.time_points)))
    
    def _load_processed_dataset(self, path):
        self.dataset = DiffractionDataset(path)
        self.processed_dataset_loaded_signal.emit(True)
        self.powder_dataset_loaded_signal.emit(False)
        self.raw_dataset_loaded_signal.emit(False)

        # Display data as close as possible to time zero
        self.display_averaged_data(timedelay = min(map(abs, self.dataset.time_points)))

class Iris(QtGui.QMainWindow):
    """
    """
    dataset_path_signal = QtCore.pyqtSignal(str, name = 'dataset_path_signal')

    def __init__(self):
        
        super(Iris, self).__init__()
        
        self.controller = IrisController()
        self.dataset_info = DatasetInfoWidget()
        self.raw_data_viewer = RawDataViewer()
        self.processed_viewer = ProcessedDataViewer()
        self.radav_viewer = RadavViewer()
        self.status_bar = IrisStatusBar()

        self._init_ui()
        self._init_actions()
        self._connect_signals()

        # Initialization
        self.controller.raw_dataset_loaded_signal.emit(False)
        self.controller.powder_dataset_loaded_signal.emit(False)
        self.controller.processed_dataset_loaded_signal.emit(False)
        self.controller.status_message_signal.emit('Ready.')
        self.controller.processing_progress_signal.emit(0)
    
    @QtCore.pyqtSlot()
    def load_raw_dataset(self):
        path = self.file_dialog.getExistingDirectory(parent = self, caption = 'Load raw dataset')
        self.dataset_path_signal.emit(path)

    @QtCore.pyqtSlot()
    def load_dataset(self):
        path = self.file_dialog.getOpenFileName(parent = self, caption = 'Load dataset')[0]
        self.dataset_path_signal.emit(path)
    
    def _init_ui(self):
        
        # UI components
        self.error_dialog = QtGui.QErrorMessage(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)      
        self.menu_bar = self.menuBar()
        self.user_controls = None
        self.viewer_stack = QtGui.QTabWidget()
        self.setStatusBar(self.status_bar)
        
        # Assemble menu from previously-defined actions
        self.file_menu = self.menu_bar.addMenu('&File')
        
        #Taskbar icon
        self.setWindowIcon(QtGui.QIcon(os.path.join(image_folder, 'eye.png')))
        
        
        self.viewer_stack.addTab(self.raw_data_viewer, 'View raw dataset')
        self.viewer_stack.addTab(self.processed_viewer, 'View processed dataset')
        self.viewer_stack.addTab(self.radav_viewer, 'View radial averages')
        self.viewer_stack.setCurrentWidget(self.raw_data_viewer)

        self.layout = QtGui.QHBoxLayout()
        self.layout.addWidget(self.dataset_info)
        self.layout.addWidget(self.viewer_stack)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        #Window settings ------------------------------------------------------
        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('Iris - UED data exploration')
        self.center_window()
        self.showMaximized()
    
    def _init_actions(self):
        self.load_raw_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load raw dataset', self)
        self.load_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load dataset', self)
        
        self.file_menu.addAction(self.load_raw_dataset_action)
        self.file_menu.addAction(self.load_dataset_action)
    
    def _connect_signals(self):

        # Status bar
        self.controller.status_message_signal.connect(self.status_bar.update_status)

        # Error handling
        self.controller.error_message_signal.connect(self.error_dialog.showMessage)

        # Loading datasets
        self.load_raw_dataset_action.triggered.connect(self.load_raw_dataset)
        self.load_dataset_action.triggered.connect(self.load_dataset)
        self.dataset_path_signal.connect(self.controller.load_dataset)

        # Update when a new dataset is loaded
        self.controller.dataset_info_signal.connect(self.dataset_info.update)
        self.controller.dataset_info_signal.connect(self.processed_viewer.update_info)

        self.controller.raw_dataset_loaded_signal.connect(lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.raw_data_viewer), x))
        self.controller.processed_dataset_loaded_signal.connect(lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.processed_viewer), x))
        self.controller.powder_dataset_loaded_signal.connect(lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.radav_viewer), x))

        # Display data
        self.controller.raw_data_signal.connect(self.raw_data_viewer.display)
        self.controller.averaged_data_signal.connect(self.processed_viewer.display)

        # Display processed data when requested
        self.raw_data_viewer.raw_data_request_signal.connect(self.controller.display_raw_data)
        self.processed_viewer.averaged_data_request_signal.connect(self.controller.display_averaged_data)

        # Processing raw dataset
        self.raw_data_viewer.process_dataset_signal.connect(self.controller.process_raw_dataset)
        self.controller.processing_progress_signal.connect(print) #self.raw_data_viewer.processing_progress_bar.setValue)
    
    def center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())