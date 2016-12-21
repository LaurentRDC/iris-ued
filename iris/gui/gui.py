# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""

from .. import RawDataset, DiffractionDataset, PowderDiffractionDataset
from .controller import IrisController, error_aware
from .widgets import IrisStatusBar, DatasetInfoWidget, ProcessedDataViewer, RawDataViewer, PowderViewer
from .utils import WorkThread

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

class Iris(QtGui.QMainWindow):
    """
    """
    dataset_path_signal = QtCore.pyqtSignal(str, name = 'dataset_path_signal')
    raw_dataset_path_signal = QtCore.pyqtSignal(str, name = 'raw_dataset_path_signal')

    def __init__(self):
        
        super(Iris, self).__init__()
        
        self.controller = IrisController(parent = self)
        self.dataset_info = DatasetInfoWidget()
        self.raw_data_viewer = RawDataViewer()
        self.processed_viewer = ProcessedDataViewer()
        self.powder_viewer = PowderViewer()
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
        self.raw_dataset_path_signal.emit(path)

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
        self.viewer_stack.addTab(self.powder_viewer, 'View radial averages')
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

        ######################################################################
        # RAW DATA INTERACTION
        # Displaying raw data when requested
        self.load_raw_dataset_action.triggered.connect(self.load_raw_dataset)
        self.raw_dataset_path_signal.connect(self.controller.load_raw_dataset)

        self.controller.raw_dataset_loaded_signal.connect(  
            lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.raw_data_viewer), x))
        self.controller.raw_dataset_loaded_signal.connect(  
            lambda x: self.viewer_stack.setCurrentIndex(self.viewer_stack.indexOf(self.raw_data_viewer)) if x else None)

        self.raw_data_viewer.display_btn.clicked.connect(   
            lambda x: self.controller.raw_data_signal.emit( 
                float(self.raw_data_viewer.timedelay_edit.text()),
                int(self.raw_data_viewer.scan_edit.text())))
        
        self.controller.raw_data_signal.connect(self.raw_data_viewer.display)

        # Processing raw dataset
        self.raw_data_viewer.process_dataset_signal.connect(self.controller.process_raw_dataset)
        self.controller.processing_progress_signal.connect(print) #self.raw_data_viewer.processing_progress_bar.setValue)

        ######################################################################
        # PROCESSED DATA INTERACTION
        self.load_dataset_action.triggered.connect(self.load_dataset)
        self.dataset_path_signal.connect(self.controller.load_dataset)

        self.controller.processed_dataset_loaded_signal.connect(lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.processed_viewer), x))
        self.controller.processed_dataset_loaded_signal.connect(lambda x: self.viewer_stack.setCurrentIndex(self.viewer_stack.indexOf(self.processed_viewer)) if x else None)
        
        self.processed_viewer.time_slider.sliderMoved.connect(lambda i: self.controller.display_averaged_data(self.controller.dataset.time_points[i]))
        self.controller.averaged_data_signal.connect(self.processed_viewer.display)

        ######################################################################
        # POWDER DATA INTERACTION
        self.controller.powder_dataset_loaded_signal.connect(lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.powder_viewer), x))
        self.controller.powder_dataset_loaded_signal.connect(lambda x: self.viewer_stack.setCurrentIndex(self.viewer_stack.indexOf(self.powder_viewer)) if x else None)
        self.controller.powder_data_signal.connect(self.powder_viewer.display_powder_data)
        self.powder_viewer.baseline_parameters_signal.connect(self.controller.compute_baseline)
        self.powder_viewer.baseline_removed_btn.toggled.connect(lambda x: self.controller.powder_data_signal.emit(*self.controller.dataset.powder_data_block(bgr = x)))

        ######################################################################
        # Update when a new dataset is loaded
        # Switch tabs as well
        self.controller.dataset_info_signal.connect(self.dataset_info.update)
        self.controller.dataset_info_signal.connect(self.processed_viewer.update_info)
        self.controller.dataset_info_signal.connect(self.powder_viewer.update_info)
            
    def center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())