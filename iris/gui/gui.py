# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""

import functools
import multiprocessing
import numpy as n
import os
from os.path import join, dirname
import sys

from . import pyqtgraph as pg
from .pyqtgraph import QtCore, QtGui
from .qdarkstyle import load_stylesheet_pyqt5
from .. import RawDataset, DiffractionDataset, PowderDiffractionDataset
from .controller import IrisController, error_aware
from .widgets import (IrisStatusBar, DatasetInfoWidget, ProcessedDataViewer, 
                      RawDataViewer, PowderViewer, FluenceCalculatorDialog)
from .utils import WorkThread

image_folder = join(dirname(__file__), 'images')

def run():
    app = QtGui.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet_pyqt5())
    app.setWindowIcon(QtGui.QIcon(join(image_folder, 'eye.png')))
    gui = Iris()
    sys.exit(app.exec_())

class Iris(QtGui.QMainWindow):
    
    dataset_path_signal = QtCore.pyqtSignal(str, name = 'dataset_path_signal')
    raw_dataset_path_signal = QtCore.pyqtSignal(str, name = 'raw_dataset_path_signal')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.controller = IrisController(parent = self)
        self.dataset_info = DatasetInfoWidget()
        self.raw_data_viewer = RawDataViewer()
        self.processed_viewer = ProcessedDataViewer()
        self.powder_viewer = PowderViewer()
        self.status_bar = IrisStatusBar()
        self.fluence_calculator = FluenceCalculatorDialog(parent = self)

        # UI components
        self.error_dialog = QtGui.QErrorMessage(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)      
        self.menu_bar = self.menuBar()
        self.viewer_stack = QtGui.QTabWidget()
        self.setStatusBar(self.status_bar)
        
        self.viewer_stack.addTab(self.raw_data_viewer, 'View raw dataset')
        self.viewer_stack.addTab(self.processed_viewer, 'View processed dataset')
        self.viewer_stack.addTab(self.powder_viewer, 'View radial averages')

        self.load_raw_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load raw dataset', self)
        self.load_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load dataset', self)
        self.file_menu = self.menu_bar.addMenu('&File')
        self.file_menu.addAction(self.load_raw_dataset_action)
        self.file_menu.addAction(self.load_dataset_action)

        self.fluence_calculator_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Fluence calculator', self)
        self.fluence_calculator_action.triggered.connect(lambda x: self.fluence_calculator.exec_())
        self.autoindexing_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Autoindexing', self)
        self.autoindexing_action.setEnabled(False)
        self.tools_menu = self.menu_bar.addMenu('&Tools')
        self.tools_menu.addAction(self.fluence_calculator_action)
        self.tools_menu.addAction(self.autoindexing_action)

        # Status bar
        self.controller.status_message_signal.connect(self.status_bar.update_status)

        # Error handling
        self.controller.error_message_signal.connect(self.error_dialog.showMessage)
        self.raw_data_viewer.error_message_signal.connect(self.error_dialog.showMessage)

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
            lambda x: self.controller.display_raw_data( 
                float(self.raw_data_viewer.timedelay_edit.text()),
                int(self.raw_data_viewer.scan_edit.text())))
        
        self.controller.raw_data_signal.connect(self.raw_data_viewer.display)

        # Processing raw dataset
        self.raw_data_viewer.process_dataset_signal.connect(self.controller.process_raw_dataset)
        self.controller.processing_progress_signal.connect(self.raw_data_viewer.processing_progress_bar.setValue)

        ######################################################################
        # PROCESSED DATA INTERACTION
        self.load_dataset_action.triggered.connect(self.load_dataset)
        self.dataset_path_signal.connect(self.controller.load_dataset)

        self.controller.processed_dataset_loaded_signal.connect(
            lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.processed_viewer), x))
        self.controller.processed_dataset_loaded_signal.connect(
            lambda x: self.viewer_stack.setCurrentIndex(self.viewer_stack.indexOf(self.processed_viewer)) if x else None)
        
        self.processed_viewer.time_slider.sliderMoved.connect(
            lambda i: self.controller.display_averaged_data(self.controller.dataset.time_points[i]))
        self.controller.averaged_data_signal.connect(self.processed_viewer.display)

        ######################################################################
        # POWDER DATA INTERACTION
        self.controller.powder_dataset_loaded_signal.connect(
            lambda x: self.viewer_stack.setTabEnabled(self.viewer_stack.indexOf(self.powder_viewer), x))
        self.controller.powder_dataset_loaded_signal.connect(
            lambda x: self.viewer_stack.setCurrentIndex(self.viewer_stack.indexOf(self.powder_viewer)) if x else None)
        self.controller.powder_data_signal.connect(self.powder_viewer.display_powder_data)
        self.powder_viewer.baseline_parameters_signal.connect(self.controller.compute_baseline)
        self.powder_viewer.baseline_removed_btn.toggled.connect(
            lambda x: self.controller.powder_data_signal.emit(*self.controller.dataset.powder_data_block(bgr = x)))

        ######################################################################
        # Update when a new dataset is loaded
        # Switch tabs as well
        self.controller.dataset_info_signal.connect(self.dataset_info.update_info)
        self.controller.dataset_info_signal.connect(self.processed_viewer.update_info)
        self.controller.dataset_info_signal.connect(self.powder_viewer.update_info)
        
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.viewer_stack)
        self.layout.addWidget(self.dataset_info)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        # Initialization
        self.controller.raw_dataset_loaded_signal.emit(False)
        self.controller.powder_dataset_loaded_signal.emit(False)
        self.controller.processed_dataset_loaded_signal.emit(False)
        self.controller.status_message_signal.emit('Ready.')
        self.controller.processing_progress_signal.emit(0)
        self.viewer_stack.setCurrentWidget(self.raw_data_viewer)
        self.dataset_info.hide()

        #Window settings ------------------------------------------------------
        self.setGeometry(500, 500, 800, 800)
        self.setWindowIcon(QtGui.QIcon(os.path.join(image_folder, 'eye.png')))
        self.setWindowTitle('Iris - UED data exploration')
        self.center_window()
        self.showMaximized()
    
    @QtCore.pyqtSlot()
    def load_raw_dataset(self):
        path = self.file_dialog.getExistingDirectory(parent = self, caption = 'Load raw dataset')
        self.raw_dataset_path_signal.emit(path)

    @QtCore.pyqtSlot()
    def load_dataset(self):
        path = self.file_dialog.getOpenFileName(parent = self, caption = 'Load dataset')[0]
        self.dataset_path_signal.emit(path)
            
    def center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())