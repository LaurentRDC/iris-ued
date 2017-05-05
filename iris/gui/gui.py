# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""

import functools
import multiprocessing
import os
import sys
from os.path import dirname, join

import numpy as n
import pyqtgraph as pg
from . import QtCore, QtGui

from .. import DiffractionDataset, PowderDiffractionDataset, RawDataset
from .controller import IrisController, error_aware
from .control_bar import ControlBar
from .fluence_calculator import FluenceCalculatorDialog
from .knife_edge_tool import KnifeEdgeToolDialog
from .powder_viewer import PowderViewer
from .processing_dialog import ProcessingDialog
from .promote_dialog import PromoteToPowderDialog
from .qdarkstyle import load_stylesheet_pyqt5
from .raw_viewer import RawDataViewer
from .data_viewer import ProcessedDataViewer

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
        
        self._controller_thread = QtCore.QThread()
        self.controller = IrisController() # No parent so that we can moveToThread
        self.controller.moveToThread(self._controller_thread)
        self._controller_thread.start()

        self.controls = ControlBar(parent = self)
        self.controls.raw_data_request.connect(self.controller.display_raw_data)
        self.controls.averaged_data_request.connect(self.controller.display_averaged_data)
        self.controls.process_dataset.connect(self.launch_processsing_dialog)
        self.controls.promote_to_powder.connect(self.launch_promote_to_powder_dialog)
        self.controls.baseline_computation_parameters.connect(self.controller.compute_baseline)
        self.controller.raw_dataset_loaded_signal.connect(self.controls.enable_raw_dataset_controls)
        self.controller.processed_dataset_loaded_signal.connect(self.controls.enable_diffraction_dataset_controls)
        self.controller.powder_dataset_loaded_signal.connect(self.controls.enable_powder_diffraction_dataset_controls)
        self.controller.processing_progress_signal.connect(self.controls.update_processing_progress)

        #########
        # Viewers
        self.raw_data_viewer = pg.ImageView(parent = self, name = 'Raw data')
        self.raw_data_viewer.resize(self.raw_data_viewer.maximumSize())
        self.controller.raw_data_signal.connect(self.raw_data_viewer.setImage)

        self.processed_viewer = ProcessedDataViewer(parent = self)
        self.controller.averaged_data_signal.connect(self.processed_viewer.display)
        self.controller.time_series_signal.connect(self.processed_viewer.update_peak_dynamics)

        self.powder_viewer = PowderViewer(parent = self)
        self.controller.powder_data_signal.connect(self.powder_viewer.display_powder_data)
        self.controller.powder_time_series_signal.connect(self.powder_viewer.display_peak_dynamics)

        # UI components
        self.error_dialog = QtGui.QErrorMessage(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)      
        self.menu_bar = self.menuBar()
        self.viewer_stack = QtGui.QTabWidget()
        
        self.viewer_stack.addTab(self.raw_data_viewer, 'View raw dataset')
        self.viewer_stack.addTab(self.processed_viewer, 'View processed dataset')
        self.viewer_stack.addTab(self.powder_viewer, 'View radial averages')

        self.load_raw_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load raw dataset', self)
        self.load_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load dataset', self)
        self.file_menu = self.menu_bar.addMenu('&File')
        self.file_menu.addAction(self.load_raw_dataset_action)
        self.file_menu.addAction(self.load_dataset_action)

        self.fluence_calculator_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Fluence calculator', self)
        self.fluence_calculator_action.triggered.connect(self.launch_fluence_calculator_tool)
        self.autoindexing_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Autoindexing', self)
        self.autoindexing_action.setEnabled(False)
        self.knife_edge_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Knife-edge analysis', self)
        self.knife_edge_action.triggered.connect(self.launch_knife_edge_tool)
        self.tools_menu = self.menu_bar.addMenu('&Tools')
        self.tools_menu.addAction(self.fluence_calculator_action)
        self.tools_menu.addAction(self.autoindexing_action)
        self.tools_menu.addAction(self.knife_edge_action)

        # Error handling
        self.controller.error_message_signal.connect(self.error_dialog.showMessage)
        
        self.layout = QtGui.QHBoxLayout()
        self.layout.addWidget(self.viewer_stack)
        self.layout.addWidget(self.controls)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        # Initialization
        self.controller.raw_dataset_loaded_signal.emit(False)
        self.controller.powder_dataset_loaded_signal.emit(False)
        self.controller.processed_dataset_loaded_signal.emit(False)
        self.controller.processing_progress_signal.emit(0)
        self.viewer_stack.setCurrentWidget(self.raw_data_viewer)

        #Window settings ------------------------------------------------------
        self.setGeometry(500, 500, 800, 800)
        self.setWindowIcon(QtGui.QIcon(os.path.join(image_folder, 'eye.png')))
        self.setWindowTitle('Iris - UED data exploration')
        self.center_window()
        self.showMaximized()
        
    def __del__(self):
        self._controller_thread.quit()
    
    @QtCore.pyqtSlot()
    def launch_processsing_dialog(self):
        processing_dialog = ProcessingDialog(parent = self, raw = self.controller.raw_dataset)
        processing_dialog.processing_parameters_signal.connect(self.controller.process_raw_dataset)
        return processing_dialog.exec_()
    
    @QtCore.pyqtSlot()
    def launch_promote_to_powder_dialog(self):
        promote_dialog = PromoteToPowderDialog(dataset_filename = self.controller.dataset.filename, parent = self)
        promote_dialog.center_signal.connect(self.controller.promote_to_powder)
        return promote_dialog.exec_()
    
    @QtCore.pyqtSlot()
    def load_raw_dataset(self):
        path = self.file_dialog.getExistingDirectory(parent = self, caption = 'Load raw dataset')
        self.raw_dataset_path_signal.emit(path)

    @QtCore.pyqtSlot()
    def load_dataset(self):
        path = self.file_dialog.getOpenFileName(parent = self, caption = 'Load dataset', filter = '*.hdf5')[0]
        self.dataset_path_signal.emit(path)
    
    @QtCore.pyqtSlot()
    def center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    @QtCore.pyqtSlot()
    def launch_knife_edge_tool(self):
        window = KnifeEdgeToolDialog(parent = self)
        return window.exec_()
    
    @QtCore.pyqtSlot()
    def launch_fluence_calculator_tool(self):
        window = FluenceCalculatorDialog(parent = self)
        return window.exec_()
