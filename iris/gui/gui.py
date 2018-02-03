# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""

import functools
import multiprocessing
import os
import sys
from os.path import dirname, join

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from qdarkstyle import load_stylesheet_pyqt5
from skimage.io import imread
from skued import mibread

from .control_bar import ControlBar
from .controller import ErrorAware, IrisController
from .data_viewer import ProcessedDataViewer
from .metadata_edit_dialog import MetadataEditDialog
from .powder_viewer import PowderViewer
from .processing_dialog import ProcessingDialog
from .angular_average_dialog import AngularAverageDialog

image_folder = join(dirname(__file__), 'images')

def run(**kwargs):
    app = QtGui.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet_pyqt5())
    app.setWindowIcon(QtGui.QIcon(join(image_folder, 'eye.png')))
    gui = Iris()
    return app.exec_()

# TODO: show progress on windows taskbar
#       http://doc.qt.io/qt-5/qtwinextras-overview.html
class Iris(QtGui.QMainWindow, metaclass = ErrorAware):
    
    dataset_path_signal             = QtCore.pyqtSignal(str)
    single_picture_path_signal      = QtCore.pyqtSignal(str)
    raw_dataset_path_signal         = QtCore.pyqtSignal(str)
    merlin_raw_dataset_path_signal  = QtCore.pyqtSignal(str)
    error_message_signal            = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.peak_dynamics_window = None

        self._controller_thread = QtCore.QThread(parent = self)
        self.controller = IrisController() # No parent so that we can moveToThread
        self.controller.moveToThread(self._controller_thread)
        self._controller_thread.start()

        self.dataset_path_signal.connect(self.controller.load_dataset)
        self.raw_dataset_path_signal.connect(self.controller.load_raw_dataset)
        self.merlin_raw_dataset_path_signal.connect(self.controller.load_raw_merlin_dataset)

        self.controls = ControlBar(parent = self)
        self.controls.raw_data_request.connect(self.controller.display_raw_data)
        self.controls.averaged_data_request.connect(self.controller.display_averaged_data)
        self.controls.baseline_computation_parameters.connect(self.controller.compute_baseline)
        self.controls.baseline_removed.connect(self.controller.powder_background_subtracted)
        self.controls.relative_powder.connect(self.controller.enable_powder_relative)
        self.controls.relative_averaged.connect(self.controller.enable_averaged_relative)
        self.controls.notes_updated.connect(self.controller.set_dataset_notes)
        self.controls.time_zero_shift.connect(self.controller.set_time_zero_shift)

        self.controller.raw_dataset_loaded_signal.connect(self.controls.enable_raw_dataset_controls)
        self.controller.raw_dataset_loaded_signal.connect(lambda b: self.viewer_stack.setCurrentWidget(self.raw_data_viewer))
        self.controller.processed_dataset_loaded_signal.connect(self.controls.enable_diffraction_dataset_controls)
        self.controller.processed_dataset_loaded_signal.connect(lambda b: self.viewer_stack.setCurrentWidget(self.processed_viewer))
        self.controller.powder_dataset_loaded_signal.connect(self.controls.enable_powder_diffraction_dataset_controls)
        self.controller.powder_dataset_loaded_signal.connect(lambda b: self.viewer_stack.setCurrentWidget(self.powder_viewer))
        self.controller.processing_progress_signal.connect(self.controls.update_processing_progress)
        self.controller.powder_promotion_progress.connect(self.controls.update_powder_promotion_progress)
        self.controller.angular_average_progress.connect(self.controls.update_angular_average_progress)
        self.controller.raw_dataset_metadata.connect(self.controls.update_raw_dataset_metadata)
        self.controller.dataset_metadata.connect(self.controls.update_dataset_metadata)

        #########
        # Viewers
        self.raw_data_viewer = pg.ImageView(parent = self, name = 'Raw data')
        self.raw_data_viewer.resize(self.raw_data_viewer.maximumSize())
        self.controller.raw_data_signal.connect(lambda obj: self.raw_data_viewer.clear() if obj is None else self.raw_data_viewer.setImage(obj))

        self.processed_viewer = ProcessedDataViewer(parent = self)
        self.processed_viewer.peak_dynamics_roi_signal.connect(self.controller.time_series)
        self.controller.averaged_data_signal.connect(self.processed_viewer.display)
        self.controller.time_series_signal.connect(self.processed_viewer.display_peak_dynamics)
        self.controls.enable_peak_dynamics.connect(self.processed_viewer.toggle_peak_dynamics)

        self.powder_viewer = PowderViewer(parent = self)
        self.powder_viewer.peak_dynamics_roi_signal.connect(self.controller.powder_time_series)
        self.controller.powder_data_signal.connect(self.powder_viewer.display_powder_data)
        self.controller.powder_time_series_signal.connect(self.powder_viewer.display_peak_dynamics)
        # UI components
        self.controller.error_message_signal.connect(self.show_error_message)
        self.error_message_signal.connect(self.show_error_message)

        self.file_dialog = QtGui.QFileDialog(parent = self)      
        self.menu_bar = self.menuBar()

        self.viewer_stack = QtGui.QTabWidget()
        self.viewer_stack.addTab(self.raw_data_viewer, 'View raw dataset')
        self.viewer_stack.addTab(self.processed_viewer, 'View processed dataset')
        self.viewer_stack.addTab(self.powder_viewer, 'View radial averages')

        self.controller.raw_dataset_loaded_signal.connect(lambda toggle: self.viewer_stack.setTabEnabled(0, toggle))
        self.controller.processed_dataset_loaded_signal.connect(lambda toggle: self.viewer_stack.setTabEnabled(1, toggle))
        self.controller.powder_dataset_loaded_signal.connect(lambda toggle: self.viewer_stack.setTabEnabled(2, toggle))

        ###################
        # Actions
        self.load_raw_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load raw dataset', self)
        self.load_raw_dataset_action.triggered.connect(self.load_raw_dataset)

        self.load_merlin_raw_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load MERLIN raw dataset', self)
        self.load_merlin_raw_action.triggered.connect(self.load_merlin_raw_dataset)

        self.load_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load dataset', self)
        self.load_dataset_action.triggered.connect(self.load_dataset)

        self.load_single_picture_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Load diffraction picture', self)
        self.load_single_picture_action.triggered.connect(self.load_single_picture)

        self.close_raw_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Close raw dataset', self)
        self.close_raw_dataset_action.triggered.connect(self.controller.close_raw_dataset)
        self.close_raw_dataset_action.setEnabled(False)
        self.controller.raw_dataset_loaded_signal.connect(self.close_raw_dataset_action.setEnabled)

        self.close_dataset_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'locator.png')), '&Close dataset', self)
        self.close_dataset_action.triggered.connect(self.controller.close_dataset)
        self.close_dataset_action.setEnabled(False)
        self.controller.powder_dataset_loaded_signal.connect(self.close_dataset_action.setEnabled)
        self.controller.processed_dataset_loaded_signal.connect(self.close_dataset_action.setEnabled)

        self.file_menu = self.menu_bar.addMenu('&File')
        self.file_menu.addAction(self.load_raw_dataset_action)
        self.file_menu.addAction(self.load_merlin_raw_action)
        self.file_menu.addAction(self.load_dataset_action)
        self.file_menu.addAction(self.load_single_picture_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.close_raw_dataset_action)
        self.file_menu.addAction(self.close_dataset_action)

        ###################
        # Operations on Diffraction Datasets
        # TODO: remove operations from control bar as they are added here
        self.processing_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Process raw data', self)
        self.processing_action.triggered.connect(self.launch_processsing_dialog)
        self.controller.raw_dataset_loaded_signal.connect(self.processing_action.setEnabled)

        self.promote_to_powder_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Calculate angular average', self)
        self.promote_to_powder_action.triggered.connect(self.launch_promote_to_powder_dialog)
        self.controller.processed_dataset_loaded_signal.connect(self.promote_to_powder_action.setEnabled)

        self.update_metadata_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'save.png')), '&Update dataset metadata', self)
        self.update_metadata_action.triggered.connect(self.launch_metadata_edit_dialog)
        self.controller.processed_dataset_loaded_signal.connect(self.update_metadata_action.setEnabled)
        self.controller.powder_dataset_loaded_signal.connect(self.update_metadata_action.setEnabled)

        self.recompute_angular_averages = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Recompute angular average', self)
        self.recompute_angular_averages.triggered.connect(self.launch_recompute_angular_average_dialog)
        self.controller.powder_dataset_loaded_signal.connect(self.recompute_angular_averages.setEnabled)

        self.diffraction_dataset_menu = self.menu_bar.addMenu('&Dataset')
        self.diffraction_dataset_menu.addAction(self.processing_action)
        self.diffraction_dataset_menu.addSeparator()
        self.diffraction_dataset_menu.addAction(self.promote_to_powder_action)
        self.diffraction_dataset_menu.addAction(self.update_metadata_action)
        self.diffraction_dataset_menu.addSeparator()
        self.diffraction_dataset_menu.addAction(self.recompute_angular_averages)

        ###################
        # Layout
        control_layout = QtGui.QVBoxLayout()
        control_layout.addWidget(self.controls)

        self.layout = QtGui.QHBoxLayout()
        self.layout.addWidget(self.viewer_stack)
        self.layout.addLayout(control_layout)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        # Initialization
        self.controller.raw_dataset_loaded_signal.emit(False)
        self.controller.powder_dataset_loaded_signal.emit(False)
        self.controller.processed_dataset_loaded_signal.emit(False)
        self.controller.processing_progress_signal.emit(0)
        self.controller.powder_promotion_progress.emit(0)
        self.controller.angular_average_progress.emit(0)
        
        self.viewer_stack.setCurrentWidget(self.raw_data_viewer)
        self.viewer_stack.resize(self.viewer_stack.maximumSize())

        #Window settings ------------------------------------------------------
        self.setGeometry(0, 0, 1920, 1080)
        self.setWindowIcon(QtGui.QIcon(os.path.join(image_folder, 'eye.png')))
        self.setWindowTitle('Iris - UED data exploration')
        self.center_window()
        self.showMaximized()
        
    def closeEvent(self, event):
        self._controller_thread.quit()
        super().closeEvent(event)
    
    @QtCore.pyqtSlot(str)
    def show_error_message(self, msg):
        self.error_dialog = QtGui.QErrorMessage(parent = self)
        self.error_dialog.showMessage(msg)
    
    @QtCore.pyqtSlot()
    def launch_processsing_dialog(self):
        processing_dialog = ProcessingDialog(parent = self, raw = self.controller.raw_dataset)
        processing_dialog.processing_parameters_signal.connect(self.controller.process_raw_dataset)
        processing_dialog.exec_()
        processing_dialog.processing_parameters_signal.disconnect(self.controller.process_raw_dataset)
    
    @QtCore.pyqtSlot()
    def launch_metadata_edit_dialog(self):
        metadata_dialog = MetadataEditDialog(parent = self, config = self.controller.dataset.metadata)
        metadata_dialog.updated_metadata_signal.connect(self.controller.update_metadata)
        metadata_dialog.exec_()
        metadata_dialog.updated_metadata_signal.disconnect(self.controller.update_metadata)

    @QtCore.pyqtSlot()
    def launch_promote_to_powder_dialog(self):
        image = self.controller.dataset.diff_data(self.controller.dataset.time_points[0])
        promote_dialog = AngularAverageDialog(image, parent = self)
        promote_dialog.angular_average_signal.connect(self.controller.promote_to_powder)
        promote_dialog.exec_()
        promote_dialog.angular_average_signal.disconnect(self.controller.promote_to_powder)
    
    @QtCore.pyqtSlot()
    def launch_recompute_angular_average_dialog(self):
        image = self.controller.dataset.diff_data(self.controller.dataset.time_points[0])
        dialog = AngularAverageDialog(image, parent = self)
        dialog.angular_average_signal.connect(self.controller.recompute_angular_average)
        dialog.exec_()
        dialog.angular_average_signal.disconnect(self.controller.recompute_angular_average)
    
    @QtCore.pyqtSlot()
    def load_raw_dataset(self):
        path = self.file_dialog.getExistingDirectory(parent = self, caption = 'Load raw dataset')
        self.raw_dataset_path_signal.emit(path)
    
    @QtCore.pyqtSlot()
    def load_merlin_raw_dataset(self):
        path = self.file_dialog.getExistingDirectory(parent = self, caption = 'Load raw dataset')
        self.merlin_raw_dataset_path_signal.emit(path)

    @QtCore.pyqtSlot()
    def load_dataset(self):
        path = self.file_dialog.getOpenFileName(parent = self, caption = 'Load dataset', filter = '*.hdf5')[0]
        self.dataset_path_signal.emit(path)
    
    @QtCore.pyqtSlot()
    def load_single_picture(self):
        path = self.file_dialog.getOpenFileName(parent = self, caption = 'Load diffraction picture', filter = 'Images (*.tif *.tiff *.mib)')[0]
        if not path:
            return
        
        return SinglePictureViewer(path).exec_()
    
    @QtCore.pyqtSlot()
    def center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class SinglePictureViewer(QtGui.QDialog):

    def __init__(self, fname, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if fname.endswith(('.tif', '.tiff')):
            im = imread(fname)
        elif fname.endswith('.mib'):
            im = mibread(fname)

        self.image_viewer = pg.ImageView(parent = self)
        self.image_viewer.setImage(im)

        layout = QtGui.QVBoxLayout()
        fname_label = QtGui.QLabel('Filename: ' + fname, parent = self)
        fname_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(fname_label)
        layout.addWidget(self.image_viewer)
        self.setLayout(layout)