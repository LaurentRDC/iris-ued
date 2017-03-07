from os.path import join, dirname
from . import pyqtgraph as pg
from .pyqtgraph import QtGui, QtCore
import numpy as n

from .utils import spectrum_colors

image_folder = join(dirname(__file__), 'images')

class RawDataViewer(QtGui.QWidget):

    display_raw_data_signal = QtCore.pyqtSignal(float, int)
    process_dataset_signal = QtCore.pyqtSignal(dict)
    error_message_signal = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_info = dict()

        self.raw_viewer = pg.ImageView(parent = self)
        self.menu_bar = QtGui.QMenuBar(parent = self)
        self.processing_progress_bar = QtGui.QProgressBar(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.display_btn = QtGui.QPushButton('Display')
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))

        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])

        self.process_btn = QtGui.QPushButton('Processing')
        self.process_btn.clicked.connect(self.process_dataset)
        self.process_btn.setEnabled(False)

        self.show_tools_btn = QtGui.QPushButton('Show centering tools')
        self.show_tools_btn.setCheckable(True)
        self.show_tools_btn.setChecked(False)
        self.show_tools_btn.toggled.connect(self.process_btn.setEnabled)
        self.show_tools_btn.toggled.connect(self.mask.setVisible)
        self.show_tools_btn.toggled.connect(self.center_finder.setVisible)

        process_bar = QtGui.QGridLayout()
        process_bar.addWidget(self.show_tools_btn,0,0)
        process_bar.addWidget(self.process_btn,0,1)
        process_bar.addWidget(self.processing_progress_bar, 0, 2, 1, 4)

        # Navigating through raw data
        self.timedelay_widget = QtGui.QComboBox(parent = self)
        self.timedelay_widget.setEditable(False)
        self.scan_widget = QtGui.QSpinBox(parent = self)

        # When scan widget or timedelay_widget are updated, get a new image
        self.timedelay_widget.currentIndexChanged.connect(lambda x: self.update_display)
        self.scan_widget.valueChanged.connect(lambda x: self.update_display)

        command_bar = QtGui.QHBoxLayout()
        command_bar.addWidget(QtGui.QLabel('Time-delay (ps):'))
        command_bar.addWidget(self.timedelay_widget)
        command_bar.addWidget(QtGui.QLabel('Scan number:'))
        command_bar.addWidget(self.scan_widget)
        command_bar.addWidget(self.display_btn)
        
        # Hide on instantiation
        self.raw_viewer.getView().addItem(self.mask)
        self.raw_viewer.getView().addItem(self.center_finder)
        self.mask.hide(), self.center_finder.hide()

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(process_bar)
        self.layout.addWidget(self.raw_viewer)
        self.layout.addLayout(command_bar)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(dict)
    def update_dataset_info(self, info_dict):
        """ Update the range of possible values of times and scans """
        self.dataset_info.update(info_dict)

        # Update range of timedelay and scan
        self.timedelay_widget.clear()
        self.timedelay_widget.addItems(list(map(str, self.dataset_info['time_points'])))

        nscans = self.dataset_info['nscans']
        self.scan_widget.setRange(min(nscans), max(nscans))
    
    @QtCore.pyqtSlot()
    def update_display(self):
        """ Request an update on the raw data display """
        try:
            timedelay = float(self.timedelay_widget.currentText())
            scan = self.scan_widget.value()
        except:
            timedelay = float(self.dataset_info['time_points'][0])
            scan = int(self.dataset_info['nscans'][0])
            
        self.display_raw_data_signal.emit(timedelay, scan)
        
    @QtCore.pyqtSlot(object)
    def display(self, data):
        """ Display a single diffraction pattern. """
        self.raw_viewer.setImage(data, autoLevels = False, autoRange = True)
    
    @QtCore.pyqtSlot()
    def process_dataset(self):
        """ 
        Assemble filename and processing information (center, etc.) and send a request
        to process a dataset.

        Parameters
        ----------
        sample_type : str {'single crystal', 'powder'}
        """
        processing_info = dict()

        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2.0
        #Flip output since image viewer plots transpose...
        processing_info['center'] = (round(corner_y + radius), round(corner_x + radius))
        processing_info['radius'] = round(radius)
            
        # Beamblock rect
        rect = self.mask.parentBounds().toRect()
        #If coordinate is negative, return 0
        x1 = round(max(0, rect.topLeft().x() ))
        x2 = round(max(0, rect.x() + rect.width() ))
        y1 = round(max(0, rect.topLeft().y() ))
        y2 = round(max(0, rect.y() + rect.height() ))
        processing_info['beamblock_rect'] = (y1, y2, x1, x2)       #Flip output since image viewer plots transpose

        self.processing_options_dialog = ProcessingOptionsDialog(parent = self, info_dict = processing_info)
        self.processing_options_dialog.processing_options_signal.connect(
            lambda info_dict: self.process_dataset_signal.emit(info_dict))
        
        success = self.processing_options_dialog.exec_()
        self.show_tools_btn.setChecked(False)   # Remove tools from view
        if not success:
            self.error_message_signal.emit('Processing options could not be set.')

class ProcessingOptionsDialog(QtGui.QDialog):
    """
    Modal dialog used to select dataset processing options.
    """
    processing_options_signal = QtCore.pyqtSignal(dict, name = 'processing_options_signal')

    def __init__(self, parent, info_dict, **kwargs):
        """
        Parameters
        ----------
        parent : QWidget or None, optional
        """
        super().__init__(parent = parent, **kwargs)
        self.info_dict = info_dict
        self.setModal(True)
        self.setWindowTitle('Processing options')

        self.save_btn = QtGui.QPushButton('Launch processing', self)
        self.save_btn.clicked.connect(self.accept)

        self.cancel_btn = QtGui.QPushButton('Cancel', self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        # Determine settings
        self.file_dialog = QtGui.QFileDialog(parent = self)

        # HDF5 compression
        # TODO: determine automatically legal values?
        self.compression_cb = QtGui.QComboBox(parent = self)
        self.compression_cb.addItems(['lzf', 'gzip'])
        self.compression_cb.setCurrentText('lzf')

        # Click either of two mutually-exclusive btns
        self.powder_type_btn = QtGui.QPushButton('Powder', parent = self)
        self.powder_type_btn.setCheckable(True)
        self.sc_type_btn = QtGui.QPushButton('Single crystal', parent = self)
        self.sc_type_btn.setCheckable(True)

        self.powder_type_btn.clicked.connect(lambda checked: self.sc_type_btn.setChecked(not checked))
        self.sc_type_btn.clicked.connect(lambda checked: self.powder_type_btn.setChecked(not checked))
        self.powder_type_btn.setChecked(True)
        
        sample_type_layout = QtGui.QHBoxLayout()
        sample_type_layout.addWidget(self.powder_type_btn)
        sample_type_layout.addWidget(self.sc_type_btn)

        self.mad_checkbox = QtGui.QCheckBox('Enable MAD filtering', parent = self)
        self.mad_checkbox.setChecked(True)
        self.center_correction_checkbox = QtGui.QCheckBox('Enable center-correction', parent = self)

        self.window_size_cb = QtGui.QComboBox(parent = self)
        self.window_size_cb.addItems(list(map(str, range(0, 20, 1))))
        self.window_size_cb.setCurrentText('10')   # TODO: get default from somewhere?

        self.ring_width_cb = QtGui.QComboBox(parent = self)
        self.ring_width_cb.addItems(list(map(str, range(3, 20, 1))))
        self.ring_width_cb.setCurrentText('5')   # TODO: get default from somewhere?

        self.center_correction_checkbox.toggled.connect(self.window_size_cb.setEnabled)
        self.center_correction_checkbox.toggled.connect(self.ring_width_cb.setEnabled)
        self.center_correction_checkbox.setChecked(True)    # Default

        items = QtGui.QVBoxLayout()
        items.setAlignment(QtCore.Qt.AlignCenter)
        items.addWidget(QtGui.QLabel('Select HDF5 Compression:', parent = self))
        items.addWidget(self.compression_cb)
        items.addWidget(QtGui.QLabel('Select sample type:', parent = self))
        items.addLayout(sample_type_layout)
        items.addWidget(self.mad_checkbox)
        items.addWidget(self.center_correction_checkbox)
        items.addWidget(QtGui.QLabel('Center-correction window size:', parent = self))
        items.addWidget(self.window_size_cb)
        items.addWidget(QtGui.QLabel('Center-correction ring width:', parent = self))
        items.addWidget(self.ring_width_cb)

        buttons = QtGui.QHBoxLayout()
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.cancel_btn)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(items)
        self.layout.addLayout(buttons)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def accept(self):
        filename = self.file_dialog.getSaveFileName(filter = '*.hdf5')[0]

        # All parameters for the function RawDataset.process go here as keywork arguments
        self.info_dict.update( {'filename':filename,
                                'compression': self.compression_cb.currentText(),
                                'sample_type': 'powder' if self.powder_type_btn.isChecked() else 'single_crystal',
                                'window_size': int(self.window_size_cb.currentText()),
                                'ring_width': int(self.ring_width_cb.currentText()),
                                'cc': self.center_correction_checkbox.isChecked(),
                                'mad': self.mad_checkbox.isChecked()} )
        self.processing_options_signal.emit(self.info_dict)
        super().accept()