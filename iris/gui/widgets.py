from collections.abc import Iterable
from ..dualtree import ALL_FIRST_STAGE, ALL_COMPLEX_WAV
from os.path import join, dirname
from . import pyqtgraph as pg
from .pyqtgraph import QtGui, QtCore
import numpy as n

from .utils import spectrum_colors
from ..utils import fluence

image_folder = join(dirname(__file__), 'images')

class RawDataViewer(QtGui.QWidget):

    process_dataset_signal = QtCore.pyqtSignal(dict, name = 'process_dataset_signal')
    error_message_signal = QtCore.pyqtSignal(str, name = 'error_message_signal')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.raw_viewer = pg.ImageView(parent = self)
        self.menu_bar = QtGui.QMenuBar(parent = self)
        self.processing_progress_bar = QtGui.QProgressBar(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.timedelay_edit = QtGui.QLineEdit()
        self.scan_edit = QtGui.QLineEdit()
        self.display_btn = QtGui.QPushButton('Display')
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))

        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])

        # Toggle view of tools
        def toggle_mask(t):
            if t: self.mask.show()
            else: self.mask.hide()
        
        def toggle_cf(t):
            if t: self.center_finder.show()
            else: self.center_finder.hide()

        self.process_btn = QtGui.QPushButton('Processing')
        self.process_btn.clicked.connect(self.process_dataset)
        self.process_btn.setEnabled(False)

        self.show_tools_btn = QtGui.QPushButton('Show centering tools')
        self.show_tools_btn.setCheckable(True)
        self.show_tools_btn.setChecked(False)
        self.show_tools_btn.toggled.connect(self.process_btn.setEnabled)
        self.show_tools_btn.toggled.connect(toggle_mask)
        self.show_tools_btn.toggled.connect(toggle_cf)

        process_bar = QtGui.QGridLayout()
        process_bar.addWidget(self.show_tools_btn,0,0)
        process_bar.addWidget(self.process_btn,0,1)
        process_bar.addWidget(self.processing_progress_bar, 0, 2, 1, 4)

        command_bar = QtGui.QHBoxLayout()
        command_bar.addWidget(QtGui.QLabel('Time-delay (ps):'))
        command_bar.addWidget(self.timedelay_edit)
        command_bar.addWidget(QtGui.QLabel('Scan number:'))
        command_bar.addWidget(self.scan_edit)
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
    
class ProcessedDataViewer(QtGui.QWidget):
    """
    Widget displaying the result of processing from RawDataset.process()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent = self)
        self.dataset_info = dict()
        
        # QSlider properties
        self.time_slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.time_slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.time_slider.setTickInterval(1)
        self.time_slider.setValue(0)

        # Final assembly
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.layout.addWidget(self.time_slider)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(dict)
    def update_info(self, info):
        """
        Update the widget with dataset information
        
        Parameters
        ----------
        info : dict 
        """
        self.dataset_info.update(info)
        self.time_slider.setRange(0, len(self.dataset_info['time_points']) - 1)
        # TODO: set tick labels to time points
    
    @QtCore.pyqtSlot(object)
    def display(self, image):
        """
        Display an image in the form a an ndarray.

        Parameters
        ----------
        image : ndarray
        """
        # autoLevels = False ensures that the colormap stays the same
        # when 'sliding' through data. This makes it easier to compare
        # data at different time points.
        # Similarly for autoRange = False
        self.image_viewer.setImage(image, autoLevels = False, autoRange = False)

class PowderViewer(QtGui.QWidget):
    baseline_parameters_signal = QtCore.pyqtSignal(dict, name = 'baseline_parameters_signal')

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
                
        self.powder_pattern_viewer = pg.PlotWidget(title = 'Radially-averaged pattern(s)', 
                                                   labels = {'left': 'Intensity (counts)', 'bottom': 'Scattering length (1/A)'})

        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 'bottom': ('time', 'ps')})
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()

        self.fourier_dynamics_viewer = pg.PlotWidget(title = 'Fourier transform', 
                                                     labels = {'left': 'Power Spectrum (a. u.)', 'bottom': ('Frequency', '1/ps?')})
        self.fourier_dynamics_viewer.getPlotItem().enableAutoRange()
       
        self.peak_dynamics_region = pg.LinearRegionItem(values = (0.2, 0.3))
        self.peak_dynamics_region.sigRegionChanged.connect(self._update_peak_dynamics)
        self.peak_dynamics_viewer.addItem(self.peak_dynamics_region)

        # Cache
        self.dataset_info = dict()
        self.powder_data_block = None
        self.scattering_length = None
        self.error_block = None
        self.peak_dynamics = None
        self.peak_dynamics_errors = None

        self._colors = None
        self._pens = None
        self._brushes = None

        # Buttons
        self.compute_baseline_btn = QtGui.QPushButton('Compute baseline', parent = self)
        self.compute_baseline_btn.setEnabled(False) # On instantiation, there is no dataset available
        self.compute_baseline_btn.clicked.connect(
            lambda x: self.baseline_parameters_signal.emit(
                {'first_stage': self.first_stage_cb.currentText(),
                 'wavelet': self.wavelet_cb.currentText(),
                 'level': 'max',
                 'max_iter': 100}
            ))

        self.baseline_removed_btn = QtGui.QPushButton('Show baseline-removed', parent = self)
        self.baseline_removed_btn.setCheckable(True)
        self.baseline_removed_btn.setChecked(False)
        self.baseline_removed_btn.setEnabled(False)

        # Scroll lists for wavelet parameters
        self.first_stage_cb = QtGui.QComboBox()
        self.first_stage_cb.addItems(ALL_FIRST_STAGE)
        self.first_stage_cb.setEnabled(False)

        self.wavelet_cb = QtGui.QComboBox()
        self.wavelet_cb.addItems(ALL_COMPLEX_WAV)
        self.wavelet_cb.setEnabled(False)

        first_stage_label = QtGui.QLabel('First stage wav.:', parent = self)
        first_stage_label.setAlignment(QtCore.Qt.AlignCenter)

        wavelet_label = QtGui.QLabel('Dual-tree wavelet:', parent = self)
        wavelet_label.setAlignment(QtCore.Qt.AlignCenter)
        
        command_layout = QtGui.QGridLayout()
        command_layout.addWidget(self.baseline_removed_btn,  0,  0,  1,  1)
        command_layout.addWidget(self.compute_baseline_btn,  1,  0,  1,  1)
        command_layout.addWidget(first_stage_label,  0,  1,  1,  1)
        command_layout.addWidget(self.first_stage_cb,  1,  1,  1,  1)
        command_layout.addWidget(wavelet_label,  0,  2,  1,  1)
        command_layout.addWidget(self.wavelet_cb,  1,  2,  1,  1)

        tabs = QtGui.QTabWidget(parent = self)
        tabs.addTab(self.peak_dynamics_viewer, 'Peak dynamics')
        tabs.addTab(self.fourier_dynamics_viewer, 'Fourier domain')

        layout = QtGui.QVBoxLayout()
        layout.addLayout(command_layout)
        layout.addWidget(self.powder_pattern_viewer)
        layout.addWidget(tabs)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(dict)
    def update_info(self, info):
        """ 
        Update the widget with dataset information
        
        Parameters
        ----------
        info : dict 
        """
        if info['sample_type'] != 'powder':
            return

        self.dataset_info.update(info)

        self.compute_baseline_btn.setEnabled(True)  # info is updated as soon as a dataset is available
        self.first_stage_cb.setEnabled(True)
        self.wavelet_cb.setEnabled(True)

        self.baseline_removed_btn.setEnabled(self.dataset_info['baseline_removed'])
        self.baseline_removed_btn.setChecked(self.dataset_info['baseline_removed'])

        if self.dataset_info['baseline_removed']:
            self.first_stage_cb.setCurrentText(self.dataset_info['first_stage'])
            self.wavelet_cb.setCurrentText(self.dataset_info['wavelet'])
    
    @property
    def time_points(self):
        return self.dataset_info['time_points']
    
    @QtCore.pyqtSlot(object, object, object)
    def display_powder_data(self, scattering_length, powder_data_block, error_block):
        """ 
        Display the radial averages of a dataset.

        Parameters
        ----------
        scattering_length : ndarray, shape (1, N)
            Scattering length of the radial patterns
        powder_data_block : ndarray, shape (M, N)
            Array for which each row is a radial pattern for a specific time-delay.
        error_block : ndarray, shape (M,N)
            Array for which each row is the error in diffracted intensity 
            at each time-delay.
        """
        # Cache
        self.powder_data_block = powder_data_block
        self.scattering_length = scattering_length
        self.error_block = error_block

        # Cache some info
        self._colors = list(spectrum_colors(powder_data_block.shape[0]))
        self._pens = list(map(pg.mkPen, self._colors))
        self._brushes = list(map(pg.mkBrush, self._colors))

        self.powder_pattern_viewer.clear() 
        self.powder_pattern_viewer.enableAutoRange()
        
        # Get timedelay colors and plot
        for pen, brush, curve in zip(self._pens, self._brushes, powder_data_block):
            self.powder_pattern_viewer.plot(scattering_length, curve, pen = None, symbol = 'o',
                                            symbolPen = pen, symbolBrush = brush, symbolSize = 3)
        
        self.peak_dynamics_region.setBounds([self.scattering_length.min(), self.scattering_length.max()])
        self.powder_pattern_viewer.addItem(self.peak_dynamics_region)
        self._update_peak_dynamics()
    
    @QtCore.pyqtSlot(object)
    def _update_peak_dynamics(self, *args):
        """ 
        Update the cached peaks dynamics corresponding to the
        integrated intensity inside the ROI. 
        """
        # Integrate signal between bounds
        min_s, max_s = self.peak_dynamics_region.getRegion()
        i_min, i_max = n.argmin(n.abs(self.scattering_length - min_s)), n.argmin(n.abs(self.scattering_length - max_s)) + 1

        integrated = self.powder_data_block[:, i_min:i_max].sum(axis = 1)
        integrated_error =  n.sqrt(n.square(self.error_block[:, i_min:i_max]).sum(axis = 1))/(i_max - i_min)
        self.peak_dynamics_errors = integrated_error/integrated.max() if integrated.max() > 0.0 else integrated_error
        self.peak_dynamics = integrated/integrated.max() if integrated.max() > 0.0 else integrated

        # compute fourier dynamics
        # TODO: THIS
        self.frequencies = n.linspace(0, 10, len(self.time_points))
        self.fourier_dynamics = n.zeros_like(self.frequencies)
        
        self._update_peak_dynamics_plot()
        self._update_fourier_dynamics_plot()
    
    def _update_peak_dynamics_plot(self):
        self.peak_dynamics_viewer.clear()

        # Display and error bars
        self.peak_dynamics_viewer.plot(self.time_points, self.peak_dynamics, pen = None, symbol = 'o', 
                                       symbolPen = self._pens, symbolBrush = self._brushes, symbolSize = 4)
        error_bars = pg.ErrorBarItem(x = self.time_points, y = self.peak_dynamics, height = self.peak_dynamics_errors)
        self.peak_dynamics_viewer.addItem(error_bars)
        
        # If the use has zoomed on the previous frame, auto range might be disabled.
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()
    
    def _update_fourier_dynamics_plot(self):
        self.fourier_dynamics_viewer.clear()

        self.fourier_dynamics_viewer.plot(self.frequencies, self.fourier_dynamics, pen = None, symbol = 'o',
                                          symbolPen = pg.mkPen('r'), symbolBrush = pg.mkBrush('r'), symbolSize = 4)

        # If the use has zoomed on the previous frame, auto range might be disabled.
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()

class IrisStatusBar(QtGui.QStatusBar):
    """
    Status bar displaying a status message.

    Slots
    -----
    update_status [str]
    """
    base_message = 'Ready.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_label = QtGui.QLabel(self.base_message)
        self.addPermanentWidget(self.status_label)
        self.timer = QtCore.QTimer(parent = self)
        self.timer.setSingleShot(True)
    
    @QtCore.pyqtSlot(str)
    def update_status(self, message):
        """ 
        Update the permanent status label with a temporary message. 
        
        Parameters
        ----------
        message : str
        """
        self.status_label.setText(message)
        self.timer.singleShot(1e5, lambda: self.update_status(self.base_message))

class DatasetInfoWidget(QtGui.QTableWidget):
    """
    Display of dataset information as a table.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_info = dict()
        self.setRowCount(2) # This will always be true, but the number of
                            # columns will change depending on the dataset

        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)

        # Resize to content
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)

    @QtCore.pyqtSlot(dict)
    def update_info(self, info):
        """ 
        Update the widget with dataset information
        
        Parameters
        ----------
        info : dict 
        """
        self.setVisible(True)
        self.dataset_info.update(info)

        self.setColumnCount(len(self.dataset_info))
        for column, (key, value) in enumerate(self.dataset_info.items()):
            # Items like time_points and nscans are long lists. Summarize using length
            key = key.replace('_', ' ')
            if isinstance(value, Iterable) and not isinstance(value, str) and len(tuple(value)) > 3:
                key += ' (len)'
                value = str(len(tuple(value)))
            
            # TODO: change font of key text to bold
            key_item = QtGui.QTableWidgetItem(str(key))
            ft = key_item.font()
            ft.setBold(True)
            key_item.setFont(ft)
            self.setItem(0, column, key_item)
            self.setItem(1, column, QtGui.QTableWidgetItem(str(value)))
        
        self.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)

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
        self.info_dict.update( {'filename':filename,
                                'compression': self.compression_cb.currentText(),
                                'sample_type': 'powder' if self.powder_type_btn.isChecked() else 'single_crystal',
                                'window_size': int(self.window_size_cb.currentText()),
                                'ring_width': int(self.ring_width_cb.currentText()),
                                'cc': self.center_correction_checkbox.isChecked()} )
        self.processing_options_signal.emit(self.info_dict)
        super().accept()

class FluenceCalculatorDialog(QtGui.QDialog):
    """
    Modal dialog to calculate fluence from laser power
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        parent : QWidget or None, optional
        """
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle('Fluence calculator')

        self.beam_size_x_edit = QtGui.QLineEdit('FWHM x [um]', parent = self)
        self.beam_size_y_edit = QtGui.QLineEdit('FHMM y [um]', parent = self)
        self.beam_size_x_edit.textChanged.connect(self.update)
        self.beam_size_y_edit.textChanged.connect(self.update)
        beam_size = QtGui.QHBoxLayout()
        beam_size.addWidget(self.beam_size_x_edit)
        beam_size.addWidget(self.beam_size_y_edit)

        self.laser_rep_rate_cb = QtGui.QComboBox(parent = self)
        self.laser_rep_rate_cb.addItems(['50', '100', '200', '250', '500', '1000'])
        self.laser_rep_rate_cb.setCurrentText('1000')
        self.laser_rep_rate_cb.currentIndexChanged.connect(self.update)

        self.incident_laser_power_edit = QtGui.QLineEdit('Laser power [mW]', parent = self)
        self.incident_laser_power_edit.textChanged.connect(self.update)

        self.fluence = QtGui.QLineEdit('Fluence (mJ/cm2)', parent = self)
        self.fluence.setReadOnly(True)

        self.done_btn = QtGui.QPushButton('Done', self)
        self.done_btn.clicked.connect(self.accept)
        self.done_btn.setDefault(True)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(beam_size)
        self.layout.addWidget(self.laser_rep_rate_cb)
        self.layout.addWidget(self.incident_laser_power_edit)
        self.layout.addWidget(self.fluence)
        self.layout.addWidget(self.done_btn)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot(int)
    def update(self, *args):
        try:
            f = fluence(float(self.incident_laser_power_edit.text()),
                        int(self.laser_rep_rate_cb.currentText()),
                        FWHM = [int(self.beam_size_x_edit.text()), int(self.beam_size_y_edit.text())])
        except ValueError:  # Could not parse 
            f = '---'
        self.fluence.setText(str(f) + ' mJ / cm^2')