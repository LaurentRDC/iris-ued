from collections.abc import Iterable
from dualtree import ALL_FIRST_STAGE, ALL_COMPLEX_WAV
from os.path import join, dirname
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
import numpy as n

from .utils import spectrum_colors

image_folder = join(dirname(__file__), 'images')

class RawDataViewer(QtGui.QWidget):
    """
    Slots
    -----
    display [object]

    process_dataset [str]

    """
    process_dataset_signal = QtCore.pyqtSignal(dict, name = 'process_dataset_signal')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.raw_viewer = pg.ImageView(parent = self)
        self.menu_bar = QtGui.QMenuBar(parent = self)
        self.processing_progress_bar = QtGui.QProgressBar(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.timedelay_edit = QtGui.QLineEdit()
        self.scan_edit = QtGui.QLineEdit()
        self.display_btn = QtGui.QPushButton('Display')

        # Radial-averaging tools
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))

        self.command_bar = QtGui.QHBoxLayout()
        self.command_bar.addWidget(QtGui.QLabel('Time-delay (ps):'))
        self.command_bar.addWidget(self.timedelay_edit)
        self.command_bar.addWidget(QtGui.QLabel('Scan number:'))
        self.command_bar.addWidget(self.scan_edit)
        self.command_bar.addWidget(self.display_btn)

        # Menus
        self.process_menu = self.menu_bar.addMenu('&Processing')

        # Add handles to the beam block mask
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
        
        # Hide on instantiation
        self.raw_viewer.getView().addItem(self.mask)
        self.raw_viewer.getView().addItem(self.center_finder)
        self.mask.hide(), self.center_finder.hide()

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.menu_bar)
        self.layout.addWidget(self.processing_progress_bar)
        self.layout.addWidget(self.raw_viewer)
        self.layout.addLayout(self.command_bar)
        self.setLayout(self.layout)
    
        self.show_centering_tools_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'diffraction.png')), '&Show centering tools', self)
        self.show_centering_tools_action.setCheckable(True)
        self.show_centering_tools_action.setChecked(False)
        self.process_menu.addAction(self.show_centering_tools_action)

        self.process_dataset_as_sc_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Process dataset as single crystal', self)
        self.process_dataset_as_sc_action.setDisabled(True)
        self.process_menu.addAction(self.process_dataset_as_sc_action)
        
        self.process_dataset_as_powder_action = QtGui.QAction(QtGui.QIcon(join(image_folder, 'analysis.png')), '&Process dataset as powder', self)
        self.process_dataset_as_powder_action.setDisabled(True)
        self.process_menu.addAction(self.process_dataset_as_powder_action)

        # Toggle view of tools
        def toggle_mask(t):
            if t: self.mask.show()
            else: self.mask.hide()
        
        def toggle_cf(t):
            if t: self.center_finder.show()
            else: self.center_finder.hide()

        self.show_centering_tools_action.toggled.connect(toggle_mask)
        self.show_centering_tools_action.toggled.connect(toggle_cf)
        self.show_centering_tools_action.toggled.connect(self.process_dataset_as_sc_action.setEnabled)
        self.show_centering_tools_action.toggled.connect(self.process_dataset_as_powder_action.setEnabled)

        # Process dataset
        self.process_dataset_as_sc_action.triggered.connect(lambda : self.process_dataset(sample_type = 'single crystal'))
        self.process_dataset_as_powder_action.triggered.connect(lambda : self.process_dataset(sample_type = 'powder'))
    
    @QtCore.pyqtSlot(object)
    def display(self, data):
        """ Display a single diffraction pattern. """
        self.raw_viewer.setImage(data)
    
    @QtCore.pyqtSlot(str)
    def process_dataset(self, sample_type):
        """ 
        Assemble filename and processing information (center, etc.) and send a request
        to process a dataset.

        Parameters
        ----------
        sample_type : str {'single crystal', 'powder'}
        """
        processing_info = dict()
        processing_info['filename'] = self.file_dialog.getSaveFileName(parent = self, filter = '*.hdf5')[0]
        processing_info['sample_type'] = sample_type

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

        self.process_dataset_signal.emit(processing_info)
    
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
        """ Update slider according to dataset info """
        self.dataset_info.update(info)
        self.time_slider.setRange(0, len(self.dataset_info['time_points']) - 1)
        # TODO: set tick labels to time points
    
    @QtCore.pyqtSlot(object)
    def display(self, image):
        # autoLevels = False ensures that the colormap stays the same
        # when 'sliding' through data. This makes it easier to compare
        # data at different time points.
        self.image_viewer.setImage(image, autoLevels = False, autoRange = True)

class PowderViewer(QtGui.QWidget):
    """
    """

    baseline_parameters_signal = QtCore.pyqtSignal(dict, name = 'baseline_parameters_signal')

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
                
        self.powder_pattern_viewer = pg.PlotWidget(title = 'Radially-averaged pattern(s)', 
                                                   labels = {'left': 'Intensity (counts)', 'bottom': 'Scattering length (1/A)'})

        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 'bottom': ('time', 'ps')})
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()
       

        self.peak_dynamics_region = pg.LinearRegionItem()
        self.peak_dynamics_region.sigRegionChanged.connect(self._update_peak_dynamics)
        self.peak_dynamics_viewer.addItem(self.peak_dynamics_region)

        # Cache
        self.dataset_info = dict()
        self.powder_data_block = None
        self.scattering_length = None
        self.error_block = None
        
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
        

        layout = QtGui.QVBoxLayout()
        layout.addLayout(command_layout)
        layout.addWidget(self.powder_pattern_viewer)
        layout.addWidget(self.peak_dynamics_viewer)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(dict)
    def update_info(self, info):
        self.dataset_info.update(info)

        self.compute_baseline_btn.setEnabled(True)  # info is updated as soon as a dataset is available
        self.first_stage_cb.setEnabled(True)
        self.wavelet_cb.setEnabled(True)

        # TODO: set combo boxes to the value that is is dataset_info

        self.baseline_removed_btn.setEnabled(self.dataset_info['baseline_removed'])
        self.baseline_removed_btn.setChecked(self.dataset_info['baseline_removed'])
    
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
        array : ndarray, shape (M, N)
            Array for which each row is a radial pattern for a specific time-delay
        """
        self.powder_data_block = powder_data_block
        self.scattering_length = scattering_length
        self.error_block = error_block

        for viewer in (self.powder_pattern_viewer, self.peak_dynamics_viewer):
            viewer.clear(), viewer.enableAutoRange()
        
        # Get timedelay colors and plot
        colors = spectrum_colors(len(self.time_points))
        for color, curve in zip(colors, powder_data_block):
            self.powder_pattern_viewer.plot(scattering_length, curve, pen = None, symbol = 'o',
                                            symbolPen = pg.mkPen(color), symbolBrush = pg.mkBrush(color), symbolSize = 3)
        self.powder_pattern_viewer.addItem(self.peak_dynamics_region)
        self._update_peak_dynamics()
    
    @QtCore.pyqtSlot(object)
    def _update_peak_dynamics(self, *args):
        """ """
        self.peak_dynamics_viewer.clear()
        
        # Integrate signal between bounds
        min_s, max_s = self.peak_dynamics_region.getRegion()
        i_min, i_max = n.argmin(n.abs(self.scattering_length - min_s)), n.argmin(n.abs(self.scattering_length - max_s)) + 1

        integrated = self.powder_data_block[:, i_min:i_max].sum(axis = 1)
        integrated /= integrated.max() if integrated.max() > 0.0 else 1

        # Errors add in quadrature
        # TODO: check...
        integrated_error = n.sqrt(n.sum(n.square(self.error_block[:, i_min:i_max]), axis = 1))/(i_max - i_min)

        # Display and error bars
        colors = list(spectrum_colors(len(self.time_points)))
        self.peak_dynamics_viewer.plot(self.time_points, integrated, pen = None, symbol = 'o', 
                                       symbolPen = [pg.mkPen(c) for c in colors], symbolBrush = [pg.mkBrush(c) for c in colors], symbolSize = 4)
        
        error_bars = pg.ErrorBarItem(x = self.time_points, y = integrated, height = integrated_error)
        self.peak_dynamics_viewer.addItem(error_bars)
        
        # If the use has zoomed on the previous frame, auto range might be disabled.
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()

class IrisStatusBar(QtGui.QStatusBar):
    """
    Status bar displaying a status message.

    Slots
    -----
    update_status [str]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permanent_status_label = QtGui.QLabel('')
        self.addPermanentWidget(self.permanent_status_label)

    @QtCore.pyqtSlot(str)
    def update_status(self, message):
        """ Update the permanent status label with a message. """
        self.permanent_status_label.setText(message)

class DatasetInfoWidget(QtGui.QWidget):
    # TODO: make into a QTable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_info = dict()
    
    @QtCore.pyqtSlot(dict)
    def update_info(self, info):
        """ Update the widget with dataset information """
        self.dataset_info.update(info)

        self.master_layout = QtGui.QHBoxLayout()
        self.keys_layout = QtGui.QVBoxLayout()
        self.values_layout = QtGui.QVBoxLayout()

        for key, value in info.items():
            # Items like time_points and nscans are long lists. Summarize using length
            if isinstance(value, Iterable) and not isinstance(value, str) and len(tuple(value)) > 3:
                key += ' (len)'
                value = len(tuple(value))
            self.keys_layout.addWidget(QtGui.QLabel(str(key)))
            self.values_layout.addWidget(QtGui.QLabel(str(value)))

        self.master_layout.addLayout(self.keys_layout)
        self.master_layout.addLayout(self.values_layout)
        self.setLayout(self.master_layout)