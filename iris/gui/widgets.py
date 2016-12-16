from collections.abc import Iterable
from os.path import join, dirname
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore

image_folder = join(dirname(__file__), 'images')

def spectrum_colors(num_colors):
    """
    Generates a set of QColor 0bjects corresponding to the visible spectrum.
    
    Parameters
    ----------
    num_colors : int
        number of colors to return
    
    Yields
    ------
    colors : QColor 
        Can be used with PyQt and PyQtGraph
    """
    # Hue values from 0 to 1
    hue_values = [i/num_colors for i in range(num_colors)]
    
    # Scale so that the maximum is 'purple':
    hue_values = [0.8*hue for hue in hue_values]
    
    colors = list()
    for h in reversed(hue_values):
        yield pg.hsvColor(hue = h, sat = 0.7, val = 0.9)

class IrisStatusBar(QtGui.QStatusBar):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permanent_status_label = QtGui.QLabel('')
        self.addPermanentWidget(self.permanent_status_label)

    @QtCore.pyqtSlot(str)
    def update_status(self, message):
        """ Update the permanent status label with a message. """
        self.permanent_status_label.setText(message)

class DatasetInfoWidget(QtGui.QWidget):
    
    @QtCore.pyqtSlot(dict)
    def update(self, info):
        """ Update the widget with dataset information """

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

class ProcessedDataViewer(QtGui.QWidget):
    """
    Widget displaying the result of processing from RawDataset.process()
    """
    averaged_data_request_signal = QtCore.pyqtSignal(float, name = 'averaged_data_request_signal')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent = self)
        self.time_slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.dataset_info = dict()

        self._init_ui()
        self._init_actions()
        self._connect_signals()
    
    @QtCore.pyqtSlot(dict)
    def update_info(self, info_dict):
        """ Update slider according to dataset info """
        self.dataset_info = info_dict
        self.time_slider.setRange(0, len(info_dict['time_points']))
    
    @QtCore.pyqtSlot(object)
    def display(self, image):
        self.image_viewer.setImage(image)
    
    def _init_ui(self):
        
        # Final assembly
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.layout.addWidget(self.time_slider)
        self.setLayout(self.layout)
    
    def _init_actions(self):
        pass

    def _connect_signals(self):
        # Emit the timedelay (float) of the corresponding slider position
        self.time_slider.sliderMoved.connect(lambda i: self.averaged_data_request_signal.emit(self.dataset_info['time_points'][i] ))

class RawDataViewer(QtGui.QWidget):
    """
    Slots
    -----
    
    display_image

    display_image_series
    """
    process_dataset_signal = QtCore.pyqtSignal(dict, name = 'process_dataset_signal')
    raw_data_request_signal = QtCore.pyqtSignal(float, int, name = 'raw_data_request_signal')

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

        self._init_ui()
        self._init_actions()
        self._connect_signals()
    
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
        processing_info['filename'] = self.file_dialog.getSaveFileName(parent = self, filter = '*.hdf5')
        processing_info['sample_type'] = sample_type

        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2.0
        #Flip output since image viewer plots transpose...
        processing_info['center'] = (corner_y + radius, corner_x + radius)
        processing_info['radius'] = radius
            
        # Beamblock rect
        rect = self.mask.parentBounds().toRect()
        #If coordinate is negative, return 0
        x1 = max(0, rect.topLeft().x() )
        x2 = max(0, rect.x() + rect.width() )
        y1 = max(0, rect.topLeft().y() )
        y2 = max(0, rect.y() + rect.height() )
        processing_info['beamblock_rect'] = (y1, y2, x1, x2)       #Flip output since image viewer plots transpose

        self.process_dataset_signal.emit(processing_info)
    
    @QtCore.pyqtSlot()
    def _request_raw_data(self):
        """ """
        timedelay = float(self.timedelay_edit.text())
        scan = int(self.scan_edit.text())
        self.raw_data_request_signal.emit( timedelay, scan )
    
    def _init_ui(self):

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
    
    def _init_actions(self):
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

    def _connect_signals(self):

        # Get raw data on display
        self.display_btn.clicked.connect(self._request_raw_data)

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
        

class RadavViewer(QtGui.QWidget):
    """
    Widgets displaying two pyqtgraph.PlotWidget widgets. The first one is for radially-averaged
    diffraction patterns, while the other is for time-dynamics of specific regions of
    the radially-averaged diffractiona patterns.
    
    Attributes
    ----------
    parent : QWidget
        Iris QWidget in which this instance is contained. While technically this
        instance could be in a QSplitter, 'parent' refers to the Iris QWidget 
        specifically.
    
    Components
    ----------
    radial_pattern_viewer : PlotWidget
        View the time-delay radial patterns. This widget is also used to specify peak
        dynamics.
    
    peak_dynamics_viewer : PlotWidget
        View the time-delay information about peaks specified in radial_pattern_viewer.
    
    menubar: QtGui.QMenuBar
        Actions are stored in a menu bar
        
    Methods
    -------
    update_peak_dynamics_plot
        Update the peak dynamics plot after changed to the radial pattern viewer
    
    set_background_curve
        Show or hide an inelastic scattering background curve
    
    display_radial_averages
        Plot the data from a PowderDiffractionDataset object
    
    compute_baseline
        Wavelet decomposition of the inelastic scattering background.
    """
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
                
        self.radial_pattern_viewer = pg.PlotWidget(title = 'Radially-averaged pattern(s)', 
                                                   labels = {'left': 'Intensity (counts)', 'bottom': 'Scattering length (1/A)'})
        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 'bottom': ('time', 'ps')})
        self.peak_dynamics_region = pg.LinearRegionItem()
        
        self._init_actions()
        self._init_ui()
        self._connect_signals()

    @QtCore.pyqtSlot(object, object)
    def display_radial_averages(self, scattering_length, array):
        """ 
        Display the radial averages of a dataset.

        Parameters
        ----------
        scattering_length : ndarray, shape (1, N)
            Scattering length of the radial patterns
        array : ndarray, shape (M, N)
            Array for which each row is a radial pattern for a specific time-delay
        """
        for viewer in (self.radial_pattern_viewer, self.peak_dynamics_viewer):
            viewer.clear(), viewer.enableAutoRange()

        #TODO: remove background from array in one operation using axis argument?
        
        # Get timedelay colors and plot
        colors = spectrum_colors(len(curves))
        for color, curve in zip(colors, array):
            self.radial_pattern_viewer.plot(scattering_length, curve, pen = None, symbol = 'o',
                                            symbolPen = pg.mkPen(color), symbolBrush = pg.mkBrush(color), symbolSize = 3)

    def _init_actions(self):
        pass

    def _init_ui(self):
        pass
    
    def _connect_signals(self):
        pass