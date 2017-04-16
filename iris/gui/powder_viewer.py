from ..dualtree import ALL_FIRST_STAGE, ALL_COMPLEX_WAV
from . import pyqtgraph as pg
from .pyqtgraph import opengl as gl
from .pyqtgraph import QtGui, QtCore
import numpy as n

from .utils import spectrum_colors
from ..utils import fluence

class PowderViewer(QtGui.QWidget):

    baseline_parameters_signal = QtCore.pyqtSignal(dict)
    peak_dynamics_roi_signal = QtCore.pyqtSignal(float, float, bool)  #left pos, right pos, background_removed

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
                
        self.powder_pattern_viewer = pg.PlotWidget(title = 'Radially-averaged pattern(s)', 
                                                   labels = {'left': 'Intensity (counts)', 
                                                             'bottom': 'Scattering length (1/A)'})
        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 
                                                            'bottom': ('time', 'ps')})
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()
       
        # Peak dynamics region-of-interest
        self.peak_dynamics_region = pg.LinearRegionItem(values = (0.2, 0.3))
        self.peak_dynamics_viewer.addItem(self.peak_dynamics_region)
        self.peak_dynamics_region.sigRegionChanged.connect(self.update_peak_dynamics)

        # Surface plot
        self.powder_surface_widget = gl.GLViewWidget(parent = self)
        self.powder_surface_widget.setCameraPosition(distance = 50)
        self.powder_surface_item = gl.GLSurfacePlotItem()
        self.powder_surface_widget.addItem(self.powder_surface_item)

        # Buttons
        self.compute_baseline_btn = QtGui.QPushButton('Compute baseline', parent = self)
        self.compute_baseline_btn.clicked.connect(self.baseline_parameters)

        self.baseline_removed_btn = QtGui.QPushButton('Show baseline-removed', parent = self)
        self.baseline_removed_btn.setCheckable(True)
        self.baseline_removed_btn.setChecked(False)

        # Scroll lists for wavelet parameters
        self.first_stage_cb = QtGui.QComboBox()
        self.first_stage_cb.addItems(ALL_FIRST_STAGE)

        self.wavelet_cb = QtGui.QComboBox()
        self.wavelet_cb.addItems(ALL_COMPLEX_WAV)

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

        powder_ensemble = QtGui.QTabWidget(parent = self)
        powder_ensemble.addTab(self.powder_pattern_viewer, 'Line plot')
        powder_ensemble.addTab(self.powder_surface_widget, 'Surface plot')

        layout = QtGui.QVBoxLayout()
        layout.addLayout(command_layout)
        layout.addWidget(powder_ensemble)
        layout.addWidget(self.peak_dynamics_viewer)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot()
    def baseline_parameters(self):
        params = {'first_stage': self.first_stage_cb.currentText(),
                  'wavelet': self.wavelet_cb.currentText(),
                  'level': 'max',
                  'max_iter': 100}
        self.baseline_parameters_signal.emit(params)
    
    @QtCore.pyqtSlot()
    def update_peak_dynamics(self):
        """ Update powder peak dynamics settings on demand. """
        self.peak_dynamics_roi_signal.emit(*self.peak_dynamics_region.getRegion(), self.baseline_removed_btn.isChecked())
    
    @QtCore.pyqtSlot(object, object, object, bool)
    def display_powder_data(self, scattering_length, powder_data_block, powder_error_block, bgr):
        """ 
        Display the radial averages of a dataset.

        Parameters
        ----------
        scattering_length : ndarray, shape (N,)
            Scattering length of the radial patterns
        powder_data_block : ndarray, shape (M, N)
            Array for which each row is an azimuthal pattern for a specific time-delay.
        powder_error_block : ndarray, shape (M, N)
            Array for which each row is the error for the corresponding azimuthal pattern.
        bgr : bool  
            Describes whether the powder_data_block is baseline-corrected (True) or not (False).
        """
        colors = list(spectrum_colors(powder_data_block.shape[0]))  # number of time-points
        pens, brushes = map(pg.mkPen, colors), map(pg.mkBrush, colors)

        self.powder_pattern_viewer.enableAutoRange()
        self.powder_pattern_viewer.clear()

        # Line plot
        for pen, brush, curve, error in zip(pens, brushes, powder_data_block, powder_error_block):
            self.powder_pattern_viewer.plot(scattering_length, curve, pen = None, symbol = 'o',
                                            symbolPen = pen, symbolBrush = brush, symbolSize = 3)
            error_bars = pg.ErrorBarItem(x = scattering_length, y = curve, height = error)
            self.powder_pattern_viewer.addItem(error_bars)
        
        # Surface plot

        self.baseline_removed_btn.setChecked(bgr)
        self.peak_dynamics_region.setBounds([scattering_length.min(), scattering_length.max()])
        self.powder_pattern_viewer.addItem(self.peak_dynamics_region)
        self.update_peak_dynamics() #Update peak dynamics plot if background has been changed, for example
    
    @QtCore.pyqtSlot(object, object)
    @QtCore.pyqtSlot(object, object, object)    # Maybe there's an error to display, maybe not
    def display_peak_dynamics(self, times, intensities, errors = None):
        """ 
        Display the time series associated with the integral between the bounds 
        of the ROI
        """
        intensities /= intensities.max()

        colors = list(spectrum_colors(len(times)))
        pens, brushes = map(pg.mkPen, colors), map(pg.mkBrush, colors)
        self.peak_dynamics_viewer.plot(times, intensities, 
                                       pen = None, symbol = 'o', 
                                       symbolPen = list(pens), symbolBrush = list(brushes), 
                                       symbolSize = 4, clear = True)
        
        if errors is not None:
            error_bars = pg.ErrorBarItem(x = times, y = intensities, height = errors)
            self.peak_dynamics_viewer.addItem(error_bars)
        
        # If the use has zoomed on the previous frame, auto range might be disabled.
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()