import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from functools import lru_cache
from skued import spectrum_colors
from skued.baseline import ALL_COMPLEX_WAV, ALL_FIRST_STAGE

@lru_cache(maxsize = 1)
def pens_and_brushes(num):
    qcolors = tuple(map(lambda c: QtGui.QColor.fromRgbF(*c), spectrum_colors(num)))
    pens = list(map(pg.mkPen, qcolors))
    brushes = list(map(pg.mkBrush, qcolors))
    return pens, brushes

class PowderViewer(QtGui.QWidget):

    baseline_parameters_signal = QtCore.pyqtSignal(dict)
    peak_dynamics_roi_signal = QtCore.pyqtSignal(float, float)  #left pos, right pos

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        # Internal that stores whether time-series points should
        # be connected or not
        self._time_series_connect = False
                
        self.powder_pattern_viewer = pg.PlotWidget(title = 'Radially-averaged pattern(s)', 
                                                   labels = {'left': 'Intensity (counts)', 
                                                             'bottom': 'Scattering vector (1/A)'})
        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 
                                                            'bottom': ('time', 'ps')})
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()

        # Peak dynamics region-of-interest
        self.peak_dynamics_region = pg.LinearRegionItem(values = (0.2, 0.3))
        self.peak_dynamics_viewer.addItem(self.peak_dynamics_region)
        self.peak_dynamics_region.sigRegionChanged.connect(self.update_peak_dynamics)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.powder_pattern_viewer)
        layout.addWidget(self.peak_dynamics_viewer)
        self.setLayout(layout)
        self.resize(self.maximumSize())
    
    @QtCore.pyqtSlot(bool)
    def set_time_series_connect(self, enable):
        self._time_series_connect = enable
        self.update_peak_dynamics()

    @QtCore.pyqtSlot()
    def update_peak_dynamics(self):
        """ Update powder peak dynamics settings on demand. """
        self.peak_dynamics_roi_signal.emit(*self.peak_dynamics_region.getRegion())
        
    @QtCore.pyqtSlot(object, object)
    def display_powder_data(self, scattering_vector, powder_data_block):
        """ 
        Display the radial averages of a dataset.

        Parameters
        ----------
        scattering_vector : ndarray, shape (N,) or None
            Scattering length of the radial patterns. If None, all
            viewers are cleared.
        powder_data_block : ndarray, shape (M, N) or None
            Array for which each row is an azimuthal pattern for a specific time-delay. If None, all
            viewers are cleared.
        powder_error_block : ndarray, shape (M, N) or None
            Array for which each row is the error for the corresponding azimuthal pattern. If None, error bars
            are not displayed.
        """
        if (scattering_vector is None) or (powder_data_block is None):
            self.powder_pattern_viewer.clear()
            self.peak_dynamics_viewer.clear()
            return
        
        pens, brushes = pens_and_brushes(num = powder_data_block.shape[0])

        self.powder_pattern_viewer.enableAutoRange()
        self.powder_pattern_viewer.clear()

        for pen, brush, curve in zip(pens, brushes, powder_data_block):
            self.powder_pattern_viewer.plot(scattering_vector, curve, pen = None, symbol = 'o',
                                            symbolPen = pen, symbolBrush = brush, symbolSize = 3)
        
        self.peak_dynamics_region.setBounds([scattering_vector.min(), scattering_vector.max()])
        self.powder_pattern_viewer.addItem(self.peak_dynamics_region)
        self.update_peak_dynamics() #Update peak dynamics plot if background has been changed, for example
    
    @QtCore.pyqtSlot(object, object)
    @QtCore.pyqtSlot(object, object)    # Maybe there's an error to display, maybe not
    def display_peak_dynamics(self, times, intensities):
        """ 
        Display the time series associated with the integral between the bounds 
        of the ROI
        """
        pens, brushes = pens_and_brushes(num = len(times))

        intensities /= intensities.max()

        connect_kwargs = {'pen': None} if not self._time_series_connect else {}

        self.peak_dynamics_viewer.plot(times, intensities, symbol = 'o', 
                                       symbolPen = pens, symbolBrush = brushes, 
                                       symbolSize = 5, clear = True, **connect_kwargs)
        
        # If the use has zoomed on the previous frame, auto range might be disabled.
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()