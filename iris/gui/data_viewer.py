# -*- coding: utf-8 -*-
"""
Viewer widgets for DiffractionDatasets
"""
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .time_series_widget import TimeSeriesWidget


class ProcessedDataViewer(QtWidgets.QWidget):
    """
    Widget displaying DiffractionDatasets
    """
    peak_dynamics_roi_signal = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent = self)
        self.time_series_widget = TimeSeriesWidget(parent = self)

        self.peak_dynamics_region = pg.ROI(pos = [0,0], size = [200,200], pen = pg.mkPen('r'))
        self.peak_dynamics_region.addScaleHandle([1, 1], [0, 0])
        self.peak_dynamics_region.addScaleHandle([0, 0], [1, 1])
        self.peak_dynamics_region.sigRegionChanged.connect(self.update_peak_dynamics)

        self.image_viewer.getView().addItem(self.peak_dynamics_region)
        self.peak_dynamics_region.hide()
        self.time_series_widget.hide()

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.layout.addWidget(self.time_series_widget)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def update_peak_dynamics(self):
        self.peak_dynamics_roi_signal.emit(self.peak_dynamics_region.parentBounds().toRect())

    @QtCore.pyqtSlot(bool)
    def toggle_peak_dynamics(self, toggle):
        if toggle: 
            self.peak_dynamics_region.show()
        else: 
            self.peak_dynamics_region.hide()
        
        self.time_series_widget.setVisible(toggle)

    @QtCore.pyqtSlot(object)
    def display(self, image):
        """
        Display an image in the form of an ndarray.

        Parameters
        ----------
        image : ndarray or None
            If None, the display is cleared.
        """
        if image is None:
            self.image_viewer.clear()
            return
        # autoLevels = False ensures that the colormap stays the same
        # when 'sliding' through data. This makes it easier to compare
        # data at different time points.
        # Similarly for autoRange = False
        self.image_viewer.setImage(image, autoLevels = False, autoRange = False)
        self.update_peak_dynamics()

    @QtCore.pyqtSlot(object, object)
    def display_peak_dynamics(self, times, intensities):
        """ 
        Display the time series associated with the integral between the bounds 
        of the ROI
        """
        self.time_series_widget.plot(times, intensities)
