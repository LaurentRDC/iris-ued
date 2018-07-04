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
    peak_dynamics_roi_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent = self)
        self.time_series_widget = TimeSeriesWidget(parent = self)

        self.peak_dynamics_region = pg.ROI(pos = [0,0], size = [200,200], pen = pg.mkPen('r'))
        self.peak_dynamics_region.addScaleHandle([1, 1], [0, 0])
        self.peak_dynamics_region.addScaleHandle([0, 0], [1, 1])
        self.peak_dynamics_region.sigRegionChanged.connect(self.update_peak_dynamics)

        self.roi_topleft_text = pg.TextItem('', anchor = (1,1))
        self.roi_bottomright_text = pg.TextItem('', anchor = (0,0))
        self.image_viewer.addItem(self.roi_topleft_text)
        self.image_viewer.addItem(self.roi_bottomright_text)

        self.image_viewer.getView().addItem(self.peak_dynamics_region)
        self.peak_dynamics_region.hide()
        self.time_series_widget.hide()

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.layout.addWidget(self.time_series_widget)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def update_peak_dynamics(self):
        rect = self.peak_dynamics_region.parentBounds().toRect()

        # Add text to show rect bounds
        # If coordinate is negative, return 0
        # Note that in case pg.getConfigOption('imageAxisOrder') is not row-major,
        # The following bounds will not work
        x1 = round(max(0, rect.topLeft().y() ))
        x2 = round(max(0, rect.y() + rect.height() ))
        y1 = round(max(0, rect.topLeft().x() ))
        y2 = round(max(0, rect.x() + rect.width() ))

        self.peak_dynamics_roi_signal.emit((x1, x2, y1, y2))

    @QtCore.pyqtSlot(bool)
    def toggle_peak_dynamics(self, toggle):
        """ Toggle interactive peak dynamics region-of-interest"""
        if toggle: 
            self.peak_dynamics_region.show()
        else: 
            self.peak_dynamics_region.hide()
            self.roi_topleft_text.setText('')
            self.roi_bottomright_text.setText('')
        
        self.time_series_widget.setVisible(toggle)
    
    @QtCore.pyqtSlot(bool)
    def toggle_roi_bounds_text(self, enable):
        """ Toggle showing array indices around the peak dynamics region-of-interest """
        if enable:
            self.peak_dynamics_roi_signal.connect(self._update_roi_bounds_text)
            self.update_peak_dynamics()
        else:
            self.roi_topleft_text.setText('')
            self.roi_bottomright_text.setText('')
            self.peak_dynamics_roi_signal.disconnect(self._update_roi_bounds_text)
    
    @QtCore.pyqtSlot(tuple)
    def _update_roi_bounds_text(self, rect):
        """ Update the ROI bounds text based on the bounds in ``rect`` """
        x1, x2, y1, y2 = rect
        self.roi_topleft_text.setPos(y1, x1)
        self.roi_topleft_text.setText(f'({y1},{x1})')

        self.roi_bottomright_text.setPos(y2, x2)
        self.roi_bottomright_text.setText(f'({y2},{x2})')

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
