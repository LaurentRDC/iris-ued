# -*- coding: utf-8 -*-
"""
Viewer widgets for PowderDiffractionDatasets
"""
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .time_series_widget import TimeSeriesWidget, pens_and_brushes


class PowderViewer(QtWidgets.QWidget):

    baseline_parameters_signal = QtCore.pyqtSignal(dict)
    peak_dynamics_roi_signal = QtCore.pyqtSignal(float, float)  # left pos, right pos

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.powder_pattern_viewer = pg.PlotWidget(
            title="Azimuthally-averaged pattern(s)",
            labels={"left": "Intensity (counts)", "bottom": "Scattering vector (1/A)"},
        )
        self.time_series_widget = TimeSeriesWidget(parent=self)

        # Peak dynamics region-of-interest
        self.peak_dynamics_region = pg.LinearRegionItem(values=(0.2, 0.3))
        self.peak_dynamics_region.sigRegionChanged.connect(self.update_peak_dynamics)

        self._text_height = 0
        self.roi_left_text = pg.TextItem("", anchor=(1, 1))
        self.roi_right_text = pg.TextItem("", anchor=(0, 1))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.powder_pattern_viewer)
        layout.addWidget(self.time_series_widget)
        self.setLayout(layout)
        self.resize(self.maximumSize())

    @QtCore.pyqtSlot()
    def update_peak_dynamics(self):
        """Update powder peak dynamics settings on demand."""
        qmin, qmax = self.peak_dynamics_region.getRegion()

        self.peak_dynamics_roi_signal.emit(qmin, qmax)

        self.roi_left_text.setText(f"{qmin:.3f}")
        self.roi_right_text.setText(f"{qmax:.3f}")
        self.roi_left_text.setPos(qmin, self._text_height)
        self.roi_right_text.setPos(qmax, self._text_height)

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
            self.time_series_widget.clear()
            return

        pens, brushes = pens_and_brushes(num=powder_data_block.shape[0])

        self.powder_pattern_viewer.enableAutoRange()
        self.powder_pattern_viewer.clear()

        for pen, brush, curve in zip(pens, brushes, powder_data_block):
            self.powder_pattern_viewer.plot(
                scattering_vector,
                curve,
                pen=None,
                symbol="o",
                symbolPen=pen,
                symbolBrush=brush,
                symbolSize=3,
            )

        # Calculate the height of the text
        # so that it never obscures data
        self._text_height = 1.1 * powder_data_block.max()

        # clearing the powder_pattern_viewer removes all items from the view box
        self.powder_pattern_viewer.addItem(self.peak_dynamics_region)
        self.powder_pattern_viewer.addItem(self.roi_left_text)
        self.powder_pattern_viewer.addItem(self.roi_right_text)

        self.peak_dynamics_region.setBounds(
            [scattering_vector.min(), scattering_vector.max()]
        )
        self.update_peak_dynamics()  # Update peak dynamics plot if background has been changed, for example

    @QtCore.pyqtSlot(object, object)
    def display_peak_dynamics(self, times, intensities):
        """
        Display the time series associated with the integral between the bounds
        of the ROI
        """
        self.time_series_widget.plot(times, intensities)
