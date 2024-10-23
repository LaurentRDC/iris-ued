# -*- coding: utf-8 -*-
"""
Time-series widget
"""

from functools import lru_cache

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget, TextItem, mkBrush, mkPen
from scipy.optimize import curve_fit
from skued import exponential, spectrum_colors


@lru_cache(maxsize=1)
def pens_and_brushes(num):
    qcolors = tuple(map(lambda c: QtGui.QColor.fromRgbF(*c), spectrum_colors(num)))
    pens = list(map(mkPen, qcolors))
    brushes = list(map(mkBrush, qcolors))
    return pens, brushes


class TimeSeriesWidget(QtWidgets.QWidget):
    """Time-series widget with built-in display controls."""

    # Internal refresh signal
    _refresh_signal = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    # Signal to change the y-axis units
    _yaxis_units_signal = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plot_widget = PlotWidget(
            parent=self,
            title="Diffraction time-series",
            labels={"left": ("Intensity", "a. u."), "bottom": ("time", "ps")},
        )
        self._yaxis_units_signal.connect(self.set_yaxis_units)

        self.plot_widget.getPlotItem().showGrid(x=True, y=True)

        # Internal state on whether or not to 'connect the dots'
        self._time_series_connect = False
        self._symbol_size = 5

        # Internal memory on last time-series info
        self._last_times = None
        self._last_intensities_abs = None
        self._last_norm_mask = None
        self._refresh_signal.connect(self.plot)

        self.connect_widget = QtWidgets.QCheckBox("Connect time-series", self)
        self.connect_widget.toggled.connect(self.enable_connect)

        self.symbol_size_widget = QtWidgets.QSpinBox(parent=self)
        self.symbol_size_widget.setRange(1, 25)
        self.symbol_size_widget.setValue(self._symbol_size)
        self.symbol_size_widget.setPrefix("Symbol size: ")
        self.symbol_size_widget.valueChanged.connect(self.set_symbol_size)

        self.absolute_intensity_widget = QtWidgets.QCheckBox("Show absolute intensity", self)
        self.absolute_intensity_widget.toggled.connect(self.toggle_absolute_intensity)

        self.exponential_fit_widget = QtWidgets.QPushButton("Calculate exponential decay", self)
        self.exponential_fit_widget.clicked.connect(self.fit_exponential_decay)

        self.export_timeseries_widget = QtWidgets.QPushButton("Export time-series data", self)
        self.export_timeseries_widget.clicked.connect(self.export_timeseries)

        self.fit_constants_label = QtWidgets.QLabel(self)

        self.controls = QtWidgets.QHBoxLayout()
        self.controls.addWidget(self.connect_widget)
        self.controls.addWidget(self.absolute_intensity_widget)
        self.controls.addWidget(self.symbol_size_widget)
        self.controls.addWidget(self.exponential_fit_widget)
        self.controls.addWidget(self.fit_constants_label)
        self.controls.addWidget(self.export_timeseries_widget)
        self.controls.addStretch(1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.plot_widget)
        layout.addLayout(self.controls)
        self.setLayout(layout)

    @QtCore.pyqtSlot(str)
    def set_yaxis_units(self, units):
        """Set the y-axis label"""
        self.plot_widget.getPlotItem().setLabel("left", text="Intensity", units=units)

    @QtCore.pyqtSlot(bool)
    def toggle_absolute_intensity(self, absolute):
        # Normalize to intensity before time-zero
        if absolute:
            self._yaxis_units_signal.emit("counts")
        else:
            self._yaxis_units_signal.emit("a.u.")

        self.refresh()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def plot(self, time_points, intensity):
        """
        Plot the a time-series. Time-series are normalized to the intensity before photoexcitation.

        Parameters
        ----------
        time_points : `~numpy.ndarray`, shape (N,)
            Time-delays [ps]
        intensity : `~numpy.ndarray`, shape (N,)
            Diffracted intensity in absolute units.
        """
        self._last_times = np.asarray(time_points)
        self._last_intensities_abs = np.asarray(intensity)
        self._last_norm_mask = self._last_times < 0
        if self._last_norm_mask.sum() == 0:
            self._last_norm_mask[0] = True

        # Normalize to intensity before time-zero
        # Note that the change in units is taken care of in the toggle_absolute_intensity method
        absolute = self.absolute_intensity_widget.isChecked()
        intensity = (
            self._last_intensities_abs
            if absolute
            else self._last_intensities_abs / np.mean(self._last_intensities_abs[self._last_norm_mask])
        )

        # Only compute the colors if number of time-points changes or first time
        pens, brushes = pens_and_brushes(len(time_points))

        connect_kwargs = {"pen": None} if not self._time_series_connect else dict()
        self.plot_widget.plot(
            x=self._last_times,
            y=intensity,
            symbol="o",
            symbolPen=pens,
            symbolBrush=brushes,
            symbolSize=self._symbol_size,
            clear=True,
            **connect_kwargs,
        )

        # Don't forget to clear the fit constants
        self.fit_constants_label.clear()

    @QtCore.pyqtSlot()
    def fit_exponential_decay(self):
        """Try to fit to a time-series with an exponential decay. If successful, plot the result."""
        times = self._last_times
        intensity = self._last_intensities_abs

        # time-zero, amplitude, time-constant, offset
        initial_guesses = (0, intensity.max(), 1, intensity.min())

        try:
            params, pcov = curve_fit(exponential, times, intensity, p0=initial_guesses)
        except RuntimeError:
            return

        fit = exponential(times, *params)
        absolute = self.absolute_intensity_widget.isChecked()
        if not absolute:
            fit /= np.mean(self._last_intensities_abs[self._last_times < 0])
        self.plot_widget.plot(x=times, y=fit, symbol=None, clear=False)

        # Write fit parameters to text items
        tconst, tconsterr = params[2], np.sqrt(pcov[2, 2])
        tzero, tzeroerr = params[0], np.sqrt(pcov[0, 0])
        self.fit_constants_label.setText(
            f"Time-constant: ({tconst:.3f}±{tconsterr:.3f} ps, Time-zero: ({tzero:.3f}±{tzeroerr:.3f}) ps"
        )

    @QtCore.pyqtSlot()
    def export_timeseries(self):
        """Allow exporting the time-series data."""
        times = self._last_times
        intensity = self._last_intensities_abs

        if (times is None) or (intensity is None):
            return QtWidgets.QMessageBox.warning("No time-series to export.")

        file_dialog = QtWidgets.QFileDialog()
        path = file_dialog.getSaveFileName(parent=self, caption="Export time-series data", filter="*.csv")[0]
        if not path:
            return

        data = np.empty(shape=(len(times), 2), dtype=float)
        data[:, 0] = times
        data[:, 1] = intensity
        np.savetxt(path, data, delimiter=",", header="times [ps], intensity [cnts]")

    @QtCore.pyqtSlot()
    def refresh(self):
        self._refresh_signal.emit(self._last_times, self._last_intensities_abs)

    @QtCore.pyqtSlot()
    def clear(self):
        self.plot_widget.clear()

    @QtCore.pyqtSlot(bool)
    def enable_connect(self, toggle):
        self._time_series_connect = toggle
        self.refresh()

    @QtCore.pyqtSlot(int)
    def set_symbol_size(self, size):
        self._symbol_size = size
        self.refresh()
