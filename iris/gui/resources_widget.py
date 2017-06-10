
from collections import deque
import colorsys
import numpy as np
from pyqtgraph import QtCore, QtGui, PlotWidget
import pyqtgraph as pg
from psutil import cpu_percent, virtual_memory
from time import time

# TODO: spinbox for update interval
# TODO: have two widgets 'on top of each other'
#       see pyqtgraph.MultiPlotWidget?
#       see https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/MultiplePlotAxes.py

class ComputationalResourceWidget(QtGui.QWidget):

    def __init__(self, interval = 500, **kwargs):
        """
        Parameters
        ----------
        interval : int
            Initial update interval [ms]
        """
        super().__init__(**kwargs)

        self.times = deque([], maxlen = 100)
        self.cpu_percent_stack = deque([], maxlen = 100)
        self.virtual_memory_stack = deque([], maxlen = 100)

        self.update_timer = QtCore.QTimer(parent = self)
        self.update_timer.setInterval(interval)
        self.update_timer.setTimerType(QtCore.Qt.CoarseTimer)
        self.update_timer.timeout.connect(self.measure_resources)

        interval_label = QtGui.QLabel('Update interval: ')
        interval_label.setAlignment(QtCore.Qt.AlignCenter)

        interval_widget = QtGui.QSpinBox(parent = self)
        interval_widget.setRange(100, 10000)
        interval_widget.setSuffix(' ms')
        interval_widget.setValue(int(interval))
        interval_widget.valueChanged.connect(self.update_timer.setInterval)

        enable_measurements_btn = QtGui.QPushButton('Enable measurements', parent = self)
        enable_measurements_btn.setCheckable(True)
        enable_measurements_btn.toggled.connect(self.enable_measurements)
        enable_measurements_btn.setChecked(True)

        self.cpu_percent_widget = PlotWidget(parent = self, title = 'CPU percent', 
                                             labels = {'left':'CPU Utilization (%)',  'bottom':'Elapsed time (s)'})
        self.cpu_percent_widget.setYRange(0, 100)

        self.virtual_memory_widget = PlotWidget(parent = self, title = 'Virtual memory',
                                                labels = {'left':'Memory Used (GB)', 'bottom': 'Elapsed time (s)'})
        self.virtual_memory_widget.setYRange(0, virtual_memory().total / 1e9)
        self.virtual_memory_widget.setXLink(self.cpu_percent_widget)

        for widget in (self.cpu_percent_widget, self.virtual_memory_widget):
            widget.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)

        commands_layout = QtGui.QHBoxLayout()
        commands_layout.addWidget(enable_measurements_btn)
        commands_layout.addWidget(interval_label)
        commands_layout.addWidget(interval_widget)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.cpu_percent_widget)
        layout.addWidget(self.virtual_memory_widget)
        layout.addLayout(commands_layout)
        self.setLayout(layout)

    @QtCore.pyqtSlot()
    def measure_resources(self):
        """ Measure cpu utilization [%] and virtual memory used [GB] """
        self.cpu_percent_stack.append(cpu_percent())
        self.virtual_memory_stack.append(virtual_memory().used / 1e9)
        self.times.append(time())
        self.update_plots()
    
    @QtCore.pyqtSlot()
    def update_plots(self):
        times = np.array(self.times)
        times -= times.max()

        cpu = np.array(self.cpu_percent_stack)
        cpu_brushes = list(map(pg.mkBrush, danger_colors(cpu)))

        memory = np.array(self.virtual_memory_stack)
        memory_brushes = list(map(pg.mkBrush, danger_colors(memory)))

        self.cpu_percent_widget.plot(times, cpu, pen = None, symbolBrush = cpu_brushes, symbolSize = 7, clear = True)
        self.virtual_memory_widget.plot(times, memory, pen = None, symbolBrush = memory_brushes, symbolSize = 7, clear = True)
    
    @QtCore.pyqtSlot(bool)
    def enable_measurements(self, enable):
        """ Start or stop the CPU utilization and virtual memory
        measurements """
        if enable: self.update_timer.start()
        else: self.update_timer.stop()

def danger_colors(arr, bounds = (0, 100)):
    """ 
    Returns a list of colors grading the intensity of the array. 
    
    Parameters
    ----------
    arr : array_like

    bounds : 2-tuple, optional

    Returns
    -------
    colors : list of QColors
    """
    bins = np.linspace(min(bounds), max(bounds), num = 20)

    # Red to Green is hue from 0 to 120 deg
    hues_bins = bins - bins.min()
    hues_bins /= hues_bins.max()
    hues_bins *= 0.33                   # Maximum is green
    hues_bins[:] = hues_bins.max() - hues_bins     # reverse red and greed

    inds = np.digitize(arr, bins, right = True)
    hues = np.take(hues_bins, inds)

    return [QtGui.QColor.fromHsvF(h, 1.0, 0.7) for h in hues]