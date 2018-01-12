"""
This module implements a modal dialog to analyze knife-edge measurements from csv files.

@author : Laurent P. Rene de Cotret
"""
import numpy as n
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from ..knife_edge import cdf, knife_edge


class KnifeEdgeToolDialog(QtGui.QDialog):

    _measurement_signal = QtCore.pyqtSignal(object, object)
    _fwhm_signal = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle('Knife-edge tool')

        self._measurement_signal.connect(self.plot_data)
        self._measurement_signal.connect(self.fit_data)

        self.fwhm_item = pg.TextItem(text = 'No data', color = 'w')
        self._fwhm_signal.connect(self.set_fwhm_text)

        self.data_item = pg.ScatterPlotItem()
        self.fit_item = pg.PlotCurveItem()

        self.viewer = pg.PlotWidget(parent = self)
        self.viewer.addItem(self.data_item)
        self.viewer.addItem(self.fit_item)
        self.viewer.addItem(self.fwhm_item)
        self.fwhm_item.setPos(1, 1)

        self.load_csv_btn = QtGui.QPushButton('Load CSV data', self)
        self.load_csv_btn.clicked.connect(self.load_measurement)

        self.done_btn = QtGui.QPushButton('Done', self)
        self.done_btn.clicked.connect(self.accept)
        self.done_btn.setDefault(True)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(self.load_csv_btn)
        btns.addWidget(self.done_btn)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.viewer)
        self.layout.addLayout(btns)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def load_measurement(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load measurement')[0]
        if not filename : 
            return
        positions, values = n.loadtxt(fname = filename, dtype = n.float, delimiter = ',', 
                                      skiprows = 2, usecols = (0,1), unpack = True)
        self._measurement_signal.emit(positions, values)
    
    @QtCore.pyqtSlot(object, object)
    def plot_data(self, position, values):
        """ Plot a measurement and fit the data """
        self.data_item.setData(position, values, symbol = 'o', symbolSize = 3, 
                               symbolPen = pg.mkPen('r'))
    
    @QtCore.pyqtSlot(object, object)
    def fit_data(self, position, values):
        params = dict()
        fwhm = knife_edge(position, values, fit_parameters = params)
        self._fwhm_signal.emit(fwhm)

        x_fit = n.linspace(position.min(), position.max(), num = 10*len(values))
        y_fit = cdf(x_fit, **params)
        self.fit_item.setData(x_fit, y_fit, pen = pg.mkPen('w'))
        
        # Set fwhm text anchor according to fit
        self.fwhm_item.setPos( x_fit.min(), n.mean(y_fit) )
    
    @QtCore.pyqtSlot(float)
    def set_fwhm_text(self, fwhm):
        self.fwhm_item.setText('FWHM: {:.4f}'.format(fwhm))
