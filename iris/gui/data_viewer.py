from collections.abc import Iterable
from os.path import dirname, join

import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from ..utils import fluence
from skued import spectrum_colors

image_folder = join(dirname(__file__), 'images')
    
class ProcessedDataViewer(QtGui.QWidget):
    """
    Widget displaying the result of processing from RawDataset.process()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent = self)
        self.dataset_info = dict()
        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 'bottom': ('time', 'ps')})
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()
        self.peak_dynamics_viewer.hide()

        # Single-crystal peak dynamics
        self.peak_dynamics_region = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.peak_dynamics_region.addScaleHandle([1, 1], [0, 0])
        self.peak_dynamics_region.addScaleHandle([0, 0], [1, 1])
        self.image_viewer.getView().addItem(self.peak_dynamics_region)
        self.peak_dynamics_region.hide()

        # Final assembly
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.layout.addWidget(self.peak_dynamics_viewer)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(object, object)
    def update_peak_dynamics(self, time_points, integrated_intensity):
        pens = list(map(pg.mkPen, spectrum_colors(len(time_points))))
        brushes = list(map(pg.mkBrush, spectrum_colors(len(time_points))))
        self.peak_dynamics_viewer.plot(time_points, integrated_intensity, pen = None, symbol = 'o', 
                                       symbolPen = pens, symbolBrush = brushes, symbolSize = 4, clear = True)
    
    @QtCore.pyqtSlot(object)
    def display(self, image):
        """
        Display an image in the form of an ndarray.

        Parameters
        ----------
        image : ndarray
        """
        # autoLevels = False ensures that the colormap stays the same
        # when 'sliding' through data. This makes it easier to compare
        # data at different time points.
        # Similarly for autoRange = False
        self.image_viewer.setImage(image, autoLevels = self.autolevel_cb.isChecked(), 
                                          autoRange = self.autorange_cb.isChecked())