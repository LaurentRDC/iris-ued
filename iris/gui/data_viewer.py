from collections.abc import Iterable
from os.path import dirname, join

from . import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

class ProcessedDataViewer(QtGui.QWidget):
    """
    Widget displaying the result of processing from McGillRawDataset
    """
    peak_dynamics_roi_signal = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent = self)

        self.peak_dynamics_region = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.peak_dynamics_region.addScaleHandle([1, 1], [0, 0])
        self.peak_dynamics_region.addScaleHandle([0, 0], [1, 1])
        self.peak_dynamics_region.sigRegionChanged.connect(
            lambda roi: self.peak_dynamics_roi_signal.emit(self.peak_dynamics_region.parentBounds().toRect()))


        self.image_viewer.getView().addItem(self.peak_dynamics_region)
        self.peak_dynamics_region.hide()

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(object, object)
    def update_peak_dynamics(self, time_points, integrated_intensity):
        self.peak_dynamics_viewer.plot(time_points, integrated_intensity)

    @QtCore.pyqtSlot(bool)
    def toggle_peak_dynamics(self, toggle):
        if toggle: 
            self.peak_dynamics_region.show()
        else: 
            self.peak_dynamics_region.hide()

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