from .pyqtgraph import QtGui, QtCore
from . import pyqtgraph as pg

from ..dataset import DiffractionDataset, PowderDiffractionDataset
#from ..utils import center_finder

class PromoteToPowderDialog(QtGui.QDialog):
    """
    Modal dialog to promote a DiffractionDataset to a
    PowderDiffractionDataset
    """
    center_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, dataset_filename, *args, **kwargs):
        """
        Parameters
        ----------
        dataset_filename : str
            Path to the DiffractionDataset.
        """
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle('Promote to powder dataset')

        with DiffractionDataset(name = dataset_filename, mode = 'r') as dataset:
            image = dataset.averaged_data(timedelay = dataset.time_points[0])
        
        self.viewer = pg.ImageView(parent = self)
        self.viewer.setImage(image)
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))
        self.viewer.getView().addItem(self.center_finder)

        self.accept_btn = QtGui.QPushButton('Promote', self)
        self.accept_btn.clicked.connect(self.accept)

        self.cancel_btn = QtGui.QPushButton('Cancel', self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(QtGui.QLabel('Drag the circle onto a diffraction ring'))
        self.layout.addWidget(self.viewer)
        self.layout.addLayout(btns)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def accept(self):
        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2
        center = (round(corner_y + radius), round(corner_x + radius)) #Flip output since image viewer plots transpose...
        
        self.center_signal.emit(center)