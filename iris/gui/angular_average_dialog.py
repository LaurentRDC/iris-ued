from pyqtgraph import QtGui, QtCore
import pyqtgraph as pg

normalize_help = """ If checked, all powder patterns will be normalized to their overall intensity.
This can get rid of systematic offsets between patterns at different time-delay. """

class AngularAverageDialog(QtGui.QDialog):
    """
    Modal dialog to promote a DiffractionDataset to a
    PowderDiffractionDataset
    """
    angular_average_signal = QtCore.pyqtSignal(dict)

    def __init__(self, image, *args, **kwargs):
        """
        Parameters
        ----------
        image : ndarray
            Diffraction pattern to be displayed.
        """
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle('Promote to powder dataset')
        
        self.viewer = pg.ImageView(parent = self)
        self.viewer.setImage(image)
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))
        self.viewer.getView().addItem(self.center_finder)

        self.accept_btn = QtGui.QPushButton('Promote', self)
        self.accept_btn.clicked.connect(self.accept)

        self.cancel_btn = QtGui.QPushButton('Cancel', self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        self.min_angular_bound_widget = QtGui.QDoubleSpinBox(parent = self)
        self.min_angular_bound_widget.setRange(0, 360)
        self.min_angular_bound_widget.setSingleStep(1)
        self.min_angular_bound_widget.setValue(0)
        self.min_angular_bound_widget.setSuffix(' deg')

        self.max_angular_bound_widget = QtGui.QDoubleSpinBox(parent = self)
        self.max_angular_bound_widget.setRange(0, 360)
        self.max_angular_bound_widget.setSingleStep
        self.max_angular_bound_widget.setValue(360)
        self.max_angular_bound_widget.setSuffix(' deg')

        self.min_angular_bound_widget.valueChanged.connect(self.max_angular_bound_widget.setMinimum)
        self.max_angular_bound_widget.valueChanged.connect(self.min_angular_bound_widget.setMaximum)

        self.normalize_widget = QtGui.QCheckBox('Normalize (?)', self)
        self.normalize_widget.setChecked(False)
        self.normalize_widget.setToolTip(normalize_help)

        angle_bounds_layout = QtGui.QFormLayout()
        angle_bounds_layout.addRow('Min. angle: ', self.min_angular_bound_widget)
        angle_bounds_layout.addRow('Max. angle: ', self.max_angular_bound_widget)

        params_layout = QtGui.QHBoxLayout()
        params_layout.addLayout(angle_bounds_layout)
        params_layout.addWidget(self.normalize_widget)
        params_layout.addStretch()

        btns = QtGui.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(QtGui.QLabel('Drag the circle onto a diffraction ring'))
        self.layout.addWidget(self.viewer)
        self.layout.addLayout(params_layout)
        self.layout.addLayout(btns)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def accept(self):
        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2
        center = (round(corner_y + radius), round(corner_x + radius)) #Flip output since image viewer plots transpose...
        
        params = {'center': center,
                  'angular_bounds': (self.min_angular_bound_widget.value(), self.max_angular_bound_widget.value()),
                  'normalized': self.normalize_widget.checked()}
        self.angular_average_signal.emit(center)
        super().accept()