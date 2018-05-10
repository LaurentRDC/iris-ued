
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

description = """ Align the circle so that its center is aligned with the diffraction center. """

class SymmetrizeDialog(QtWidgets.QDialog):
    """
    Modal dialog used to symmetrize datasets.
    """

    error_message_signal         = QtCore.pyqtSignal(str)
    symmetrize_parameters_signal = QtCore.pyqtSignal(str, dict)

    def __init__(self, image, **kwargs):
        super().__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Data Symmetrization')

        description_label = QtWidgets.QLabel(parent)
        description_label.setText(description)

        self.viewer = pg.ImageView(parent = self)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                  QtWidgets.QSizePolicy.MinimumExpanding)
        self.viewer.setImage(image)
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))
        self.viewer.getView().addItem(self.center_finder)

        self.mod_widget = QtWidgets.QComboBox(parent = self)
        self.mod_widget.addItems(['2', '3', '4', '6'])
        self.mod_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, 
                                      QtWidgets.QSizePolicy.Maximum)

        self.accept_btn = QtWidgets.QPushButton('Symmetrize', self)
        self.accept_btn.clicked.connect(self.accept)

        self.cancel_btn = QtWidgets.QPushButton('Cancel', self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        self.error_message_signal.connect(self.show_error_message)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        params = QtWidgets.QFormLayout()
        params.addRow('Rotational multiplicity: ', self.mod_widget)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(description_label)
        self.layout.addWidget(self.viewer)
        self.layout.addLayout(params)
        self.layout.addLayout(btns)
        self.setLayout(self.layout)

    @QtCore.pyqtSlot(str)
    def show_error_message(self, msg):
        self.error_dialog = QtGui.QErrorMessage(parent = self)
        self.error_dialog.showMessage(msg)

    @QtCore.pyqtSlot()
    def accept(self):
        self.file_dialog = QtWidgets.QFileDialog(parent = self)
        filename = self.file_dialog.getSaveFileName(filter = '*.hdf5')[0]
        if filename == '':
            return

        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2
        center = (round(corner_y + radius), round(corner_x + radius)) #Flip output since image viewer plots transpose...
        
        # In case the images a row-order, the image will be
        # transposed with respect to what is expected.
        if pg.getConfigOption('imageAxisOrder') == 'row-major':
            center = tuple(reversed(center))

        params = {'center': center,
                  'mod'   : int(self.mod_widget.currentText())}
        
        self.symmetrize_parameters_signal.emit(filename, params)
        super().accept()