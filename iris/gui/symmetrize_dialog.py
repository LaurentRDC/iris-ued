# -*- coding: utf-8 -*-
"""
Dialog for symmetrization of DiffractionDataset
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

description = """Align the circle so that its center is aligned with the diffraction center. """

class SymmetrizeDialog(QtWidgets.QDialog):
    """
    Modal dialog used to symmetrize datasets.
    """

    error_message_signal         = QtCore.pyqtSignal(str)
    symmetrize_parameters_signal = QtCore.pyqtSignal(str, dict)

    def __init__(self, image, **kwargs):
        super().__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Symmetrization')

        title = QtWidgets.QLabel('<h2>Symmetrization Options<\h2>')
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        description_label = QtWidgets.QLabel(description, parent = self)
        description_label.setWordWrap(True)
        description_label.setAlignment(QtCore.Qt.AlignCenter)

        self.viewer = pg.ImageView(parent = self)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                  QtWidgets.QSizePolicy.MinimumExpanding)
        self.viewer.setImage(image)
        self.center_finder = pg.CircleROI(pos = np.array(image.shape)/2 - 100, size = [200,200], pen = pg.mkPen('r'))
        self.viewer.getView().addItem(self.center_finder)

        self.mod_widget = QtWidgets.QComboBox(parent = self)
        self.mod_widget.addItems(['2', '3', '4', '6'])
        self.mod_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, 
                                      QtWidgets.QSizePolicy.Maximum)

        self.smoothing_kernel_widget = QtWidgets.QSpinBox(parent = self)
        self.smoothing_kernel_widget.setRange(0, 100)
        self.smoothing_kernel_widget.setValue(5)
        self.smoothing_kernel_widget.setSuffix(' px')
        self.smoothing_kernel_widget.setEnabled(False)
        
        self.enable_smoothing_widget = QtWidgets.QCheckBox('Enable gaussian smoothing')
        self.enable_smoothing_widget.setChecked(False)
        self.enable_smoothing_widget.toggled.connect(self.smoothing_kernel_widget.setEnabled)

        self.accept_btn = QtWidgets.QPushButton('Symmetrize', self)
        self.accept_btn.clicked.connect(self.accept)
        self.accept_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, 
                                      QtWidgets.QSizePolicy.Maximum)

        self.cancel_btn = QtWidgets.QPushButton('Cancel', self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)
        self.cancel_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, 
                                      QtWidgets.QSizePolicy.Maximum)

        self.error_message_signal.connect(self.show_error_message)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        multiplicity_layout = QtWidgets.QFormLayout()
        multiplicity_layout.addRow('Rotational multiplicity: ', self.mod_widget)

        smoothing_layout = QtWidgets.QFormLayout()
        smoothing_layout.addRow(self.enable_smoothing_widget)
        smoothing_layout.addRow('Kernel standard deviation: ', self.smoothing_kernel_widget)

        params_layout = QtWidgets.QVBoxLayout()
        params_layout.addWidget(title)
        params_layout.addWidget(description_label)
        params_layout.addLayout(multiplicity_layout)
        params_layout.addLayout(smoothing_layout)
        params_layout.addLayout(btns)

        params_widget = QtWidgets.QFrame(parent = self)
        params_widget.setLayout(params_layout)
        params_widget.setFrameShadow(QtWidgets.QFrame.Sunken)
        params_widget.setFrameShape(QtWidgets.QFrame.Panel)
        params_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(params_widget)
        right_layout.addStretch()

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.viewer)
        self.layout.addLayout(right_layout)
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

        corner_x, corner_y = self.center_finder.pos().y(), self.center_finder.pos().x()
        radius = self.center_finder.size().x()/2
        center = (round(corner_y + radius), round(corner_x + radius)) #Flip output since image viewer plots transpose...
        
        # In case the images a row-order, the image will be
        # transposed with respect to what is expected.
        if pg.getConfigOption('imageAxisOrder') == 'row-major':
            center = tuple(reversed(center))

        params = {'center': center,
                  'mod'   : int(self.mod_widget.currentText())}

        if self.enable_smoothing_widget.isChecked():
            params['kernel_size'] = self.smoothing_kernel_widget.value()
        
        self.symmetrize_parameters_signal.emit(filename, params)
        super().accept()
