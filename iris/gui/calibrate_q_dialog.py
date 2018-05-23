# -*- coding: utf-8 -*-
"""
Dialog for Q-vector calibration of powder diffraction data
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from skued import Crystal, powder_calq


EXPLANATION = """Calibrate the scattering vector range of polycrystalline data. Select two peaks of known Miller indices by dragging the vertical lines, 
and use an appropriate structure. Some structures are built-in, but you can also use a CIF of your own. Make sure the structure 
parameters are what you expect before calibration. """.replace('\n', '')

class MillerIndexWidget(QtWidgets.QWidget):
    """
    Widget for specifying a peak's Miller indices
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.h_widget = QtWidgets.QSpinBox(parent = self)
        self.h_widget.setPrefix('h: ')
        self.h_widget.setRange(-999, 999)
        self.h_widget.setValue(0)
        self.h_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        self.k_widget = QtWidgets.QSpinBox(parent = self)
        self.k_widget.setPrefix('k: ')
        self.k_widget.setRange(-999, 999)
        self.k_widget.setValue(0)
        self.k_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        self.l_widget = QtWidgets.QSpinBox(parent = self)
        self.l_widget.setPrefix('l: ')
        self.l_widget.setRange(-999, 999)
        self.l_widget.setValue(0)
        self.l_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
    
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.h_widget)
        self.layout.addWidget(self.k_widget)
        self.layout.addWidget(self.l_widget)

        self.setLayout(self.layout)
    
    @property
    def miller_indices(self):
        return self.h_widget.value(), self.k_widget.value(), self.l_widget.value()

class QCalibratorDialog(QtWidgets.QDialog):
    """
    Calibrate the scattering vector range from a polycrystalline diffraction pattern.

    Parameters
    ----------
    q : `~numpy.ndarray`
        Scattering vector array.
    I : `~numpy.ndarray`
        Powder diffraction pattern defined on the vector ``q``.
    """
    error_message = QtCore.pyqtSignal(str)
    new_crystal = QtCore.pyqtSignal(Crystal)
    calibration_parameters = QtCore.pyqtSignal(dict)

    def __init__(self, I, **kwargs):
        super().__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Scattering vector range calibration')

        explanation_label = QtWidgets.QLabel(EXPLANATION)
        explanation_label.setAlignment(QtCore.Qt.AlignHCenter)
        explanation_label.setWordWrap(True)

        self.intensity = I
        self.crystal = None

        self.error_message.connect(self.show_error_message)

        plot_widget = pg.PlotWidget(parent = self)
        plot_widget.plot(np.arange(0, len(self.intensity)), self.intensity)
        plot_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)

        self.peak1_indicator = pg.InfiniteLine(0, movable = True)
        self.peak2_indicator = pg.InfiniteLine(len(I), movable = True)

        plot_widget.addItem(self.peak1_indicator)
        plot_widget.addItem(self.peak2_indicator)

        # Crystal creation ----------------------------------------------------
        database_title = QtWidgets.QLabel('<h3>Structure description</h3>')
        database_title.setAlignment(QtCore.Qt.AlignHCenter)

        database_widget = QtWidgets.QComboBox(parent = self)
        database_widget.addItems(sorted(Crystal.builtins))
        database_widget.currentTextChanged.connect(self.create_database_crystal)
        database_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        structure_file_btn = QtWidgets.QPushButton('Open explorer', self)
        structure_file_btn.clicked.connect(self.load_cif)
        structure_file_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        crystal_label = QtWidgets.QLabel(parent = self)
        crystal_label.setWordWrap(True)
        crystal_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        crystal_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        crystal_label.setFrameShape(QtWidgets.QFrame.Box)
        self.new_crystal.connect(lambda c : crystal_label.setText(str(c)))  # Use str, not repr, for shorter representation

        crystal_label_title = QtWidgets.QLabel('<h3>Selected crystal structure</h3>')
        crystal_label_title.setAlignment(QtCore.Qt.AlignHCenter)

        crystal_creation_layout = QtWidgets.QFormLayout()
        crystal_creation_layout.addRow(database_title)
        crystal_creation_layout.addRow('Select a database structure: ', database_widget)
        crystal_creation_layout.addRow('Load structure from file: ', structure_file_btn)
        crystal_creation_layout.addRow(crystal_label_title)
        crystal_creation_layout.addRow(crystal_label)

        # Peak specifications -------------------------------------------------
        self.left_peak_miller = MillerIndexWidget(parent = self)
        self.left_peak_miller.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        self.right_peak_miller = MillerIndexWidget(parent = self)
        self.right_peak_miller.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        peak_layout = QtWidgets.QFormLayout()
        peak_layout.addRow('Left peak: ', self.left_peak_miller)
        peak_layout.addRow('Right peak: ', self.right_peak_miller)

        self.accept_btn = QtWidgets.QPushButton('Calibrate', parent = self)
        self.accept_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.accept_btn.clicked.connect(self.accept)

        self.cancel_btn = QtWidgets.QPushButton('Cancel', parent = self)
        self.cancel_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(explanation_label)
        right_layout.addLayout(crystal_creation_layout)
        right_layout.addLayout(peak_layout)
        right_layout.addStretch()
        right_layout.addLayout(btns)

        # Put left layout in a widget for better control on size 
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(plot_widget)
        layout.addWidget(right_widget)
        self.setLayout(layout)

        plot_widget.resize(plot_widget.maximumSize())

    @QtCore.pyqtSlot(str)
    def create_database_crystal(self, name):
        crystal = Crystal.from_database(name)
        self.crystal = crystal
        self.new_crystal.emit(self.crystal)
    
    @QtCore.pyqtSlot()
    def load_cif(self):
        path, *_ = QtWidgets.QFileDialog.getOpenFileName(parent = self, caption = 'Load structure from CIF', filter = '*.cif')
        if not path:
            return
        
        self.crystal = Crystal.from_cif(path)
        self.new_crystal.emit(self.crystal)
    
    @QtCore.pyqtSlot(str)
    def show_error_message(self, msg):
        self.error_dialog = QtGui.QErrorMessage(parent = self)
        self.error_dialog.showMessage(msg)
    
    @QtCore.pyqtSlot()
    def accept(self):
        if self.crystal is None:
            self.show_error_message('Missing structure. Select a Crystal from the database or load a structure file (CIF).')
            return
            
        positions = self.peak1_indicator.getXPos(), self.peak2_indicator.getXPos()
        left, right = min(positions), max(positions)

        params = {'crystal'      : self.crystal,
                  'peak_indices' : (int(left), int(right)),
                  'miller_indices': (self.left_peak_miller.miller_indices, 
                                     self.right_peak_miller.miller_indices) }
        self.calibration_parameters.emit(params)
        super().accept()

if __name__ == '__main__':

    from PyQt5 import QtGui
    import sys
    from qdarkstyle import load_stylesheet_pyqt5
    from skued import Crystal, powdersim
    import numpy as np

    cryst = Crystal.from_database('vo2-m1')
    s = np.linspace(0.1, 1, 512)
    I = powdersim(cryst, s)

    app = QtGui.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet_pyqt5())
    gui = QCalibratorDialog(I)
    gui.show()
    sys.exit(app.exec_())
