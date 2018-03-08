
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from skued import Crystal, calibrate_scattvector

class CrystalMakerWidget(QtWidgets.QWidget):
    """
    Widget responsible for building Crystal either from 
    file or from database.
    """
    new_crystal = QtCore.pyqtSignal(Crystal)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        crystal_label = QtWidgets.QLabel('', parent = self)
        crystal_label.setWordWrap(True)
        crystal_label.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.new_crystal.connect(lambda c : crystal_label.setText(str(c)))

        database_widget = QtWidgets.QListWidget(parent = self)
        database_widget.addItems(sorted(Crystal.builtins))
        database_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding)
        database_widget.currentTextChanged.connect(self.create_database_crystal)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(database_widget)
        layout.addWidget(crystal_label)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(str)
    def create_database_crystal(self, name):
        self.new_crystal.emit(
            Crystal.from_database(name)
        )

class QCalibratorDialog(QtWidgets.QDialog):
    """
    Calibrate the scattering vector range from a polycrystalline diffraction pattern.

    Parameters
    ----------
    I : `~numpy.ndarray`
        Powder diffraction pattern assumed to be defined on a regular grid.
    """
    error_message = QtCore.pyqtSignal(str)

    def __init__(self, I, **kwargs):

        super().__init__(**kwargs)
        self.setModal(True)

        self.error_message.connect(self.show_error_message)

        plot_widget = pg.PlotWidget(parent = self)
        plot_widget.plot(np.arange(0, len(I)), I)

        self.peak1_indicator = pg.InfiniteLine(0, movable = True)
        self.peak2_indicator = pg.InfiniteLine(len(I), movable = True)

        plot_widget.addItem(self.peak1_indicator)
        plot_widget.addItem(self.peak2_indicator)

        crystal_maker = CrystalMakerWidget(parent = self)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(plot_widget)
        layout.addWidget(crystal_maker)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(str)
    def show_error_message(self, msg):
        self.error_dialog = QtGui.QErrorMessage(parent = self)
        self.error_dialog.showMessage(msg)

if __name__ == '__main__':
    
    from qdarkstyle import load_stylesheet_pyqt5
    import sys
    from skued import powdersim, Crystal

    c = Crystal.from_database('vo2-m1')
    I = powdersim(c, np.linspace(0.1, 0.8, 2048))

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet_pyqt5())
    gui = QCalibratorDialog(I)
    gui.show()
    app.exec_()