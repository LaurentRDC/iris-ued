from pyqtgraph import QtGui, QtCore
from ..utils import fluence

class FluenceCalculatorDialog(QtGui.QDialog):
    """
    Modal dialog to calculate fluence from laser power
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        parent : QWidget or None, optional
        """
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle('Fluence calculator')

        self.beam_size_x_edit = QtGui.QLineEdit('', parent = self)
        self.beam_size_y_edit = QtGui.QLineEdit('', parent = self)
        self.beam_size_x_edit.textChanged.connect(self.update)
        self.beam_size_y_edit.textChanged.connect(self.update)
        beam_size = QtGui.QHBoxLayout()
        beam_size.addWidget(self.beam_size_x_edit)
        beam_size.addWidget(self.beam_size_y_edit)

        self.laser_rep_rate_cb = QtGui.QComboBox(parent = self)
        self.laser_rep_rate_cb.addItems(['50', '100', '200', '250', '500', '1000'])
        self.laser_rep_rate_cb.setCurrentText('1000')
        self.laser_rep_rate_cb.currentIndexChanged.connect(self.update)

        self.window_size_cb = QtGui.QComboBox(parent = self)
        self.window_size_cb.addItems(['250', '500'])
        self.window_size_cb.setCurrentText('250')
        self.window_size_cb.currentIndexChanged.connect(self.update)

        self.incident_laser_power_edit = QtGui.QLineEdit('', parent = self)
        self.incident_laser_power_edit.textChanged.connect(self.update)

        self.fluence = QtGui.QLineEdit('', parent = self)
        self.fluence.setReadOnly(True)

        self.done_btn = QtGui.QPushButton('Done', self)
        self.done_btn.clicked.connect(self.accept)
        self.done_btn.setDefault(True)

        self.layout = QtGui.QFormLayout()
        self.layout.addRow('FWHM x [μm]', self.beam_size_x_edit)
        self.layout.addRow('FWHM y [μm]', self.beam_size_y_edit)
        self.layout.addRow('Laser reprate [Hz]', self.laser_rep_rate_cb)
        self.layout.addRow('Window size [μm]', self.window_size_cb)
        self.layout.addRow('Laser power [mW]', self.incident_laser_power_edit)
        self.layout.addRow('Fluence [mJ/cm2]', self.fluence)
        self.layout.addRow(self.done_btn)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot(int)
    def update(self, *args):
        try:
            f = fluence(float(self.incident_laser_power_edit.text()),
                        int(self.laser_rep_rate_cb.currentText()),
                        sample_size = float(self.window_size_cb.currentText()),
                        FWHM = [int(self.beam_size_x_edit.text()), int(self.beam_size_y_edit.text())])
            self.fluence.setText('{:.2f}'.format(f) + ' mJ / cm^2')
        except:  # Could not parse 
            self.fluence.setText('-----')