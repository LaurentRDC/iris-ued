
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from .. import beam_properties

class ElectronBeamPropertiesDialog(QtGui.QDialog):
    """
    Modal dialog used to calculate electron count and other electron
    beam properties
    """
    _calculation_progress = QtCore.pyqtSignal(int)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Electron Beam Properties')

        self.progress_bar = QtGui.QProgressBar(parent = self)
        self.progress_bar.hide()
        self._calculation_progress.connect(self.progress_bar.setValue)

        self.directory_finder = QtGui.QPushButton('Select directory', self)
        self.directory_finder.clicked.connect(self.evaluate_beam_properties)

        count_label = QtGui.QLabel('Electrons per shot:')
        count_label.setAlignment(QtCore.Qt.AlignCenter)

        self.count_widget = QtGui.QLabel('< --- >')
        self.count_widget.setAlignment(QtCore.Qt.AlignCenter)

        stability_label = QtGui.QLabel('Electron number stability (%):')
        stability_label.setAlignment(QtCore.Qt.AlignCenter)

        self.stability_widget = QtGui.QLabel('< --- >')
        self.stability_widget.setAlignment(QtCore.Qt.AlignCenter)

        self.accept_btn = QtGui.QPushButton('Done', self)
        self.accept_btn.clicked.connect(self.accept)

        properties_layout = QtGui.QGridLayout()
        properties_layout.addWidget(count_label, 0, 0, 1, 1)
        properties_layout.addWidget(self.count_widget, 0, 1, 1, 1)
        properties_layout.addWidget(stability_label, 1, 0, 1, 1)
        properties_layout.addWidget(self.stability_widget, 1, 1, 1, 1)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.directory_finder)
        self.layout.addWidget(self.progress_bar)
        self.layout.addLayout(properties_layout)
        self.layout.addWidget(self.accept_btn)
        self.setLayout(self.layout)
    
    def evaluate_beam_properties(self):
        directory = QtGui.QFileDialog.getExistingDirectory(caption = 'Select directory')
        if not directory:   # e.g. ''
            return
        
        self.progress_bar.show()
        self._calculation_progress.emit(0)
        properties = beam_properties(directory, callback = self._calculation_progress.emit)
        self._calculation_progress.emit(100)

        self.count_widget.setText('{:.3e}'.format(properties['count']))
        self.stability_widget.setText('{:.2f}'.format(properties['stability']))