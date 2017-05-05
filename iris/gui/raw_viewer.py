from os.path import dirname, join

import pyqtgraph as pg
from . import QtCore, QtGui

image_folder = join(dirname(__file__), 'images')

class RawDataViewer(QtGui.QWidget):

    error_message_signal = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_info = dict()

        self.raw_viewer = pg.ImageView(parent = self)
        self.menu_bar = QtGui.QMenuBar(parent = self)
        self.processing_progress_bar = QtGui.QProgressBar(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.display_btn = QtGui.QPushButton('Display')

        self.process_btn = QtGui.QPushButton('Processing')
        self.process_btn.clicked.connect(self.process_dataset_signal)

        process_bar = QtGui.QGridLayout()
        process_bar.addWidget(self.process_btn,0,0)
        process_bar.addWidget(self.processing_progress_bar, 0, 1, 1, 4)

        # Navigating through raw data
        self.timedelay_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.timedelay_widget.setMinimum(0)
        self.timedelay_widget.setTracking(True)
        self.timedelay_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.timedelay_widget.setTickInterval(1)
        self.timedelay_widget.sliderMoved.connect(lambda x: self.update_display())

        self.scan_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.scan_widget.setMinimum(0)
        self.scan_widget.setTracking(True)
        self.scan_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.scan_widget.setTickInterval(1)
        self.scan_widget.sliderMoved.connect(lambda x: self.update_display())

        command_bar = QtGui.QHBoxLayout()
        command_bar.addWidget(QtGui.QLabel('Time-delay (ps):'))
        command_bar.addWidget(self.timedelay_widget)
        command_bar.addWidget(QtGui.QLabel('Scan number:'))
        command_bar.addWidget(self.scan_widget)
        command_bar.addWidget(self.display_btn)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(process_bar)
        self.layout.addWidget(self.raw_viewer)
        self.layout.addLayout(command_bar)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(dict)
    def update_dataset_info(self, info_dict):
        """ Update the range of possible values of times and scans """
        self.dataset_info.update(info_dict)

        # Update range of timedelay and scan
        self.timedelay_widget.setMaximum(len(self.dataset_info['time_points']) - 1)
        self.scan_widget.setMaximum(len(self.dataset_info['nscans']) - 1)
    
    @QtCore.pyqtSlot()
    def update_display(self):
        """ Request an update on the raw data display """
        timedelay = self.dataset_info['time_points'][self.timedelay_widget.value()]
        scan = self.dataset_info['nscans'][self.scan_widget.value()]
        self.display_raw_data_signal.emit(float(timedelay), int(scan))
        
    @QtCore.pyqtSlot(object)
    def display(self, data):
        """ Display a single diffraction pattern. """
        data[data < 0] = 0
        self.raw_viewer.setImage(data, autoLevels = False, autoRange = True)
