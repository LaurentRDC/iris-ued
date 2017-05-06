"""
Control bar for all Iris's controls

"""
from collections import Iterable
from contextlib import suppress
from . import QtGui, QtCore
from skued.baseline import ALL_COMPLEX_WAV, ALL_FIRST_STAGE

class ControlBar(QtGui.QWidget):
    """
    Signals
    -------
    raw_data_request[int, int]

    averaged_data_request[int]

    process_dataset[]

    promote_to_powder[]

    enable_peak_dynamics[bool]

    baseline_computation_parameters[dict]

    notes_updated[str]

    Slots
    -----
    enable_raw_dataset_controls [bool]

    enable_diffraction_dataset_controls [bool]

    enable_powder_diffraction_dataset_controls [bool]
    """
    raw_data_request = QtCore.pyqtSignal(int, int)  # timedelay index, scan
    averaged_data_request = QtCore.pyqtSignal(int)  # timedelay index
    process_dataset = QtCore.pyqtSignal()
    promote_to_powder = QtCore.pyqtSignal()
    enable_peak_dynamics = QtCore.pyqtSignal(bool)
    baseline_computation_parameters = QtCore.pyqtSignal(dict)
    notes_updated = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.raw_dataset_controls = RawDatasetControl(parent = self)
        self.raw_dataset_controls.timedelay_widget.sliderMoved.connect(self.request_raw_data)
        self.raw_dataset_controls.scan_widget.sliderMoved.connect(self.request_raw_data)
        self.raw_dataset_controls.process_btn.clicked.connect(lambda x: self.process_dataset.emit())

        self.diffraction_dataset_controls = DiffractionDatasetControl(parent = self)
        self.diffraction_dataset_controls.timedelay_widget.sliderMoved.connect(self.averaged_data_request)
        self.diffraction_dataset_controls.show_pd_btn.toggled.connect(self.enable_peak_dynamics)
        self.diffraction_dataset_controls.promote_to_powder_btn.clicked.connect(lambda x: self.promote_to_powder.emit())

        self.powder_diffraction_dataset_controls = PowderDiffractionDatasetControl(parent = self)
        self.powder_diffraction_dataset_controls.compute_baseline_btn.clicked.connect(self.request_baseline_computation)

        self.notes_editor = NotesEditor(parent = self)
        self.notes_editor.notes_updated.connect(self.notes_updated)

        self.metadata_widget = QtGui.QTableWidget(parent = self)
        self.metadata_widget.setColumnCount(2)
        self.metadata_widget.horizontalHeader().hide()
        self.metadata_widget.verticalHeader().hide()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.raw_dataset_controls)
        layout.addWidget(self.diffraction_dataset_controls)
        layout.addWidget(self.powder_diffraction_dataset_controls)
        layout.addWidget(self.metadata_widget)
        layout.addWidget(self.notes_editor)
        self.setLayout(layout)
        self.resize(self.minimumSize())
    
    @QtCore.pyqtSlot(dict)
    def update_raw_dataset_metadata(self, metadata):
        self.raw_dataset_controls.timedelay_widget.setRange(0, len(metadata['time_points']) - 1)
        self.raw_dataset_controls.scan_widget.setRange(1, len(metadata['nscans']))
    
    @QtCore.pyqtSlot(dict)
    def update_dataset_metadata(self, metadata):
        self.diffraction_dataset_controls.timedelay_widget.setRange(0, len(metadata['time_points']))
        self.notes_editor.editor.setPlainText(metadata.pop('notes', 'No notes available'))

        self.metadata_widget.clear()
        self.metadata_widget.setRowCount(len(metadata) - 1 if 'notes' in metadata else len(metadata))
        for row, (key, value) in enumerate(metadata.items()):
            if isinstance(value, Iterable) and key not in ('acquisition_date', 'sample_type'):
                if len(value) > 4:
                    key += ' (length)'
                    value = len(tuple(value))
            self.metadata_widget.setItem(row, 0, QtGui.QTableWidgetItem(key))
            self.metadata_widget.setItem(row, 1, QtGui.QTableWidgetItem(str(value)))
        
        self.metadata_widget.resizeColumnsToContents()

    @QtCore.pyqtSlot(int)
    def update_processing_progress(self, value):
        self.raw_dataset_controls.processing_progress_bar.setValue(value)

    @QtCore.pyqtSlot(int)
    def request_raw_data(self, wtv):
        self.raw_data_request.emit(self.raw_dataset_controls.timedelay_widget.value(), 
                                   self.raw_dataset_controls.scan_widget.value())
    
    @QtCore.pyqtSlot()
    def request_baseline_computation(self):
        self.baseline_computation_parameters.emit(self.powder_diffraction_dataset_controls.baseline_parameters())
    
    @QtCore.pyqtSlot(bool)
    def enable_raw_dataset_controls(self, enable):
        self.raw_dataset_controls.setEnabled(enable)
    
    @QtCore.pyqtSlot(bool)
    def enable_diffraction_dataset_controls(self, enable):
        self.diffraction_dataset_controls.setEnabled(enable)
        self.notes_editor.setEnabled(enable)
    
    @QtCore.pyqtSlot(bool)
    def enable_powder_diffraction_dataset_controls(self, enable):
        self.powder_diffraction_dataset_controls.setEnabled(enable)
        self.notes_editor.setEnabled(enable)

class RawDatasetControl(QtGui.QWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #############################
        # Navigating through raw data
        # TODO: show time point next to sliders
        self.timedelay_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.timedelay_widget.setMinimum(0)
        self.timedelay_widget.setTracking(False)
        self.timedelay_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.timedelay_widget.setTickInterval(1)
        td_label = QtGui.QLabel('Time-delay: ')
        td_label.setAlignment(QtCore.Qt.AlignCenter)

        self.scan_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.scan_widget.setMinimum(0)
        self.scan_widget.setTracking(False)
        self.scan_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.scan_widget.setTickInterval(1)
        s_label = QtGui.QLabel('Scan: ')
        s_label.setAlignment(QtCore.Qt.AlignCenter)

        sliders = QtGui.QGridLayout()
        sliders.addWidget(td_label, 0, 0, 1, 1)
        sliders.addWidget(s_label, 1, 0, 1, 1)
        sliders.addWidget(self.timedelay_widget, 0, 1, 1, 2)
        sliders.addWidget(self.scan_widget, 1, 1, 1, 2)

        #####################
        # Processing raw data
        self.process_btn = QtGui.QPushButton('Processing')
        self.processing_progress_bar = QtGui.QProgressBar(parent = self)

        processing = QtGui.QGridLayout()
        processing.addWidget(self.process_btn, 0, 0, 1, 1)
        processing.addWidget(self.processing_progress_bar, 0, 1, 1, 2)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(sliders)
        layout.addLayout(processing)
        self.setLayout(layout)
        self.resize(self.minimumSize())

class DiffractionDatasetControl(QtGui.QWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ################################
        # Diffraction dataset navigation
        # TODO: show time point next to sliders
        self.timedelay_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.timedelay_widget.setMinimum(0)
        self.timedelay_widget.setTracking(False)
        self.timedelay_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.timedelay_widget.setTickInterval(1)
        td_label = QtGui.QLabel('Time-delay: ')
        td_label.setAlignment(QtCore.Qt.AlignCenter)

        sliders = QtGui.QGridLayout()
        sliders.addWidget(td_label, 0, 0, 1, 1)
        sliders.addWidget(self.timedelay_widget, 0, 1, 1, 2)

        ################################
        # Enable/disable time-series ROI
        self.show_pd_btn = QtGui.QPushButton('Show/hide peak dynamics', parent = self)
        self.show_pd_btn.setCheckable(True)
        self.show_pd_btn.setChecked(False)

        ################################
        # Promote DiffractionDataset to PowderDiffractionDataset
        self.promote_to_powder_btn = QtGui.QPushButton('Promote dataset to powder', parent = self)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(self.show_pd_btn)
        btns.addWidget(self.promote_to_powder_btn)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(sliders)
        layout.addLayout(btns)
        self.setLayout(layout)
        self.resize(self.minimumSize())

class PowderDiffractionDatasetControl(QtGui.QWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ######################
        # baseline computation
        self.first_stage_cb = QtGui.QComboBox()
        self.first_stage_cb.addItems(ALL_FIRST_STAGE)

        self.wavelet_cb = QtGui.QComboBox()
        self.wavelet_cb.addItems(ALL_COMPLEX_WAV)

        first_stage_label = QtGui.QLabel('First stage wav.:', parent = self)
        first_stage_label.setAlignment(QtCore.Qt.AlignCenter)

        wavelet_label = QtGui.QLabel('Dual-tree wavelet:', parent = self)
        wavelet_label.setAlignment(QtCore.Qt.AlignCenter)

        self.compute_baseline_btn = QtGui.QPushButton('Compute baseline', parent = self)

        self.baseline_removed_btn = QtGui.QPushButton('Show baseline-removed', parent = self)
        self.baseline_removed_btn.setCheckable(True)
        self.baseline_removed_btn.setChecked(False)

        baseline = QtGui.QGridLayout()
        baseline.addWidget(self.baseline_removed_btn, 0, 0, 1, 2)
        baseline.addWidget(first_stage_label, 1, 0, 1, 1)
        baseline.addWidget(self.first_stage_cb, 1, 1, 1, 1)
        baseline.addWidget(wavelet_label, 2, 0, 1, 1)
        baseline.addWidget(self.wavelet_cb, 2, 1, 1, 1)
        baseline.addWidget(self.compute_baseline_btn, 3, 0, 1, 2)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(baseline)
        self.setLayout(layout)
        self.resize(self.minimumSize())
    
    def baseline_parameters(self):
        """ Returns a dictionary of baseline-computation parameters """
        return {'first_stage': self.first_stage_cb.currentText(),
                'wavelet': self.wavelet_cb.currentText(),
                'level': 'max', 'max_iter': 100}

class NotesEditor(QtGui.QWidget):

    notes_updated = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        title = QtGui.QLabel('Dataset notes and remarks', parent = self)
        title.setAlignment(QtCore.Qt.AlignCenter)

        update_btn = QtGui.QPushButton('Update notes', self)
        update_btn.clicked.connect(self.update_notes)

        self.editor = QtGui.QTextEdit(parent = self)
        self.editor.setLineWrapMode(3)  # Fixed column width
        self.editor.setLineWrapColumnOrWidth(80)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.editor)
        layout.addWidget(update_btn)

        self.setLayout(layout)
        self.resize(self.minimumSize())
    
    @QtCore.pyqtSlot()
    def update_notes(self):
        self.notes_updated.emit(self.editor.toPlainText())