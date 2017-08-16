"""
Control bar for all Iris's controls
"""
from collections import Iterable
from contextlib import suppress
from pyqtgraph import QtGui, QtCore
from skued.baseline import ALL_COMPLEX_WAV, ALL_FIRST_STAGE

try:
    from PyQt5.QtWinExtras import QWinTaskbarProgress
    WITH_TASKBAR = True
except ImportError:
    WITH_TASKBAR = False

from pywt import Modes

class ControlBar(QtGui.QWidget):
    
    raw_data_request = QtCore.pyqtSignal(int, int)  # timedelay index, scan
    averaged_data_request = QtCore.pyqtSignal(int)  # timedelay index

    enable_peak_dynamics = QtCore.pyqtSignal(bool)
    baseline_removed = QtCore.pyqtSignal(bool)
    relative_powder = QtCore.pyqtSignal(bool)
    relative_averaged = QtCore.pyqtSignal(bool)
    baseline_computation_parameters = QtCore.pyqtSignal(dict)
    time_zero_shift = QtCore.pyqtSignal(float)
    notes_updated = QtCore.pyqtSignal(str)
    enable_connect_time_series = QtCore.pyqtSignal(bool)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.raw_dataset_controls = RawDatasetControl(parent = self)
        self.raw_dataset_controls.timedelay_widget.valueChanged.connect(self.request_raw_data)
        self.raw_dataset_controls.scan_widget.valueChanged.connect(self.request_raw_data)

        self.diffraction_dataset_controls = DiffractionDatasetControl(parent = self)
        self.diffraction_dataset_controls.timedelay_widget.valueChanged.connect(self.averaged_data_request)
        self.diffraction_dataset_controls.show_pd_btn.toggled.connect(self.enable_peak_dynamics)
        self.diffraction_dataset_controls.relative_btn.toggled.connect(self.relative_averaged)
        self.diffraction_dataset_controls.time_zero_shift_widget.editingFinished.connect(self.shift_time_zero)
        self.diffraction_dataset_controls.clear_time_zero_shift_btn.clicked.connect(lambda _: self.time_zero_shift.emit(0))

        self.powder_diffraction_dataset_controls = PowderDiffractionDatasetControl(parent = self)
        self.powder_diffraction_dataset_controls.compute_baseline_btn.clicked.connect(self.request_baseline_computation)
        self.powder_diffraction_dataset_controls.baseline_removed_btn.toggled.connect(self.baseline_removed)
        self.powder_diffraction_dataset_controls.relative_btn.toggled.connect(self.relative_powder)

        self.progress_bar = QtGui.QProgressBar(self)
        if WITH_TASKBAR:
            self.win_progress_bar = QWinTaskbarProgress(self)
            self.win_progress_bar.setVisible(True)
            self.progress_bar.valueChanged.connect(self.win_progress_bar.setValue)

        self.metadata_widget = MetadataWidget(parent = self)

        self.notes_editor = NotesEditor(parent = self)
        self.notes_editor.notes_updated.connect(self.notes_updated)

        self.stack = QtGui.QTabWidget(parent = self)
        self.stack.addTab(self.metadata_widget, 'Dataset metadata')
        self.stack.addTab(self.notes_editor, 'Dataset notes')

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.raw_dataset_controls)
        layout.addWidget(self.diffraction_dataset_controls)
        layout.addWidget(self.powder_diffraction_dataset_controls)
        layout.addWidget(self.stack)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        for frame in (self.raw_dataset_controls, self.diffraction_dataset_controls, self.powder_diffraction_dataset_controls):
            frame.setFrameShadow(QtGui.QFrame.Sunken)
            frame.setFrameShape(QtGui.QFrame.Panel)

        self.setMaximumWidth(self.notes_editor.maximumWidth())
    
    @QtCore.pyqtSlot(dict)
    def update_raw_dataset_metadata(self, metadata):
        self.raw_dataset_controls.update_dataset_metadata(metadata)
    
    @QtCore.pyqtSlot(dict)
    def update_dataset_metadata(self, metadata):
        self.diffraction_dataset_controls.update_dataset_metadata(metadata)
        self.notes_editor.editor.setPlainText(metadata.pop('notes', 'No notes available'))
        self.metadata_widget.set_metadata(metadata)

    @QtCore.pyqtSlot(int)
    def update_processing_progress(self, value):
        self.progress_bar.setValue(value)
        
    
    @QtCore.pyqtSlot(int)
    def update_powder_promotion_progress(self, value):
        self.progress_bar.setValue(value)

    @QtCore.pyqtSlot(int)
    def update_angular_average_progress(self, value):
        self.progress_bar.setValue(value)

    @QtCore.pyqtSlot(int)
    def request_raw_data(self, wtv):
        self.raw_data_request.emit(self.raw_dataset_controls.timedelay_widget.value(), 
                                   self.raw_dataset_controls.scan_widget.value() + 1) #scans are numbered starting at 1
    
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
        self.metadata_widget.setEnabled(enable)
    
    @QtCore.pyqtSlot(bool)
    def enable_powder_diffraction_dataset_controls(self, enable):
        self.powder_diffraction_dataset_controls.setEnabled(enable)
        self.notes_editor.setEnabled(enable)
        self.metadata_widget.setEnabled(enable)
    
    @QtCore.pyqtSlot()
    def shift_time_zero(self):
        shift = self.diffraction_dataset_controls.time_zero_shift_widget.value()
        self.time_zero_shift.emit(shift)

class RawDatasetControl(QtGui.QFrame):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #############################
        # Navigating through raw data
        self.td_label = QtGui.QLabel('Time-delay: ')
        self.td_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timedelay_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.timedelay_widget.setMinimum(0)
        self.timedelay_widget.setTracking(False)
        self.timedelay_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.timedelay_widget.setTickInterval(1)
        self.timedelay_widget.sliderMoved.connect(
            lambda pos: self.td_label.setText('Time-delay: {:.3f}ps'.format(self.time_points[pos])))

        self.s_label = QtGui.QLabel('Scan: ')
        self.s_label.setAlignment(QtCore.Qt.AlignCenter)
        self.scan_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.scan_widget.setMinimum(0)
        self.scan_widget.setTracking(False)
        self.scan_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.scan_widget.setTickInterval(1)
        self.scan_widget.sliderMoved.connect(
            lambda pos: self.s_label.setText('Scan: {:d}'.format(self.nscans[pos])))

        prev_timedelay_btn = QtGui.QPushButton('<', self)
        prev_timedelay_btn.clicked.connect(self.goto_prev_timedelay)

        next_timedelay_btn = QtGui.QPushButton('>', self)
        next_timedelay_btn.clicked.connect(self.goto_next_timedelay)

        prev_scan_btn = QtGui.QPushButton('<', self)
        prev_scan_btn.clicked.connect(self.goto_prev_scan)

        next_scan_btn = QtGui.QPushButton('>', self)
        next_scan_btn.clicked.connect(self.goto_next_scan)

        time_layout = QtGui.QHBoxLayout()
        time_layout.addWidget(self.td_label)
        time_layout.addWidget(self.timedelay_widget)
        time_layout.addWidget(prev_timedelay_btn)
        time_layout.addWidget(next_timedelay_btn)

        scan_layout = QtGui.QHBoxLayout()
        scan_layout.addWidget(self.s_label)
        scan_layout.addWidget(self.scan_widget)
        scan_layout.addWidget(prev_scan_btn)
        scan_layout.addWidget(next_scan_btn)

        sliders = QtGui.QVBoxLayout()
        sliders.addLayout(time_layout)
        sliders.addLayout(scan_layout)

        title = QtGui.QLabel('<h2>Raw dataset controls<\h2>')
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(sliders)
        self.setLayout(layout)
        self.resize(self.minimumSize())
    
    def update_dataset_metadata(self, metadata):
        self.time_points = metadata.get('time_points')
        self.nscans = metadata.get('nscans')

        self.timedelay_widget.setRange(0, len(self.time_points) - 1)
        self.scan_widget.setRange(0, len(self.nscans) - 1)
        self.timedelay_widget.triggerAction(5)
        self.timedelay_widget.sliderMoved.emit(0)
        self.scan_widget.triggerAction(5)
        self.scan_widget.sliderMoved.emit(0)

    @QtCore.pyqtSlot()
    def goto_prev_timedelay(self):
        self.timedelay_widget.setSliderDown(True)
        self.timedelay_widget.triggerAction(2)
        self.timedelay_widget.setSliderDown(False)
    
    @QtCore.pyqtSlot()
    def goto_next_timedelay(self):
        self.timedelay_widget.setSliderDown(True)
        self.timedelay_widget.triggerAction(1)
        self.timedelay_widget.setSliderDown(False)

    @QtCore.pyqtSlot()
    def goto_prev_scan(self):
        self.scan_widget.setSliderDown(True)
        self.scan_widget.triggerAction(2)
        self.scan_widget.setSliderDown(False)
    
    @QtCore.pyqtSlot()
    def goto_next_scan(self):
        self.scan_widget.setSliderDown(True)
        self.scan_widget.triggerAction(1)
        self.scan_widget.setSliderDown(False)

class DiffractionDatasetControl(QtGui.QFrame):

    time_zero_shift = QtCore.pyqtSignal(float)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ################################
        # Diffraction dataset navigation
        self.td_label = QtGui.QLabel('Time-delay: ')
        self.td_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timedelay_widget = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.timedelay_widget.setMinimum(0)
        self.timedelay_widget.setTracking(False)
        self.timedelay_widget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.timedelay_widget.setTickInterval(1)
        self.timedelay_widget.sliderMoved.connect(
            lambda pos: self.td_label.setText('Time-delay: {:.3f}ps'.format(self.time_points[pos])))

        # Time-zero shift control
        # QDoubleSpinbox does not have a slot that sets the value without notifying everybody
        self.time_zero_shift_widget = QtGui.QDoubleSpinBox(parent = self)
        self.time_zero_shift_widget.setRange(-1000, 1000)
        self.time_zero_shift_widget.setDecimals(3)
        self.time_zero_shift_widget.setSingleStep(0.5)
        self.time_zero_shift_widget.setSuffix(' ps')
        self.time_zero_shift_widget.setValue(0.0)

        self.clear_time_zero_shift_btn = QtGui.QPushButton('Clear time-zero shift', parent = self)

        prev_btn = QtGui.QPushButton('<', self)
        prev_btn.clicked.connect(self.goto_prev)

        next_btn = QtGui.QPushButton('>', self)
        next_btn.clicked.connect(self.goto_next)

        sliders = QtGui.QHBoxLayout()
        sliders.addWidget(self.td_label)
        sliders.addWidget(self.timedelay_widget)
        sliders.addWidget(prev_btn)
        sliders.addWidget(next_btn)

        self.show_pd_btn = QtGui.QPushButton('Show/hide peak dynamics', parent = self)
        self.show_pd_btn.setCheckable(True)
        self.show_pd_btn.setChecked(False)

        self.relative_btn = QtGui.QPushButton('show relative data (?)', parent = self)
        self.relative_btn.setToolTip('Subtract pre-time-zero average from the data.')
        self.relative_btn.setCheckable(True)
        self.relative_btn.setChecked(False)

        display_controls = QtGui.QGroupBox(title = 'Display options', parent = self)
        display_controls_layout = QtGui.QHBoxLayout()
        display_controls_layout.addWidget(self.show_pd_btn)
        display_controls_layout.addWidget(self.relative_btn)
        display_controls.setLayout(display_controls_layout)

        time_zero_shift_layout = QtGui.QHBoxLayout()
        time_zero_shift_layout.addWidget(QtGui.QLabel('Time-zero shift: ', parent = self))
        time_zero_shift_layout.addWidget(self.time_zero_shift_widget)
        time_zero_shift_layout.addWidget(self.clear_time_zero_shift_btn)

        title = QtGui.QLabel('<h2>Diffraction dataset controls<\h2>')
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(sliders)
        layout.addLayout(time_zero_shift_layout)
        layout.addWidget(display_controls)
        self.setLayout(layout)
        self.resize(self.minimumSize())
    
    def update_dataset_metadata(self, metadata):
        self.time_points = metadata.get('time_points')
        t0_shift = metadata.get('time_zero_shift')

        self.timedelay_widget.setRange(0, len(self.time_points) - 1)
        self.timedelay_widget.triggerAction(5) # SliderToMinimum
        self.timedelay_widget.sliderMoved.emit(0)

        if t0_shift != self.time_zero_shift_widget.value():
            self.time_zero_shift_widget.setValue(t0_shift)
    
    @QtCore.pyqtSlot()
    def goto_prev(self):
        self.timedelay_widget.setSliderDown(True)
        self.timedelay_widget.triggerAction(2)
        self.timedelay_widget.setSliderDown(False)
    
    @QtCore.pyqtSlot()
    def goto_next(self):
        self.timedelay_widget.setSliderDown(True)
        self.timedelay_widget.triggerAction(1)
        self.timedelay_widget.setSliderDown(False)

class PowderDiffractionDatasetControl(QtGui.QFrame):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ######################
        # baseline computation
        self.first_stage_cb = QtGui.QComboBox()
        self.first_stage_cb.addItems(ALL_FIRST_STAGE)
        if 'sym6' in ALL_FIRST_STAGE:
            self.first_stage_cb.setCurrentText('sym6')

        self.wavelet_cb = QtGui.QComboBox()
        self.wavelet_cb.addItems(ALL_COMPLEX_WAV)
        if 'qshift3' in ALL_COMPLEX_WAV:
            self.wavelet_cb.setCurrentText('qshift3')

        self.mode_cb = QtGui.QComboBox()
        self.mode_cb.addItems(Modes.modes)
        if 'smooth' in Modes.modes:
            self.mode_cb.setCurrentText('constant')
        
        self.max_iter_widget = QtGui.QSpinBox()
        self.max_iter_widget.setRange(0, 1000)
        self.max_iter_widget.setValue(100)

        self.compute_baseline_btn = QtGui.QPushButton('Compute baseline', parent = self)

        self.baseline_removed_btn = QtGui.QPushButton('Show baseline-removed', parent = self)
        self.baseline_removed_btn.setCheckable(True)
        self.baseline_removed_btn.setChecked(False)

        self.relative_btn = QtGui.QPushButton('Show relative', parent = self)
        self.relative_btn.setToolTip('Subtract pre-time-zero average from the data.')
        self.relative_btn.setCheckable(True)
        self.relative_btn.setChecked(False)

        baseline_controls = QtGui.QFormLayout()
        baseline_controls.addRow('First stage wavelet: ', self.first_stage_cb)
        baseline_controls.addRow('Dual-tree wavelet: ', self.wavelet_cb)
        baseline_controls.addRow('Extensions mode: ', self.mode_cb)
        baseline_controls.addRow('Iterations: ', self.max_iter_widget)
        baseline_controls.addRow(self.compute_baseline_btn)

        baseline_computation = QtGui.QGroupBox(title = 'Baseline parameters', parent = self)
        baseline_computation.setLayout(baseline_controls)

        display_controls = QtGui.QGroupBox(title = 'Display options', parent = self)
        display_controls_layout = QtGui.QHBoxLayout()
        display_controls_layout.addWidget(self.baseline_removed_btn)
        display_controls_layout.addWidget(self.relative_btn)
        display_controls.setLayout(display_controls_layout)

        # TODO: add callback and progressbar for computing the baseline?

        title = QtGui.QLabel('<h2>Powder dataset controls<\h2>')
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(baseline_computation)
        layout.addWidget(display_controls)
        self.setLayout(layout)
        self.resize(self.minimumSize())
    
    def baseline_parameters(self):
        """ Returns a dictionary of baseline-computation parameters """
        return {'first_stage': self.first_stage_cb.currentText(),
                'wavelet': self.wavelet_cb.currentText(),
                'mode': self.mode_cb.currentText(),
                'max_iter': self.max_iter_widget.value(),
                'level': None,
                'callback': lambda : self.baseline_removed_btn.setChecked(True)}

class MetadataWidget(QtGui.QWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        title = QtGui.QLabel('<h2>Dataset metadata<\h2>', parent = self)
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        self.table = QtGui.QTableWidget(parent = self)
        self.table.setColumnCount(2)
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)   #no edit triggers, see QAbstractItemViews

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.resize(self.minimumSize())
    
    @QtCore.pyqtSlot(dict)
    def set_metadata(self, metadata):
        self.table.clear()
        self.table.setRowCount(len(metadata) - 1 if 'notes' in metadata else len(metadata))
        for row, (key, value) in enumerate(metadata.items()):
            if isinstance(value, Iterable) and (not isinstance(value, str)):
                if len(value) > 4:
                    key += ' (length)'
                    value = len(tuple(value))

            self.table.setItem(row, 0, QtGui.QTableWidgetItem(key))
            self.table.setItem(row, 1, QtGui.QTableWidgetItem(str(value)))
        
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

class NotesEditor(QtGui.QFrame):

    notes_updated = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        title = QtGui.QLabel('<h2>Dataset notes and remarks<\h2>', parent = self)
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        update_btn = QtGui.QPushButton('Update notes', self)
        update_btn.clicked.connect(self.update_notes)

        self.editor = QtGui.QTextEdit(parent = self)

        # Set editor size such that 60 characters will fit
        font_info = QtGui.QFontInfo(self.editor.currentFont())
        self.editor.setMaximumWidth(40*font_info.pixelSize())
        self.editor.setLineWrapMode(QtGui.QTextEdit.WidgetWidth)  # widget width

        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.editor)
        layout.addWidget(update_btn)
        self.setLayout(layout)
        self.setMaximumWidth(self.editor.maximumWidth())
        self.resize(self.minimumSize())
    
    @QtCore.pyqtSlot()
    def update_notes(self):
        self.notes_updated.emit(self.editor.toPlainText())