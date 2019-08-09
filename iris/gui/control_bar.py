# -*- coding: utf-8 -*-
"""
Control bar for all Iris's controls
"""
from collections import Iterable

from PyQt5 import QtCore, QtGui, QtWidgets
from pywt import Modes

from skued import available_dt_filters, available_first_stage_filters


class ControlBar(QtWidgets.QWidget):

    raw_data_request = QtCore.pyqtSignal(int, int)  # timedelay index, scan
    averaged_data_request = QtCore.pyqtSignal(int)  # timedelay index

    baseline_computation_parameters = QtCore.pyqtSignal(dict)
    time_zero_shift = QtCore.pyqtSignal(float)
    notes_updated = QtCore.pyqtSignal(str)
    enable_connect_time_series = QtCore.pyqtSignal(bool)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.raw_dataset_controls = RawDatasetControl(parent=self)
        self.raw_dataset_controls.timedelay_widget.valueChanged.connect(
            self.request_raw_data
        )
        self.raw_dataset_controls.scan_widget.valueChanged.connect(
            self.request_raw_data
        )
        #        self.raw_dataset_controls.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.diffraction_dataset_controls = DiffractionDatasetControl(parent=self)
        self.diffraction_dataset_controls.timedelay_widget.valueChanged.connect(
            self.averaged_data_request
        )
        self.diffraction_dataset_controls.time_zero_shift_widget.editingFinished.connect(
            self.shift_time_zero
        )
        self.diffraction_dataset_controls.clear_time_zero_shift_btn.clicked.connect(
            lambda _: self.time_zero_shift.emit(0)
        )
        #        self.diffraction_dataset_controls.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.powder_diffraction_dataset_controls = PowderDiffractionDatasetControl(
            parent=self
        )
        self.powder_diffraction_dataset_controls.compute_baseline_btn.clicked.connect(
            self.request_baseline_computation
        )
        #        self.powder_diffraction_dataset_controls.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.metadata_widget = MetadataWidget(parent=self)

        self.notes_editor = NotesEditor(parent=self)
        self.notes_editor.notes_updated.connect(self.notes_updated)

        self.metadata_and_notes_widget = QtWidgets.QFrame(parent=self)
        _layout = QtWidgets.QVBoxLayout()
        _layout.addWidget(self.metadata_widget)
        _layout.addWidget(self.notes_editor)
        self.metadata_and_notes_widget.setLayout(_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.raw_dataset_controls)
        layout.addWidget(self.diffraction_dataset_controls)
        layout.addWidget(self.powder_diffraction_dataset_controls)
        layout.addWidget(self.metadata_and_notes_widget)
        layout.addStretch()
        self.setLayout(layout)

        for frame in (
            self.raw_dataset_controls,
            self.diffraction_dataset_controls,
            self.powder_diffraction_dataset_controls,
            self.metadata_and_notes_widget,
        ):
            frame.setFrameShadow(QtWidgets.QFrame.Sunken)
            frame.setFrameShape(QtWidgets.QFrame.Panel)

        self.setMaximumWidth(self.notes_editor.maximumWidth())

    @QtCore.pyqtSlot(dict)
    def update_raw_dataset_metadata(self, metadata):
        self.raw_dataset_controls.update_dataset_metadata(metadata)

    @QtCore.pyqtSlot(dict)
    def update_dataset_metadata(self, metadata):
        self.diffraction_dataset_controls.update_dataset_metadata(metadata)
        self.notes_editor.editor.setPlainText(
            metadata.pop("notes", "No notes available")
        )
        self.metadata_widget.set_metadata(metadata)

    @QtCore.pyqtSlot(int)
    def request_raw_data(self, wtv):
        self.raw_data_request.emit(
            self.raw_dataset_controls.timedelay_widget.value(),
            self.raw_dataset_controls.scan_widget.value() + 1,
        )  # scans are numbered starting at 1

    @QtCore.pyqtSlot()
    def request_baseline_computation(self):
        self.baseline_computation_parameters.emit(
            self.powder_diffraction_dataset_controls.baseline_parameters()
        )

    @QtCore.pyqtSlot()
    def shift_time_zero(self):
        shift = self.diffraction_dataset_controls.time_zero_shift_widget.value()
        self.time_zero_shift.emit(shift)


class RawDatasetControl(QtWidgets.QFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #############################
        # Navigating through raw data
        self.td_label = QtWidgets.QLabel("Time-delay: ")
        self.td_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timedelay_widget = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent=self)
        self.timedelay_widget.setMinimum(0)
        self.timedelay_widget.setTracking(False)
        self.timedelay_widget.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.timedelay_widget.setTickInterval(1)
        self.timedelay_widget.sliderMoved.connect(
            lambda pos: self.td_label.setText(
                "Time-delay: {:.3f}ps".format(self.time_points[pos])
            )
        )

        self.s_label = QtWidgets.QLabel("Scan: ")
        self.s_label.setAlignment(QtCore.Qt.AlignCenter)
        self.scan_widget = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent=self)
        self.scan_widget.setMinimum(0)
        self.scan_widget.setTracking(False)
        self.scan_widget.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.scan_widget.setTickInterval(1)
        self.scan_widget.sliderMoved.connect(
            lambda pos: self.s_label.setText("Scan: {:d}".format(self.scans[pos]))
        )

        prev_timedelay_btn = QtWidgets.QPushButton("<", self)
        prev_timedelay_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        prev_timedelay_btn.clicked.connect(self.goto_prev_timedelay)

        next_timedelay_btn = QtWidgets.QPushButton(">", self)
        next_timedelay_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        next_timedelay_btn.clicked.connect(self.goto_next_timedelay)

        prev_scan_btn = QtWidgets.QPushButton("<", self)
        prev_scan_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        prev_scan_btn.clicked.connect(self.goto_prev_scan)

        next_scan_btn = QtWidgets.QPushButton(">", self)
        next_scan_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        next_scan_btn.clicked.connect(self.goto_next_scan)

        time_layout = QtWidgets.QHBoxLayout()
        time_layout.addWidget(self.td_label)
        time_layout.addWidget(self.timedelay_widget)
        time_layout.addWidget(prev_timedelay_btn)
        time_layout.addWidget(next_timedelay_btn)

        scan_layout = QtWidgets.QHBoxLayout()
        scan_layout.addWidget(self.s_label)
        scan_layout.addWidget(self.scan_widget)
        scan_layout.addWidget(prev_scan_btn)
        scan_layout.addWidget(next_scan_btn)

        sliders = QtWidgets.QVBoxLayout()
        sliders.addLayout(time_layout)
        sliders.addLayout(scan_layout)

        title = QtWidgets.QLabel("<h2>Raw dataset controls<\h2>")
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(sliders)
        self.setLayout(layout)
        self.resize(self.minimumSize())

    def update_dataset_metadata(self, metadata):
        self.time_points = metadata.get("time_points")
        self.scans = metadata.get("scans")

        self.timedelay_widget.setRange(0, len(self.time_points) - 1)
        self.scan_widget.setRange(0, len(self.scans) - 1)
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


class DiffractionDatasetControl(QtWidgets.QFrame):

    time_zero_shift = QtCore.pyqtSignal(float)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ################################
        # Diffraction dataset navigation
        self.td_label = QtWidgets.QLabel("Time-delay: ")
        self.td_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timedelay_widget = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent=self)
        self.timedelay_widget.setMinimum(0)
        self.timedelay_widget.setTracking(False)
        self.timedelay_widget.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.timedelay_widget.setTickInterval(1)
        self.timedelay_widget.sliderMoved.connect(
            lambda pos: self.td_label.setText(
                "Time-delay: {:.3f}ps".format(self.time_points[pos])
            )
        )

        # Time-zero shift control
        # QDoubleSpinbox does not have a slot that sets the value without notifying everybody
        self.time_zero_shift_widget = QtWidgets.QDoubleSpinBox(parent=self)
        self.time_zero_shift_widget.setRange(-1000, 1000)
        self.time_zero_shift_widget.setDecimals(3)
        self.time_zero_shift_widget.setSingleStep(0.5)
        self.time_zero_shift_widget.setSuffix(" ps")
        self.time_zero_shift_widget.setValue(0.0)

        self.clear_time_zero_shift_btn = QtWidgets.QPushButton(
            "Clear time-zero shift", parent=self
        )
        self.clear_time_zero_shift_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )

        prev_btn = QtWidgets.QPushButton("<", self)
        prev_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        prev_btn.clicked.connect(self.goto_prev)

        next_btn = QtWidgets.QPushButton(">", self)
        next_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        next_btn.clicked.connect(self.goto_next)

        sliders = QtWidgets.QHBoxLayout()
        sliders.addWidget(self.td_label)
        sliders.addWidget(self.timedelay_widget)
        sliders.addWidget(prev_btn)
        sliders.addWidget(next_btn)

        time_zero_shift_layout = QtWidgets.QHBoxLayout()
        time_zero_shift_layout.addWidget(
            QtWidgets.QLabel("Time-zero shift: ", parent=self)
        )
        time_zero_shift_layout.addWidget(self.time_zero_shift_widget)
        time_zero_shift_layout.addWidget(self.clear_time_zero_shift_btn)

        title = QtWidgets.QLabel("<h2>Diffraction dataset controls<\h2>")
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(sliders)
        layout.addLayout(time_zero_shift_layout)
        self.setLayout(layout)
        self.resize(self.minimumSize())

    def update_dataset_metadata(self, metadata):
        self.time_points = metadata.get("time_points")
        t0_shift = metadata.get("time_zero_shift")

        self.timedelay_widget.setRange(0, len(self.time_points) - 1)
        self.timedelay_widget.triggerAction(5)  # SliderToMinimum
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


class PowderDiffractionDatasetControl(QtWidgets.QFrame):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ######################
        # baseline computation
        self.first_stage_cb = QtWidgets.QComboBox()
        self.first_stage_cb.addItems(available_first_stage_filters())
        if "sym6" in available_first_stage_filters():
            self.first_stage_cb.setCurrentText("sym6")

        self.wavelet_cb = QtWidgets.QComboBox()
        self.wavelet_cb.addItems(available_dt_filters())
        if "qshift3" in available_dt_filters():
            self.wavelet_cb.setCurrentText("qshift3")

        self.mode_cb = QtWidgets.QComboBox()
        self.mode_cb.addItems(Modes.modes)
        if "smooth" in Modes.modes:
            self.mode_cb.setCurrentText("constant")

        self.max_iter_widget = QtWidgets.QSpinBox()
        self.max_iter_widget.setRange(0, 1000)
        self.max_iter_widget.setValue(100)

        self.compute_baseline_btn = QtWidgets.QPushButton(
            "Compute baseline", parent=self
        )
        self.compute_baseline_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )

        baseline_controls = QtWidgets.QFormLayout()
        baseline_controls.addRow("First stage wavelet: ", self.first_stage_cb)
        baseline_controls.addRow("Dual-tree wavelet: ", self.wavelet_cb)
        baseline_controls.addRow("Extensions mode: ", self.mode_cb)
        baseline_controls.addRow("Iterations: ", self.max_iter_widget)
        baseline_controls.addWidget(self.compute_baseline_btn)

        baseline_computation = QtWidgets.QGroupBox(
            title="Baseline parameters", parent=self
        )
        baseline_computation.setLayout(baseline_controls)

        # TODO: add callback and progressbar for computing the baseline?

        title = QtWidgets.QLabel("<h2>Powder dataset controls<\h2>")
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(baseline_computation)
        self.setLayout(layout)
        self.resize(self.minimumSize())

    def baseline_parameters(self):
        """ Returns a dictionary of baseline-computation parameters """
        return {
            "first_stage": self.first_stage_cb.currentText(),
            "wavelet": self.wavelet_cb.currentText(),
            "mode": self.mode_cb.currentText(),
            "max_iter": self.max_iter_widget.value(),
            "level": None,
        }


class MetadataWidget(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        title = QtWidgets.QLabel("<h2>Dataset metadata<\h2>", parent=self)
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        self.table = QtWidgets.QTableWidget(parent=self)
        self.table.setColumnCount(2)
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )  # no edit triggers, see QAbstractItemViews

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.resize(self.minimumSize())

    @QtCore.pyqtSlot(dict)
    def set_metadata(self, metadata):
        self.table.clear()
        self.table.setRowCount(
            len(metadata) - 1 if "notes" in metadata else len(metadata)
        )
        for row, (key, value) in enumerate(sorted(metadata.items())):
            if isinstance(value, Iterable) and (not isinstance(value, str)):
                if len(value) > 4:
                    key += " (length)"
                    value = len(tuple(value))

            value = str(value)
            # We show tool tip on hover because
            # some values are long (e.g. filenames)
            key_item = QtWidgets.QTableWidgetItem(key)
            key_item.setToolTip(key)

            value_item = QtWidgets.QTableWidgetItem(value)
            value_item.setToolTip(value)

            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, value_item)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)


class NotesEditor(QtWidgets.QFrame):

    notes_updated = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        title = QtWidgets.QLabel("<h2>Dataset notes and remarks<\h2>", parent=self)
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        update_btn = QtWidgets.QPushButton("Update notes", self)
        update_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        update_btn.clicked.connect(self.update_notes)

        self.editor = QtWidgets.QTextEdit(parent=self)

        # Set editor size such that 60 characters will fit
        font_info = QtGui.QFontInfo(self.editor.currentFont())
        self.editor.setMaximumWidth(40 * font_info.pixelSize())
        self.editor.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)  # widget width

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.editor)
        layout.addWidget(update_btn, 1, QtCore.Qt.AlignHCenter)
        self.setLayout(layout)
        self.setMaximumWidth(self.editor.maximumWidth())
        self.resize(self.minimumSize())

    @QtCore.pyqtSlot()
    def update_notes(self):
        self.notes_updated.emit(self.editor.toPlainText())
