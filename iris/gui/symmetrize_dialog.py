# -*- coding: utf-8 -*-
"""
Dialog for symmetrization of DiffractionDataset
"""
from os import cpu_count
from skued import autocenter
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from .controller import WorkThread
from .qbusyindicator import QBusyIndicator

description = (
    """Align the circle so that its center is aligned with the diffraction center. """
)


class CircleROIWithCenter(pg.CircleROI):

    # Calculating the center position assumes that PyQtGraph is configured
    # such that imageAxisOrder == 'row-major'
    def center(self):
        corner_x, corner_y = self.pos().x(), self.pos().y()
        radius = self.size().x() / 2
        return (round(corner_x + radius), round(corner_y + radius))

    def set_center(self, x, y):
        radius = self.size().x() / 2
        left = x - radius
        bottom = y - radius
        self.setPos(left, bottom)


class SymmetrizeDialog(QtWidgets.QDialog):
    """
    Modal dialog used to symmetrize datasets.
    """

    error_message_signal = QtCore.pyqtSignal(str)
    symmetrize_parameters_signal = QtCore.pyqtSignal(str, dict)

    def __init__(self, image, mask, center=None, **kwargs):
        super().__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle("Symmetrization")

        # For use with autocenter
        self._image = image
        self._mask = mask

        title = QtWidgets.QLabel("<h2>Symmetrization Options<\\h2>")
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        description_label = QtWidgets.QLabel(description, parent=self)
        description_label.setWordWrap(True)
        description_label.setAlignment(QtCore.Qt.AlignCenter)

        self.viewer = pg.ImageView(parent=self)
        self.viewer.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.viewer.setImage(image)
        self.center_finder = CircleROIWithCenter(
            pos=np.array(image.shape) / 2 - 100, size=[200, 200], pen=pg.mkPen("r")
        )
        if center is not None and center != (0, 0):
            self.center_finder.set_center(*center)

        self.viewer.getView().addItem(self.center_finder)

        self.mod_widget = QtWidgets.QComboBox(parent=self)
        self.mod_widget.addItems(["2", "3", "4", "6"])
        self.mod_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )

        self.smoothing_kernel_widget = QtWidgets.QSpinBox(parent=self)
        self.smoothing_kernel_widget.setRange(0, 100)
        self.smoothing_kernel_widget.setValue(5)
        self.smoothing_kernel_widget.setSuffix(" px")
        self.smoothing_kernel_widget.setEnabled(False)

        self.enable_smoothing_widget = QtWidgets.QCheckBox("Enable gaussian smoothing")
        self.enable_smoothing_widget.setChecked(False)
        self.enable_smoothing_widget.toggled.connect(
            self.smoothing_kernel_widget.setEnabled
        )

        self.processes_widget = QtWidgets.QSpinBox(parent=self)
        self.processes_widget.setRange(1, cpu_count() - 1)
        self.processes_widget.setValue(1)

        self.autocenter_btn = QtWidgets.QPushButton("Autocenter", self)
        self.autocenter_btn.clicked.connect(self.initiate_autocenter)
        self.autocenter_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        self.busy_indicator = QBusyIndicator(parent=self)
        autocenter_layout = QtWidgets.QHBoxLayout()
        autocenter_layout.addWidget(self.autocenter_btn)
        autocenter_layout.addWidget(self.busy_indicator)

        self.accept_btn = QtWidgets.QPushButton("Symmetrize", self)
        self.accept_btn.clicked.connect(self.accept)
        self.accept_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )

        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)
        self.cancel_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )

        self.error_message_signal.connect(self.show_error_message)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.busy_indicator)

        multiplicity_layout = QtWidgets.QFormLayout()
        multiplicity_layout.addRow("Rotational multiplicity: ", self.mod_widget)
        multiplicity_layout.addRow("Number of CPU cores:", self.processes_widget)

        smoothing_layout = QtWidgets.QFormLayout()
        smoothing_layout.addRow(self.enable_smoothing_widget)
        smoothing_layout.addRow(
            "Kernel standard deviation: ", self.smoothing_kernel_widget
        )

        params_layout = QtWidgets.QVBoxLayout()
        params_layout.addWidget(title)
        params_layout.addWidget(description_label)
        params_layout.addLayout(autocenter_layout)
        params_layout.addLayout(multiplicity_layout)
        params_layout.addLayout(smoothing_layout)
        params_layout.addLayout(btns)

        params_widget = QtWidgets.QFrame(parent=self)
        params_widget.setLayout(params_layout)
        params_widget.setFrameShadow(QtWidgets.QFrame.Sunken)
        params_widget.setFrameShape(QtWidgets.QFrame.Panel)
        params_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(params_widget)
        right_layout.addStretch()

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.viewer)
        self.layout.addLayout(right_layout)
        self.setLayout(self.layout)

        self.initiate_autocenter()

    @QtCore.pyqtSlot(str)
    def show_error_message(self, msg):
        self.error_dialog = QtGui.QErrorMessage(parent=self)
        self.error_dialog.showMessage(msg)

    @QtCore.pyqtSlot()
    def accept(self):
        self.file_dialog = QtWidgets.QFileDialog(parent=self)
        filename = self.file_dialog.getSaveFileName(filter="*.hdf5")[0]
        if filename == "":
            return

        center = self.center_finder.center()

        params = {
            "center": center,
            "mod": int(self.mod_widget.currentText()),
            "processes": self.processes_widget.value(),
        }

        if self.enable_smoothing_widget.isChecked():
            params["kernel_size"] = self.smoothing_kernel_widget.value()

        self.symmetrize_parameters_signal.emit(filename, params)
        super().accept()

    @QtCore.pyqtSlot()
    def initiate_autocenter(self):
        """Automatically determine the center of an image
        and move the center-finder accordingly"""
        self._worker = AutocenteringThread(
            function=autocenter, kwargs=dict(im=self._image, mask=self._mask)
        )
        self._worker.results_signal.connect(self.set_center)
        self._worker.in_progress_signal.connect(self.busy_indicator.toggle_animation)
        self._worker.in_progress_signal.connect(self.autocenter_btn.setDisabled)
        self._worker.start()

    @QtCore.pyqtSlot(object)
    def set_center(self, rc):
        row, col = rc
        self.center_finder.set_center(col, row)


class AutocenteringThread(WorkThread):
    results_signal = QtCore.pyqtSignal(object)
