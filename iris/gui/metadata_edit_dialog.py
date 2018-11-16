# -*- coding: utf-8 -*-
"""
Dialog for editing a restricted set of metadata
"""
from PyQt5 import QtCore, QtWidgets


class MetadataEditDialog(QtWidgets.QDialog):
    """ Modal dialog to specify modify metadata
    like camera distance, etc. """

    updated_metadata_signal = QtCore.pyqtSignal(dict)

    def __init__(self, config, *args, **kwargs):
        """
        Parameters
        ----------
        config : dict
            Dictionary of the currently-saved experimental setup.
        """
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle("Experimental set-up")

        self.camera_distance_widget = QtWidgets.QDoubleSpinBox(parent=self)
        self.camera_distance_widget.setRange(0, 100)
        self.camera_distance_widget.setDecimals(4)
        self.camera_distance_widget.setSingleStep(0.0001)
        self.camera_distance_widget.setSuffix(" m")
        self.camera_distance_widget.setValue(float(config["camera_length"]))

        self.pixel_width_widget = QtWidgets.QDoubleSpinBox(parent=self)
        self.pixel_width_widget.setRange(0, 1000)
        self.pixel_width_widget.setDecimals(2)
        self.pixel_width_widget.setSingleStep(0.01)
        self.pixel_width_widget.setSuffix(" μm")
        self.pixel_width_widget.setValue(float(config["pixel_width"]) * 1e6)

        self.temperature_widget = QtWidgets.QDoubleSpinBox(parent=self)
        self.temperature_widget.setRange(0, 1e4)
        self.temperature_widget.setDecimals(2)
        self.temperature_widget.setSingleStep(1)
        self.temperature_widget.setSuffix(" K")
        self.temperature_widget.setValue(float(config["temperature"]))

        self.fluence_widget = QtWidgets.QDoubleSpinBox(parent=self)
        self.fluence_widget.setRange(0, 1000)
        self.fluence_widget.setDecimals(2)
        self.fluence_widget.setSingleStep(1)
        self.fluence_widget.setSuffix(" mJ/cm²")
        self.fluence_widget.setValue(float(config["fluence"]))

        self.accept_btn = QtWidgets.QPushButton("Confirm", self)
        self.accept_btn.clicked.connect(self.accept)

        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        params = QtWidgets.QFormLayout()
        params.addRow("Camera distance: ", self.camera_distance_widget)
        params.addRow("Pixel width: ", self.pixel_width_widget)
        params.addRow("Temperature: ", self.temperature_widget)
        params.addRow("Fluence: ", self.fluence_widget)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(params)
        self.layout.addLayout(btns)
        self.setLayout(self.layout)

    @QtCore.pyqtSlot()
    def accept(self):
        params = {
            "camera_length": float(self.camera_distance_widget.value()),
            "pixel_width": float(self.pixel_width_widget.value()) * 1e-6,
            "temperature": float(self.temperature_widget.value()),
            "fluence": float(self.fluence_widget.value()),
        }

        self.updated_metadata_signal.emit(params)
        super().accept()
