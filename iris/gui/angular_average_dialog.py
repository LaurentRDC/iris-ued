# -*- coding: utf-8 -*-
"""
Dialog for azimuthal average of diffraction data
"""
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


normalize_help = """ If checked, all powder patterns will be normalized to their overall intensity.
This can get rid of systematic offsets between patterns at different time-delay. """

explanation = """Drag and resize the red circle until it sits on top of a diffraction ring. 
This allows for easy determination of the picture center. """.replace(
    "\n", ""
)


class AngularAverageDialog(QtWidgets.QDialog):
    """
    Modal dialog to promote a DiffractionDataset to a
    PowderDiffractionDataset
    """

    angular_average_signal = QtCore.pyqtSignal(dict)

    def __init__(self, image, *args, **kwargs):
        """
        Parameters
        ----------
        image : ndarray
            Diffraction pattern to be displayed.
        """
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle("Calculate azimuthal averages")

        title = QtWidgets.QLabel("<h2>Azimuthal Average Options<\h2>")
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)

        explanation_label = QtWidgets.QLabel(explanation, parent=self)
        explanation_label.setWordWrap(True)

        self.viewer = pg.ImageView(parent=self)
        self.viewer.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.viewer.setImage(image)
        self.center_finder = pg.CircleROI(
            pos=np.array(image.shape) / 2 - 100, size=[200, 200], pen=pg.mkPen("r")
        )
        self.viewer.getView().addItem(self.center_finder)

        self.partial_circle_btn = QtWidgets.QCheckBox("Restrict azimuthal angle", self)
        self.partial_circle_btn.setChecked(False)

        self.accept_btn = QtWidgets.QPushButton("Calculate", self)
        self.accept_btn.clicked.connect(self.accept)

        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        self.min_angular_bound_widget = QtWidgets.QDoubleSpinBox(parent=self)
        self.min_angular_bound_widget.setRange(0, 360)
        self.min_angular_bound_widget.setSingleStep(1)
        self.min_angular_bound_widget.setValue(0)
        self.min_angular_bound_widget.setSuffix(" deg")
        self.min_angular_bound_widget.setEnabled(False)
        self.partial_circle_btn.toggled.connect(
            self.min_angular_bound_widget.setEnabled
        )

        self.max_angular_bound_widget = QtWidgets.QDoubleSpinBox(parent=self)
        self.max_angular_bound_widget.setRange(0, 360)
        self.max_angular_bound_widget.setSingleStep
        self.max_angular_bound_widget.setValue(360)
        self.max_angular_bound_widget.setSuffix(" deg")
        self.max_angular_bound_widget.setEnabled(False)
        self.partial_circle_btn.toggled.connect(
            self.max_angular_bound_widget.setEnabled
        )

        self.min_angular_bound_widget.valueChanged.connect(
            self.max_angular_bound_widget.setMinimum
        )
        self.max_angular_bound_widget.valueChanged.connect(
            self.min_angular_bound_widget.setMaximum
        )

        self.normalize_widget = QtWidgets.QCheckBox("Normalize (?)", self)
        self.normalize_widget.setChecked(False)
        self.normalize_widget.setToolTip(normalize_help)

        angle_bounds_layout = QtWidgets.QFormLayout()
        angle_bounds_layout.addRow(self.partial_circle_btn)
        angle_bounds_layout.addRow("Min. angle: ", self.min_angular_bound_widget)
        angle_bounds_layout.addRow("Max. angle: ", self.max_angular_bound_widget)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        params_layout = QtWidgets.QVBoxLayout()
        params_layout.addWidget(title)
        params_layout.addWidget(explanation_label)
        params_layout.addWidget(self.normalize_widget)
        params_layout.addLayout(angle_bounds_layout)
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

    @QtCore.pyqtSlot()
    def accept(self):
        # Calculating the center position assumes that PyQtGraph is configured
        # such that imageAxisOrder == 'row-major'
        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x() / 2
        center = (round(corner_x + radius), round(corner_y + radius))

        params = {
            "center": center,
            "normalized": self.normalize_widget.isChecked(),
            "angular_bounds": None,
        }  # default

        if self.partial_circle_btn.isChecked():
            params["angular_bounds"] = (
                self.min_angular_bound_widget.value(),
                self.max_angular_bound_widget.value(),
            )

        self.angular_average_signal.emit(params)
        super().accept()
