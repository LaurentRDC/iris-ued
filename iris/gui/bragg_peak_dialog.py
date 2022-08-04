# -*- coding: utf-8 -*-
"""
Dialog for processing between AbstractRawDataset and DiffractionDataset
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from skued import bragg_peaks, autocenter
from .controller import WorkThread
from .qbusyindicator import QBusyIndicator
from .processing_dialog import MaskCreator

class RectROIWithCenter(pg.RectROI):

    # Calculating the center position assumes that PyQtGraph is configured
    # such that imageAxisOrder == 'row-major'
    def center(self):
        corner_x, corner_y = self.pos().x(), self.pos().y()
        return (round(corner_x + self.size().x() / 2), round(corner_y + self.size().y() / 2))

    def set_center(self, x, y):
        left = x - self.size().x() / 2
        bottom = y - self.size().y() / 2
        self.setPos(left, bottom)
# -*- coding: utf-8 -*-

description = (
    """Auto-determine the Bragg peak positions and add manually those not found. Subsequent calculation will optimize and realign the Bragg peaks to local maxima. """
)
vertices_help = """This is the number of vertices you expect the projection of the Brilluoin
zone to have. Since this tool is just a visualization aid, this ensures only robustly determined
Brilluoin zones will be computed."""

class BraggPeakDialog(QtWidgets.QDialog):
    """
    Modal dialog to determine Bragg peak locations and corresponding
    2D projections of Brilluoin zones
    """

    bragg_peak_signal = QtCore.pyqtSignal(dict)

    def __init__(self, image, mask, pixel_width, center=None, *args, **kwargs):
        """
        Parameters
        ----------
        image : ndarray
            Diffraction pattern to be displayed.
        """
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setWindowTitle("Calculate Bragg peaks")
        self.__bragg_peak_items = list()
        self.__bragg_peaks = list()
        # For use with autopeak
        self._image = image
        self._mask = mask
        self._resolution = np.array(self._image.shape)
        title = QtWidgets.QLabel("<h2>Bragg peak options<\\h2>")
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
        if pixel_width is None:
            pixel_width = 1.4e-5 #default to Gatan pixel width
        self.bbox_size = int(7e-4 / pixel_width) #7e-4 m^-1 is about what looks right to cover most bragg peaks
        self.center_finder = RectROIWithCenter(
            pos=np.array(image.shape) / 2 - 100, size=[self.bbox_size, self.bbox_size]#, pen=pg.mkPen("r")
        )
        # self.autocenter = autocenter(self._image, self._mask)
        # self.center_finder.set_center(self.autocenter[1], self.autocenter[0])
        if center is not None and center != (0, 0):
            self.center_finder.set_center(*center)

        self.viewer.getView().addItem(self.center_finder)


        self.accept_btn = QtWidgets.QPushButton("Calculate", self)
        self.accept_btn.clicked.connect(self.accept)

        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        self.autopeak_btn = QtWidgets.QPushButton("Auto-detection", self)
        self.autopeak_btn.clicked.connect(self.initiate_autopeaks)
        self.autopeak_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        self.busy_indicator = QBusyIndicator(parent=self)
        autopeak_layout = QtWidgets.QHBoxLayout()
        autopeak_layout.addWidget(self.autopeak_btn)
        autopeak_layout.addWidget(self.busy_indicator)

        self.add_circ_mask_btn = QtWidgets.QPushButton("Add new peak", self)
        self.add_circ_mask_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        self.add_circ_mask_btn.clicked.connect(self.add_bragg_peak)
        self.add_circ_mask_btn.setEnabled(False)

        self.vertices_widget = QtWidgets.QSpinBox(parent=self)
        self.vertices_widget.setRange(1, 12)
        self.vertices_widget.setValue(6)
        self.vertices_widget.setToolTip(vertices_help)
    
        self.vertices_layout = QtWidgets.QFormLayout()
        self.vertices_layout.addRow("Number of vertices:", self.vertices_widget)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.cancel_btn)

        params_layout = QtWidgets.QVBoxLayout()
        params_layout.addWidget(title)
        params_layout.addWidget(description_label)
        params_layout.addLayout(autopeak_layout)
        params_layout.addWidget(self.add_circ_mask_btn)
        params_layout.addLayout(self.vertices_layout)
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

        self.initiate_autopeaks()

    @QtCore.pyqtSlot()
    def add_bragg_peak(self):
        new_roi = pg.RectROI(
            pos=self._resolution/2,
            size=(self.bbox_size,self.bbox_size),
            resizable=False
        )
        self.viewer.addItem(new_roi)
        self.__bragg_peak_items.append(new_roi)


    @QtCore.pyqtSlot()
    def accept(self):
        # Calculating the center position assumes that PyQtGraph is configured
        # such that imageAxisOrder == 'row-major'
        
        for item in self.__bragg_peak_items:
            self.__bragg_peaks.append([item.pos().x(), item.pos().y()])
        params = {
            "peaks" : self.__bragg_peaks,
            "sym" : int(self.vertices_widget.value())
        }
        self.bragg_peak_signal.emit(params)
        super().accept()

    @QtCore.pyqtSlot()
    def initiate_autopeaks(self):
        """Automatically determine the bragg peaks
        and move the center-finder accordingly"""
        self._worker = AutobraggpeakThread(
            function=bragg_peaks, kwargs=dict(im=self._image, mask=self._mask, min_dist=self.bbox_size)
        )

        self._worker.results_signal.connect(self.set_peaks)
        self._worker.in_progress_signal.connect(self.busy_indicator.toggle_animation)
        self._worker.in_progress_signal.connect(self.autopeak_btn.setDisabled)
        self._worker.start()

    @QtCore.pyqtSlot(object)
    def set_peaks(self, rc):
        for item in self.__bragg_peak_items:
            self.image_viewer.removeItem(item)
        self.__bragg_peak_items.clear()

        # self.__bragg_peak_items.append(self.center_finder)
        for idx, peak in enumerate(rc):
            r, c = peak
            self.__bragg_peak_items.append(
                pg.RectROI(
                    pos=(c-self.bbox_size//2, r-self.bbox_size//2), size=(self.bbox_size,self.bbox_size), movable=True, resizable=False, removable=True
                )
            )
        for item in self.__bragg_peak_items:
            self.viewer.addItem(item)
        
        self.add_circ_mask_btn.setEnabled(True)
        self._worker.exit()

class AutobraggpeakThread(WorkThread):
    results_signal = QtCore.pyqtSignal(object)



def parse_range(range_str):
    """
    Parse an integer range into a list of numbers.

    Parameters
    ----------
    range_str : str
        String of the form : "-10, 1:5, 10:50, 100, 101".
        Ranges are inclusive (the endpoint is included). Can also be
        an empty string.

    Returns
    -------
    range : iterable of ints
        Iterable of integers (possibly empty). Guaranteed to be sorted and unique.

    Raises
    ------
    ValueError
        if the input ``range_str`` is unparseable.
    """
    range_str = str(range_str)
    range_str = range_str.replace(" ", "")
    if not range_str:
        return list()

    elements = range_str.split(",")
    if not elements:
        return list()

    iterable = list()

    # Two possibilities : ints or ranges
    # Either elem = int
    # or     elem = start:stop
    # Note : stop + 1 because inclusive ranges
    for elem in elements:
        try:
            fl = int(elem)
            iterable.append(fl)
        except ValueError:
            try:
                start, stop = tuple(map(int, elem.split(":")))
                iterable.extend(range(start, stop + 1))
            except:
                # Raise exception from None because full traceback is not useful
                # especially in terms of GUI error messages
                raise ValueError("Unparseable input: ", range_str) from None

    return list(sorted(set(iterable)))
