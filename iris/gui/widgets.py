from collections.abc import Iterable
from ..dualtree import ALL_FIRST_STAGE, ALL_COMPLEX_WAV
from os.path import join, dirname
from . import pyqtgraph as pg
from .pyqtgraph import QtGui, QtCore
import numpy as n

from .utils import spectrum_colors
from ..utils import fluence

image_folder = join(dirname(__file__), 'images')
    
class ProcessedDataViewer(QtGui.QWidget):
    """
    Widget displaying the result of processing from RawDataset.process()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_viewer = pg.ImageView(parent = self)
        self.dataset_info = dict()
        self.peak_dynamics_viewer = pg.PlotWidget(title = 'Peak dynamics measurement', 
                                                  labels = {'left': 'Intensity (a. u.)', 'bottom': ('time', 'ps')})
        self.peak_dynamics_viewer.getPlotItem().enableAutoRange()
        self.peak_dynamics_viewer.hide()

        # Single-crystal peak dynamics
        self.peak_dynamics_region = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.peak_dynamics_region.addScaleHandle([1, 1], [0, 0])
        self.peak_dynamics_region.addScaleHandle([0, 0], [1, 1])
        self.image_viewer.getView().addItem(self.peak_dynamics_region)
        self.peak_dynamics_region.hide()

        # QSlider properties
        self.time_slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent = self)
        self.time_slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.time_slider.setTickInterval(1)
        self.time_slider.setValue(0)

        self.autolevel_cb = QtGui.QCheckBox('Enable auto-level', parent = self)
        self.autolevel_cb.setChecked(False)

        self.autorange_cb = QtGui.QCheckBox('Enable auto-range', parent = self)
        self.autolevel_cb.setChecked(False)

        self.show_pd_btn = QtGui.QPushButton('Show/hide peak dynamics', parent = self)
        self.show_pd_btn.setCheckable(True)
        self.show_pd_btn.setChecked(False)
        self.show_pd_btn.toggled.connect(self.peak_dynamics_viewer.setVisible)
        self.show_pd_btn.toggled.connect(self.peak_dynamics_region.setVisible)

        # Final assembly
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.image_viewer)
        self.layout.addWidget(self.peak_dynamics_viewer)

        checkboxes = QtGui.QVBoxLayout()
        checkboxes.addWidget(self.autolevel_cb)
        checkboxes.addWidget(self.autorange_cb)

        cmd_bar = QtGui.QHBoxLayout()
        cmd_bar.addLayout(checkboxes)
        cmd_bar.addWidget(self.show_pd_btn)
        cmd_bar.addWidget(self.time_slider)
        self.layout.addLayout(cmd_bar)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(dict)
    def update_info(self, info):
        """
        Update the widget with dataset information
        
        Parameters
        ----------
        info : dict 
        """
        self.dataset_info.update(info)
        self.time_slider.setRange(0, len(self.dataset_info['time_points']) - 1)

        # TODO: set tick labels to time points
    
    @QtCore.pyqtSlot(object, object)
    def update_peak_dynamics(self, time_points, integrated_intensity):
        pens = list(map(pg.mkPen, spectrum_colors(len(time_points))))
        brushes = list(map(pg.mkBrush, spectrum_colors(len(time_points))))
        self.peak_dynamics_viewer.plot(time_points, integrated_intensity, pen = None, symbol = 'o', 
                                       symbolPen = pens, symbolBrush = brushes, symbolSize = 4, clear = True)
    
    @QtCore.pyqtSlot(object)
    def display(self, image):
        """
        Display an image in the form of an ndarray.

        Parameters
        ----------
        image : ndarray
        """
        # autoLevels = False ensures that the colormap stays the same
        # when 'sliding' through data. This makes it easier to compare
        # data at different time points.
        # Similarly for autoRange = False
        self.image_viewer.setImage(image, autoLevels = self.autolevel_cb.isChecked(), 
                                          autoRange = self.autorange_cb.isChecked())

class IrisStatusBar(QtGui.QStatusBar):
    """
    Status bar displaying a status message.

    Slots
    -----
    update_status [str]
    """
    base_message = 'Ready.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_label = QtGui.QLabel(self.base_message)
        self.addPermanentWidget(self.status_label)
        self.timer = QtCore.QTimer(parent = self)
        self.timer.setSingleShot(True)
    
    @QtCore.pyqtSlot(str)
    def update_status(self, message):
        """ 
        Update the permanent status label with a temporary message. 
        
        Parameters
        ----------
        message : str
        """
        self.status_label.setText(message)
        self.timer.singleShot(1e5, lambda: self.update_status(self.base_message))

class DatasetInfoWidget(QtGui.QTableWidget):
    """
    Display of dataset information as a table.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_info = dict()
        self.setRowCount(2) # This will always be true, but the number of
                            # columns will change depending on the dataset

        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)

        # Resize to content
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)

    @QtCore.pyqtSlot(dict)
    def update_info(self, info):
        """ 
        Update the widget with dataset information
        
        Parameters
        ----------
        info : dict 
        """
        self.setVisible(True)
        self.dataset_info.update(info)

        self.setColumnCount(len(self.dataset_info))
        for column, (key, value) in enumerate(self.dataset_info.items()):
            # Items like time_points and nscans are long lists. Summarize using length
            key = key.replace('_', ' ')
            if isinstance(value, Iterable) and not isinstance(value, str) and len(tuple(value)) > 3:
                key += ' (len)'
                value = str(len(tuple(value)))
            
            # TODO: change font of key text to bold
            key_item = QtGui.QTableWidgetItem(str(key))
            ft = key_item.font()
            ft.setBold(True)
            key_item.setFont(ft)
            self.setItem(0, column, key_item)
            self.setItem(1, column, QtGui.QTableWidgetItem(str(value)))
        
        self.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)

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

        self.beam_size_x_edit = QtGui.QLineEdit('FWHM x [um]', parent = self)
        self.beam_size_y_edit = QtGui.QLineEdit('FHMM y [um]', parent = self)
        self.beam_size_x_edit.textChanged.connect(self.update)
        self.beam_size_y_edit.textChanged.connect(self.update)
        beam_size = QtGui.QHBoxLayout()
        beam_size.addWidget(self.beam_size_x_edit)
        beam_size.addWidget(self.beam_size_y_edit)

        self.laser_rep_rate_cb = QtGui.QComboBox(parent = self)
        self.laser_rep_rate_cb.addItems(['50', '100', '200', '250', '500', '1000'])
        self.laser_rep_rate_cb.setCurrentText('1000')
        self.laser_rep_rate_cb.currentIndexChanged.connect(self.update)

        self.incident_laser_power_edit = QtGui.QLineEdit('Laser power [mW]', parent = self)
        self.incident_laser_power_edit.textChanged.connect(self.update)

        self.fluence = QtGui.QLineEdit('Fluence (mJ/cm2)', parent = self)
        self.fluence.setReadOnly(True)

        self.done_btn = QtGui.QPushButton('Done', self)
        self.done_btn.clicked.connect(self.accept)
        self.done_btn.setDefault(True)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(beam_size)
        self.layout.addWidget(self.laser_rep_rate_cb)
        self.layout.addWidget(self.incident_laser_power_edit)
        self.layout.addWidget(self.fluence)
        self.layout.addWidget(self.done_btn)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot(int)
    def update(self, *args):
        try:
            f = fluence(float(self.incident_laser_power_edit.text()),
                        int(self.laser_rep_rate_cb.currentText()),
                        FWHM = [int(self.beam_size_x_edit.text()), int(self.beam_size_y_edit.text())])
            self.fluence.setText('{:.2f}'.format(f) + ' mJ / cm^2')
        except:  # Could not parse 
            self.fluence.setText('-----')