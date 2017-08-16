
from os import cpu_count
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
import numpy as np

from ..raw import McGillRawDataset

fletcher32_help = """ Adds a checksum to each chunk to detect data corruption. 
Attempts to read corrupted chunks will fail with an error. 
No significant speed penalty """

shuffle_help = """ Block-oriented compressors like GZIP or LZF work better
when presented with runs of similar values. Enabling the
shuffle filter rearranges the bytes in the chunk and may
improve compression ratio. """

alignment_help = """If checked, diffraction patterns will be aligned 
using masked normalized cross-correlation. This can double the processing time. """

DTYPE_NAMES = {'Auto': None,
               '64-bit floats': np.float64,
               '32-bit floats': np.float32,
               '16-bit floats': np.float16,
               '64-bit integers': np.int64,
               '32-bit integers': np.int32,
               '16-bit integers': np.int16}

class H5FileWidget(QtGui.QWidget):
    """ Widget specifying all HDF5 file properties """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.compression_filter_widget = QtGui.QComboBox()
        self.compression_filter_widget.addItems(['None', 'lzf', 'gzip'])
        self.compression_filter_widget.setCurrentText('None')

        self.fletcher32_widget = QtGui.QCheckBox('Enable Fletcher32 filter (?)', parent = self)
        self.fletcher32_widget.setToolTip(fletcher32_help)

        self.shuffle_filter_widget = QtGui.QCheckBox('Enable shuffle filter (?)', parent = self)
        self.shuffle_filter_widget.setToolTip(shuffle_help)
        self.shuffle_filter_widget.setChecked(True)

        layout = QtGui.QFormLayout()
        layout.addRow('Compression filter: ', self.compression_filter_widget)
        layout.addRow(self.fletcher32_widget)
        layout.addRow(self.shuffle_filter_widget)

        self.setLayout(layout)

    def file_params(self):
        """ Returns a dictionary with HDF5 file parameters """
        compression = self.compression_filter_widget.currentText()
        fletcher32 = self.fletcher32_widget.isChecked()
        shuffle = self.shuffle_filter_widget.isChecked()

        if compression == 'None':
            return dict()

        return {'compression': compression, 'fletcher32': fletcher32, 'shuffle': shuffle}

class ProcessingDialog(QtGui.QDialog):
    """
    Modal dialog used to select dataset processing options.
    """
    processing_parameters_signal = QtCore.pyqtSignal(dict)

    def __init__(self, raw, **kwargs):
        """
        Parameters
        ----------
        raw : McGillRawDataset
        """
        super().__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Diffraction Dataset Processing')
        self.h5_file_widget = H5FileWidget(parent = self)
        self.raw = raw

        image = self.raw.raw_data(timedelay = raw.time_points[0], 
                                  scan = raw.nscans[0], bgr = True)

        self.viewer = pg.ImageView(parent = self)
        self.viewer.setImage(image)

        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
        self.viewer.getView().addItem(self.mask)

        self.processes_widget = QtGui.QSpinBox(parent = self)
        self.processes_widget.setRange(1, cpu_count() - 1)
        self.processes_widget.setValue(min([cpu_count(), 7]))

        self.alignment_tf_widget = QtGui.QCheckBox('Perform alignment (?)', parent = self)
        self.alignment_tf_widget.setToolTip(alignment_help)
        self.alignment_tf_widget.setChecked(True)

        self.dtype_widget = QtGui.QComboBox(parent = self)
        self.dtype_widget.addItems(DTYPE_NAMES.keys())
        self.dtype_widget.setCurrentText('Auto')

        # Set exclude scan widget with a validator
        # integers separated by commas only
        self.exclude_scans_widget = QtGui.QLineEdit('', parent = self)
        self.exclude_scans_widget.setPlaceholderText('[comma-separated]')
        self.exclude_scans_widget.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp('^\d{1,4}(?:[,]\d{1,4})*$')))

        self.save_btn = QtGui.QPushButton('Launch processing', self)
        self.save_btn.clicked.connect(self.accept)

        self.cancel_btn = QtGui.QPushButton('Cancel', self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        # Determine settings
        self.file_dialog = QtGui.QFileDialog(parent = self)

        processing_options = QtGui.QFormLayout()
        processing_options.addRow('Number of cores to use:',self.processes_widget)
        processing_options.addRow('Scans to exclude: ', self.exclude_scans_widget)
        processing_options.addRow(self.alignment_tf_widget)
        processing_options.addRow('Data type: ', self.dtype_widget)

        params_layout = QtGui.QHBoxLayout()
        params_layout.addLayout(processing_options)
        params_layout.addWidget(self.h5_file_widget)

        buttons = QtGui.QHBoxLayout()
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.cancel_btn)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.viewer)
        self.layout.addLayout(params_layout)
        self.layout.addLayout(buttons)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def accept(self):

        # Beamblock rect
        rect = self.mask.parentBounds().toRect()
        #If coordinate is negative, return 0
        x1 = round(max(0, rect.topLeft().x() ))
        x2 = round(max(0, rect.x() + rect.width() ))
        y1 = round(max(0, rect.topLeft().y() ))
        y2 = round(max(0, rect.y() + rect.height() ))

        valid_mask = np.ones(self.raw.resolution, dtype = np.bool)
        valid_mask[x1:x2, y1:y2] = False

        filename = self.file_dialog.getSaveFileName(filter = '*.hdf5')[0]
        if filename == '':
            return

        # HDF5 compression kwargs
        ckwargs = self.h5_file_widget.file_params()

        # Force data type
        dtype = DTYPE_NAMES[self.dtype_widget.currentText()]
        
        exclude_scans_text = self.exclude_scans_widget.text()
        try:
            exclude_scans = [int(exclude_scans_text)]
        except ValueError:
            exclude_scans_text = exclude_scans_text.split(',')
            try:
                exclude_scans = list(map(int, exclude_scans_text))
            except:
                exclude_scans = []
        
        # The arguments to the iris.processing.process function
        # more arguments will be added by controller
        kwargs = {'filename':filename, 
                  'valid_mask': valid_mask,
                  'processes': self.processes_widget.value(),
                  'exclude_scans': exclude_scans,
                  'align': self.alignment_tf_widget.isChecked(),
                  'dtype': dtype,
                  'ckwargs': ckwargs}
        
        self.processing_parameters_signal.emit(kwargs)
        super().accept()
