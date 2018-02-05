
from os import cpu_count
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np

from .. import McGillRawDataset

fletcher32_help = """ Adds a checksum to each chunk to detect data corruption. 
Attempts to read corrupted chunks will fail with an error. 
No significant speed penalty """.replace('\n', '')

shuffle_help = """ Block-oriented compressors like GZIP or LZF work better
when presented with runs of similar values. Enabling the 
shuffle filter rearranges the bytes in the chunk and may 
improve compression ratio. """.replace('\n', '')

alignment_help = """If checked, diffraction patterns will be aligned 
using masked normalized cross-correlation. 
This can double the processing time. """ .replace('\n', '')

normalization_help = """If checked, diffraction patterns are normalized so that the total
 intensity is equal for each picture at the same scan. For this to be effective, a good mask 
must be provided. """ .replace('\n', '')

exclude_scans_help = """ Specify scans to exclude comma separated,
e.g. 3,4, 5, 10, 32. """ .replace('\n', '')

DTYPE_NAMES = {'Auto': None,
               '64-bit floats': np.float64,
               '32-bit floats': np.float32,
               '16-bit floats': np.float16,
               '64-bit integers': np.int64,
               '32-bit integers': np.int32,
               '16-bit integers': np.int16}

class H5FileWidget(QtWidgets.QWidget):
    """ Widget specifying all HDF5 file properties """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.enable_compression_widget = QtWidgets.QCheckBox('Enable compression')
        self.enable_compression_widget.setChecked(False)

        self.lzf_btn = QtWidgets.QRadioButton('LZF', self)
        self.lzf_btn.setChecked(True)

        self.gzip_btn = QtWidgets.QRadioButton('GZIP', self)
        
        filter_btns = QtWidgets.QVBoxLayout()
        filter_btns.addWidget(self.lzf_btn)
        filter_btns.addWidget(self.gzip_btn)

        self.filters = QtWidgets.QGroupBox('Compression filters', parent = self)
        self.filters.setLayout(filter_btns)
        self.filters.setFlat(True)

        self.gzip_level_widget = QtWidgets.QSpinBox(self)
        self.gzip_level_widget.setRange(0, 9)
        self.gzip_level_widget.setValue(4)
        self.gzip_btn.toggled.connect(self.gzip_level_widget.setEnabled)
        self.gzip_level_widget.setEnabled(False)

        self.fletcher32_widget = QtWidgets.QCheckBox('Enable Fletcher32 filter (?)', parent = self)
        self.fletcher32_widget.setToolTip(fletcher32_help)

        self.shuffle_filter_widget = QtWidgets.QCheckBox('Enable shuffle filter (?)', parent = self)
        self.shuffle_filter_widget.setToolTip(shuffle_help)
        self.shuffle_filter_widget.setChecked(True)

        params_layout = QtWidgets.QFormLayout()
        params_layout.addRow(self.filters)
        params_layout.addRow('GZIP level: ', self.gzip_level_widget)
        params_layout.addRow(self.fletcher32_widget)
        params_layout.addRow(self.shuffle_filter_widget)

        params_widget = QtWidgets.QWidget(parent = self)
        params_widget.setLayout(params_layout)
        params_widget.setEnabled(False)
        self.enable_compression_widget.toggled.connect(params_widget.setEnabled)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.enable_compression_widget)
        layout.addWidget(params_widget)

        self.setLayout(layout)

    def file_params(self):
        """ Returns a dictionary with HDF5 file parameters """
        if not self.enable_compression_widget.isChecked():
            return dict()
        
        compression = 'lzf' if self.lzf_btn.isChecked() else 'gzip'
        fletcher32 = self.fletcher32_widget.isChecked()
        shuffle = self.shuffle_filter_widget.isChecked()
        
        params = {'compression': compression, 
                  'fletcher32': fletcher32, 
                  'shuffle': shuffle}

        if compression == 'gzip':
            params['compression_opts'] = self.gzip_level_widget.value()

        return params

class DataTransformationsWidget(QtWidgets.QWidget):
    """ Widgets specifying all data transformations, 
    e.g. clipping, normalization, alignment, etc. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.alignment_tf_widget = QtWidgets.QCheckBox('Perform alignment (?)', parent = self)
        self.alignment_tf_widget.setToolTip(alignment_help)
        self.alignment_tf_widget.setChecked(False)

        self.normalization_tf_widget = QtWidgets.QCheckBox('Normalize (?)', parent = self)
        self.normalization_tf_widget.setToolTip(normalization_help)
        self.normalization_tf_widget.setChecked(True)

        self.enable_clip_widget = QtWidgets.QCheckBox('Clip pixel values', parent = self)
        self.enable_clip_widget.setChecked(True)

        self.clip_min_widget = QtWidgets.QDoubleSpinBox(parent = self)
        self.clip_min_widget.setMinimum(0.0)
        self.clip_min_widget.setValue(0.0)

        self.clip_max_widget = QtWidgets.QDoubleSpinBox(parent = self)
        self.clip_max_widget.setMinimum(0)
        self.clip_max_widget.setValue(30000)

        # Make sure minimum is never larger than maximum
        self.clip_min_widget.valueChanged.connect(self.clip_max_widget.setMinimum)
        self.clip_max_widget.valueChanged.connect(self.clip_min_widget.setMaximum)

        clip_layout = QtWidgets.QFormLayout()
        clip_layout.addRow('Minimum:', self.clip_min_widget)
        clip_layout.addRow('Maximum:', self.clip_max_widget)

        clip_widget = QtWidgets.QGroupBox('Clipping limits', parent = self)
        clip_widget.setLayout(clip_layout)
        clip_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.enable_clip_widget.toggled.connect(clip_widget.setEnabled)

        checks = QtWidgets.QFormLayout()
        checks.addRow(self.alignment_tf_widget)
        checks.addRow(self.normalization_tf_widget)
        checks.addRow(self.enable_clip_widget)
        checks.addRow(clip_widget)

        #layout = QtWidgets.QVBoxLayout()
        #layout.addLayout(checks)
        #layout.addWidget(clip_widget)
        self.setLayout(checks)
    
    def data_transformations(self):

        params = dict()
        params['align']     = self.alignment_tf_widget.isChecked()
        params['normalize'] = self.normalization_tf_widget.isChecked()

        if self.enable_clip_widget.isChecked:
            params['clip'] = [self.clip_min_widget.value(), self.clip_max_widget.value()]
        else:
            params['clip'] = None
        
        return params



class ProcessingDialog(QtWidgets.QDialog):
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

        self.data_transformations_widget = DataTransformationsWidget(parent = self)

        self.raw = raw

        image = self.raw.raw_data(timedelay = raw.time_points[0], 
                                  scan = raw.scans[0], bgr = True)

        self.viewer = pg.ImageView(parent = self)
        self.viewer.setImage(image)
        self.viewer.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                  QtWidgets.QSizePolicy.MinimumExpanding)

        self.mask = pg.ROI(pos = [0,0], size = [200,200], pen = pg.mkPen('r'))
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
        self.viewer.getView().addItem(self.mask)

        self.processes_widget = QtWidgets.QSpinBox(parent = self)
        self.processes_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.processes_widget.setRange(1, cpu_count() - 1)
        self.processes_widget.setValue(min([cpu_count(), 7]))

        self.dtype_widget = QtWidgets.QComboBox(parent = self)
        self.dtype_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.dtype_widget.addItems(DTYPE_NAMES.keys())
        self.dtype_widget.setCurrentText('Auto')

        # Set exclude scan widget with a validator
        # integers separated by commas only
        self.exclude_scans_widget = QtWidgets.QLineEdit(parent = self)
        self.exclude_scans_widget.setPlaceholderText('[comma-separated]')
        self.exclude_scans_widget.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp('^\d{1,4}(?:[,]\d{1,4})*$')))

        save_btn = QtWidgets.QPushButton('Launch processing', self)
        save_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        save_btn.clicked.connect(self.accept)

        cancel_btn = QtWidgets.QPushButton('Cancel', self)
        cancel_btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setDefault(True)

        # Determine settings
        self.file_dialog = QtWidgets.QFileDialog(parent = self)

        processing_options = QtWidgets.QFormLayout()
        processing_options.addRow('Number of cores:',   self.processes_widget)
        processing_options.addRow('Scans to exclude: ', self.exclude_scans_widget)
        processing_options.addRow('Data type: ',        self.dtype_widget)

        params_layout = QtWidgets.QHBoxLayout()
        params_layout.addLayout(processing_options)
        params_layout.addWidget(self.data_transformations_widget)
        params_layout.addWidget(self.h5_file_widget)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)

        self.layout = QtWidgets.QVBoxLayout()
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

        # TODO: use self.mask.getArrayRegion
        valid_mask = np.ones(self.raw.resolution, dtype = np.bool)
        valid_mask[x1:x2, y1:y2] = False

        filename = self.file_dialog.getSaveFileName(filter = '*.hdf5')[0]
        if filename == '':
            return

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
        kwargs = {'filename'     : filename, 
                  'valid_mask'   : valid_mask,
                  'processes'    : self.processes_widget.value(),
                  'exclude_scans': exclude_scans,
                  'dtype'        : dtype}
        
        # Some parameters are from different widgets
        # HDF5 compression kwargs
        kwargs['ckwargs'] = self.h5_file_widget.file_params()
        kwargs.update(self.data_transformations_widget.data_transformations())
        
        self.processing_parameters_signal.emit(kwargs)
        super().accept()

if __name__ == '__main__':
    
    from qdarkstyle import load_stylesheet_pyqt5
    import sys
    from .. import McGillRawDataset

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet_pyqt5())
    gui = ProcessingDialog(raw = McGillRawDataset('D:\\Diffraction data\\2017.08.12.15.47.VO2_50nm_29mJcm2_50Hz_18hrs'))
    gui.show()
    app.exec_()