# -*- coding: utf-8 -*-

import sys
import os.path
import numpy as n
from PIL import Image

#Core functions
from core import *

#GUI backends
from pyqtgraph.Qt import QtCore, QtGui
from image_viewer import ImageViewer

# -----------------------------------------------------------------------------
#           COLOR PALETTE FOR PROGRESS
# -----------------------------------------------------------------------------

#Palette parameters
complete_palette = QtGui.QPalette()
complete_palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.darkGreen)

incomplete_palette = QtGui.QPalette()
incomplete_palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)

in_progress_palette = QtGui.QPalette()
in_progress_palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.darkYellow)

# -----------------------------------------------------------------------------
#           WORKING CLASS
# -----------------------------------------------------------------------------

class WorkThread(QtCore.QThread):
    """
    Object taking care of threading computations
    """
    done_signal = QtCore.pyqtSignal(bool, name = 'done_signal')
    in_progress_signal = QtCore.pyqtSignal(bool, name = 'in_progress_signal')
    results_signal = QtCore.pyqtSignal(object, name = 'results_signal')
    
    def __init__(self, function, *args, **kwargs):
        
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def __del__(self):
        self.wait()
    
    def run(self):
        """ Compute and emit a 'done' signal."""
        
        self.in_progress_signal.emit(True)
        self.result = self.function(*self.args, **self.kwargs)
        self.results_signal.emit(self.result)
        self.done_signal.emit(True)
    
# -----------------------------------------------------------------------------
#           DIRECTORY HANDLER OBJECT
# -----------------------------------------------------------------------------

class DirectoryHandler(QtGui.QWidget):
    
    dataset_directory_signal = QtCore.pyqtSignal(str, name = 'dataset_directory_signal')
    preprocess_signal = QtCore.pyqtSignal(bool, name = 'preprocess_signal')
    reset_signal = QtCore.pyqtSignal(bool, name = 'reset_signal')
    
    def __init__(self, parent = None):
        
        super(DirectoryHandler, self).__init__()
        
        self.initUI()
        self.initLayout()
        self.connectSignals()    
    
    def initUI(self):
        
        self.file_dialog = QtGui.QFileDialog(parent = self)
        
        self.directory_btn = QtGui.QPushButton('Select directory', parent = self)
        self.preprocess_btn = QtGui.QPushButton('Preprocess images', parent = self)
        self.preprocess_progress_label = QtGui.QLabel('Incomplete', parent = self)
        
        self.preprocess_btn.setChecked(False)
        self.preprocess_progress_label.setPalette(incomplete_palette)
    
    def initLayout(self):
        
        self.file_dialog_row = QtGui.QHBoxLayout()
        self.file_dialog_row.addWidget(self.directory_btn)
        
        self.preprocess_row = QtGui.QHBoxLayout()
        self.preprocess_row.addWidget(self.preprocess_btn)
        self.preprocess_row.addWidget(self.preprocess_progress_label)
        
        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(self.file_dialog_row)
        self.layout.addLayout(self.preprocess_row)
        
        self.setLayout(self.layout)
    
    def connectSignals(self):
        self.directory_btn.clicked.connect(self.directoryLocator)
        self.preprocess_btn.clicked.connect(self.preprocessImages)
        
        #When preprocess_btn is clicked, send signal to change preprocess_label to 'in progress' state
        self.preprocess_signal.connect(self.preprocessInProgress)
    
    def preprocessImages(self):
        self.preprocess_signal.emit(True)
        
    def directoryLocator(self):
        """ 
        Activates a file dialog that selects the data directory to be processed. If the folder
        selected is one with processed images (then the directory name is C:\\...\\processed\\),
        return data 'root' directory.
        """
        
        possible_directory = self.file_dialog.getExistingDirectory(self, 'Open diffraction dataset', 'C:\\')
        possible_directory = os.path.abspath(possible_directory)
        
        #Check whether the directory name ends in 'processed'. If so, return previous directory
        last_directory = possible_directory.split('\\')[-1]
        if last_directory == 'processed':
            directory = os.path.dirname(possible_directory) #If directory is 'processed', back up one directory
        else:
            directory = possible_directory
        
        self.reset_signal.emit(True)
        self.reset()
        self.dataset_directory_signal.emit(directory)
    
    @QtCore.pyqtSlot(bool)
    def preprocessInProgress(self, in_progress = False):
        if in_progress:
            self.preprocess_progress_label.setText('In progress...')
            self.preprocess_progress_label.setPalette(in_progress_palette)
    
    @QtCore.pyqtSlot(bool)
    def preprocessComplete(self, complete = False):
        if complete:
            palette = complete_palette
            self.preprocess_progress_label.setText('Complete.')
        else:
            palette = incomplete_palette
            self.preprocess_progress_label.setText('Incomplete.')
        self.preprocess_progress_label.setPalette(palette)
    
    @QtCore.pyqtSlot(bool)
    def reset(self, hmm_really = True):
        if hmm_really:
            self.preprocess_btn.setChecked(False)
            self.preprocess_progress_label.setText('Incomplete.')
            self.preprocess_progress_label.setPalette(incomplete_palette)

# -----------------------------------------------------------------------------
#           DATA HANDLER OBJECT
# -----------------------------------------------------------------------------

class DataHandler(QtCore.QObject):
    """
    Object tasked with all things computations and data.
    """
    image_preprocessing_in_progress_signal = QtCore.pyqtSignal(bool, name = 'image_preprocessing_in_progress_signal')
    image_preprocessed_signal = QtCore.pyqtSignal(bool, name = 'image_preprocessed_signal')
    
    has_image_center_signal = QtCore.pyqtSignal(bool, name = 'has_image_center_signal')
    has_mask_rect_signal = QtCore.pyqtSignal(bool, name = 'has_mask_rect_signal')
    has_cutoff_signal = QtCore.pyqtSignal(bool, name = 'has_cutoff_signal')
    has_inelasticBG_signal = QtCore.pyqtSignal(bool, name = 'has_inelasticBG_signal')
    
    def __init__(self, parent = None):
        
        super(DataHandler, self).__init__()
        
        #Data attributes
        self.diffraction_dataset = None
        self.image_center = list()
        self.mask_rect = list()
        self.cutoff = list()
        self.inelasticBGCurve = None
        
        #State attributes
        self.has_image_center = False
        self.has_mask_rect = False
        self.has_cutoff = False
        self.has_inelasticBG = False
        
        self.connectSignals()
    
    def connectSignals(self):
        self.has_image_center_signal.connect(self.hasImageCenter)
        self.has_mask_rect_signal.connect(self.hasMaskRect)
        self.has_cutoff_signal.connect(self.hasCutoff)
        self.has_inelasticBG_signal.connect(self.hasInelasticBG)
    
    @QtCore.pyqtSlot(tuple)
    def setImageCenter(self, center):
        self.image_center = center
        self.has_image_center_signal.emit(True)
    
    @QtCore.pyqtSlot(tuple)
    def setMaskRect(self, mask_rect):
        self.mask_rect = mask_rect
        self.has_mask_rect_signal.emit(True)
    
    @QtCore.pyqtSlot(tuple)
    def setCutoff(self, cutoff):
        self.cutoff = cutoff
        self.has_cutoff_signal.emit(True)
    
    @QtCore.pyqtSlot(object)
    def setInelasticBG(self, inelasticBGCurve):
        self.inelasticBGCurve = inelasticBGCurve
        self.has_inelasticBG_signal.emit(True)
        
    # -------------------------------------------------------------------------
    #           INTERNAL SELF-UPDATING SLOTS
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot(bool)
    def hasImageCenter(self, flag):
        self.has_image_center = flag
    
    @QtCore.pyqtSlot(bool)
    def hasMaskRect(self, flag):
        self.has_imask_rect = flag
    
    @QtCore.pyqtSlot(bool)
    def hasCutoff(self, flag):
        self.has_cutoff = flag
    
    @QtCore.pyqtSlot(bool)
    def hasInelasticBG(self, flag):
        self.has_inelasticBG = flag
        
    # -------------------------------------------------------------------------
    #           DIFFRACTION DATASET 
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot(str)
    def createDiffractionDataset(self, directory):
        #TODO: Allow various resolutions
        self.diffraction_dataset = DiffractionDataset( directory, resolution = (2048, 2048) )
        
    @QtCore.pyqtSlot(bool)
    def preprocessImages(self, flag):
        self.work_thread = WorkThread(self.diffraction_dataset.batchAverage, True) #Check if averaged has been computed before
        self.work_thread.in_progress_signal.connect(self.preprocessInProgress)
        self.work_thread.done_signal.connect(self.preprocessDone)
        #No need to connect the results_signal since batchAverage returns None
        self.work_thread.start()
    
    # -------------------------------------------------------------------------
    #           IMAGE PREPROCESSING SIGNAL HANDLING
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot(bool)
    def preprocessInProgress(self, flag):
        self.image_preprocessing_in_progress_signal.emit(flag)
    
    @QtCore.pyqtSlot()
    def preprocessDone(self):
        self.image_preprocessed_signal.emit(True)
    
    @QtCore.pyqtSlot(bool)
    def reset(self, hmm_really):
        """ 
        Reset all attributes and emit signals to the effect
        that all attributes are 'empty'
        """
        if hmm_really:
            self.image_center = list()
            self.mask_rect = list()
            self.cutoff = list()
            self.inelasticBGCurve = None
            
            self.has_image_center_signal.emit(False)
            self.has_mask_rect_signal.emit(False)
            self.has_cutoff_signal.emit(False)
            self.has_inelasticBG_signal.emit(False)
        
# -----------------------------------------------------------------------------
#           STATUS WIDGET 
# -----------------------------------------------------------------------------

class StatusWidget(QtGui.QWidget):
    
    def __init__(self, parent = None):
        
        super(StatusWidget, self).__init__()
        self.labels = list()
        self.statuses = list()
        
        #Components attributes
        self.initUI()
        
        #Finish layout
        self.initLayout()
        self.initStatuses()
    
    def initUI(self):
        """ Creates UI components. """
        self.image_center_status = QtGui.QLabel(parent = self)
        self.image_center_label = QtGui.QLabel('Image center: ', parent = self)
        
        self.mask_rect_status = QtGui.QLabel(parent = self)
        self.mask_rect_label = QtGui.QLabel('Beamblock mask: ', parent = self)
        
        self.cutoff_status = QtGui.QLabel(parent = self)
        self.cutoff_label = QtGui.QLabel('Cutoff: ', parent = self)
        
        self.inelasticBG_status = QtGui.QLabel(parent = self)
        self.inelasticBG_label = QtGui.QLabel('Inelastic BG: ', parent = self)
        
        self.labels = [self.image_center_label, self.mask_rect_label, self.cutoff_label, self.inelasticBG_label]
        self.statuses = [self.image_center_status, self.mask_rect_status, self.cutoff_status, self.inelasticBG_status]
    
    def initLayout(self):
        """ Lays out components on the widget. """
        self.layout = QtGui.QVBoxLayout()        
        for label, status in zip(self.labels, self.statuses):
            layout = QtGui.QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(status)
            self.layout.addLayout(layout)
        
        self.setLayout(self.layout)
    
    def initStatuses(self):
        for status in self.statuses:
            status.setText('Incomplete')
            status.setPalette(incomplete_palette)
    
    # -------------------------------------------------------------------------
    #           CHANGE STATUS METHODS
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot(bool)
    def imageCenterToggle(self, complete = False):
        if complete:
            palette = complete_palette
            label = 'Complete.'
        else:
            palette = incomplete_palette
            label = 'Incomplete.'
        
        self.image_center_status.setText(label)
        self.image_center_status.setPalette(palette)
    
    @QtCore.pyqtSlot(bool)
    def imageCenterInProgress(self, in_progress = False):
        if in_progress:
            self.image_center_status.setText('In progress...')
            self.image_center_status.setPalette(in_progress_palette)
    
    @QtCore.pyqtSlot(bool)
    def maskRectToggle(self, complete = False):
        if complete:
            palette = complete_palette
            label = 'Complete.'
        else:
            palette = incomplete_palette
            label = 'Incomplete.'
        
        self.mask_rect_status.setText(label)
        self.mask_rect_status.setPalette(palette)
    
    @QtCore.pyqtSlot(bool)
    def maskRectInProgress(self, in_progress = False):
        if in_progress:
            self.mask_rect_status.setText('In progress...')
            self.mask_rect_status.setPalette(in_progress_palette)

    @QtCore.pyqtSlot(bool)
    def cutoffToggle(self, complete = False):
        if complete:
            palette = complete_palette
            label = 'Complete.'
        else:
            palette = incomplete_palette
            label = 'Incomplete.'
        
        self.cutoff_status.setText(label)
        self.cutoff_status.setPalette(palette)
    
    @QtCore.pyqtSlot(bool)
    def cutoffInProgress(self, in_progress = False):
        if in_progress:
            self.cutoff_status.setText('In progress...')
            self.cutoff_status.setPalette(in_progress_palette)
    
    @QtCore.pyqtSlot(bool)
    def inelasticBGToggle(self, complete = False):
        if complete:
            palette = complete_palette
            label = 'Complete.'
        else:
            palette = incomplete_palette
            label = 'Incomplete.'
        
        self.inelasticBG_status.setText(label)
        self.inelasticBG_status.setPalette(palette)
    
    @QtCore.pyqtSlot(bool)
    def inelasticBGInProgress(self, in_progress = False):
        if in_progress:
            self.inelasticBG_status.setText('In progress...')
            self.inelasticBG_status.setPalette(in_progress_palette)
    
# -----------------------------------------------------------------------------
#           COMMAND CENTER WIDGET
# -----------------------------------------------------------------------------

class CommandCenter(QtGui.QWidget):
    
    #Action signals
    # 'get' signals are to tell the image viewer to let the user get the value
    get_image_center_signal = QtCore.pyqtSignal(name = 'get_image_center_signal')
    get_mask_rect_signal = QtCore.pyqtSignal(name = 'get_mask_rect_signal')
    get_cutoff_signal = QtCore.pyqtSignal(name = 'get_cutoff_signal')    
    get_inelasticBG_signal = QtCore.pyqtSignal(name = 'get_inelasticBG_signal')
    
    # 'set' signals are to tell the image viewer to return the value
    set_image_center_signal = QtCore.pyqtSignal(name = 'set_image_center_signal')
    set_mask_rect_signal = QtCore.pyqtSignal(name = 'set_mask_rect_signal')
    set_cutoff_signal = QtCore.pyqtSignal(name = 'set_cutoff_signal')
    set_inelasticBG_signal = QtCore.pyqtSignal(name = 'set_inelasticBG_signal')
    
    #Incomplete / complete signals
    image_center_signal = QtCore.pyqtSignal(bool, name = 'image_center_signal')
    mask_rect_signal = QtCore.pyqtSignal(bool, name = 'mask_rect_signal')
    cutoff_signal = QtCore.pyqtSignal(bool, name = 'cutoff_signal')
    inelasticBG_signal = QtCore.pyqtSignal(bool, name = 'inelasticBG_signal')
    
    #In progress signals
    image_center_in_progress = QtCore.pyqtSignal(bool, name = 'image_center_in_progress')
    mask_rect_in_progress = QtCore.pyqtSignal(bool, name = 'mask_rect_in_progress')
    cutoff_in_progress = QtCore.pyqtSignal(bool, name = 'cutoff_in_progress')
    inelasticBG_in_progress = QtCore.pyqtSignal(bool, name = 'inelasticBG_in_progress')
    
    def __init__(self, parent = None):
        
        super(CommandCenter, self).__init__()
        
        self.buttons = list()
        
        self.initUI()
        self.initLayout()
        self.connectSignals()
    
    def initUI(self):
        
        #Buttons
        self.image_center_btn = QtGui.QPushButton('Find image center', parent = self)
        self.mask_rect_btn = QtGui.QPushButton('Set beamblock mask', parent = self)
        self.cutoff_btn = QtGui.QPushButton('Set cutoff', parent = self)
        self.inelastic_btn = QtGui.QPushButton('Fit inelastic BG', parent = self)
        
        self.operation_buttons = [self.image_center_btn, self.mask_rect_btn, self.cutoff_btn, self.inelastic_btn]
        for btn in self.operation_buttons:
            btn.setCheckable(True)
        
        self.confirm_btn = QtGui.QPushButton('Apply', parent = self)
        
    def initLayout(self):
        self.layout = QtGui.QVBoxLayout()
        for operation_btn in self.operation_buttons:
            self.layout.addWidget(operation_btn)
        
        self.layout.addWidget(self.confirm_btn)
        self.setLayout(self.layout)
        
    def connectSignals(self):
        
        self.image_center_btn.toggled.connect(self.handleImageCenter)
        self.mask_rect_btn.toggled.connect(self.handleMaskRect)
        self.cutoff_btn.toggled.connect(self.handleCutoff)
        self.inelastic_btn.toggled.connect(self.handleInelasticBG)
    
    @QtCore.pyqtSlot(bool)
    def handleImageCenter(self, is_checked):
        if is_checked:
            self.get_image_center_signal.emit()
        else:
            self.set_image_center_signal.emit()
        self.image_center_in_progress.emit(is_checked)
    
    @QtCore.pyqtSlot(bool)
    def handleMaskRect(self, is_checked):
        if is_checked:
            self.get_mask_rect_signal.emit()
        else:
            self.set_mask_rect_signal.emit()
        self.mask_rect_in_progress.emit(is_checked)
    
    @QtCore.pyqtSlot(bool)
    def handleCutoff(self, is_checked):
        if is_checked:
            self.get_cutoff_signal.emit()
        else:
            self.set_cutoff_signal.emit()
        self.cutoff_in_progress.emit(is_checked)
    
    @QtCore.pyqtSlot(bool)
    def handleInelasticBG(self, is_checked):
        if is_checked:
            self.get_inelasticBG_signal.emit()
        else:
            self.set_inelasticBG_signal.emit()
        self.inelasticBG_in_progress.emit(is_checked)

class Shell(QtGui.QMainWindow):
    
    def __init__(self):
        """ 
        This object acts as the glue between other components. 
        Most importantly, it connects signals between components
        """
        
        super(Shell, self).__init__()
        
        self.initUI()
        self.initLayout()
        self.connectSignals()
    
    def initUI(self):
        
        self.directory_widget = DirectoryHandler(parent = self)
        self.status_widget = StatusWidget(parent = self)
        self.command_center = CommandCenter(parent = self)
        self.data_handler = DataHandler(parent = self)
        self.image_viewer = ImageViewer(parent = self)
    
    def initLayout(self):
                
        self.command_pane = QtGui.QVBoxLayout()
        self.command_pane.addWidget(self.directory_widget)
        self.command_pane.addWidget(self.status_widget)
        self.command_pane.addWidget(self.command_center)
        
        self.viewer_pane = QtGui.QHBoxLayout()
        self.viewer_pane.addWidget(self.image_viewer)
        
        self.layout = QtGui.QHBoxLayout()
        self.layout.addLayout(self.command_pane)
        self.layout.addLayout(self.viewer_pane)
        
        #Create window
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        #Window settings ------------------------------------------------------
        self.setGeometry(600, 600, 350, 300)
        self.setWindowTitle('UED Powder Analysis Software')
        self.centerWindow()
        self.show()
    
    def connectSignals(self):
        
        #Connect directory_handler to data handler
        self.directory_widget.dataset_directory_signal.connect(self.data_handler.createDiffractionDataset)
        self.directory_widget.preprocess_signal.connect(self.data_handler.preprocessImages)
        
        #Connect data handler to status widget
        self.data_handler.has_image_center_signal.connect(self.status_widget.imageCenterToggle)
        self.data_handler.has_mask_rect_signal.connect(self.status_widget.maskRectToggle)
        self.data_handler.has_cutoff_signal.connect(self.status_widget.cutoffToggle)
        self.data_handler.has_inelasticBG_signal.connect(self.status_widget.inelasticBGToggle)
        
        #Connect commands from the command center to image viewer
        self.command_center.get_image_center_signal.connect(self.image_viewer.displayCenterFinder)
        self.command_center.get_mask_rect_signal.connect(self.image_viewer.displayMask)
        self.command_center.get_cutoff_signal.connect(self.image_viewer.displayCutoff)
        self.command_center.get_inelasticBG_signal.connect(self.image_viewer.displayInelasticBG)
        
        self.command_center.set_image_center_signal.connect(self.image_viewer.returnImageCenter)
        self.command_center.set_mask_rect_signal.connect(self.image_viewer.returnMaskRect)
        self.command_center.set_cutoff_signal.connect(self.image_viewer.returnCutoff)
        self.command_center.set_inelasticBG_signal.connect(self.image_viewer.returnInelasticBG)
        
        #Connect data transfer from image viewer to data handler
        self.image_viewer.image_center_signal.connect(self.data_handler.setImageCenter)
        self.image_viewer.mask_rect_signal.connect(self.data_handler.setMaskRect)
        self.image_viewer.cutoff_signal.connect(self.data_handler.setCutoff)
        self.image_viewer.inelastic_BG_signal.connect(self.data_handler.setInelasticBG)
        
        #connect in-progress to status widget
        self.data_handler.image_preprocessing_in_progress_signal.connect(self.directory_widget.preprocessInProgress)
        self.data_handler.image_preprocessed_signal.connect(self.directory_widget.preprocessComplete)
        
        self.command_center.image_center_in_progress.connect(self.status_widget.imageCenterInProgress)
        self.command_center.mask_rect_in_progress.connect(self.status_widget.maskRectInProgress)
        self.command_center.cutoff_in_progress.connect(self.status_widget.cutoffInProgress)
        self.command_center.inelasticBG_in_progress.connect(self.status_widget.inelasticBGInProgress)
        

        
    def centerWindow(self):
        """ Centers the window """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
# -----------------------------------------------------------------------------
#           MAIN WINDOW CLASS
# -----------------------------------------------------------------------------
    
class UEDpowder(QtGui.QMainWindow):
    """
    Main application window
    
    Attributes
    ----------
    image_center : list, shape (2,)
        [x,y]-coordinates of the image center
    image : ndarray, shape (N,N)
        ndarray corresponding to the data TIFF file
    radial_average = [r,i, name] : list, shape (3,)
        list of 2 ndarrays, shape (M,). r is the radius array, and i is the radially-averaged intensity. 'name' is a string used to make the plot legend.
    state : string
        Value describing in what state the software is. Possible values are:
            state in ['initial','data loaded', 'center guessed', 'radius guessed', 'center found', 'radial averaged', 'radial cutoff', 'background guessed', 'background determined', 'background substracted']
    raw_radial_average : list of lists
        List of the form [[radii1, pattern1, name1], [radii2, pattern2, name2], ... ], : list of ndarrays, shapes (M,), and an ID string
    """
    def __init__(self):
        
        #Attributes
        self.work_thread = None
        self.image_center = list()
        self.image = None
        self.guess_center = None
        self.guess_radius = None
        self.cutoff = list()
        self.background_fit = list()
        self.substrate_image = None
        self.raw_radial_average = None       #Before inelastic background substraction
        self.voigt_profiles = list()
        self.radial_average = list()         #After inelastic background substraction
        self.background_guesses = list()
        self.background_fit = None
        self._state = None
        
        self.diffractionDataset = None
        
        #Initialize
        super(UEDpowder, self).__init__()     #inherit from the constructor of QMainWindow        
        self.initUI()
        self.state = 'initial'
        
    # -------------------------------------------------------------------------
    #           SETTER METHODS FOR THREADING
    # -------------------------------------------------------------------------
        
    def setRawRadialAverage(self, value):
        """ Handles the radial averages for both the diffraction pattern adn the substrate from a thread. """
        self.raw_radial_average = value        
        self.viewer.displayRadialPattern(self.raw_radial_average)
        self.state = 'radial averaged'
    
    def setImageCenter(self, value):
        self.image_center = value[0:2] 
        print 'Calculated center: {0}'.format(self.image_center)
        circle = generateCircle(value[0], value[1], value[2])
        self.state = 'center found'
        self.viewer.displayMask()
        self.viewer.displayImage(self.image, overlay = circle, overlay_color = 'g')
    # -------------------------------------------------------------------------
        
        
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        assert value in ['initial','data loaded', 'center guessed', 'radius guessed', 'center found', 'radial averaged', 'cutoff set', 'radial cutoff', 'background guessed','background determined', 'background substracted']
        self._state = value
        print 'New state: ' + self._state
        self.updateButtonAvailability()     #Update which buttons are valid
        self.updateInstructions()
        
    def initUI(self):
        
        # ---------------------------------------------------------------------
        #       WIDGETS
        # ---------------------------------------------------------------------
        
        self.statusBar().showMessage('Ready')    #Top status bar
        
        #Set up state buttons
        self.acceptBtn = QtGui.QPushButton('Accept', self)
        self.acceptBtn.setIcon(QtGui.QIcon('images\checkmark.png'))
        
        self.rejectBtn = QtGui.QPushButton('Reject', self)
        self.rejectBtn.setIcon(QtGui.QIcon('images\cancel.png'))
        
        self.turboBtn = QtGui.QPushButton('Turbo Mode', self)
        self.turboBtn.setIcon(QtGui.QIcon('images\close.png'))
        self.turboBtn.setCheckable(True)

        self.batchAverageBtn = QtGui.QPushButton('Preprocess images', self)
        self.batchAverageBtn.setIcon(QtGui.QIcon('images\save.png'))
        self.batchAverageBtn.setEnabled(False)
        
        #Set up message boxes
        self.initial_message = QtGui.QLabel('Step 1: Select TIFF image.')
        
        #For initial state box
        initial_box = QtGui.QHBoxLayout()
        initial_box.addWidget(self.initial_message)
        self.imageLocatorBtn = QtGui.QPushButton('Open diffraction image', self)
        self.imageLocatorBtn.setIcon(QtGui.QIcon('images\diffraction.png'))
        initial_box.addWidget(self.imageLocatorBtn)

        #For image center select box
        center_box = QtGui.QHBoxLayout()
        center_box.addWidget(QtGui.QLabel('Step 2: Find the center of the image'))
        self.executeCenterBtn = QtGui.QPushButton('Find center', self)
        self.executeCenterBtn.setIcon(QtGui.QIcon('images\science.png'))
        center_box.addWidget(self.executeCenterBtn)
        
        #For beamblock radial cutoff box
        beamblock_box = QtGui.QHBoxLayout()
        beamblock_box.addWidget(QtGui.QLabel('Step 3: Select a radial cutoff for the beamblock'))
        self.executeRadialCutoffBtn = QtGui.QPushButton('Cutoff radial patterns', self)
        self.executeRadialCutoffBtn.setIcon(QtGui.QIcon('images\science.png'))
        beamblock_box.addWidget(self.executeRadialCutoffBtn)
        
        #For the inelastic scattering correction
        inelastic_box = QtGui.QHBoxLayout()
        inelastic_box.addWidget(QtGui.QLabel('Step 4: Remove inelastic scattering background from data.'))
        self.executeInelasticBtn = QtGui.QPushButton('Fit', self)
        self.executeInelasticBtn.setIcon(QtGui.QIcon('images\science.png'))
        inelastic_box.addWidget(self.executeInelasticBtn)
        
        #For the batch processing box
        batch_process_box = QtGui.QHBoxLayout()
        batch_process_box.addWidget(QtGui.QLabel('Step 5: Initiate batch processing.'))
        self.executeBatchProcessingBtn = QtGui.QPushButton('Batch process', self)
        self.executeBatchProcessingBtn.setIcon(QtGui.QIcon('images\science.png'))
        batch_process_box.addWidget(self.executeBatchProcessingBtn)
        
        #For instructions and printing
        instruction_box = QtGui.QVBoxLayout()
        instruction_box.addWidget(QtGui.QLabel('Instructions:'))
        self.instructions = QtGui.QTextEdit()
        self.instructions.setOverwriteMode(False)
        self.instructions.setReadOnly(True)
        self.instructions.setMaximumWidth(500)
        instruction_box.addWidget(self.instructions)

        #Set up the ImageViewer and CurveViewer
        self.viewer = ImageViewer(parent = self)
        
        #File dialog
        self.file_dialog = QtGui.QFileDialog()

        # ---------------------------------------------------------------------
        #       SIGNALS
        # ---------------------------------------------------------------------
        
        #Connect the image locator button to the file dialog
        self.imageLocatorBtn.clicked.connect(self.imageLocator)
        self.acceptBtn.clicked.connect(self.acceptState)
        self.rejectBtn.clicked.connect(self.rejectState)
        self.executeCenterBtn.clicked.connect(self.executeStateOperation)
        self.executeRadialCutoffBtn.clicked.connect(self.executeStateOperation)
        self.executeInelasticBtn.clicked.connect(self.executeStateOperation)
        self.executeBatchProcessingBtn.clicked.connect(self.executeStateOperation)
        self.batchAverageBtn.clicked.connect(self.batchAverageOperation)
        
        #Click events from the ImageViewer
        self.viewer.image_clicked.connect(self.click)
        self.viewer.curve_clicked.connect(self.click)
        
        # ---------------------------------------------------------------------
        #       LAYOUT
        # ---------------------------------------------------------------------
        
        #Accept - reject buttons combo
        state_controls = QtGui.QHBoxLayout()
        state_controls.addWidget(self.acceptBtn)
        state_controls.addWidget(self.rejectBtn)
        state_controls.addWidget(self.batchAverageBtn)
        state_controls.addWidget(self.turboBtn)
        
        # State boxes ---------------------------------------------------------
        self.state_boxes = QtGui.QVBoxLayout()
        self.state_boxes.addLayout(initial_box)
        self.state_boxes.addLayout(center_box)
        self.state_boxes.addLayout(beamblock_box)
        self.state_boxes.addLayout(inelastic_box)
        self.state_boxes.addLayout(batch_process_box)
        self.state_boxes.addLayout(instruction_box)
        self.state_boxes.addLayout(state_controls)
        
        #Image viewer pane ----------------------------------------------------
        self.right_pane = QtGui.QHBoxLayout()
        self.right_pane.addWidget(self.viewer)
        
        #Master Layout --------------------------------------------------------
        grid = QtGui.QHBoxLayout()
        
        grid.addLayout(self.state_boxes)
        grid.addLayout(self.right_pane)
        
        #Set master layout  ---------------------------------------------------
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)
        
        #Window settings ------------------------------------------------------
        self.setGeometry(600, 600, 350, 300)
        self.setWindowTitle('UED Powder Analysis Software')
        self.centerWindow()
        self.show()
    
    @QtCore.pyqtSlot(tuple)
    def click(self, pos):
        """
        Executes actions based on click signals from the image viewer
        """

        print '(x, y) = ({0}, {1})'.format(pos[0], pos[1]) 
            
        if self.state == 'center guessed':
            self.guess_radius = n.linalg.norm(n.asarray(self.guess_center) - n.asarray(pos))
            circle_guess = generateCircle(self.guess_center[0], self.guess_center[1], self.guess_radius)
            self.viewer.displayImage(self.image, overlay = circle_guess, overlay_color = 'r')
            self.state = 'radius guessed'
            
        elif self.state == 'radial averaged' or self.state == 'cutoff set':
            self.cutoff = pos
            self.axes.axvline(self.cutoff[0],ymax = self.axes.get_ylim()[1], color = 'k')
            self.draw()
            self.state = 'cutoff set'
                
        elif self.state == 'radial cutoff' or self.state == 'background guessed':
            self.background_guesses.append(pos)
            self.axes.axvline(pos[0],ymax = self.axes.get_ylim()[1])
            self.draw()
            #After 6th guess, change state to 'background guessed', but leave the possibility of continue guessing
            if len(self.background_guesses) == 6:
                self.state = 'background guessed'
    
    def updateInstructions(self, message = None):
        """ Handles the instructions text, either a specific message or a preset message depending on the state """
        
        if message is not None:
            self.instructions.append(message)
        else:           #Handle state changes
            if self.state == 'initial':
                self.instructions.append('\n Click the "Locate diffraction image" button to import a diffraction image.')
            elif self.state == 'data loaded':
                self.instructions.append('\n Click on the image to guess a diffraction pattern center.')
            elif self.state == 'center found':
                self.instructions.append('\n "Accept" to radially average the data, or click "Reject" to guess for a center again.')
            elif self.state == 'radial averaged':
                self.instructions.append('\n Click on the pattern to cutoff the area affected by the beam block.')
            elif self.state == 'cutoff set':
                self.instructions.append('\n Click on "Cutoff radial patterns" if you are satisfied with the cutoff or click again on the image viewer to set a new cutoff.')
            elif self.state == 'radial cutoff':
                self.instructions.append('\n Click on the image viewer to select sampling points to fit a background to.')
            elif self.state == 'background guessed':
                self.instructions.append('\n Click on the "Fit" button to fit the inelastic background, or click on the image viewer for more sampling points.')
            elif self.state == 'background determined':
                self.instructions.append('\n Accept the background fit to substract it from the data, or reject to sample the background again.')
            elif self.state == 'background substracted':
                pass

    def updateButtonAvailability(self):
        """
        """
        #Create list of buttons to be disabled and enables
        availableButtons = list()
        unavailableButtons = [self.imageLocatorBtn, self.executeCenterBtn, self.executeInelasticBtn, self.acceptBtn, self.rejectBtn, self.batchAverageBtn, self.executeRadialCutoffBtn, self.executeBatchProcessingBtn]
        
        if self.state == 'initial':
            availableButtons = [self.imageLocatorBtn]
        elif self.state == 'data loaded':
            availableButtons = [self.imageLocatorBtn, self.batchAverageBtn, self.executeCenterBtn]
        elif self.state == 'center found':
            availableButtons = [self.imageLocatorBtn, self.acceptBtn, self.rejectBtn, self.batchAverageBtn]
        elif self.state == 'radial averaged':
            availableButtons = [self.imageLocatorBtn, self.batchAverageBtn]
        elif self.state == 'cutoff set':
            availableButtons = [self.imageLocatorBtn, self.executeRadialCutoffBtn, self.batchAverageBtn]
        elif self.state == 'radial cutoff':
            availableButtons = [self.imageLocatorBtn, self.batchAverageBtn, self.acceptBtn]
        elif self.state == 'background guessed':
            availableButtons = [self.imageLocatorBtn, self.executeInelasticBtn, self.batchAverageBtn]
        elif self.state == 'background determined':
            availableButtons = [self.imageLocatorBtn, self.acceptBtn, self.rejectBtn, self.batchAverageBtn]
        elif self.state == 'background substracted':
            availableButtons = [self.imageLocatorBtn, self.batchAverageBtn, self.executeBatchProcessingBtn]
            
        #Act!
        for btn in unavailableButtons:
            btn.setEnabled(False)
        for btn in availableButtons:
            btn.setEnabled(True)
        
    def imageLocator(self):
        """ File dialog that selects the TIFF image file to be processed. """
        filename = self.file_dialog.getOpenFileName(self, 'Open image', 'C:\\')
        filename = os.path.abspath(filename)
        #Create diffraction dataset for upcoming batch processing
        #Check if folder is 'processed'. If so, back up one directory. This
        #makes it possible to re-analyze averaged images without having to keep
        #unaverages data on the computer (it takes lots of space)
        last_directory = os.path.dirname(filename).split('\\')[-1]
        if last_directory == 'processed':
            directory = os.path.dirname(os.path.dirname(filename)) #If directory is 'processed', back up one directory
        else:
            directory = os.path.dirname(filename)
            
        self.diffractionDataset = DiffractionDataset(directory, resolution = (2048,2048))
        self.batchAverageBtn.setEnabled(True)
        self.loadImage(filename)
        self.viewer.displayImage(self.image)     #display raw image
    
    def batchAverageOperation(self):
        #TODO: set up some way of notifying users that the process is done. Popup screen?
        # http://stackoverflow.com/questions/4838890/python-pyqt-popup-window
        self.work_thread = WorkThread(self.diffractionDataset.batchAverage, True) 
        self.work_thread.start()
        
    def loadImage(self, filename):
        """ Loads an image (and the associated background image) and sets the first state. """
        self.image = n.array(Image.open(filename), dtype = n.float)    
        self.state = 'data loaded'
        self.viewer.displayCenterFinder()
        
    def acceptState(self):
        """ Master accept function that validates a state and proceeds to the next one. """
        
        if self.state == 'center found':
            mask = self.viewer.maskPosition()
            print 'Mask dimensions: x1, x2, y1, y2 = {0}'.format(mask)
            self.work_thread = WorkThread(radialAverage, self.image, 'Sample', self.image_center, mask)              #Create thread with a specific function and arguments
            self.connect(self.work_thread, QtCore.SIGNAL('Computation done'), self.setRawRadialAverage) #Connect the outcome with a setter method
            self.connect(self.work_thread, QtCore.SIGNAL('Display Loading'), self.updateInstructions)
            self.connect(self.work_thread, QtCore.SIGNAL('Remove Loading'), self.updateInstructions)

            self.work_thread.start()                                                                    #Compute stuff        
            self.viewer.hideMask()
        
        elif self.state == 'radial cutoff':
            self.background_fit = RadialCurve()
            self.state = 'background substracted'
        elif self.state == 'background determined':
            #Subtract backgrounds
            self.radial_average = self.raw_radial_average - self.background_fit            #Substract inelastic background from data
            self.viewer.displayRadialPattern(self.radial_average)
            self.state = 'background substracted'
            
    def rejectState(self):
        """ Master reject function that invalidates a state and reverts to an appropriate state. """
        if self.state == 'center found':
            #Go back to the data loaded state and forget the guessed for the center and radius
            self.viewer.hideMask()
            self.state = 'data loaded'
            self.viewer.center, self.viewer.circle = None, None
            self.viewer.displayImage(self.image)
        elif self.state == 'background determined':
            self.viewer.displayRadialPattern(self.raw_radial_average)
            self.background_guesses = list()
            self.state = 'radial cutoff'

    def executeStateOperation(self):
        """ Placeholder function to confirm that computation may proceed in certain cases. """
            
        if self.state == 'data loaded':
            #Get center from fitted circle
            self.image_center = self.viewer.centerPosition()
            print 'center: {0}'.format(self.image_center)
            self.viewer.hideCenterFinder()
            self.viewer.displayMask()
            self.state = 'center found'
        elif self.state == 'cutoff set':
            #Cutoff radial patterns
            self.raw_radial_average = self.raw_radial_average.cutoff(self.cutoff)
            self.viewer.displayRadialPattern(self.raw_radial_average)
            self.state = 'radial cutoff'        
        elif self.state == 'background guessed':
            #Create guess data
            self.background_fit = self.raw_radial_average.inelasticBG(self.background_guesses)
            self.viewer.displayRadialPattern([self.raw_radial_average, self.background_fit])
            self.state = 'background determined'
        elif self.state == 'background substracted':
            # UEDPowder will skip reaveraging images if the folder already exists
            self.diffractionDataset.batchAverage(check_for_averages = True)
            #self.diffractionDataset.batchProcess(self.image_center, self.cutoff, self.background_fit, 'pumpon')
            self.diffractionDataset.batchProcess(self.image_center, self.cutoff)
            self.instructions.append('\n Batch radial averages exported. See processed/radial_averages.hdf5 in the data directory.')

    def centerWindow(self):
        """ Centers the window """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()
    

def run():
    app = QtGui.QApplication(sys.argv)
    analysisWindow = UEDpowder()
    analysisWindow.showMaximized()
    sys.exit(app.exec_())

def testDirectoryHandler():
    app = QtGui.QApplication(sys.argv)
    test = DirectoryHandler()
    test.show()
    sys.exit(app.exec_())
    
def testStatus():
    app = QtGui.QApplication(sys.argv)
    test = StatusWidget()
    test.show()
    sys.exit(app.exec_())

def testCommandCenter():
    app = QtGui.QApplication(sys.argv)
    test = CommandCenter()
    test.show()
    sys.exit(app.exec_())

def testShell():
    app = QtGui.QApplication(sys.argv)
    test = Shell()
    test.show()
    sys.exit(app.exec_())
    
#Run
if __name__ == '__main__':
    pass