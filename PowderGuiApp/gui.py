# -*- coding: utf-8 -*-

import sys
import os.path

#Core functions
from dataset import DiffractionDataset

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
    reset_signal = QtCore.pyqtSignal(bool, name = 'reset_signal')
    
    def __init__(self, parent = None):
        
        super(DirectoryHandler, self).__init__()
        
        self.initUI()
        self.initLayout()
        self.connectSignals()    
    
    def initUI(self):
        
        self.file_dialog = QtGui.QFileDialog(parent = self)        
        self.directory_btn = QtGui.QPushButton('Select directory', parent = self)

    def initLayout(self):
        
        self.file_dialog_row = QtGui.QHBoxLayout()
        self.file_dialog_row.addWidget(self.directory_btn)
        
        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(self.file_dialog_row)
        
        self.setLayout(self.layout)
    
    def connectSignals(self):
        self.directory_btn.clicked.connect(self.directoryLocator)        
        self.reset_signal.connect(self.reset)    
        
    def directoryLocator(self):
        """ 
        Activates a file dialog that selects the data directory to be processed. If the folder
        selected is one with processed images (then the directory name is C:\\...\\processed\\),
        return data 'root' directory.
        """
        
        directory = self.file_dialog.getExistingDirectory(self, 'Open diffraction dataset', 'C:\\')
        directory = os.path.abspath(directory)
        
#        #Check whether the directory name ends in 'processed'. If so, return previous directory
#        last_directory = possible_directory.split('\\')[-1]
#        if last_directory == 'processed':
#            directory = os.path.dirname(possible_directory) #If directory is 'processed', back up one directory
#        else:
#            directory = possible_directory
        
        self.reset_signal.emit(True)
        self.dataset_directory_signal.emit(directory)
    
    @QtCore.pyqtSlot(bool)
    def reset(self, hmm_really = True):
        pass

# -----------------------------------------------------------------------------
#           DATA HANDLER OBJECT
# -----------------------------------------------------------------------------

class DataHandler(QtCore.QObject):
    """
    Object tasked with all things computations and data.
    """
    image_preprocessing_in_progress_signal = QtCore.pyqtSignal(bool, name = 'image_preprocessing_in_progress_signal')
    image_preprocessed_signal = QtCore.pyqtSignal(bool, name = 'image_preprocessed_signal')
    
    #Export green light
    export_ready_signal = QtCore.pyqtSignal(bool, name = 'export_ready_signal')
    
    #Data to be plotted by the image viewer
    radial_average_signal = QtCore.pyqtSignal(object, name = 'radial_average_signal')
    image_signal = QtCore.pyqtSignal(object, name = 'image_signal')     #Three arguments to match ImageViewer.displayImage arguments
    
    has_image_center_signal = QtCore.pyqtSignal(bool, name = 'has_image_center_signal')
    has_mask_rect_signal = QtCore.pyqtSignal(bool, name = 'has_mask_rect_signal')
    has_cutoff_signal = QtCore.pyqtSignal(bool, name = 'has_cutoff_signal')
    has_inelasticBG_signal = QtCore.pyqtSignal(bool, name = 'has_inelasticBG_signal')
    
    def __init__(self, parent = None):
        
        super(DataHandler, self).__init__()
        
        #Data attributes
        self.image = None
        self.time = 0
        self.radial_curve = None            #Keep an unmodified copy as a reference 
        self.background_curve = None
        self.modified_radial_curve = None   #Modified from raw radial curve
        self.curve_shown = 0                #0 -> raw radial curve is shown. 1 -> modified curve
        
        self.diffraction_dataset = None
        self.image_center = tuple()
        self.mask_rect = tuple()
        self.cutoff = tuple()
        self._inelasticBGIntersects = list()
        
        #State attributes
        self.has_image_center = False
        self.has_mask_rect = False
        self.has_cutoff = False
        self.has_inelasticBG = False
        
        self.connectSignals()
    
    #Property to make sure that an inelastic BG curve is set every time intersects are set
    @property
    def inelasticBGIntersects(self):
        return self._inelasticBGIntersects
    
    @inelasticBGIntersects.setter
    def inelasticBGIntersects(self, list_of_points):
        self._inelasticBGIntersects = list_of_points
        if not list_of_points: #List is empty
            self.background_curve = None
        elif self.radial_curve is not None:
            self.background_curve = self.radial_curve.inelasticBG(points = list_of_points)
    
    def connectSignals(self):
        
        #Internal signals
        self.has_image_center_signal.connect(self.hasImageCenter)
        self.has_mask_rect_signal.connect(self.hasMaskRect)
        self.has_cutoff_signal.connect(self.hasCutoff)
        self.has_inelasticBG_signal.connect(self.hasInelasticBG)
        
        #Check radial average conditions at every update of image_center and mask_rect
        self.has_image_center_signal.connect(self.checkConditionsForRadialAverage)
        self.has_mask_rect_signal.connect(self.checkConditionsForRadialAverage)
        
        #Check if ready to export
        self.has_image_center_signal.connect(self.checkConditionForExport)
        self.has_mask_rect_signal.connect(self.checkConditionForExport)
        self.has_cutoff_signal.connect(self.checkConditionForExport)
        self.has_inelasticBG_signal.connect(self.checkConditionForExport)
    
    def getImage(self):
        """ 
        Returns an image from the directory//processed folder. Use images before 
        time 0 if possible.
        
        This method assumes that the images have been preprocessed (and so 
        directory//processed exists)
        """
        return self.diffraction_dataset.image(0)
        
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
        self.modified_radial_curve = self.radial_curve.cutoff(cutoff)
        self.has_cutoff_signal.emit(True)
    
    @QtCore.pyqtSlot(list)
    def setInelasticBG(self, intersects):
        self.inelasticBGIntersects = intersects
        # self.background_curve will be computed directly through the property
        self.modified_radial_curve = self.modified_radial_curve - self.background_curve
        self.has_inelasticBG_signal.emit(True)
    
    # -------------------------------------------------------------------------
    @QtCore.pyqtSlot(int)
    def whichCurveToPlot(self, index):
        curve_dict = {0: [self.radial_curve, self.background_curve], 1: self.modified_radial_curve}
        self.sendRadialAverage(curve_dict[index])
        
    # -------------------------------------------------------------------------
    #           INTERNAL SELF-UPDATING SLOTS
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot(bool)
    def hasImageCenter(self, flag):
        self.has_image_center = flag
    
    @QtCore.pyqtSlot(bool)
    def hasMaskRect(self, flag):
        self.has_mask_rect = flag
    
    @QtCore.pyqtSlot(bool)
    def hasCutoff(self, flag):
        self.has_cutoff = flag
    
    @QtCore.pyqtSlot(bool)
    def hasInelasticBG(self, flag):
        self.has_inelasticBG = flag
    
    #This slot takes in a boolean because of has_image_center_signal and
    #has_mask_rect_signal
    @QtCore.pyqtSlot(bool)      
    def checkConditionsForRadialAverage(self, flag):
        if self.has_image_center and self.has_mask_rect:
            self.radialAverage()
    
    #This slot takes in a boolean because of has_image_center_signal and
    #has_mask_rect_signal, etc...
    @QtCore.pyqtSlot(bool)
    def checkConditionForExport(self, flag):
        if (self.has_image_center and self.has_mask_rect and self.has_cutoff and self.has_inelasticBG):
            self.export_ready_signal.emit(True)
        
    # -------------------------------------------------------------------------
    #           HEAVY LIFTING 
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot(str)
    def createDiffractionDataset(self, directory):
        #TODO: Allow various resolutions. Setting page?
        self.diffraction_dataset = DiffractionDataset( directory )
        self.sendImage()
    
    def radialAverage(self):
        self.work_thread = WorkThread(self.diffraction_dataset.radial_average, self.time, self.image_center, self.mask_rect)
        #TODO: have some way of checking radial averaging progress
        self.work_thread.results_signal.connect(self.setRadialCurve)
        self.work_thread.start()
    
    @QtCore.pyqtSlot()
    def processDataset(self):
        if (self.has_image_center and self.has_mask_rect and self.has_cutoff and self.has_inelasticBG):
            self.work_thread = WorkThread(self.diffraction_dataset.process, center = self.image_center, cutoff = self.cutoff, inelastic_background = self.background_curve, mask_rect = self.mask_rect)
            self.work_thread.start()
            #TODO: What to do after processing?
            
    # -------------------------------------------------------------------------
    #           IMAGE PREPROCESSING SIGNAL HANDLING
    # -------------------------------------------------------------------------

    @QtCore.pyqtSlot(object)
    def setRadialCurve(self, curve):
        #Set radial curve and reset all other stuff
        self.radial_curve = curve
        self.modified_radial_curve = curve
        self.inelasticBGIntersects = list()
        
        #Plot
        self.sendRadialAverage([self.radial_curve, self.background_curve])
    
    @QtCore.pyqtSlot()
    def resetRadialCurve(self):
        self.modified_radial_curve = self.radial_curve.__copy__()
        
        self.cutoff = tuple()
        self.inelasticBGIntersects = list()
        
        self.has_cutoff_signal.emit(False)
        self.has_inelasticBG_signal.emit(False)
        
        self.resetRadialCurve()
        
    def sendImage(self):
        self.image = self.getImage()
        self.image_signal.emit(self.image)
        
    @QtCore.pyqtSlot(object)
    def sendRadialAverage(self, curve):
        self.radial_average_signal.emit(curve)
    
    # -------------------------------------------------------------------------
    
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
        
        self.ready_to_export_status = QtGui.QLabel(parent = self)
        self.ready_to_export_label = QtGui.QLabel('Ready to export:', parent = self)
        
        self.labels = [self.image_center_label, self.mask_rect_label, self.cutoff_label, self.inelasticBG_label, self.ready_to_export_label]
        self.statuses = [self.image_center_status, self.mask_rect_status, self.cutoff_status, self.inelasticBG_status, self.ready_to_export_status]
    
    def initLayout(self):
        """ Lays out components on the widget. """
        self.layout = QtGui.QHBoxLayout()
        labels_layout = QtGui.QVBoxLayout()
        statuses_layout = QtGui.QVBoxLayout()        
        for label, status in zip(self.labels, self.statuses):
            label.setAlignment(QtCore.Qt.AlignHCenter)
            status.setAlignment(QtCore.Qt.AlignHCenter)
            
            labels_layout.addWidget(label)
            statuses_layout.addWidget(status)
        
        self.layout.addLayout(labels_layout)
        self.layout.addLayout(statuses_layout)
        self.setLayout(self.layout)
    
    def initStatuses(self):
        toggle_methods = [self.imageCenterToggle, self.maskRectToggle, self.cutoffToggle, self.inelasticBGToggle, self.readyToExportToggle]
        for method in toggle_methods:
            method(complete = False)
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
    
    @QtCore.pyqtSlot(bool)
    def readyToExportToggle(self, complete = False):
        if complete:
            palette = complete_palette
            label = 'Ready'
        else:
            palette = incomplete_palette
            label = 'Not ready.'
            
        self.ready_to_export_status.setText(label)
        self.ready_to_export_status.setPalette(palette)
    
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
    
    #Switch between displaying raw radial curve or modified
    #   use: 0 = see raw radial curve
    #        1  = see modified radial curve
    switch_curve_signal = QtCore.pyqtSignal(int, name = 'switch_curve_signal')
    reset_curve_modifications_signal = QtCore.pyqtSignal(name = 'reset_curve_modifications')
    
    #Final export
    process_dataset_signal = QtCore.pyqtSignal(name = 'process_dataset_signal')
    
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
        self.reset_curve_mod_btn = QtGui.QPushButton('Reset curve modifications', parent = self)
        self.process_dataset_btn = QtGui.QPushButton('Process dataset', parent = self)
        
        #Curve view slider
        self.curve_picker = QtGui.QComboBox(parent = self)
        self.curve_picker.addItem('Raw radial average')
        self.curve_picker.addItem('Modified curve')
        
        self.operation_buttons = [self.image_center_btn, self.mask_rect_btn, self.cutoff_btn, self.inelastic_btn]
        for btn in self.operation_buttons:
            btn.setCheckable(True)
        
    def initLayout(self):
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.curve_picker)
        
        #Buttons
        for operation_btn in self.operation_buttons:
            self.layout.addWidget(operation_btn)
        self.layout.addWidget(self.reset_curve_mod_btn)
        self.layout.addWidget(self.process_dataset_btn)

        self.setLayout(self.layout)
        
    def connectSignals(self):
        
        self.image_center_btn.toggled.connect(self.handleImageCenter)
        self.mask_rect_btn.toggled.connect(self.handleMaskRect)
        self.cutoff_btn.toggled.connect(self.handleCutoff)
        self.inelastic_btn.toggled.connect(self.handleInelasticBG)
        
        self.curve_picker.activated.connect(self.curvePicked)
        self.reset_curve_mod_btn.clicked.connect(self.resetCurveModifications)
        self.process_dataset_btn.clicked.connect(self.processDataset)
        
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot(str)
    def curvePicked(self, item_index):        
        self.switch_curve_signal.emit(item_index)
    
    @QtCore.pyqtSlot()
    def resetCurveModifications(self):
        self.reset_curve_modifications_signal.emit()
    
    @QtCore.pyqtSlot()
    def processDataset(self):
        self.process_dataset_signal.emit()
    
    # -------------------------------------------------------------------------
    
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
        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('UED Powder Analysis Software')
        self.centerWindow()
        self.show()
    
    def connectSignals(self):
        
        #Connect curve picker, reset and final export buttons to data_handler
        self.command_center.switch_curve_signal.connect(self.data_handler.whichCurveToPlot)
        self.command_center.reset_curve_modifications_signal.connect(self.data_handler.resetRadialCurve)
        self.command_center.process_dataset_signal.connect(self.data_handler.processDataset)
        
        #Connect directory_handler to data handler
        self.directory_widget.dataset_directory_signal.connect(self.data_handler.createDiffractionDataset)
        
        #Connect data handler to status widget
        self.data_handler.has_image_center_signal.connect(self.status_widget.imageCenterToggle)
        self.data_handler.has_mask_rect_signal.connect(self.status_widget.maskRectToggle)
        self.data_handler.has_cutoff_signal.connect(self.status_widget.cutoffToggle)
        self.data_handler.has_inelasticBG_signal.connect(self.status_widget.inelasticBGToggle)
        self.data_handler.export_ready_signal.connect(self.status_widget.readyToExportToggle)
        
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
        
        self.data_handler.image_signal.connect(self.image_viewer.displayImage)
        self.data_handler.radial_average_signal.connect(self.image_viewer.displayRadialPattern)
        
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
    
    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()    
    
# -----------------------------------------------------------------------------
#           INDIVIDUAL COMPONENTS TESTING
# -----------------------------------------------------------------------------

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

def run():
    app = QtGui.QApplication(sys.argv)
    gui = Shell()
    gui.showMaximized()
    sys.exit(app.exec_())
    
#Run
if __name__ == '__main__':
    run()