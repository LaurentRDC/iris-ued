# -*- coding: utf-8 -*-

import sys
import os.path
import numpy as n
from PIL import Image

#Core functions
from core import *

#plotting backends
from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.backends.backend_qt4agg as qt4agg
from matplotlib.figure import Figure
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE

#GUI backends
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore


# -----------------------------------------------------------------------------
#           WORKING CLASS
# -----------------------------------------------------------------------------

class WorkThread(QtCore.QThread):
    """
    Object taking care of computations
    """
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
        self.emit(QtCore.SIGNAL('Display Loading'), '\n Computing...')
        self.result = self.function(*self.args, **self.kwargs)
        self.emit(QtCore.SIGNAL('Remove Loading'), '\n Done.')
        self.emit(QtCore.SIGNAL('Computation done'), self.result)
        
# -----------------------------------------------------------------------------
#           IMAGE VIEWER CLASSES AND FUNCTIONS
# -----------------------------------------------------------------------------

class ImageViewer(FigureCanvas):
    """
    Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    This object displays any plot we want: TIFF image, radial-averaged pattern, etc.
    
    Non-plotting Attributes
    ----------------------- 
    last_click_position : list, shape (2,)
        [x,y] coordinates of the last click. Clicking outside the data of the ImageViewer set last_click = [0,0]
    
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        #Non-plotting attributes
        self.parent = parent
        self.last_click_position = [0,0]
        
        #plot setup
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.hold(True)

        self.initialFigure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        #Set toolbar
        
        #connect clicking events
        self.mpl_connect('button_press_event', self.click)
    
    def click(self, event):
        
        self.parent.current_state.click(event)

 
    def displayImage(self, image, circle = None, center = None, colour = 'red'):
        pass
    def displayRadialPattern(self, data, **kwargs):
        pass

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
            state in ['initial','data loaded', 'center guessed', 'radius guessed', 'center found', 'radial averaged', 'radial cutoff', 'substrate background guessed', 'substrate background determined', 'background guessed', 'background determined']
    raw_radial_averages : list of lists
        List of the form [[radii1, pattern1, name1], [radii2, pattern2, name2], ... ], : list of ndarrays, shapes (M,), and an ID string
    """
    def __init__(self):
        
        #Attributes
        self.work_thread = None
        self.state = None
        
        #Initialize
        super(UEDpowder, self).__init__()     #inherit from the constructor of QMainWindow
        self.setupStates()        
        self.initUI()
        
    def setupStates(self):
        
        #Create instances
        initial_state = State(application = self)
        data_loaded_state = State(application = self)
        center_guessed_state = State(application = self)
        radius_guessed_state = State(application = self)
        center_found_state = State(application = self)
        radial_averaged_state = State(application = self)
        radial_cutoff_state = State(application = self)
        substrate_background_guessed = State(application = self)
        substrate_background_determined = State(application = self)
        data_background_guessed = State(application = self)
        data_background_determined = State(application = self)
        
        #Define previous states
        data_loaded_state.previous_state = initial_state
        center_guessed_state.previous_state = data_loaded_state
        radius_guessed_state.previous_state = center_guessed_state
        center_found_state.previous_state = radius_guessed_state
        radial_averaged_state.previous_state = radius_guessed_state
        radial_cutoff_state.previous_state = radial_averaged_state
        substrate_background_guessed.previous_state = radial_cutoff_state
        substrate_background_determined.previous_state = substrate_background_guessed
        data_background_guessed.previous_state = substrate_background_determined
        data_background_determined.previous_state = data_background_guessed
        
        #Define next states
        initial_state.next_state = data_loaded_state
        data_loaded_state.next_state = center_guessed_state
        center_guessed_state.next_state = radius_guessed_state
        radius_guessed_state.next_state = center_found_state
        center_found_state.next_state = radial_averaged_state
        radial_averaged_state.next_state = radial_cutoff_state
        radial_cutoff_state.next_state = substrate_background_guessed
        substrate_background_guessed.next_state = substrate_background_determined
        substrate_background_determined.next_state = data_background_guessed
        data_background_guessed.next_state = data_background_determined
        
#        #Define onclick methods
# TODO: Create on_click functions
#        initial_state.on_click = 
#        data_loaded_state.on_click = 
#        center_guessed_state.on_click = 
#        radius_guessed_state.on_click = 
#        center_found_state.on_click = 
#        radial_averaged_state.on_click =
#        radial_cutoff_state.on_click = 
#        substrate_background_guessed.on_click = 
#        substrate_background_determined.on_click = 
#        data_background_guessed.on_click = 
        
        #Define other attribute dictionnaries
        data_loaded.other = {'guess center': None}
        center_guessed_state.other = {'guess radius': None}
        radius_guessed_state.other = {'center': None}
        radial_averaged_state.other = {'cutoff': None}
        radial_cutoff_state.other = {'guess substrate background': None}
        
    def initUI(self):
        
        # ---------------------------------------------------------------------
        #       WIDGETS
        # ---------------------------------------------------------------------
        
        self.statusBar()    #Top status bar
        
        #Set up state buttons
        self.acceptBtn = QtGui.QPushButton('Accept', self)
        self.acceptBtn.setIcon(QtGui.QIcon('images\checkmark.png'))
        
        self.rejectBtn = QtGui.QPushButton('Reject', self)
        self.rejectBtn.setIcon(QtGui.QIcon('images\cancel.png'))
        
        self.turboBtn = QtGui.QPushButton('Turbo Mode', self)
        self.turboBtn.setIcon(QtGui.QIcon('images\close.png'))
        self.turboBtn.setCheckable(True)
        
        self.saveBtn = QtGui.QPushButton('Save', self)
        self.saveBtn.setIcon(QtGui.QIcon('images\save.png'))
        
        self.loadBtn = QtGui.QPushButton('Load', self)
        self.loadBtn.setIcon(QtGui.QIcon('images\locator.png'))
        
        #Set up message boxes
        self.initial_message = QtGui.QLabel('Step 1: Select TIFF image.')
        
        #Save-load box
        save_load_box = QtGui.QVBoxLayout()
        save_load_box.addWidget(QtGui.QLabel('Save state:'))
        save_load_box.addWidget(self.saveBtn)
        
        #For initial state box
        initial_box = QtGui.QHBoxLayout()
        initial_box.addWidget(self.initial_message)
        self.imageLocatorBtn = QtGui.QPushButton('Open diffraction image', self)
        self.imageLocatorBtn.setIcon(QtGui.QIcon('images\diffraction.png'))
        initial_box.addWidget(self.imageLocatorBtn)
        self.directoryLocatorBtn = QtGui.QPushButton('Open data directory', self)
        self.directoryLocatorBtn.setIcon(QtGui.QIcon('images\locator.png'))
        initial_box.addWidget(self.directoryLocatorBtn)
        
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
        substrate_inelastic_box = QtGui.QHBoxLayout()
        substrate_inelastic_box.addWidget(QtGui.QLabel('Step 4: Remove inelastic scattering background from substrate.'))
        self.executeSubstrateInelasticBtn = QtGui.QPushButton('Fit', self)
        self.executeSubstrateInelasticBtn.setIcon(QtGui.QIcon('images\science.png'))
        substrate_inelastic_box.addWidget(self.executeSubstrateInelasticBtn)
        
        #For the inelastic scattering correction
        inelastic_box = QtGui.QHBoxLayout()
        inelastic_box.addWidget(QtGui.QLabel('Step 5: Remove inelastic scattering background from data.'))
        self.executeInelasticBtn = QtGui.QPushButton('Fit', self)
        self.executeInelasticBtn.setIcon(QtGui.QIcon('images\science.png'))
        inelastic_box.addWidget(self.executeInelasticBtn)
        
        #For instructions and printing
        instruction_box = QtGui.QVBoxLayout()
        instruction_box.addWidget(QtGui.QLabel('Instructions:'))
        self.instructions = QtGui.QTextEdit()
        self.instructions.setOverwriteMode(False)
        self.instructions.setReadOnly(True)
        instruction_box.addWidget(self.instructions)

        #Set up ImageViewer
        self.image_viewer = ImageViewer(parent = self)
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
        self.executeSubstrateInelasticBtn.clicked.connect(self.executeStateOperation)
        self.executeInelasticBtn.clicked.connect(self.executeStateOperation)
        self.saveBtn.clicked.connect(self.save)
        self.loadBtn.clicked.connect(self.load)
        
        # ---------------------------------------------------------------------
        #       LAYOUT
        # ---------------------------------------------------------------------
        
        #Accept - reject buttons combo
        state_controls = QtGui.QHBoxLayout()
        state_controls.addWidget(self.acceptBtn)
        state_controls.addWidget(self.rejectBtn)
        state_controls.addWidget(self.saveBtn)
        state_controls.addWidget(self.loadBtn)
        state_controls.addWidget(self.turboBtn)
        
        # State boxes ---------------------------------------------------------
        state_boxes = QtGui.QVBoxLayout()
        state_boxes.addLayout(initial_box)
        state_boxes.addLayout(center_box)
        state_boxes.addLayout(beamblock_box)
        state_boxes.addLayout(substrate_inelastic_box)
        state_boxes.addLayout(inelastic_box)
        state_boxes.addLayout(instruction_box)
        state_boxes.addLayout(state_controls)
        
        #Image viewer pane ----------------------------------------------------
        right_pane = QtGui.QVBoxLayout()
        right_pane.addWidget(self.image_viewer)
        
        #Master Layout --------------------------------------------------------
        grid = QtGui.QHBoxLayout()
        grid.addLayout(state_boxes)
        grid.addLayout(right_pane)
        
        #Set master layout  ---------------------------------------------------
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)
        
        #Window settings ------------------------------------------------------
        self.setGeometry(600, 600, 350, 300)
        self.setWindowTitle('UED Powder Analysis Software')
        self.centerWindow()
        self.show()
    
    def updateInstructions(self, message = None):
        """ Handles the instructions text, either a specific message or a preset message depending on the state """
        
        if message != None:
            assert isinstance(message, str)
            self.instructions.append(message)
        else:           #Handle state changes
            if self.state == 'initial':
                self.instructions.append('\n Click the "Locate diffraction image" button to import a diffraction image.')
            elif self.state == 'data loaded':
                self.instructions.append('\n Click on the image to guess a diffraction pattern center.')
            elif self.state == 'center guessed':
                self.instructions.append('\n Click on a diffraction ring to guess a diffraction pattern radius.')
            elif self.state == 'radius guessed':
                self.instructions.append('\n Click on the "Find center" button to fit a circle to the diffraction ring you selected.')
            elif self.state == 'center found':
                self.instructions.append('\n Click the "Accept" button to radially average the data, or click "Reject" to guess for a center again.')
            elif self.state == 'radial averaged':
                self.instructions.append('\n Click on the pattern to cutoff the area affected by the beam block.')
            elif self.state == 'cutoff set':
                self.instructions.append('\n Click on the "Cutoff radial patterns" button if you are satisfied with the cutoff or click again on the image viewer to set a new cutoff.')
            elif self.state == 'radial cutoff':
                self.instructions.append('\n Click on the image viewer to select sampling points to fit a background to.')
            elif self.state == 'background guessed':
                self.instructions.append('\n Click on the "Fit" button to fit the inelastic background.')
            elif self.state == 'background determined':
                pass
            elif self.state == 'background substracted':
                pass

    def updateButtonAvailability(self):
        """
        """
        #Create list of buttons to be disabled and enables
        availableButtons = list()
        unavailableButtons = list()
        
        if self.state == 'initial':
            availableButtons = [self.imageLocatorBtn, self.loadBtn]
            unavailableButtons = [self.executeCenterBtn, self.executeInelasticBtn, self.acceptBtn, self.rejectBtn, self.saveBtn, self.executeRadialCutoffBtn, self.executeSubstrateInelasticBtn]
        elif self.state == 'data loaded':
            availableButtons = [self.imageLocatorBtn, self.loadBtn]
            unavailableButtons = [self.executeCenterBtn, self.executeInelasticBtn, self.acceptBtn, self.rejectBtn, self.saveBtn, self.executeRadialCutoffBtn, self.executeSubstrateInelasticBtn]
        elif self.state == 'center guessed':
            availableButtons = [self.imageLocatorBtn, self.acceptBtn, self.loadBtn]
            unavailableButtons = [self.executeCenterBtn, self.executeInelasticBtn, self.rejectBtn, self.saveBtn,self.executeSubstrateInelasticBtn, self.executeRadialCutoffBtn]
        elif self.state == 'radius guessed':
            availableButtons = [self.imageLocatorBtn, self.executeCenterBtn, self.loadBtn]
            unavailableButtons = [self.executeInelasticBtn, self.acceptBtn, self.rejectBtn, self.saveBtn,self.executeSubstrateInelasticBtn, self.executeRadialCutoffBtn]
        elif self.state == 'center found':
            availableButtons = [self.imageLocatorBtn, self.acceptBtn, self.rejectBtn, self.loadBtn]
            unavailableButtons = [self.executeCenterBtn, self.executeInelasticBtn, self.saveBtn,self.executeSubstrateInelasticBtn, self.executeRadialCutoffBtn]
        elif self.state == 'radial averaged':
            availableButtons = [self.imageLocatorBtn,  self.saveBtn, self.loadBtn]
            unavailableButtons = [self.acceptBtn, self.rejectBtn, self.executeCenterBtn, self.executeInelasticBtn,self.executeSubstrateInelasticBtn, self.executeRadialCutoffBtn]
        elif self.state == 'cutoff set':
            availableButtons = [self.imageLocatorBtn, self.executeRadialCutoffBtn, self.loadBtn]
            unavailableButtons = [self.acceptBtn, self.rejectBtn, self.executeCenterBtn, self.executeInelasticBtn,self.executeSubstrateInelasticBtn, self.saveBtn]
        elif self.state == 'radial cutoff':
            availableButtons = [self.imageLocatorBtn, self.saveBtn, self.loadBtn]
            unavailableButtons = [self.acceptBtn, self.rejectBtn, self.executeCenterBtn, self.executeInelasticBtn, self.executeRadialCutoffBtn]
        elif self.state == 'substrate background guessed':
            availableButtons = [self.imageLocatorBtn, self.executeSubstrateInelasticBtn, self.loadBtn]
            unavailableButtons = [self.acceptBtn, self.rejectBtn, self.executeInelasticBtn, self.executeCenterBtn, self.saveBtn, self.executeRadialCutoffBtn]
        elif self.state == 'substrate background determined':
            availableButtons = [self.imageLocatorBtn, self.acceptBtn, self.rejectBtn, self.saveBtn, self.loadBtn]
            unavailableButtons = [self.executeInelasticBtn, self.executeCenterBtn, self.executeSubstrateInelasticBtn, self.executeRadialCutoffBtn]
        elif self.state == 'background guessed':
            availableButtons = [self.imageLocatorBtn, self.executeInelasticBtn, self.loadBtn]
            unavailableButtons = [self.acceptBtn, self.rejectBtn, self.executeSubstrateInelasticBtn, self.executeCenterBtn, self.saveBtn, self.executeRadialCutoffBtn]
        elif self.state == 'background determined':
            availableButtons = [self.imageLocatorBtn, self.acceptBtn, self.rejectBtn, self.saveBtn, self.loadBtn]
            unavailableButtons = [self.executeInelasticBtn, self.executeCenterBtn, self.executeSubstrateInelasticBtn, self.executeRadialCutoffBtn]
        elif self.state == 'inelastic background substracted':
            availableButtons = [self.imageLocatorBtn, self.loadBtn]
            unavailableButtons = [self.executeInelasticBtn, self.executeCenterBtn, self.executeSubstrateInelasticBtn, self.executeRadialCutoffBtn, self.acceptBtn, self.rejectBtn, self.saveBtn]
        #Act!
        for btn in availableButtons:
            btn.setEnabled(True)
        for btn in unavailableButtons:
            btn.setEnabled(False)
        
    def imageLocator(self):
        """ File dialog that selects the TIFF image file to be processed. """
        filename = self.file_dialog.getOpenFileName(self, 'Open image', 'C:\\')
        self.loadImage(filename)
        self.image_viewer.displayImage(self.image)     #display raw image
        
    def acceptState(self):
        """ Master accept function that validates a state and proceeds to the next one. """
        
        if self.state == 'center guessed':
            #To speedup debugging, accept the guessed center as the true center and move on
            self.image_center = self.guess_center
            self.state = 'center found'
            self.acceptState()
            
        elif self.state == 'center found':

            self.work_thread = WorkThread(fc.radialAverage, [self.image, self.substrate_image], ['Raw', 'Subtrate'], self.image_center)              #Create thread with a specific function and arguments
            self.connect(self.work_thread, QtCore.SIGNAL('Computation done'), self.setRawRadialAverages) #Connect the outcome with a setter method
            self.connect(self.work_thread, QtCore.SIGNAL('Display Loading'), self.updateInstructions)
            self.connect(self.work_thread, QtCore.SIGNAL('Remove Loading'), self.updateInstructions)

            self.work_thread.start()                                                                    #Compute stuff
        elif self.state == 'substrate background determined':
            self.image_viewer.displayRadialPattern([self.raw_radial_averages[0]])
            self.background_guesses = list()
        
        elif self.state == 'background determined':
            #Subtract backgrounds
            self.radial_averages = list(self.raw_radial_averages)   #Copy list
            self.radial_averages[1][1] -= self.substrate_background_fit[1]  #Substract inelastic background from substrate data
            self.radial_averages[0][1] -= self.background_fit[1]            #Substract inelastic background from data
            self.image_viewer.displayRadialPattern(self.radial_averages)
            self.state = 'inelastic background substracted'
            
            
    def rejectState(self):
        """ Master reject function that invalidates a state and reverts to an appropriate state. """
        if self.state == 'center found':
            #Go back to the data loaded state and forget the guessed for the center and radius
            self.state = 'data loaded'
            self.image_viewer.guess_center, self.image_viewer.guess_radius = None, None
            self.image_viewer.displayImage(self.image)
        elif self.state == 'substrate background determined':
            #Go back to choosing the guesses for the background
            self.image_viewer.displayRadialPattern([self.raw_radial_averages[1]])
            self.background_guesses = list() #Clear guesses
            self.state = 'radial cutoff'
        elif self.state == 'background determined':
            self.image_viewer.displayRadialPattern([self.raw_radial_averages[0]])
            self.background_guesses = list()
            self.state = 'substrate background determined'

    def executeStateOperation(self):
        """ Placeholder function to confirm that computation may proceed in certain cases. """
        if self.state == 'center guessed':        
           pass
        elif self.state == 'radius guessed':
            #Compute center
            xg, yg = self.guess_center
            self.work_thread = WorkThread(fc.fCenter, xg, yg, self.guess_radius, self.image)
            self.connect(self.work_thread, QtCore.SIGNAL('Computation done'), self.setImageCenter)            
            self.connect(self.work_thread, QtCore.SIGNAL('Display Loading'), self.updateInstructions)
            self.connect(self.work_thread, QtCore.SIGNAL('Remove Loading'), self.updateInstructions)
            self.work_thread.start()
        elif self.state == 'cutoff set':
            #Cutoff radial patterns
            self.raw_radial_averages = fc.cutoff(self.raw_radial_averages, self.cutoff)
            self.image_viewer.displayRadialPattern([self.raw_radial_averages[1]], color = 'g')
            self.state = 'radial cutoff'
        elif self.state == 'substrate background guessed':
            #Create guess data
            self.substrate_background_fit = fc.inelasticBG(self.raw_radial_averages[1], self.background_guesses)
            self.image_viewer.displayRadialPattern([self.raw_radial_averages[1], self.substrate_background_fit])
            self.background_guesses = list()
            self.state = 'substrate background determined'
        
        elif self.state == 'background guessed':
            #Create guess data
            self.background_fit = fc.inelasticBG(self.raw_radial_averages[0], self.background_guesses)
            self.image_viewer.displayRadialPattern([self.raw_radial_averages[0], self.background_fit])
            self.state = 'background determined'
    
    def save(self):
        """ Determines what to do when the save button is clicked """
        from scipy.io import savemat
        if self.state == 'radial averaged':
            filename = self.file_dialog.getSaveFileName(self, 'Save radial averages', 'C:\\')
    
            mdict = {'rav_x':self.raw_radial_averages[0][0], 'rav_y': self.raw_radial_averages[0][1],
                     'rav_subs_x':self.raw_radial_averages[1][0], 'rav_subs_y': self.raw_radial_averages[1][1]}
            savemat(filename, mdict)
        self.instructions.append('\n Radial averages saved.')
            
    def load(self, filename = None):
        """ Determines what to do when the load button is clicked """
        from scipy.io import loadmat
        
        filename = self.file_dialog.getOpenFileName(self, 'Load radial averages', 'C:\\')
        
        mdict = dict()
        mdict = loadmat(filename)
        rav = [ mdict['rav_x'][0], mdict['rav_y'][0], 'Raw' ]
        rav_subs = [ mdict['rav_subs_x'][0], mdict['rav_subs_y'][0], 'Subs' ]
        self.raw_radial_averages = [rav, rav_subs]
        self.image_viewer.displayRadialPattern(self.raw_radial_averages)
        self.state = 'radial averaged'
        self.instructions.append('\n Radial averages loaded.')

    def centerWindow(self):
        """ Centers the window """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def loadImage(self, filename):
        """ Loads an image (and the associated background image) and sets the first state. """
        substrate_filename = os.path.normpath(os.path.join(os.path.dirname(filename), 'subs.tif'))
        background_filename = os.path.normpath(os.path.join(os.path.dirname(filename), 'bg.tif'))
        #Load images
        background = n.array(Image.open(background_filename), dtype = n.float)
        self.image = n.array(Image.open(filename), dtype = n.float) - background
        self.substrate_image = n.array(Image.open(substrate_filename), dtype = n.float) - background
        
        #Process data
        self.image[self.image < 0] = 0
        self.substrate_image[self.substrate_image < 0] = 0
    
        self.state = 'data loaded'
        
    def startLoading(self):
        self.instructions.append('\n loading...')
        
    def endLoading(self):
        self.instructions.append('\n loading...done')
    
    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()
        
#Run
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    analysisWindow = UEDpowder()
    analysisWindow.showMaximized()
    sys.exit(app.exec_())
