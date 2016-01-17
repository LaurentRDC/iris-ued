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
    Object taking care of threading computations
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
    
def generateCircle(xc, yc, radius):
    """
    Generates scatter value for a cicle centered at [xc,yc] of radius 'radius'.
    """
    xvals = xc+ radius*n.cos(n.linspace(0,2*n.pi,100))
    yvals = yc+ radius*n.sin(n.linspace(0,2*n.pi,100))
    return [xvals,yvals]

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
        #Setting determining the contrast of diffraction images. Pixel values above this will not be displayed, but still be used in the analysis
        self.max_count = 30000
        
        self.center = None
        self.circle = None
        self.overlay_color = 'red'
        
        #plot setup
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.hold(True)

        self.initialFigure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        #connect clicking events
        self.mpl_connect('button_press_event', self.click)
    
    def click(self, event):
        """
        Saves the position of the last click on the canvas
        """
        if event.xdata == None or event.ydata == None:
            self.last_click_position = [0,0]
        else:
            self.last_click_position = [event.xdata, event.ydata]
        
        if self.parent.state == 'data loaded':
            self.parent.guess_center = n.asarray(self.last_click_position)
            self.center = n.asarray(self.last_click_position); self.overlay_color = 'red'
            self.displayImage()
            self.parent.state = 'center guessed'
            
        elif self.parent.state == 'center guessed':
            ring_position = n.asarray(self.last_click_position)
            self.parent.guess_radius = n.linalg.norm(self.parent.guess_center - ring_position)
            circle_guess = generateCircle(self.parent.guess_center[0], self.parent.guess_center[1], self.parent.guess_radius)
            self.circle = circle_guess; self.overlay_color = 'red'
            self.displayImage()
            self.parent.state = 'radius guessed'
        
        elif self.parent.state == 'radial averaged' or self.parent.state == 'cutoff set':
            self.parent.cutoff = self.last_click_position
            self.axes.axvline(self.parent.cutoff[0],ymax = self.axes.get_ylim()[1], color = 'k')
            self.draw()
            self.parent.state = 'cutoff set'
                
        elif self.parent.state == 'radial cutoff' or self.parent.state == 'background guessed':
            self.parent.background_guesses.append(self.last_click_position)
            self.axes.axvline(self.last_click_position[0],ymax = self.axes.get_ylim()[1])
            self.draw()
            #After 6th guess, change state to 'background guessed', but leave the possibility of continue guessing
            if len(self.parent.background_guesses) == 6:
                self.parent.state = 'background guessed'

    def initialFigure(self):
        """ Plots a placeholder image until an image file is selected """
        missing_image = n.zeros(shape = (1024,1024), dtype = n.uint8)
        self.axes.cla()     #Clear axes
        self.axes.imshow(missing_image)
 
    def displayImage(self):
        """ 
        This method displays a raw TIFF image from the instrument. Optional arguments can be used to overlay a circle.
        """
        image = self.parent.image
        if image is None:
            self.initialFigure()
        else:
            self.axes.cla()     #Clear axes
            self.axes.imshow(image, vmin = image.min(), vmax = self.max_count)
            if self.center != None:
                self.axes.scatter(self.center[0],self.center[1], color = self.overlay_color)
                self.axes.set_xlim(0, image.shape[0])
                self.axes.set_ylim(image.shape[1],0)
            if self.circle != None:  #Overlay circle if provided
                xvals, yvals = self.circle
                self.axes.scatter(xvals, yvals, color = self.overlay_color)
                #Restrict view to the plotted circle (to better evaluate the fit)
                self.axes.set_xlim(xvals.min() - 10, xvals.max() + 10)
                self.axes.set_ylim(yvals.max() + 10, yvals.min() - 10)
            
            self.axes.set_title('Raw TIFF image')
            self.draw()
    
    def setContrast(self, max_val):
        self.max_count = max_val
        
    def updateContrast(self):
        if not self.axes.images:
            pass #image_viewer is currently displaying something else than a CCD image
        else:
            self.displayImage()
    
    def displayRadialPattern(self, data, **kwargs):
        """
        Plots one or more diffraction patterns.
        
        Parameters
        ----------
        *args : lists of RadialCurve objects
        """
        self.axes.cla()
        
        #Create a list with only one element if the data is not already in list form
        if isinstance(data, RadialCurve):
            data = [data]
        
        #Plot list of objects
        for item in data:
            item.plot(self.axes, **kwargs)
        
        self.draw()

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
        self.image_viewer.displayRadialPattern(self.raw_radial_average)
        self.state = 'radial averaged'
    
    def setImageCenter(self, value):
        self.image_center = value[0:2] 
        self.image_viewer.center = value[0:2]
        self.image_viewer.circle = generateCircle(value[0], value[1], value[2])
        self.state = 'center found'
        self.image_viewer.overlay_color = 'green'
        self.image_viewer.displayImage()
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
        instruction_box.addWidget(self.instructions)

        #Set up ImageViewer
        self.image_viewer = ImageViewer(parent = self)
        self.file_dialog = QtGui.QFileDialog()
        
        #Set up contrast slider
        self.contrast_slider = QtGui.QSlider(self)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(30000)
        self.contrast_slider.setValue(25000)
        
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
        self.connect(self.contrast_slider, QtCore.SIGNAL('valueChanged(int)'), self.image_viewer.setContrast)
        self.connect(self.contrast_slider, QtCore.SIGNAL('sliderReleased()'), self.image_viewer.updateContrast)
        
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
        state_boxes = QtGui.QVBoxLayout()
        state_boxes.addLayout(initial_box)
        state_boxes.addLayout(center_box)
        state_boxes.addLayout(beamblock_box)
        state_boxes.addLayout(inelastic_box)
        state_boxes.addLayout(batch_process_box)
        state_boxes.addLayout(instruction_box)
        state_boxes.addLayout(state_controls)
        
        #Image viewer pane ----------------------------------------------------
        right_pane = QtGui.QHBoxLayout()
        right_pane.addWidget(self.contrast_slider)
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
                self.instructions.append('\n Click on "Find center" to fit a circle to the diffraction ring you selected.')
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
            availableButtons = [self.imageLocatorBtn, self.batchAverageBtn]
        elif self.state == 'center guessed':
            availableButtons = [self.imageLocatorBtn, self.acceptBtn, self.batchAverageBtn]
        elif self.state == 'radius guessed':
            availableButtons = [self.imageLocatorBtn, self.executeCenterBtn, self.batchAverageBtn]
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
        self.image_viewer.displayImage()     #display raw image
    
    def batchAverageOperation(self):
        self.work_thread = WorkThread(self.diffractionDataset.batchAverage, True) 
        self.work_thread.start()
        
    def loadImage(self, filename):
        """ Loads an image (and the associated background image) and sets the first state. """
        image = n.array(Image.open(filename), dtype = n.float)
        
        #Load images if they exist
        background = self.diffractionDataset.pumpon_background
        substrate_image = self.diffractionDataset.substrate
                    
        #Process data
        #remove hot spots
        for im in [image, substrate_image, background]:
            im[im > 30000] = 0
        
        for im in [image, substrate_image]:
            im -= background
            im[im < 0] = 0
        
        #Substract substrate effects weighted by exposure
        self.image = image - (2.0/5.0)*substrate_image
    
        self.state = 'data loaded'
        
    def acceptState(self):
        """ Master accept function that validates a state and proceeds to the next one. """
        
        if self.state == 'center guessed':
            #To speedup debugging, accept the guessed center as the true center and move on
            self.image_center = self.guess_center
            self.state = 'center found'
            self.acceptState()
            
        elif self.state == 'center found':

            self.work_thread = WorkThread(radialAverage, self.image, 'Sample', self.image_center)              #Create thread with a specific function and arguments
            self.connect(self.work_thread, QtCore.SIGNAL('Computation done'), self.setRawRadialAverage) #Connect the outcome with a setter method
            self.connect(self.work_thread, QtCore.SIGNAL('Display Loading'), self.updateInstructions)
            self.connect(self.work_thread, QtCore.SIGNAL('Remove Loading'), self.updateInstructions)

            self.work_thread.start()                                                                    #Compute stuff        
        
        elif self.state == 'radial cutoff':
            self.background_fit = RadialCurve()
            self.state = 'background substracted'
        elif self.state == 'background determined':
            #Subtract backgrounds
            self.radial_average = self.raw_radial_average - self.background_fit            #Substract inelastic background from data
            self.image_viewer.displayRadialPattern(self.radial_average)
            self.state = 'background substracted'
            
    def rejectState(self):
        """ Master reject function that invalidates a state and reverts to an appropriate state. """
        if self.state == 'center found':
            #Go back to the data loaded state and forget the guessed for the center and radius
            self.state = 'data loaded'
            self.image_viewer.center, self.image_viewer.circle = None, None
            self.image_viewer.displayImage()
        elif self.state == 'background determined':
            self.image_viewer.displayRadialPattern(self.raw_radial_average)
            self.background_guesses = list()
            self.state = 'radial cutoff'

    def executeStateOperation(self):
        """ Placeholder function to confirm that computation may proceed in certain cases. """
        if self.state == 'center guessed':        
           pass
        elif self.state == 'radius guessed':
            #Compute center
            xg, yg = self.guess_center
            self.work_thread = WorkThread(fCenter, xg, yg, self.guess_radius, self.image)
            self.connect(self.work_thread, QtCore.SIGNAL('Computation done'), self.setImageCenter)            
            self.connect(self.work_thread, QtCore.SIGNAL('Display Loading'), self.updateInstructions)
            self.connect(self.work_thread, QtCore.SIGNAL('Remove Loading'), self.updateInstructions)
            self.work_thread.start()
        elif self.state == 'cutoff set':
            #Cutoff radial patterns
            self.raw_radial_average = self.raw_radial_average.cutoff(self.cutoff)
            self.image_viewer.displayRadialPattern(self.raw_radial_average)
            self.state = 'radial cutoff'        
        elif self.state == 'background guessed':
            #Create guess data
            self.background_fit = self.raw_radial_average.inelasticBG(self.background_guesses)
            self.image_viewer.displayRadialPattern([self.raw_radial_average, self.background_fit])
            self.state = 'background determined'
        elif self.state == 'background substracted':
            # UEDPowder will skip reaveraging images if the folder already exists
            self.diffractionDataset.batchAverage(check_for_averages = True)
            #self.diffractionDataset.batchProcess(self.image_center, self.cutoff, self.background_fit, 'pumpon')
            self.diffractionDataset.batchProcess(self.image_center, self.cutoff)
            self.instructions.append('\n Batch radial averages exported. See processed/radial_averages.hdf5 in the data directory.')
    
    def save(self):
        """ Determines what to do when the save button is clicked """
        from scipy.io import savemat
        if self.state == 'radial averaged':
            filename = self.file_dialog.getSaveFileName(self, 'Save radial averages', 'C:\\')
    
            mdict = {'rav_x':self.raw_radial_average.xdata, 'rav_y': self.raw_radial_average.ydata}
            savemat(filename, mdict)
        self.instructions.append('\n Radial averages saved.')
            
    def load(self, filename = None):
        """ Determines what to do when the load button is clicked """
        from scipy.io import loadmat
        
        filename = self.file_dialog.getOpenFileName(self, 'Load radial averages', 'C:\\')
        
        mdict = dict()
        mdict = loadmat(filename)
        self.raw_radial_average = RadialCurve(mdict['rav_x'][0], mdict['rav_y'][0], 'Raw' )
        self.image_viewer.displayRadialPattern(self.raw_radial_average)
        self.state = 'radial averaged'
        self.instructions.append('\n Radial averages loaded.')

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
   
#Run
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    analysisWindow = UEDpowder()
    analysisWindow.showMaximized()
    sys.exit(app.exec_())
