# -*- coding: utf-8 -*-

import sys
import os.path
import numpy as n
from PIL.Image import open

#Core functions
from coreV2 import *

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
        
        #connect clicking events
        self.mpl_connect('button_press_event', self.click)
    
    def initialFigure(self):
        """ Plots a placeholder image until an image file is selected """
        missing_image = n.zeros(shape = (1024,1024), dtype = n.uint8)
        self.axes.cla()     #Clear axes
        self.axes.imshow(missing_image)
    
    def click(self, event):
        #Record last click position
        if event.xdata == None or event.ydata == None:
            self.last_click_position = [0,0]
        else:
            self.last_click_position = [event.xdata, event.ydata]
            
        #Run specific click method
        self.parent.current_state.click(event)

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
        self._current_state = None
        
        #Initialize
        super(UEDpowder, self).__init__()     #inherit from the constructor of QMainWindow
        self.initUI()
        self.setupStates()
        self.current_state = self.initial_state        
        self.initLayout()

    @property
    def current_state(self):
        return self._current_state
    
    @current_state.setter
    def current_state(self, value):
        self._current_state = value
        
        #Plotting
        self.image_viewer.axes.cla()
        self.current_state.plot(self.image_viewer.axes)
        #TODO: figure out what to do when changing state
        print value     
        
    def setupStates(self):
        
        #Create instances
        self.initial_state = State(application = self, name = 'initial')
        self.data_loaded_state = State(application = self, name = 'data loaded')
        self.center_found_state = State(application = self, name = 'center found')
        self.radial_averaged_state = State(application = self, name = 'radial averaged')
        self.data_baseline_state = State(application = self, name = 'data baseline')
        
        #Define previous states
        self.data_loaded_state.previous_state = self.initial_state
        self.center_found_state.previous_state = self.data_loaded_state
        self.radial_averaged_state.previous_state = self.center_found_state
        self.data_baseline_state.previous_state = self.radial_averaged_state
        
        #Define next states
        self.initial_state.next_state = self.data_loaded_state
        self.data_loaded_state.next_state = self.center_found_state
        self.center_found_state.next_state = self.radial_averaged_state
        self.radial_averaged_state.next_state = self.data_baseline_state
        
        #Define execute method
        self.data_loaded_state.execute_method = computeCenter
        self.center_found_state.execute_method = radiallyAverage
        
        #Connect methods
        self.data_loaded_execute_btn.clicked.connect(self.executeClick)
        self.center_found_execute_btn.clicked.connect(self.executeClick)
        
        #Define onclick methods (default is nothingHappens)
        self.data_loaded_state.on_click = guessCenterOrRadius
        self.radial_averaged_state.on_click = cutoff
        
        #Define other attribute dictionnaries
        self.data_loaded_state.others = {'guess center': None, 'guess radius': None, 'substrate image': None}
        self.center_found_state.others = {'center': None, 'radius': None, 'substrate image': self.data_loaded_state.others['substrate image']}
        self.radial_averaged_state.others = {'cutoff': None}
        
        #Define specific plotting methods
        self.data_loaded_state.plotting_method = plotGuessCenter
        self.center_found_state.plotting_method = plotComputedCenter
        
        #Define layouts
        for state, button in zip([self.initial_state, self.data_loaded_state, self.center_found_state, self.radial_averaged_state, self.data_baseline_state],
                                 [[self.imageLocatorBtn, self.revertBtn], self.data_loaded_execute_btn, self.center_found_execute_btn, self.radial_averaged_execute_btn, self.data_baseline_execute_btn]):
            #Layout
            layout = QtGui.QVBoxLayout()
            layout.addWidget(QtGui.QLabel(state.instructions))
            if button != None:
                if isinstance(button, list):
                    for btn in button:
                        layout.addWidget(btn)
                else:    
                    layout.addWidget(button)
            self.bottom_pane.addLayout(layout)
    
    def executeClick(self):
        self.current_state.execute_method(self.current_state)
        
    def initUI(self):
        
        # ---------------------------------------------------------------------
        #       WIDGETS
        # ---------------------------------------------------------------------
        
        self.statusBar()    #Top status bar
        
        self.revertBtn = QtGui.QPushButton('Revert', self)
        self.revertBtn.setIcon(QtGui.QIcon('images\revert.png'))

        self.imageLocatorBtn = QtGui.QPushButton('Open diffraction image', self)
        self.imageLocatorBtn.setIcon(QtGui.QIcon('images\diffraction.png'))
        
        #State buttons
        self.data_loaded_execute_btn = QtGui.QPushButton('Find center')
        self.center_found_execute_btn = QtGui.QPushButton('Radially average')
        self.radial_averaged_execute_btn = QtGui.QPushButton('Remove background')
        self.data_baseline_execute_btn = QtGui.QPushButton('Dunno yet')
        
        #Set up ImageViewer
        self.image_viewer = ImageViewer(parent = self)
        self.file_dialog = QtGui.QFileDialog()
        
        self.bottom_pane = QtGui.QHBoxLayout()
        
        # ---------------------------------------------------------------------
        #       SIGNALS
        # ---------------------------------------------------------------------
        
        #Connect the image locator button to the file dialog
        self.imageLocatorBtn.clicked.connect(self.imageLocator)
        self.revertBtn.clicked.connect(self.revertState)
        
    def initLayout(self):
        #Image viewer pane ----------------------------------------------------
        top_pane = QtGui.QVBoxLayout()
        top_pane.addWidget(self.image_viewer)
        
        #Master Layout --------------------------------------------------------
        grid = QtGui.QVBoxLayout()
        grid.addLayout(top_pane)
        grid.addLayout(self.bottom_pane)
        
        #Set master layout  ---------------------------------------------------
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)
        
        #Window settings ------------------------------------------------------
        self.setGeometry(600, 600, 350, 300)
        self.setWindowTitle('UED Powder Analysis Software')
        self.centerWindow()
        self.show()
    
    def revertState(self):
        if self.current_state != self.initial_state:
            self.current_state = self.current_state.previous_state
            self.current_state.reset()

    def imageLocator(self):
        """ File dialog that selects the TIFF image file to be processed. """
        filename = self.file_dialog.getOpenFileName(self, 'Open image', 'C:\\')
        images = self.loadImage(filename)
        self.data_loaded_state.data = images[0]     #Sample image
        self.data_loaded_state.others['substrate image'] = images[1]    #Keeping substrate image
        self.current_state = self.data_loaded_state
        print 'data loaded'
    
    def loadImage(self, filename):
        """ Loads an image (and the associated background image) and sets the first state. """
        substrate_filename = os.path.normpath(os.path.join(os.path.dirname(filename), 'subs.tif'))
        background_filename = os.path.normpath(os.path.join(os.path.dirname(filename), 'bg.tif'))
        #Load images
        background = n.array(open(background_filename), dtype = n.float)
        image = n.array(open(filename), dtype = n.float) - background
        substrate_image = n.array(open(substrate_filename), dtype = n.float) - background
        
        #Process data
        image[image < 0] = 0
        substrate_image[substrate_image < 0] = 0
    
        return [Image(image, 'Sample'), Image(substrate_image, 'Substrate')]

    def centerWindow(self):
        """ Centers the window """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

#Run
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    analysisWindow = UEDpowder()
    analysisWindow.showMaximized()
    sys.exit(app.exec_())
