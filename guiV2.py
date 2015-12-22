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
        #Records last click position
        if event.xdata is None or event.ydata is None:
            self.last_click_position = [0,0]
        else:
            self.last_click_position = [event.xdata, event.ydata]
        
        self.parent.dataHandler.onClick()
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
        self.dataHandler = None
        
        #Initialize
        super(UEDpowder, self).__init__()     #inherit from the constructor of QMainWindow
        self.initUI()
        self.initLayout()
        self.dataHandler = DataHandler(self)
        
    def initUI(self):
        
        # ---------------------------------------------------------------------
        #       WIDGETS
        # ---------------------------------------------------------------------
        
        self.statusBar()    #Top status bar
        
        self.revertBtn = QtGui.QPushButton('Revert', self)
        self.revertBtn.setIcon(QtGui.QIcon('images\revert.png'))

        self.imageLocatorBtn = QtGui.QPushButton('Open diffraction image', self)
        self.imageLocatorBtn.setIcon(QtGui.QIcon('images\diffraction.png'))
        
        self.executeBtn = QtGui.QPushButton('Execute Operation')
        
        #State buttons
        self.findCenterBtn = QtGui.QPushButton('Find center')
        self.radiallyAverageBtn = QtGui.QPushButton('Radially average')
        self.cutoffBtn = QtGui.QPushButton('Cutoff')
        self.fitBackgroundBtn = QtGui.QPushButton('Fit background')
        self.batchProcessBtn = QtGui.QPushButton('Accept processing')
        
        #Set checkable
        for btn in [self.findCenterBtn, self.radiallyAverageBtn, self.cutoffBtn, self.fitBackgroundBtn, self.batchProcessBtn]:
            btn.setCheckable(True)
        
        #Connect toggleable buttons
        for btn, handle_function in zip([self.findCenterBtn, self.radiallyAverageBtn, self.cutoffBtn, self.fitBackgroundBtn, self.batchProcessBtn],
                                        [self.handleFindCenter, self.handleRadiallyAverage, self.handleCutoff, self.handleFitBackground, self.handleBatchProcess]):
            btn.toggled.connect(handle_function)
        
        #Set up ImageViewer
        self.image_viewer = ImageViewer(parent = self)
        self.file_dialog = QtGui.QFileDialog()
        
        # ---------------------------------------------------------------------
        #       SIGNALS
        # ---------------------------------------------------------------------
        
        self.imageLocatorBtn.clicked.connect(self.imageLocator)
        self.revertBtn.clicked.connect(self.revertState)
        
    def initLayout(self):
        #Image viewer pane ----------------------------------------------------
        top_pane = QtGui.QVBoxLayout()
        top_pane.addWidget(self.image_viewer)
               
        master_buttons = QtGui.QVBoxLayout()
        master_buttons.addWidget(self.imageLocatorBtn)
        master_buttons.addWidget(self.executeBtn)
        master_buttons.addWidget(self.revertBtn)
        
        action_buttons = QtGui.QVBoxLayout()
        for btn in [self.findCenterBtn, self.radiallyAverageBtn, self.cutoffBtn, self.fitBackgroundBtn, self.batchProcessBtn]:
            action_buttons.addWidget(btn)
            
        self.bottom_pane = QtGui.QHBoxLayout()
        self.bottom_pane.addLayout(master_buttons)
        self.bottom_pane.addSpacing(10)
        self.bottom_pane.addLayout(action_buttons)
        
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
    
    def handleFindCenter(self):
        if self.findCenterBtn.isChecked() == True:
            self.uncheckActionButtons(self.findCenterBtn)       #Uncheck all other buttons
            self.executeBtn.setText('Compute center')           #Change text on execute button
            
            self.dataHandler.on_click = guessCenter             
            self.dataHandler.execute_function = None
    
    def handleRadiallyAverage(self):
        if self.radiallyAverageBtn.isChecked() == True:
            self.uncheckActionButtons(self.radiallyAverageBtn)       #Uncheck all other buttons
            self.executeBtn.setText('Radial average image')           #Change text on execute button
            
            self.dataHandler.on_click = None
            self.dataHandler.execute_function = None
    
    def handleCutoff(self):
        if self.cutoffBtn.isChecked() == True:
            self.uncheckActionButtons(self.cutoffBtn)       #Uncheck all other buttons
            self.executeBtn.setText('Cutoff curve')           #Change text on execute button
            
            self.dataHandler.on_click = None
            self.dataHandler.execute_function = None
    
    def handleFitBackground(self):
        if self.fitBackgroundBtn.isChecked() == True:
            self.uncheckActionButtons(self.fitBackgroundBtn)       #Uncheck all other buttons
            self.executeBtn.setText('Fit to background')           #Change text on execute button
            self.dataHandler.on_click = None
            self.dataHandler.execute_function = None
    
    def handleBatchProcess(self):
        if self.batchProcessBtn.isChecked() == True:
            self.uncheckActionButtons(self.batchProcessBtn)       #Uncheck all other buttons
            self.executeBtn.setText('Accept')           #Change text on execute button
            
            self.dataHandler.on_click = None
            self.dataHandler.execute_function = None
    
    def uncheckActionButtons(self, exception):
        """ Unckecks all buttons except the exception. """
        for btn in [self.findCenterBtn, self.radiallyAverageBtn, self.cutoffBtn, self.fitBackgroundBtn, self.batchProcessBtn]:
            if btn == exception:
                continue
            btn.setChecked(False)
    
    def revertState(self):
        self.dataHandler.revert()

    def imageLocator(self):
        """ File dialog that selects the TIFF image file to be processed. """
        filename = self.file_dialog.getOpenFileName(self, 'Open image', 'C:\\')
        self.dataHandler.data = self.loadImage(filename)
    
    def loadImage(self, filename):
        """ Loads an image (and the associated background image) and sets the first state. """
        substrate_filename = os.path.normpath(os.path.join(os.path.dirname(filename), 'subs.tif'))
        background_filename = os.path.normpath(os.path.join(os.path.dirname(filename), 'bg.tif'))
        #Load images
        background = n.array(open(background_filename), dtype = n.float)
        image = n.array(open(filename), dtype = n.float) - background
        
        #TODO: Implement a way to change the exposure scaling (currently set to 2/5)
        substrate_image = (2.0/5.0)*(n.array(open(substrate_filename), dtype = n.float) - background)
        
        #Process data
        image[image < 0] = 0
        substrate_image[substrate_image < 0] = 0
    
        return Image(image - substrate_image, source = None, name = 'Sample') #, Image(substrate_image, None, 'Substrate')]

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
