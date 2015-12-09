# -*- coding: utf-8 -*-

import sys
import numpy as n
from PIL import Image

#plotting backends
from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE

#GUI backends
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore
    
# -----------------------------------------------------------------------------
#           IMAGE VIEWER CLASSES AND FUNCTIONS
# -----------------------------------------------------------------------------

filename = 'C:\Users\Laurent\Dropbox\Powder\VO2\NicVO2\subs.tif'
test_image = n.array( Image.open(filename) )

class ImageViewer(FigureCanvas):
    """
    Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    This object displays any plot we want: TIFF image, radial-averaged pattern, etc.
    
    Non-plotting Attributes
    -----------------------
    last_click : list, shape (2,)
        [x,y] coordinates of the last click. Clicking outside the data of the ImageViewer set last_click = [0,0]
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        #Non-plotting attributes
        self.last_click = [0,0]
        
        #plot setup
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.initialFigure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        #connect clicking events
        self.mpl_connect('button_press_event', self.clickPosition)
    
    def clickPosition(self, event):
        """
        Saves the position of the last click on the canvas
        """
        if event.xdata == None or event.ydata == None:
            self.last_click_position = [0,0]
        else:
            self.last_click = [event.xdata, event.ydata]

    def initialFigure(self):
        """ Plots a placeholder image until an image file is selected """
        missing_image = n.zeros(shape = (1024,1024), dtype = n.uint8)
        self.axes.imshow(missing_image)
    
    def displayImage(self, filename, *args):
        """ 
        This method displays a raw TIFF image from the instrument. Optional arguments can be used to overlay a circle.
        
        Parameters
        ----------
        filename : string
            Filename of the image to be displayed.
        *args :
        """
        if filename is None:
            self.initialFigure()
        else:
            image = n.array(Image.open(filename))
            self.axes.imshow(image)
            self.axes.set_title('Raw TIFF image')
            self.draw()
    
    def displayRadialPattern(self, *args):
        """
        Plots one or more diffraction patterns.
        
        Parameters
        ----------
        *args : lists of the form [s, pattern, name]
        """
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Diffraction pattern')
        ax.set_xlabel('s = |G|/4pi')
        ax.set_ylabel('Normalized intensity')
        
        for l in args:
            s, pattern, name = l        
            ax.plot(s, pattern, '.', label = name)
        
        ax.legend( loc = 'upper right', numpoints = 1)
            
# -----------------------------------------------------------------------------
#           MAIN WINDOW CLASS
# -----------------------------------------------------------------------------
    
class UEDpowder(QtGui.QMainWindow):
    """
    Main application window
    """
    def __init__(self):
        
        #Attributes
        self.image_filename = None
        
        #Methods
        super(UEDpowder, self).__init__()     #inherit from the constructor of QMainWindow        
        self.initUI()
        
    def initUI(self):
        
        # ---------------------------------------------------------------------
        #       WIDGETS
        # ---------------------------------------------------------------------
        
        self.statusBar()    #Top status bar
        
        #Set up state buttons
        self.acceptBtn = QtGui.QPushButton('Accept', self)
        self.rejectBtn = QtGui.QPushButton('Reject', self)
        
        #Set-up file dialog dialog buttons
        self.imageLocatorBtn = QtGui.QPushButton('Locate image', self)

        #Set up ImageViewer
        self.image_viewer = ImageViewer()
        self.file_dialog = QtGui.QFileDialog()
        
        # ---------------------------------------------------------------------
        #       SIGNALS
        # ---------------------------------------------------------------------
        
        #Connect the image locator button to the file dialog
        self.imageLocatorBtn.clicked.connect(self.imageLocator)
        self.acceptBtn.clicked.connect(self.acceptState)
        self.rejectBtn.clicked.connect(self.rejectState)
        
        # ---------------------------------------------------------------------
        #       LAYOUT
        # ---------------------------------------------------------------------
        
        #Accept - reject buttons combo
        state_controls = QtGui.QHBoxLayout()
        state_controls.addStretch(1)
        state_controls.addWidget(self.acceptBtn)
        state_controls.addWidget(self.rejectBtn)
        
        #Import and view data
        dataset_interaction_controls = QtGui.QVBoxLayout()
        dataset_interaction_controls.addWidget(self.imageLocatorBtn)
        dataset_interaction_controls.addWidget(self.image_viewer)
        
        #Master layout
        grid = QtGui.QGridLayout()
        grid.addLayout(dataset_interaction_controls, 0, 0)
        grid.addLayout(state_controls, 1, 0)
        
        #Don't know what that does        
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)
        
        #Window settings
        self.setGeometry(600, 600, 350, 300)
        self.setWindowTitle('UED Powder Analysis Software')
        self.centerWindow()
        self.show()
        
    def imageLocator(self):
        """ File dialog """
        self.image_filename = self.file_dialog.getOpenFileName(self, 'Open image', 'C:\\')
        self.image_viewer.displayImage(self.image_filename)     #display raw image
        
    def acceptState(self):
        pass
    
    def rejectState(self):
        pass
            
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
    analysisWindow.show()
    sys.exit(app.exec_())
