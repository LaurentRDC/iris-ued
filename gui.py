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
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
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

    def compute_initial_figure(self):
        pass
    
class TIFFImageViewer(ImageViewer):
    """ For plotting the raw TIFF images """

    def initialFigure(self):
        """ Plots a placeholder image until an image file is selected """
        missing_image = n.zeros(shape = (1024,1024), dtype = n.uint8)
        self.axes.imshow(missing_image)
    
    def displayImage(self, image = test_image):
        """
        Parameters
        ----------
        image - ndarray, shape (N,N)
            TIFF file imported into a ndarray using PIL.Image.open()
        """
        self.axes.imshow(image)
        self.axes.set_title('Raw TIFF image from the instrument')

# -----------------------------------------------------------------------------
#           MAIN WINDOW CLASS
# -----------------------------------------------------------------------------
    
class UEDpowder(QtGui.QMainWindow):
    """
    Attributes
    ----------
    dataset_location - string
    
    saved_file_location - string
    
    image - MATLAB file    
    """
    def __init__(self):
        
        #Attributes
        self._image_filename = None
        self.saved_file_location = None
        self.image = None
        
        #Methods
        super(UEDpowder, self).__init__()     #inherit from the constructor of QMainWindow        
        self.initUI()
    
    @property
    def image_filename(self):
        """ This property makes sure that the image is the correct format (.tif file) """
        return self._image_filename
    
    @image_filename.setter
    def image_filename(self, value):
        """ This property makes sure that the image is the correct format (.tif file) """
        assert isinstance(self.image_filename, str)
        if not self.image_filename.lower().endswith('.tif'):
            raise ValueError('The input file is not a TIFF image file.')
            self.image_filename = None
        
    def initUI(self):
        
        #Set Status bar
        self.statusBar()
        
        #Set-up dialog buttons
        self.imageLocatorBtn = QtGui.QPushButton('Locate image', self)
        self.imageLocatorBtn.clicked.connect(self.imageLocator)
        
        #
        
        #Set up ImageViewer
        self.image_viewer = TIFFImageViewer()
        
        #Set up vertical layout
        self.vert_box = QtGui.QVBoxLayout()
        self.vert_box.addWidget(self.imageLocatorBtn)
        self.vert_box.addWidget(self.image_viewer)
        
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.vert_box)
        self.setCentralWidget(self.central_widget)
        
        #Window settings
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('UED Powder Analysis Software')
        self.centerWindow()
        self.show()
    
    def loadImage(self):
        """ This method loads the .mat file in the UI. """
        #Check that a filename has been given
        assert not self.image_filename == None
        
        #Load image
        from PIL import Image
        self.image = n.array( Image.open(self.image_filename) )
        
    def imageLocator(self):
        """ Opens a file dialog to locate the appropriate dataset"""
        file_dialog = QtGui.QFileDialog()
        self.image_filename = file_dialog.getOpenFileName(self, 'Open File', 'C:\\')
        self.loadImage(self.image_filename)
        
    def handleBtn(self):
        """ Confirms and executes the HDF5 export"""
        try:
            pass
        except():
            print 'You must select a save location first'
            
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
app = QtGui.QApplication(sys.argv)
analysisWindow = UEDpowder()
analysisWindow.show()
sys.exit(app.exec_())
