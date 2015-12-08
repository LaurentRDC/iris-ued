# -*- coding: utf-8 -*-

import sys
import numpy as n

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

class ImageViewer(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

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

    def compute_initial_figure(self, image):
        """
        Parameters
        ----------
        image - ndarray, shape (N,N)
            TIFF file imported into a ndarray using PIL.Image.open()
        """
        self.axes.imshow(image)
    
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
        self.datasetLocatorBtn = QtGui.QPushButton('Locate data', self)
        self.datasetLocatorBtn.clicked.connect(self.datasetLocator)
        
        self.saveLocatorBtn = QtGui.QPushButton('Set export location and filename', self)
        self.saveLocatorBtn.clicked.connect(self.saveLocator)
        
        self.exportBtn = QtGui.QPushButton('Compute radial average', self)
        self.exportBtn.clicked.connect(self.handleRadialAverageBtn)
        
        #Set up vertical layout
        self.vert_box = QtGui.QVBoxLayout()
        self.vert_box.addWidget(self.datasetLocatorBtn)
        self.vert_box.addWidget(self.saveLocatorBtn)
        self.vert_box.addWidget(self.exportBtn)
        
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.vert_box)
        self.setCentralWidget(self.central_widget)
        
        #Window settings
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('pyImgLoader')
        self.centerWindow()
        self.show()
    
    def loadImage(self):
        """ This method loads the .mat file in the UI. """
        #Check that a filename has been given
        assert not self.image_filename == None
        
        #Load image
        from PIL import Image
        self.image = n.array( Image.open(self.image_filename) )
        
    def datasetLocator(self):
        """ Opens a file dialog to locate the appropriate dataset"""
        self.datasetLocation = QtGui.QFileDialog.getExistingDirectory(self, 'Open Directory', 'C:\\')
        
    def saveLocator(self):
        """ Opens a file dialog to save the exported HDF5 file """
        self.saved_file_location = QtGui.QFileDialog.getSaveFileName(self, 'Save file', 'C:\\')
        
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

#Run
app = QtGui.QApplication(sys.argv)
analysisWindow = UEDpowder()
analysisWindow.show()
sys.exit(app.exec_())
