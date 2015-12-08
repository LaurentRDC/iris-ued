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
    
app = QtGui.QApplication(sys.argv)

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

    def initialFigure(self):
        pass
    
class TIFFImageViewer(ImageViewer):
    """ For plotting the raw TIFF images """

    def initialFigure(self):
        """ Plots a placeholder image until an image file is selected """
        missing_image = n.zeros(shape = (1024,1024), dtype = n.uint8)
        self.axes.imshow(missing_image)
    
    def displayImage(self, filename):
        """  """
        if filename is None:
            self.initialFigure()
        else:
            image = n.array(Image.open(filename), 'r')
            self.axes.imshow(image)
            self.axes.set_title('Raw TIFF image from the instrument')
    
class RadialPatternImageViewer(ImageViewer):
    """ For displaying radially-averaged diffraction pattern. """
    
    def initialFigure(self):
        pass
    
    def displayImage(self):
        pass

# -----------------------------------------------------------------------------
#           MAIN WINDOW CLASS
# -----------------------------------------------------------------------------
    
class UEDpowder(QtGui.QMainWindow):
    """
    Attributes
    ----------

    """
    def __init__(self):
        
        #Attributes
        self._image_filename = None
        
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
        
        #Set up ImageViewer
        #At first, we want to display the raw image
        self.image_viewer = TIFFImageViewer()
        self.file_dialog = QtGui.QFileDialog()
        app.connect(self.file_dialog, QtCore.SIGNAL('fileSelected(QString)'), self.image_viewer.displayImage(self.image_filename))
        
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
        
    def imageLocator(self):
        """ File dialog """
        self.image_filename = self.file_dialog.getOpenFileName(self, 'Open image', 'C:\\')
            
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


analysisWindow = UEDpowder()
analysisWindow.show()
sys.exit(app.exec_())
