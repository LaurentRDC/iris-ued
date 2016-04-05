# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:18:35 2016

@author: Laurent
"""

#Core functions
from dataset import DiffractionDataset, read, cast_to_16_bits
import os

#GUI backends
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
import numpy as n

image_folder = os.path.join(os.path.dirname(__file__), 'images')

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

class Explorer(QtGui.QMainWindow):
    """
    Time-delay data viewer for averaged data.
    """
    
    def __init__(self):
        
        super(Explorer, self).__init__()
        
        self.worker = None
        self.dataset = None
        
        self.viewer = pg.ImageView(parent = self)
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))       
        
        self._init_ui()
    
    def _init_ui(self):
        
        # UI components
        self.file_dialog = QtGui.QFileDialog(parent = self)        
        self._create_menubar()
        
        # Masks
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
              
        # Label for crosshair
        self.image_position_label = pg.LabelItem()
        self.viewer.addItem(self.image_position_label)
        
        #Create window        
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.viewer)
        
        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        #Window settings ------------------------------------------------------
        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('UED Powder Analysis Software')
        self.center_window()
        self.show()
    
    def _create_menubar(self):
        self.menubar = self.menuBar()
        
        # Actions
        directory_action = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'locator.png')), '&Dataset', self)
        directory_action.triggered.connect(self.directory_locator)
        
        picture_action = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'diffraction.png')), '&Picture', self)
        picture_action.triggered.connect(self.picture_locator)
        
        show_radav_tools = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'analysis.png')), '&Show beamblock mask and center finder', self)
        show_radav_tools.triggered.connect(self.display_radav_tools)
        
        set_radav_tools = QtGui.QAction(QtGui.QIcon(os.path.join(image_folder, 'analysis.png')), '&Set beamblock mask and center finder', self)
        set_radav_tools.triggered.connect(self.compute_radial_average)
        
        file_menu = self.menubar.addMenu('&File')
        file_menu.addAction(directory_action)
        file_menu.addAction(picture_action)
        
        powder_menu = self.menubar.addMenu('&Powder')
        powder_menu.addAction(show_radav_tools)
        powder_menu.addAction(set_radav_tools)
        
    def directory_locator(self):
        """ 
        Activates a file dialog that selects the data directory to be processed. If the folder
        selected is one with processed images (then the directory name is C:\\...\\processed\\),
        return data 'root' directory.
        """
        
        directory = self.file_dialog.getExistingDirectory(self, 'Open diffraction dataset', 'C:\\')
        directory = os.path.abspath(directory)
        self.dataset = DiffractionDataset(directory)
        self.display_data(dataset = self.dataset)
    
    def picture_locator(self):
        """ Open a file dialog to select an image to view """
        filename = self.file_dialog.getOpenFileName(self, 'Open diffraction picture', 'C:\\')
        filename = os.path.abspath(filename)
        self.display_data(image = cast_to_16_bits(read(filename)))
    
    def display_radav_tools(self):
        """ Display diffraction center finder and beam block mask on the viewer. """
        self._display_center_finder()
        self._display_mask()
    
    def hide_radav_tools(self):
        """ Hide diffraction center finder and beam block mask from the viewer. """
        self._hide_center_finder()
        self._hide_mask()
    
    def _display_mask(self):
        self.viewer.addItem(self.mask)
    
    def _hide_mask(self):
        self.viewer.removeItem(self.mask)
    
    def _display_center_finder(self):
        self.viewer.addItem(self.center_finder)
    
    def _hide_center_finder(self):
        self.viewer.removeItem(self.center_finder)
    
    def compute_radial_average(self):
        if self.dataset is not None:
            mask_rectangle = self.mask_position
            center = self.center_position
            self.hide_radav_tools()
            
            #Thread the computation
            self.worker = WorkThread(self.dataset.radial_average_series, center, mask_rectangle)
            self.worker.start()
    
    @property
    def mask_position(self):
        """
        Returns the x,y limits of the rectangular beam block mask. Due to the 
        inversion of plotting axes, y-axis in the image_viewer is actually
        the x-xis when analyzing data.
        
        Returns
        -------
        xmin, xmax, ymin, ymax : tuple
            The limits determining the shape of the rectangular beamblock mask
        """
        rect = self.mask.parentBounds().toRect()
        
        #If coordinate is negative, return 0
        x1 = max(0, rect.topLeft().x() )
        x2 = max(0, rect.x() + rect.width() )
        y1 = max(0, rect.topLeft().y() )
        y2 = max(0, rect.y() + rect.height() )
               
        return y1, y2, x1, x2       #Flip output since image viewer plots transpose
    
    @property
    def center_position(self):
        """
        Returns
        -------
        x, y : tuple
            center coordinates of the center_finder Region-of-Interest object.
        """
        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2.0
        
        #Flip output since image viewer plots transpose...
        return corner_y + radius, corner_x + radius
    
    def display_data(self, image = None, dataset = None):
        if dataset is not None:
            self._display_dataset(dataset)
        elif image is not None:
            self._display_image(image)
        else:
            self._display_image(image = n.zeros((2048, 2048)))
    
    def _display_dataset(self, dataset):
        time = n.array(list(map(float, dataset.time_points)))
        # By using reduced_memory = True, RAM usage drops by 75%
        self.viewer.setImage(dataset.image_series(reduced_memory = True), xvals = time, axes = {'x':0, 'y':1, 't':2})
    
    def _display_image(self, image):
        self.viewer.setImage(image)
    
    def center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def run():
    import sys
    
    app = QtGui.QApplication(sys.argv)    
    gui = Explorer()
    gui.showMaximized()
    
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    run()