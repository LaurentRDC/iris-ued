# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:18:35 2016

@author: Laurent
"""

#Core functions
from dataset import DiffractionDataset

#GUI backends
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
import numpy as n

pg.mkQApp()

class TIFFViewer(QtGui.QMainWindow):
    """
    Time-delay data viewer for averaged data.
    """
    
    def __init__(self):
        
        super(TIFFViewer, self).__init__()
        
        self.dataset = None
        self.viewer = pg.ImageView(parent = self)
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))
        
        self._init_ui()
        self.display_data()
    
    def _init_ui(self):
        
        # Masks
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
               
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
    
    def display_mask(self):
        self.viewer.addItem(self.mask)
    
    def hide_mask(self):
        self.viewer.removeItem(self.mask)
    
    def display_center_finder(self):
        self.viewer.addItem(self.center_finder)
    
    def hide_center_finder(self):
        self.viewer.removeItem(self.center_finder)
    
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
               
        return y1, y2, x1, x2       #Flip output since image viewer plots transpose...
    
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
        self.viewer.setImage(dataset.image_series(), xvals = time, axes = {'x':0, 'y':1, 't':2})
    
    def _display_image(self, image):
        self.viewer.setImage(image)
    
    def center_window(self):
        """ Centers the window """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def run():
    import sys
    directory = 'K:\\2016.03.01.16.57.VO2_1500uW_Pump_50Hz - Copy'
    d = DiffractionDataset(directory)
    
    app = QtGui.QApplication(sys.argv)
    
    gui = TIFFViewer()
    gui.display_data(dataset = d)
    gui.showMaximized()
    
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    run()