import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
import numpy as n

pg.mkQApp()

class ImageViewer(pg.GraphicsLayoutWidget):
    
    #New-style signal definitiona
    image_clicked = QtCore.pyqtSignal(tuple, name = 'image_clicked')
    curve_clicked = QtCore.pyqtSignal(tuple, name = 'curve_clicked')
    
    def __init__(self, parent = None):
        
        super(ImageViewer, self).__init__()
        self.image = pg.ImageItem()        
        self.curve = pg.PlotDataItem()
        self.setupUI()
        
        #Initialize display
        self.displayImage( image = None )
        
        #Signals
        self.image.mouseClickEvent = self.imageClick
        self.curve.mouseClickEvent = self.curveClick
    
    def imageClick(self, event):
        pos = event.pos()
        click_position = ( int(pos.x()), int(pos.y()) )
        self.image_clicked.emit(click_position)
    
    def curveClick(self, event):
        pos = event.pos()
        click_position = ( int(pos.x()), int(pos.y()) )
        self.curve_clicked.emit(click_position)
        
    def setupUI(self):
        # A plot area (ViewBox + axes) for displaying the image
        self.image_area = self.addPlot()
        self.image_area.getViewBox().setAspectLocked(lock = True, ratio = 1)
        self.image_area.getViewBox().enableAutoRange()
        self.image_area.addItem(self.image)
        
        # Contrast/color control
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.image)
        self.addItem(hist)
        
        self.nextRow()
        self.curve_area = self.addPlot(colspan = 2)
        self.curve_area.addItem(self.curve)
        self.curve_area.setMaximumHeight(250)
    
    def displayImage(self, image):
        if image is None:
            image = n.zeros(shape = (2048, 2048), dtype = n.float)
        self.image.setImage(image)
    
    def displayRadialPattern(self, curve):
        self.curve.setData(xValues = curve.xdata, yValues = curve.ydata)
        
if __name__ == '__main__':  
    im = ImageViewer()
    im.show()