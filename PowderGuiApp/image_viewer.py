import pyqtgraph as pg
import numpy as n

pg.mkQApp()

class ImageViewer(pg.GraphicsLayoutWidget):
    
    def __init__(self, parent = None):
        
        super(ImageViewer, self).__init__()
        self.image = pg.ImageItem()        
        self.curve = pg.PlotDataItem()
        self.setupUI()
        
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