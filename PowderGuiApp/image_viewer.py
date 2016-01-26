import pyqtgraph as pg
import numpy as n

pg.mkQApp()

class ImageViewer(pg.GraphicsLayoutWidget):
    
    def __init__(self):
        
        super(ImageViewer, self).__init__()
        self.image = pg.ImageItem()
        self.curve = pg.PlotDataItem()
        # A plot area (ViewBox + axes) for displaying the image
        p1 = self.addPlot()
        p1.addItem(self.image)
        
        # Contrast/color control
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.image)
        self.addItem(hist)
        
        self.nextRow()
        p2 = self.addPlot(colspan = 2)
        p2.addItem(self.curve)
        p2.setMaximumHeight(250)
        
        # Another plot area for displaying ROI data
        self.resize(800, 800)
        self.show()

        # zoom to fit imageo
        p1.autoRange()  
    
    def displayImage(self, image):
        if image is None:
            image = n.zeros(shape = (2048, 2048), dtype = n.float)
        image = image.astype(n.float)
        #TODO: Is downsampling a good idea?
        self.image.setImage(image, autoDownsample = True)       
    
    def displayCurve(self, curve):
        self.curve.setData(xValues = curve.xdata, yValues = curve.ydata)
        
## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    im = ImageViewer()
    im.show()