import pyqtgraph as pg
import numpy as n

pg.mkQApp()

class ImageViewer(pg.GraphicsLayoutWidget):
    
    def __init__(self):
        
        super(ImageViewer, self).__init__()
        self.image = pg.ImageItem()
        self.curve = pg.PlotDataItem()
        
        # A plot area (ViewBox + axes) for displaying the image
        self.image_area = self.addPlot()
        self.image_area.addItem(self.image)
        
        # Contrast/color control
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.image)
        self.addItem(hist)
        
        self.nextRow()
        self.curve_area = self.addPlot(colspan = 2)
        self.curve_area.addItem(self.curve)
        self.curve_area.setMaximumHeight(250)
        
        # Another plot area for displaying ROI data
        self.resize(800, 800)

        # zoom to fit image
        self.image_area.autoRange()
        self.curve_area.autoRange()
    
    def displayImage(self, image, *args, **kwargs):
        if image is None:
            image = n.zeros(shape = (2048, 2048), dtype = n.float)
            
        self.image.setImage(image, *args, **kwargs)
        self.image_area.autoRange() # zoom to fit image
    
    def displayCurve(self, curve, *args, **kwargs):
        self.curve.setData(xValues = curve.xdata, yValues = curve.ydata, *args, **kwargs)

if __name__ == '__main__':
    im = ImageViewer()
    im.show()