import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
import numpy as n
import core

pg.mkQApp()

class ImageViewer(pg.GraphicsLayoutWidget):
    
    #New-style signal definitiona
    image_clicked = QtCore.pyqtSignal(tuple, name = 'image_clicked')
    curve_clicked = QtCore.pyqtSignal(tuple, name = 'curve_clicked')
    
    def __init__(self, parent = None):
        
        super(ImageViewer, self).__init__()
        self.image_position_label = pg.LabelItem()
        self.image = pg.ImageItem()
        self.image_overlay = pg.ScatterPlotItem()        
        self.curve = pg.PlotDataItem()
        
        self.setupUI()
        
        #Initialize display
        self.displayImage( image = None )
        
        #Signals
        self.image.mouseClickEvent = self.imageClick
        self.curve.mouseClickEvent = self.curveClick
        self.image_area.scene().sigMouseMoved.connect(self.updateCrosshair)
    
    def imageClick(self, event):
        pos = event.pos()
        click_position = ( int(pos.x()), int(pos.y()) )
        self.image_clicked.emit(click_position)
    
    def curveClick(self, event):
        pos = event.pos()
        print pos
        click_position = ( int(pos.x()), int(pos.y()) )
        self.curve_clicked.emit(click_position)
        
    def setupUI(self):
        
        # ---------------------------------------------------------------------
        #       LAYOUT
        # ---------------------------------------------------------------------
        
        self.addItem(self.image_position_label)
        
        self.nextRow()
        
        self.image_area = self.addPlot()
        self.image_area.getViewBox().setAspectLocked(lock = True, ratio = 1)
        self.image_area.getViewBox().enableAutoRange()
        self.image_area.addItem(self.image)
        self.image_area.addItem(self.image_overlay)
        
        # Contrast/color control
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.image)
        self.addItem(hist)
        
        self.nextRow()
        
        self.curve_area = self.addPlot(colspan = 2)
        self.curve_area.addItem(self.curve)
        self.curve_area.setMaximumHeight(300)
        
        # ---------------------------------------------------------------------
        #           CROSSHAIR
        # ---------------------------------------------------------------------
        
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.image_area.addItem(self.vLine, ignoreBounds=True)
        self.image_area.addItem(self.hLine, ignoreBounds=True)
    
    def setupCrossHair(self):
        """ """

    
    def displayImage(self, image, overlay = list(), overlay_color = 'r'):
        if image is None:
            image = n.zeros(shape = (2048, 2048), dtype = n.float)
        self.image.setImage(image)
        
        #Add overlays
        brush = pg.mkBrush(color = overlay_color)
        self.image_overlay.setData(pos = overlay, size = 5, brush = brush)
    
    def displayRadialPattern(self, curve):
        pen = pg.mkPen(curve.color)
        self.curve.setData(x = curve.xdata, y = curve.ydata, pen = pen)
    
    def updateCrosshair(self, event):
        pos = event  ## using signal proxy turns original arguments into a tuple
        if self.image_area.sceneBoundingRect().contains(pos):
            
            #Get cursor position within image
            mousePoint = self.image_area.getViewBox().mapSceneToView(pos)
            x_lims, y_lims = self.image_area.getViewBox().childrenBounds()
            mx, my = int(mousePoint.x()), int(mousePoint.y())
            
            #Update text label
            if mx >= x_lims[0] and mx <= x_lims[1] and my >= y_lims[0] and my <= y_lims[1]:
                self.image_position_label.setText( "<span style='font-size: 12pt' style='color:green'> (x, y) = ({0}, {1}) </span>".format(mx,my) )
            else:
                self.image_position_label.setText("<span style='font-size: 12pt' style='color:red'> Crosshair outside image </span>")
            
            #Change crosshair position
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
        
if __name__ == '__main__':  
    im = ImageViewer()
    im.show()
    
    test_image = n.random.normal(size = (1000,1000))
    test_curve = core.RadialCurve(xdata = n.arange(0, 100,0.1), ydata = n.sin(n.arange(0, 100,0.1)), color = 'r')
    
    #Test
    im.displayImage(image = test_image, overlay = [(500,500)], overlay_color = 'r')
    im.displayRadialPattern(test_curve)