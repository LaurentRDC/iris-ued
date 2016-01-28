import pyqtgraph as pg
from pyqtgraph import QtCore
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
        #self.curve.scene().sigMouseClicked.connect(self.curveClick)
        self.image_area.scene().sigMouseMoved.connect(self.updateCrosshair)
    
    def imageClick(self, event):
        pos = event.pos()
        click_position = ( int(pos.x()), int(pos.y()) )
        self.image_clicked.emit(click_position)
    
    def curveClick(self, event):
        pos = event.pos()
        mousePoint = self.curve_area.getViewBox().mapToView(self.curve_area, pos) # Get cursor position within image
        x_lims, y_lims = self.curve_area.getViewBox().childrenBounds()            # Plot limits
        mx, my = mousePoint.x(), mousePoint.y()
        if mx >= int(x_lims[0]) and mx <= int(x_lims[1]) and my >= int(y_lims[0]) and my <= int(y_lims[1]): 
            print (mx, my)
            self.curve_clicked.emit( (mx, my) )
        
    def setupUI(self):
        
        # ---------------------------------------------------------------------
        #       LAYOUT
        # ---------------------------------------------------------------------
        
        # Let's go with white background
        self.setBackgroundBrush(pg.mkBrush('w'))
        
        self.addItem(self.image_position_label)
        
        self.nextRow()
        
        self.image_area = self.addPlot()
        self.image_area.getViewBox().setAspectLocked(lock = True, ratio = 1)
        self.image_area.getViewBox().enableAutoRange()
        self.image_area.addItem(self.image)
        self.image_area.addItem(self.image_overlay)
        
        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image)
        self.addItem(self.hist)
        
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
        
        # ---------------------------------------------------------------------
        #           BEAMBLOCK MASK
        # ---------------------------------------------------------------------
        
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
        
        # ---------------------------------------------------------------------
        #           CENTER FINDER
        # ---------------------------------------------------------------------
        
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))
    
    # -------------------------------------------------------------------------
    #           DISPLAY (and HIDE) OBJECTS in IMAGE AREA
    # -------------------------------------------------------------------------

    def displayMask(self):
        self.image_area.getViewBox().addItem(self.mask)
    
    def displayCenterFinder(self):
        self.image_area.getViewBox().addItem(self.center_finder)
    
    def hideCenterFinder(self):
        self.image_area.getViewBox().removeItem(self.center_finder)
    
    def hideMask(self):
        self.image_area.getViewBox().removeItem(self.mask)
    
    # -------------------------------------------------------------------------
    #           POSITION METHODS
    # -------------------------------------------------------------------------
    
    def maskPosition(self):
        rect = self.mask.parentBounds().toRect()
        
        #If coordinate is negative, return 0
        x1 = max(0, rect.topLeft().x() )
        x2 = max(0, rect.x() + rect.width() )
        y1 = max(0, rect.topLeft().y() )
        y2 = max(0, rect.y() + rect.height() )
               
        return x1, x2, y1, y2
    
    def centerPosition(self):
        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2.0
        return corner_x + radius, corner_y + radius
        
    
    # -------------------------------------------------------------------------
    #           PLOTTING METHODS
    # -------------------------------------------------------------------------
    
    def displayImage(self, image, overlay = list(), overlay_color = 'r'):
        if image is None:
            image = n.zeros(shape = (2048, 2048), dtype = n.float)
        self.image.setImage(image)
        
        #Add overlays
        brush = pg.mkBrush(color = overlay_color)
        self.image_overlay.setData(pos = overlay, size = 3, brush = brush)
    
    def displayRadialPattern(self, curve):
        pen = pg.mkPen(curve.color)
        self.curve.setData(x = curve.xdata, y = curve.ydata, pen = pen)
    
    # -------------------------------------------------------------------------
    #           SIGNAL METHODS
    # -------------------------------------------------------------------------
    
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
    im.displayMask()
    im.displayCenterFinder()
    
    test_image = n.random.normal(size = (2048, 2048))
    test_curve = core.RadialCurve(xdata = n.arange(0, 100,0.1), ydata = n.sin(n.arange(0, 100,0.1)), color = 'r')
    
    #Test
    im.displayImage(image = test_image, overlay = [(500,500)], overlay_color = 'r')
    im.displayRadialPattern(test_curve)