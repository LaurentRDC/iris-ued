import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as n
import core

pg.mkQApp()

class ImageViewer(pg.GraphicsLayoutWidget):
    """
    Widget for displaying diffraction data. This widget has two sections:
    an image area and a curve area.
    
    The ImageViewer can display an image and two curves at the same time.
    """
    
    #Data transfer signals
    image_center_signal = QtCore.pyqtSignal(tuple, name = 'image_center_signal')
    mask_rect_signal = QtCore.pyqtSignal(tuple, name = 'mask_rect_signal')
    cutoff_signal = QtCore.pyqtSignal(tuple, name = 'cutoff_signal')
    inelastic_BG_signal = QtCore.pyqtSignal(list, name = 'inelastic_BG_signal')
    
    def __init__(self, parent = None):
        
        super(ImageViewer, self).__init__()
        
        #Dat amanipulation attributes
        self.center_finder = None
        self.mask = None
        self.cutoff_line = None
        self.inelasticBG_lines = list()
        
        
        self.curve = pg.PlotDataItem()
        self.curve_overlay = pg.PlotDataItem()  #Will be used to display a second curve if necessary
        
        #Initialize display
        self.initUI()
        
        #Signals

        
    def initUI(self):
        
        # ---------------------------------------------------------------------
        #       LAYOUT
        # ---------------------------------------------------------------------
        self.setBackgroundBrush(pg.mkBrush('b'))
        self.test_label = self.addLabel('Hello World')
        
        self.nextCol()
        # Let's go with white background
        self.setBackgroundBrush(pg.mkBrush('w'))
        
        self.nextRow()
        
        self.curve_area = self.addPlot(colspan = 2)
        self.curve_area.addItem(self.curve)
        self.curve_area.addItem(self.curve_overlay)
        self.curve_area.setMaximumHeight(400)
        
        # ---------------------------------------------------------------------
        #           DATA INTERACTION ITEMS MASK
        # ---------------------------------------------------------------------
        
        self.mask = pg.ROI(pos = [800,800], size = [200,200], pen = pg.mkPen('r'))
        self.mask.addScaleHandle([1, 1], [0, 0])
        self.mask.addScaleHandle([0, 0], [1, 1])
        
        self.center_finder = pg.CircleROI(pos = [1000,1000], size = [200,200], pen = pg.mkPen('r'))
        self.cutoff_line = pg.InfiniteLine(angle = 90, movable = True, pen = pg.mkPen('r'))
        self.inelasticBG_lines = [pg.InfiniteLine(angle = 90, movable = True, pen = pg.mkPen('b')) for i in range(6)]
    
    # -------------------------------------------------------------------------
    #           DISPLAY (and HIDE) OBJECTS
    # -------------------------------------------------------------------------
    
        
    @QtCore.pyqtSlot()
    def displayCutoff(self):
        self.curve_area.getViewBox().addItem(self.cutoff_line)
    
    @QtCore.pyqtSlot()
    def displayInelasticBG(self):
        #Determine curve range
        xmin, xmax = self.curve.dataBounds(ax = 0)
        dist_between_lines = float(xmax - xmin)/len(self.inelasticBG_lines)
        #Distribute lines equidistantly
        pos = xmin
        for line in self.inelasticBG_lines:
            line.setValue(pos)
            self.curve_area.getViewBox().addItem(line)
            pos += dist_between_lines
    
    def hideCutoff(self):
        self.curve_area.getViewBox().removeItem(self.cutoff_line)
    
    def hideInelasticBG(self):
        for line in self.inelasticBG_lines:
            self.curve_area.getViewBox().removeItem(line)
    
    # -------------------------------------------------------------------------
    #           POSITION METHODS
    # -------------------------------------------------------------------------
    
    def maskPosition(self):
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
    
    def centerPosition(self, return_radius = False):
        """
        Returns
        -------
        x, y : tuple
            center coordinates of the centerFinder object.
        """
        corner_x, corner_y = self.center_finder.pos().x(), self.center_finder.pos().y()
        radius = self.center_finder.size().x()/2.0
        
        #Flip output since image viewer plots transpose...
        if return_radius:
            return corner_y + radius, corner_x + radius, radius
        else:
            return corner_y + radius, corner_x + radius
    
    def cutoffPosition(self):
        """
        Returns
        -------
        x,y : tuple
            (x,y)-coordinate of the cutoff line with respect to the plotted data.
            Only the x-value is meaningful.
        """
        x = self.cutoff_line.value()
        return (x, 0)
    
    def inelasticBGPosition(self):
        """
        Returns
        -------
        intersects : list of tuples
            list of (x,y) tuples representing the intersects of the inelastic BG fit
            lines. Only the x-coordinates are meaningful.
        """
        intersects = list()
        for line in self.inelasticBG_lines:
            x = line.value()
            intersects.append( (x, 0) )
        return intersects
    # -------------------------------------------------------------------------
    #           RETURN DATA SLOTS
    #   These slots are used to emit data. 
    # -------------------------------------------------------------------------
    
    @QtCore.pyqtSlot()
    def returnImageCenter(self):
        self.image_center_signal.emit(self.centerPosition())
        self.hideCenterFinder()
    
    @QtCore.pyqtSlot()
    def returnMaskRect(self):
        self.mask_rect_signal.emit(self.maskPosition())
        self.hideMask()
    
    @QtCore.pyqtSlot()
    def returnCutoff(self):
        self.cutoff_signal.emit(self.cutoffPosition())
        self.hideCutoff()
        
    @QtCore.pyqtSlot()
    def returnInelasticBG(self):
        self.inelastic_BG_signal.emit(self.inelasticBGPosition())
        self.hideInelasticBG()
    
    # -------------------------------------------------------------------------
    #           PLOTTING METHODS
    # -------------------------------------------------------------------------
    

    
    @QtCore.pyqtSlot(object)
    def displayRadialPattern(self, curve):
        """ 
        Displays one or two curves (overlayed) in to curve area.
        
        Parameters
        ----------
        curve : core.RadialCurve object or list of RadialCurve objects
            Curves to be plotted. If 'curve' is a list, only the first two objects
            will be plotted.
        """
        #Distribute inputs
        if isinstance(curve, list):
            main_curve = curve[0]
            overlay_curve = curve[1]
        else:
            main_curve = curve
            overlay_curve = None
        
        #Display  curves
        self.curve.setData(x = main_curve.xdata, y = main_curve.ydata, pen = pg.mkPen(main_curve.color))
        
        if overlay_curve is not None:
            self.curve_overlay.setData(x = overlay_curve.xdata, y = overlay_curve.ydata, pen = pg.mkPen(overlay_curve.color))
        else:
            self.curve_overlay.setData(x = [], y = [])      #Reset plot
    
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
    
    test_curve = core.RadialCurve(xdata = n.arange(0, 100,0.1), ydata = n.sin(n.arange(0, 100,0.1)), color = 'r')
    test_background = core.RadialCurve(xdata = n.arange(0,100, 0.1), ydata = n.cos(10*n.arange(0, 100, 0.1)), color = 'b')
    
    two_curve_test = [test_curve, test_background]
    
    #Test
    im.displayRadialPattern(test_curve)
    im.displayInelasticBG()
