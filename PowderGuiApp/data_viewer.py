import pyqtgraph as pg
import os.path
from pyqtgraph.Qt import QtCore,QtGui,uic
import numpy as n
import core


pg.mkQApp()

class DataViewer(QtGui.QMainWindow):
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
        
        super(DataViewer, self).__init__()              

        self.plot_view = pg.PlotWidget()
        self.test_curve = core.RadialCurve(xdata = n.arange(0, 100,0.1), ydata = n.sin(n.arange(0, 100,0.1)), color = 'b')
       
        #Initialize display
        self.initUI()
        
        #Signals

        
    def initUI(self):
        
        # ---------------------------------------------------------------------
        #       LAYOUT
        # ---------------------------------------------------------------------
        uic.loadUi('testlayout3.ui', self)        
     
    # -------------------------------------------------------------------------
    #           FILE IO
    # -------------------------------------------------------------------------

    def menuOpen(self):
        """ 
        Activates a file dialog that selects the data directory to be processed. If the folder
        selected is one with processed images (then the directory name is C:\\...\\processed\\),
        return data 'root' directory.
        """
        
        possible_directory = QtGui.QFileDialog.getExistingDirectory(self, 'Open diffraction dataset', 'C:\\')
        possible_directory = os.path.abspath(possible_directory)
        
        #Check whether the directory name ends in 'processed'. If so, return previous directory
        last_directory = possible_directory.split('\\')[-1]
        if last_directory == 'processed':
            directory = os.path.dirname(possible_directory) #If directory is 'processed', back up one directory
        else:
            directory = possible_directory
       
    # -------------------------------------------------------------------------
    #           DISPLAY (and HIDE) OBJECTS
    # -------------------------------------------------------------------------
    
        

    # -------------------------------------------------------------------------
    #           PLOTTING METHODS
    # -------------------------------------------------------------------------
    

    
    @QtCore.pyqtSlot(object)
    def displayRadialPattern(self):
        """ 
        Displays one or two curves (overlayed) in to curve area.
        
        Parameters
        ----------
        curve : core.RadialCurve object or list of RadialCurve objects
            Curves to be plotted. If 'curve' is a list, only the first two objects
            will be plotted.
        """
        #Distribute inputs
        self.plot_view.plot(self.test_curve.xdata,self.test_curve.ydata)
        #Reset plot
    
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
    im = DataViewer()
    im.show()
