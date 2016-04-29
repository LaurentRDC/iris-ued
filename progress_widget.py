# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""
from pyqtgraph import QtGui, QtCore
from math import sin, cos, pi

class InProgressWidget(QtGui.QWidget):
    """ Spinning wheel with transparent background to overlay over other widgets. """
    def __init__(self, parent = None):
        
        super(InProgressWidget, self).__init__(parent)        
        self._init_ui()
    
    def _init_ui(self):
        
        # Number of dots to display as part of the 'spinning wheel'
        self._num_points = 12
        
        # Set background color to be transparent
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)
    
    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        
        #Overlay color is half-transparent white
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        
        # Loop over dots in the 'wheel'
        # At any time, a single ellipse is colored bright
        # Other ellipses are darker
        for i in range(self._num_points):
            if  i == self.counter % self._num_points :  # Color this ellipse bright
                painter.setBrush(QtGui.QBrush(QtGui.QColor(229, 33, 33)))
            else:   # Color this ellipse dark
               painter.setBrush(QtGui.QBrush(QtGui.QColor(114, 15, 15)))
              
            # Draw the ellipse with the right color
            painter.drawEllipse(
                self.width()/2 + 30 * cos(2 * pi * i / self._num_points),
                self.height()/2 + 30 * sin(2 * pi * i / self._num_points),
                10, 10)

        painter.end()
    
    def showEvent(self, event):
        """ Starts an updating timer, called every 50 ms. """
        self.timer = self.startTimer(50)
        self.counter = 0
    
    def timerEvent(self, event):
        """ At every timer step, this method is called. """
        self.counter += 1
        self.update()       # Calls a paintEvent