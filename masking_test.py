import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np

class MaskingTest(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.viewer1 = pg.ImageView()
        self.viewer1.setImage(np.random.random(size = (512, 512)) + 1)

        self.rect_mask = pg.RectROI(pos = [256, 256], size = [100, 100], pen = pg.mkPen('r', width = 4))
        self.rect_mask.sigRegionChanged.connect(self.update_mask)
        self.viewer1.addItem(self.rect_mask)

        self.circ_mask = pg.CircleROI(pos = [100, 100], size = [100, 100], pen = pg.mkPen('r', width = 4))
        self.rect_mask.sigRegionChanged.connect(self.update_mask)
        self.viewer1.addItem(self.circ_mask)

        self.viewer2 = pg.ImageView()
        self.viewer2.setImage(np.zeros((512, 512)))

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.viewer1)
        layout.addWidget(self.viewer2)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def rect_mask_image(self):
        (x_slice, y_slice), _ = self.rect_mask.getArraySlice(data = np.empty(shape = (512, 512)), 
                                                             img = self.viewer1.getImageItem())
        mask = np.zeros((512, 512))
        mask[x_slice, y_slice] = 1
        return mask

    def circ_mask_image(self):        
        radius = self.circ_mask.size().x()/2
        corner_x, corner_y = self.circ_mask.pos().x(), self.circ_mask.pos().y()

        xc, yc = (round(corner_x + radius), round(corner_y + radius)) 
        xx, yy = np.mgrid[0:512, 0:512]
        rr = np.hypot(xx - xc, yy - yc)

        im = np.zeros((512, 512))
        #im[x_slice, y_slice] = 1
        im[rr <= radius] = 1
        return im
    
    @QtCore.pyqtSlot()
    def update_mask(self):
        mask = np.logical_or(self.rect_mask_image(), self.circ_mask_image())
        self.viewer2.setImage(mask)

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    gui = MaskingTest()
    gui.show()
    app.exec_()