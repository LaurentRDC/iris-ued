# -*- coding: utf-8 -*-

import os
from os.path import join
import sys
from contextlib import contextmanager

from PyQt5 import QtGui
from qdarkstyle import load_stylesheet_from_environment

from .gui import Iris, image_folder

@contextmanager
def pyqt5_environment():
    """ Set the PyQtGraph QT library to PyQt5 while Iris GUI is running. Revert back when done. """
    old_qt_lib = os.environ['PYQTGRAPH_QT_LIB']
    os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    yield
    os.environ['PYQTGRAPH_QT_LIB'] = old_qt_lib

def run(**kwargs):
    with pyqt5_environment():
        app = QtGui.QApplication(sys.argv)
        app.setStyleSheet(load_stylesheet_from_environment(is_pyqtgraph = True))
        app.setWindowIcon(QtGui.QIcon(join(image_folder, 'eye.png')))
        gui = Iris()
        return app.exec_()
