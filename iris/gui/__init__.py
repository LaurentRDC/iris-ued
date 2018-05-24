# -*- coding: utf-8 -*-

import os
import platform
import sys
from contextlib import contextmanager
from os.path import join
from subprocess import CREATE_NEW_PROCESS_GROUP, Popen

from PyQt5 import QtGui
from qdarkstyle import load_stylesheet_from_environment

from .gui import Iris, image_folder

DETACHED_PROCESS = 0x00000008          # 0x8 | 0x200 == 0x208

@contextmanager
def pyqt5_environment():
    """ Set the PyQtGraph QT library to PyQt5 while Iris GUI is running. Revert back when done. """
    old_qt_lib = os.environ['PYQTGRAPH_QT_LIB']
    os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    yield
    os.environ['PYQTGRAPH_QT_LIB'] = old_qt_lib

def run(**kwargs):
    """ Run the iris GUI with the correct environment """
    with pyqt5_environment():
        app = QtGui.QApplication(sys.argv)
        app.setStyleSheet(load_stylesheet_from_environment(is_pyqtgraph = True))
        app.setWindowIcon(QtGui.QIcon(join(image_folder, 'eye.png')))
        gui = Iris()

        # Possibility to restart. A complete new interpreter must
        # be used so that new plug-ins are loaded correctly.
        gui.restart_signal.connect(lambda: restart(app))
        return app.exec_()

def restart(application):
    """ Restart an application in a separate process. A new python interpreter is used, which
    means that plug-ins are reloaded. """
    application.quit()
    return Popen(['pythonw', '-m', 'iris'], creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
