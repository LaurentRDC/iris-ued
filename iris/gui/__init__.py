# -*- coding: utf-8 -*-

import os
import platform
import sys
from contextlib import contextmanager
from os.path import join
from subprocess import Popen
from pathlib import Path

try:
    from subprocess import CREATE_NEW_PROCESS_GROUP
    WINDOWS = True
except ImportError:
    WINDOWS = False 

from PyQt5 import QtGui

from .qdarkstyle import load_stylesheet_from_environment
from .gui import Iris, image_folder
from ..raw import open_raw

DETACHED_PROCESS = 0x00000008          # 0x8 | 0x200 == 0x208

@contextmanager
def pyqt5_environment():
    """ Set the PyQtGraph QT library to PyQt5 while Iris GUI is running. Revert back when done. """
    old_qt_lib = os.environ.get('PYQTGRAPH_QT_LIB', 'PyQt5')    # environment variable might not exist
    os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    yield
    os.environ['PYQTGRAPH_QT_LIB'] = old_qt_lib

def run(path = None, **kwargs):
    """ 
    Run the iris GUI with the correct environment, and open a dataset. Invalid
    datasets are ignored.
    
    Parameters
    ----------
    path : path-like or None, optional
        Path to either a raw dataset or a processed datasets. 
        Raw dataset formats will be guessed.
    """

    with pyqt5_environment():
        app = QtGui.QApplication(sys.argv)
        app.setStyleSheet(load_stylesheet_from_environment(is_pyqtgraph = True))
        app.setWindowIcon(QtGui.QIcon(join(image_folder, 'eye.png')))
        gui = Iris()

        # If a path is provided, we try to load
        if path:
            path = Path(path)
            if path.suffix in {'.h5', '.hdf5'}:
                gui.dataset_path_signal.emit(str(path)) # signal has signature [str]
            else:
                # For raw datasets, we need to guess the AbstractRawDataset subclass
                try:
                    with open_raw(path) as dset:
                        dataformat = dset.__class__
                except RuntimeError:
                    pass
                else:
                    # No errors, valid dataset
                    # note : signal has signature [str, object]
                    gui.raw_dataset_path_signal.emit(str(path), dataformat)

        # Possibility to restart. A complete new interpreter must
        # be used so that new plug-ins are loaded correctly.
        gui.restart_signal.connect(lambda: restart(app))
        return app.exec_()

def restart(application):
    """ Restart an application in a separate process. A new python interpreter is used, which
    means that plug-ins are reloaded. """
    application.quit()
    flags = DETACHED_PROCESS
    if WINDOWS:
        flags = flags | CREATE_NEW_PROCESS_GROUP
    return Popen(['pythonw', '-m', 'iris'], creationflags = flags)
