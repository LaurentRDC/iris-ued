# -*- coding: utf-8 -*-
from setuptools import setup
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst --install-script post_installation_script.py

image_list = glob.glob('App\\images\\*.png')

setup(
    name = 'Iris', 
    version = 'v1.4',
    packages = ['iris.tifffile','iris.pyqtgraph'],
    description = 'UED data exploration', 
    author = 'Laurent P. René de Cotret',
    url = 'www.physics.mcgill.ca/siwicklab',
    scripts = ['post_installation_script.py'],
    py_modules = ['iris.dataset','iris.curve', 'iris.iris', 'iris.progress_widget'], 
    install_requires = ['numpy', 'pyqt4', 'scipy', 'h5py'],
    data_files = [('App\\images', image_list)]
    )