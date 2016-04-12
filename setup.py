# -*- coding: utf-8 -*-
from distutils.core import setup
import py2exe
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst --install-script post_installation_script.py

image_list = glob.glob('App\\images\\*.png')

setup(
    name = 'Iris - UED data exploration', 
    version = 'v1.1',
    packages = ['App.tifffile','App.pyqtgraph', 'App.tqdm'],
    description = 'UED data exploration', 
    author = 'Laurent P. Rene de Cotret',
    url = 'www.physics.mcgill.ca/siwicklab',
    scripts = ['post_installation_script.py'],
    py_modules = ['App.dataset', 'App.curve', 'App.iris'], 
    install_requires = ['numpy', 'pyqt4', 'scipy', 'h5py'],
    data_files = [('App\\images', image_list)]
    )