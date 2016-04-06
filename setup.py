# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst --install-script post_installation_script.py

image_list = glob.glob('App\\images\\*.png')

setup(
    name = 'Iris', 
    version = 'v1.0',
    packages = find_packages(),
    description = 'UED data exploration', 
    author = 'Laurent P. Rene de Cotret',
    url = 'www.physics.mcgill.ca/siwicklab',
    scripts = ['post_installation_script.py'],
    py_modules = ['App.dataset', 'App.curve', 'App.iris', 'App.tifffile.tifffile'], 
    install_requires = ['tqdm', 'numpy', 'pyqt4', 'pyqtgraph', 'scipy', 'h5py'],
    data_files = [('PowderGuiApp\\images', image_list)]
    )