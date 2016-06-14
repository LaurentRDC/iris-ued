# -*- coding: utf-8 -*-
from setuptools import setup
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst --install-script post_installation_script.py

image_list = glob.glob('App\\images\\*.png')

setup(
    name = 'Iris', 
    version = 'v1.6',
    packages = ['iris', 'iris.tifffile','iris.pyqtgraph'],
    description = 'UED data exploration', 
    author = 'Laurent P. René de Cotret',
    author_email = 'laurent.renedecotret@mail.mcgill.ca',
    url = 'www.physics.mcgill.ca/siwicklab',
    scripts = ['post_installation_script.py'],
    py_modules = ['iris.dataset','iris.pattern', 'iris.gui', 'iris.progress_widget', 
                  'iris.hough', 'iris.preprocess', 'iris.wavelet'], 
    install_requires = ['numpy', 'pyqt4', 'scipy', 'h5py', 'PyWavelets'],
    data_files = [('App\\images', image_list)]
    )