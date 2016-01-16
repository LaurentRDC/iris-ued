# -*- coding: utf-8 -*-
from distutils.core import setup
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst

image_list = glob.glob('images\\*.png')

setup(
    name = 'PowderGui', 
    version = 'v0.5', 
    description = 'UED Powder Diffraction Data Processing', 
    author = 'Laurent P. René de Cotret and Mark J. Stern',
    url = '', 
    py_modules = ['gui', 'core'], 
    data_files = [('images', image_list)])