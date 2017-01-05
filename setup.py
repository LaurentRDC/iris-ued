# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst

image_list = glob.glob('App\\images\\*.png')

setup(
    name = 'iris', 
    version = 'v2.0',
    packages = find_packages(),
    description = 'UED data exploration', 
    author = 'Laurent P. René de Cotret',
    author_email = 'laurent.renedecotret@mail.mcgill.ca',
    url = 'www.physics.mcgill.ca/siwicklab',
    install_requires = ['numpy', 'scipy', 'h5py', 'PyWavelets', 'tifffile'],
    data_files = [('App\\images', image_list)]
    )