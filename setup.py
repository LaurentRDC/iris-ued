# -*- coding: utf-8 -*-
from itertools import chain
from setuptools import setup, find_packages
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst

image_list = glob.glob('iris\\gui\\images\\*.png')
wavelets = chain.from_iterable([glob.glob('iris\\dualtree\\data\\*.npy'), 
                                glob.glob('iris\\dualtree\\data\\*.npz')])
rc = chain.from_iterable([glob.glob('iris\\gui\\qdarkstyle\\*.qrc'),
                          glob.glob('iris\\gui\\qdarkstyle\\*.qss')])

setup(
    name = 'iris', 
    version = '2.0.1',
    packages = find_packages(),
    description = 'UED data exploration', 
    author = 'Laurent P. René de Cotret',
    author_email = 'laurent.renedecotret@mail.mcgill.ca',
    url = 'www.physics.mcgill.ca/siwicklab',
    install_requires = ['numpy', 'scipy', 'h5py', 'PyWavelets', 'tifffile'],
    data_files = [('iris\\gui\\images', image_list),
                  ('iris\\dualtree\\data', wavelets),
                  ('iris\\gui\\qdarkstyle', rc)]
    )