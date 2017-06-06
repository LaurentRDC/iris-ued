# -*- coding: utf-8 -*-
from itertools import chain
import numpy
from setuptools import setup, find_packages
import glob

__version__ = '4.0'
__author__ = 'Laurent P. René de Cotret'


#To create a Windows installer for this, run:
# >>> python setup.py bdist_wininst

image_list = glob.glob('iris\\gui\\images\\*.png')
rc = chain.from_iterable([glob.glob('iris\\gui\\qdarkstyle\\*.qrc'),
                          glob.glob('iris\\gui\\qdarkstyle\\*.qss')])

setup(
    name = 'iris', 
    version = __version__,
    packages = find_packages(),
    description = 'UED data exploration', 
    author = __author__,
    author_email = 'laurent.renedecotret@mail.mcgill.ca',
    url = 'www.physics.mcgill.ca/siwicklab',
    install_requires = ['numpy >= 1.11.2', 
                        'scipy', 
                        'h5py >= 2.6.0',
                        'scikit-image',
                        'scikit-ued',
                        'pyqtgraph >= 0.10'],
    data_files = [('iris\\gui\\images', image_list),
                  ('iris\\gui\\qdarkstyle', rc)],
    entry_points = {'gui_scripts': ['iris = iris.gui:run']}
    )