# -*- coding: utf-8 -*
import numpy
from setuptools import setup, find_packages
import glob

__version__ = '4.0.1'
__author__ = 'Laurent P. René de Cotret'


#To create a Windows installer for this, run:
# >>> python setup.py bdist_wininst

image_list = glob.glob('iris\\gui\\images\\*.png')

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
                        'scikit-ued >= 0.4.2',
                        'pyqtgraph >= 0.10',
                        'qdarkstyle >= 2.3',
                        'psutil'],
    data_files = [('iris\\gui\\images', image_list)],
    entry_points = {'gui_scripts': ['iris = iris.gui:run']}
    )