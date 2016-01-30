# -*- coding: utf-8 -*-
from distutils.core import setup
import glob

#To create a Windows installer for this, run:
#
# > python setup.py bdist_wininst --install-script post_installation_script.py

image_list = glob.glob('PowderGuiApp\\images\\*.png')

setup(
    name = 'PowderGui', 
    version = 'v0.8', 
    description = 'UED Powder Diffraction Data Processing', 
    author = 'Laurent P. Rene de Cotret',
    url = 'www.physics.mcgill.ca/siwicklab',
    download_url = 'http://1drv.ms/1OVX2ac',
    scripts = ['post_installation_script.py'],
    py_modules = ['PowderGuiApp.core', 'PowderGuiApp.gui', 'PowderGuiApp.image_viewer'], 
    install_requires = ['tqdm', 'numpy', 'pyqt4', 'pyqtgraph', 'scipy', 'tifffile', 'h5py'],
    data_files = [('PowderGuiApp\\images', image_list)]
    )