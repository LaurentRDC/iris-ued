""" 
The dask package contains configuration information.

This is a transitive dependency of scikit-image
"""
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files("dask")