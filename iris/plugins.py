# -*- coding: utf-8 -*-
"""
Plugin managemenent
===================

Plug-ins are modules implementing a subclass of AbstractRawDataset. 
Plug-ins should be placed in ~\iris_plugins.

Plug-in classes can be imported from ``iris.plugins``.
"""

from pathlib import Path
from runpy import run_path
from shutil import copy2

# Pluging location is ~\iris_plugins
PLUGIN_DIR = Path.home() / Path('iris_plugins')

# Create the installation directory if it does not exist
if not PLUGIN_DIR.exists():
    PLUGIN_DIR.mkdir()

for fname in PLUGIN_DIR.rglob('*.py'):
    globals().update(run_path(PLUGIN_DIR / fname, run_name = 'iris.plugins'))

def install_plugin(path):
    """ 
    Install and load an iris plug-in.

    .. versionadded:: 5.0.4
    
    Parameters
    ----------
    path : path-like
        Path to the plug-in. This plug-in file will be copied. """
    path = Path(path)
    new_path = PLUGIN_DIR / path.name
    copy2(path, new_path)

    globals().update(run_path(new_path, run_name = 'iris.plugins'))
