# -*- coding: utf-8 -*-
"""
Plugin managemenent
===================

Plug-ins are modules implementing a subclass of AbstractRawDataset. 
Plug-ins should be placed in ~\iris_plugins.

Plug-in classes can be imported from ``iris.plugins``.
"""

from runpy import run_path

from pathlib import Path

# Pluging location is ~\iris_plugins
PLUGIN_DIR = Path.home() / Path('iris_plugins')

# Create the installation directory if it does not exist
if not PLUGIN_DIR.exists():
    PLUGIN_DIR.mkdir()

for fname in PLUGIN_DIR.rglob('*.py'):
    globals().update(run_path(PLUGIN_DIR / fname, run_name = 'iris.plugins'))