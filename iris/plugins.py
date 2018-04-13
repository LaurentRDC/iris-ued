# -*- coding: utf-8 -*-
"""
Plugin managemenent
===================

Plug-ins are modules implementing a subclass of AbstractRawDataset. 
Plug-ins should be placed in ~\iris_plugins.
"""

from pathlib import Path

from pluginbase import PluginBase

# Pluging location is ~\iris_plugins
PLUGIN_DIR = Path.home() / Path('iris_plugins')

# Create the installation directory if it does not exist
if not PLUGIN_DIR.exists():
    PLUGIN_DIR.mkdir()

source = PluginBase(package = 'iris.plugins').make_plugin_source(searchpath = [str(PLUGIN_DIR)])

for plugin_name in source.list_plugins():
    source.load_plugin(plugin_name)