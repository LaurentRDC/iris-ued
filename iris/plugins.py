# -*- coding: utf-8 -*-
"""
Plugin managemenent
===================

Plug-ins are modules implementing a subclass of AbstractRawDataset. 
Plug-ins should be placed in ~\\iris_plugins.

Plug-in classes can be imported from ``iris.plugins``.
"""

from pathlib import Path
from runpy import run_path
from shutil import copy2

PLUGIN_DIR = Path.home() / Path("iris_plugins")
PLUGIN_DIR.mkdir(exist_ok=True)


def load_plugin(path):
    """
    Load an iris plug-in, but do not install it. The plug-in
    will not be available in the next session.

    .. versionadded:: 5.2.0

    Parameters
    ----------
    path : path-like
        Path to the plug-in.
    """
    path = Path(path)
    if not path.exists():
        raise OSError(f"The plug-in file {path} does not exist.")

    try:
        globals().update(run_path(str(path), run_name="iris.plugins"))
    except Exception as e:
        print(f"Failed loading plugin {path} with error: {e}")


def install_plugin(path):
    """
    Install and load an iris plug-in. Installed plug-ins are loaded
    at every iris start-up.

    .. versionadded:: 5.0.4

    Parameters
    ----------
    path : path-like
        Path to the plug-in. This plug-in file will be copied.
    """
    path = Path(path)
    new_path = PLUGIN_DIR / path.name
    copy2(path, new_path)

    load_plugin(new_path)


for fname in PLUGIN_DIR.rglob("*.py"):
    load_plugin(PLUGIN_DIR / fname)
