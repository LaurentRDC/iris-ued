# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

from . import __version__

from iris.gui import run

DESCRIPTION = """Iris is both a library for interacting with ultrafast electron 
diffraction data, as well as a GUI frontend for interactively exploring this data."""

EPILOG = """Documentation is available here: https://iris-ued.readthedocs.io/"""

PATH_HELP = """Raw or reduced dataset to open, if any. 
The type of dataset will be guessed based on currently-installed plugins."""

parser = argparse.ArgumentParser(prog = 'iris', 
                                 description = DESCRIPTION,
                                 epilog = EPILOG)
parser.add_argument('-v', '--version', action = 'version', version = f'iris {__version__}')
parser.add_argument('path', help = PATH_HELP, type = Path, nargs = '?', default = None)

if __name__ == '__main__':
    args = parser.parse_args()
    sys.exit(
        run(path = args.path)
    )
