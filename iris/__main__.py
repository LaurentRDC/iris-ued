# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

from . import __version__
from .pack import pack

from iris.gui import run

DESCRIPTION = """Iris is both a library for interacting with ultrafast electron 
diffraction data, as well as a GUI frontend for interactively exploring this data."""

EPILOG = """Documentation is available here: https://iris-ued.readthedocs.io/"""

PATH_HELP = """Raw or reduced dataset to open, if any. 
The type of dataset will be guessed based on currently-installed plugins."""

PACK_HELP = """ Pack a raw dataset into a compressed HDF5 archive, resulting in space 
savings of up to 4x. Archives can be loaded into iris."""

parser = argparse.ArgumentParser(prog="iris", description=DESCRIPTION, epilog=EPILOG)
parser.add_argument("-v", "--version", action="version", version=f"iris {__version__}")
parser.add_argument("path", help=PATH_HELP, type=Path, nargs="?", default=None)

subparsers = parser.add_subparsers(help="Available sub-commands")

# Parser for pack command
pack_parser = subparsers.add_parser('pack', help=PACK_HELP)
pack_parser.add_argument('--src', type=Path, required=True)
pack_parser.add_argument('--dst', type=Path, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    
    # If source and destination paths are provided, this is because we
    # want to pack
    if hasattr(args, 'src') and hasattr(args, 'dst'):
        try:
            pack(args.src, args.dst)
        except RuntimeError as e:
            print('[iris pack] the following fatal error occured: ', str(e))
            sys.exit(1)
        else:
            print(f'Dataset {args.src} has been successfully packed into {args.dst}')
            sys.exit(0)

    # Otherwise, default behavior
    sys.exit(run(path=args.path))
