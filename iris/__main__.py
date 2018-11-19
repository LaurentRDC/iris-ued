# -*- coding: utf-8 -*-

import argparse
import sys
import webbrowser
from pathlib import Path

from iris.gui import run

from . import __version__
from .pack import pack

DESCRIPTION = """Iris is both a library for interacting with ultrafast electron 
diffraction data, as well as a GUI frontend for interactively exploring this data.

Below are some helpful commands. """

EPILOG = """Running this command without any parameters will 
launch the graphical user interface."""

OPEN_HELP = """Dataset to open with iris start-up. """

PACK_HELP = """Pack a raw dataset into a compressed HDF5 archive, resulting in space 
savings of up to 4x. Archives can be loaded into iris. """

DOCS_HELP = """Open online documentation in your default web browser."""

parser = argparse.ArgumentParser(prog="iris", description=DESCRIPTION, epilog=EPILOG)
parser.add_argument("-v", "--version", action="version", version=f"iris {__version__}")

subparsers = parser.add_subparsers(
    title="Subcommands", help="Available sub-commands", dest="subcmd"
)

# Parser to open a path
# To facilitate format determination, we need flags specifying whether the
# path points to a raw, compact, or reduced dataset
open_parser = subparsers.add_parser("open", help=OPEN_HELP)
open_parser.add_argument(
    "path", help="Path to the dataset", type=Path, nargs="?", default=None
)
dset_modes = open_parser.add_mutually_exclusive_group(required=True)
dset_modes.add_argument(
    "--compact",
    action="store_true",
    help="This flag indicates that the path should be considered an compact raw dataset",
)
dset_modes.add_argument(
    "--raw",
    action="store_true",
    help="This flag indicates that the path should be considered a raw dataset. Raw dataset format will be inferred from the installed plugins",
)
dset_modes.add_argument(
    "--reduced",
    action="store_true",
    help="This flag indicates that the path should be considered a reduced dataset.",
)

# Parser for pack command
pack_parser = subparsers.add_parser("pack", help=PACK_HELP)
pack_parser.add_argument("--src", type=Path, required=True)
pack_parser.add_argument("--dst", type=Path, required=True)

# Parser to reach documentation
docs_parser = subparsers.add_parser("docs", help=DOCS_HELP)

if __name__ == "__main__":
    args = parser.parse_args()

    # If source and destination paths are provided, this is because we
    # want to pack
    if args.subcmd == "pack":
        try:
            pack(args.src, args.dst)
        except RuntimeError as e:
            print("[iris pack] the following fatal error occured: ", str(e))
            sys.exit(1)
        else:
            print(f"Dataset {args.src} has been successfully packed into {args.dst}")
            sys.exit(0)

    elif args.subcmd == "open":

        # Otherwise, default behavior
        sys.exit(run(path=args.path))

    elif args.subcmd == "docs":
        webbrowser.open("https://iris-ued.readthedocs.io")
        sys.exit(0)

    # Default behavior : open gui without loading any data
    else:
        sys.exit(run(path=None))
