# -*- coding: utf-8 -*-

import argparse
import sys
import webbrowser
from pathlib import Path
from multiprocessing import freeze_support

from iris import __version__
from iris.gui import run

DESCRIPTION = """Iris is both a library for interacting with ultrafast electron 
diffraction data, as well as a GUI frontend for interactively exploring this data.

Below are some helpful commands. """

EPILOG = """Running this command without any parameters will 
launch the graphical user interface."""

OPEN_HELP = """Dataset to open with iris start-up. """

DOCS_HELP = """Open online documentation in your default web browser."""

parser = argparse.ArgumentParser(prog="iris", description=DESCRIPTION, epilog=EPILOG)
parser.add_argument("-v", "--version", action="version", version=__version__)

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
    "--raw",
    action="store_true",
    help="This flag indicates that the path should be considered a raw dataset. Raw dataset format will be inferred from the installed plugins",
)
dset_modes.add_argument(
    "--reduced",
    action="store_true",
    help="This flag indicates that the path should be considered a reduced dataset.",
)

# Parser to reach documentation
docs_parser = subparsers.add_parser("docs", help=DOCS_HELP)


def main():
    # This is to support the pynsist-built executables
    # as described here:
    #   https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
    freeze_support()

    args = parser.parse_args()

    if args.subcmd == "open":

        # Otherwise, default behavior
        sys.exit(run(path=args.path))

    elif args.subcmd == "docs":
        webbrowser.open("https://iris-ued.readthedocs.io")
        sys.exit(0)

    # Default behavior : open gui without loading any data
    else:
        sys.exit(run(path=None))


if __name__ == "__main__":
    main()
