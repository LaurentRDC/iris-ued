import os, glob

from PyInstaller.compat import string_types
from PyInstaller.utils.hooks import get_package_paths, remove_prefix, collect_data_files
from PyInstaller import log as logging

logger = logging.getLogger(__name__)

PY_DYLIB_PATTERNS = ["*.pyd", "*.dll", "*.dylib", "lib*.so"]

# The default function from PyInstaller
# does not collect *.pyd libraries!
# Therefore, the function below is identical to
# the PyInstaller.utils.hooks.collect_dynamic_libs function
# except for the addition of *.pyd files
def collect_dynamic_libs(package, destdir=None):
    """
    This routine produces a list of (source, dest) of dynamic library
    files which reside in package. Its results can be directly assigned to
    ``binaries`` in a hook script. The package parameter must be a string which
    names the package.

    :param destdir: Relative path to ./dist/APPNAME where the libraries
                    should be put.
    """
    # Accept only strings as packages.
    if not isinstance(package, string_types):
        raise ValueError

    logger.debug("Collecting dynamic libraries for %s" % package)
    pkg_base, pkg_dir = get_package_paths(package)
    # Walk through all file in the given package, looking for dynamic libraries.
    dylibs = []
    for dirpath, _, __ in os.walk(pkg_dir):
        # Try all file patterns in a given directory.
        for pattern in PY_DYLIB_PATTERNS:
            files = glob.glob(os.path.join(dirpath, pattern))
            for source in files:
                # Produce the tuple
                # (/abs/path/to/source/mod/submod/file.pyd,
                #  mod/submod/file.pyd)
                if destdir:
                    # Libraries will be put in the same directory.
                    dest = destdir
                else:
                    # The directory hierarchy is preserved as in the original package.
                    dest = remove_prefix(dirpath, os.path.dirname(pkg_base) + os.sep)
                logger.debug(" %s, %s" % (source, dest))
                dylibs.append((source, dest))
    return dylibs


binaries = []
binaries += collect_dynamic_libs("pywt")

datas = []
datas += collect_data_files("pywt", subdir="data")
