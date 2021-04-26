# -*- coding: utf-8 -*-
"""
This script creates a fresh Python environment and create an installer.

Inspired from the Spyder installer script:
https://github.com/spyder-ide/spyder/tree/master/installers/Windows
"""
from pathlib import Path
import sys
import subprocess
from functools import wraps
import logging
import tempfile
import importlib.util as iutil
import shutil
import argparse

logging.basicConfig(encoding="utf-8", level=logging.INFO)

REPO_ROOT = Path(__file__).parent.parent
DESTINATION = REPO_ROOT / "dist"
INSTALLER_REQUIREMENTS_FILE = Path(__file__).parent / "inst-requirements.txt"
PYNSIST_CFG_TEMPLATE = """
[Application]
name=iris
version={version}
entry_point=iris.gui:run
icon={icon_file}
publisher={publisher}
license_file={license_file}

[Python]
version={python_version}
bitness=64
format=bundled

[Include]
pypi_wheels= 
    pyqt5==5.15.4
    pyqt5-sip==12.8.1
packages=
    {packages}
[Build]
installer_name={installer_name}
directory=build/nsis/
"""

parser = argparse.ArgumentParser(
    prog="build.py", description="Iris Windows installer build script."
)
parser.add_argument(
    "exe_name",
    metavar="TARGET",
    help="Name of the resulting installer executable (e.g. 'iris-installer.exe')",
    type=str,
    default=None,
)

@wraps(subprocess.run)
def run(cmd, *args, **kwargs):
    logging.info("Running " + cmd)
    return subprocess.run(cmd, *args, **kwargs)

@wraps(subprocess.check_output)
def check_output(cmd, *args, **kwargs):
    logging.info("Checking output of " + cmd)
    return subprocess.check_output(cmd, *args, **kwargs)


def importable_name(pkg):
    """
    Translate package name to importable name, e.g. "scikit-image" -> "skimage".
    """
    # TODO: find a way to determine this list programatically
    translations = {
        "scikit-image": "skimage",
        "scikit-ued": "skued",
        "iris-ued": "iris",
        "pyqt5-sip": "sip",
        "pyqt5-qt5": "PyQt5",
        "pillow": "PIL",
        "pycifrw": "CifFile",
        "pywavelets": "pywt",
        "python-dateutil": "dateutil",
        "pyyaml": "yaml",
        "qdarkstyle": "qdarkstyle",  # lowercase important
        "qtpy": "qtpy",  # lowercase important
    }
    return translations.get(pkg.lower(), pkg)


def generate_pynsist_config(python_exe, filename, exe_name):
    """
    Create a pynsist configuration file. Note that all required packages need
    to be installed before calling this function.

    Parameters
    ----------
    python_exe : path-like
        Full path to the Python executable used to generate the installer.
    filename : path-like
        Full path to the generated config file.
    """
    package_name = lambda t: t.partition("==")[0].split("@")[0].strip()

    freeze = check_output(f"{python_exe} -m pip freeze --all").decode("latin1")
    # PyQt5/PyQt5-sip requirements are baked in the template string
    requirements = [
        line
        for line in freeze.splitlines()
        if package_name(line) not in {"iris-ued", "PyQt5", "PyQt5-sip"}
    ]
    packages = [package_name(p) for p in requirements]

    python_version = (
        check_output(f"{env_python} --version").decode("latin1").split(" ")[-1].strip()
    )

    iris_version = check_output(
        f'{python_exe} -c "import iris; print(iris.__version__)"'
    ).decode("latin1")
    iris_authors = check_output(
        f'{python_exe} -c "import iris; print(iris.__author__)"'
    ).decode("latin1")

    pynsist_cfg_payload = PYNSIST_CFG_TEMPLATE.format(
        version=iris_version,
        icon_file=Path(__file__).parent / "iris.ico",
        license_file=REPO_ROOT / "LICENSE.txt",
        python_version=python_version,
        publisher=iris_authors,
        packages="\n    ".join([importable_name(p) for p in packages]),
        installer_name=exe_name,
    )
    with open(filename, mode="wt", encoding="latin1") as f:
        f.write(pynsist_cfg_payload)


if __name__ == "__main__":
    args = parser.parse_args()
    exe_name = args.exe_name

    with tempfile.TemporaryDirectory(prefix="installer-iris-ued-") as work_dir:
        work_dir = Path(work_dir)
        run(f"python -m venv {work_dir} --upgrade --upgrade-deps --copies")

        env_python = work_dir / "Scripts" / "python.exe"
        run(f"{env_python} -m pip install --upgrade wheel --no-warn-script-location")
        run(f"{env_python} -m pip install {REPO_ROOT} --no-warn-script-location")

        # Generating configuration file BEFORE installing pynsist ensures that
        # we bundle the requirements for iris
        cfg_path = work_dir / "pynsist.cfg"
        generate_pynsist_config(
            python_exe=env_python, filename=cfg_path, exe_name=exe_name
        )
        print(open(cfg_path, "rt").read())

        run(
            f"{env_python} -m pip install -r {INSTALLER_REQUIREMENTS_FILE} --no-warn-script-location"
        )

        run(f"{env_python} -m nsist {cfg_path}")

        DESTINATION.mkdir(exist_ok=True)
        shutil.copy(work_dir / "build" / "nsis" / exe_name, DESTINATION)
