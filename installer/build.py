# -*- coding: utf-8 -*-
"""
This script creates a fresh Python environment and create an installer.

Inspired from the Spyder installer script:
https://github.com/spyder-ide/spyder/tree/master/installers/Windows
"""
from pathlib import Path
from subprocess import run, check_output
import tempfile
import importlib.util as iutil
import shutil

INSTALLER_NAME = "iris_64bit.exe"
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
pypi_wheels= PyQt5==5.15.2
    PyQt5-sip==12.8.1
packages=
    {packages}
[Build]
installer_name={installer_name}
directory=build/nsis/
"""


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
        "pillow": "PIL",
        "pycifrw": "CifFile",
        "pywavelets": "pywt",
        "python-dateutil": "dateutil",
        "pyyaml": "yaml",
        "qdarkstyle": "qdarkstyle",  # lowercase important
        "qtpy": "qtpy",  # lowercase important
    }
    return translations.get(pkg.lower(), pkg)


def generate_pynsist_config(python_exe, filename):
    """
    Create a pynsist configuration file

    Parameters
    ----------
    python_exe : path-like
        Full path to the Python executable used to generate the installer.
    filename : path-like
        Full path to the generated config file.
    """
    package_name = lambda t: t.partition("==")[0].split("@")[0].strip()

    freeze = check_output(f"{python_exe} -m pip freeze --all").decode("latin1")
    # PyQt requirements are baked in the template string
    requirements = [
        line
        for line in freeze.splitlines()
        if package_name(line) not in {"iris-ued", "PyQt5", "PyQt5-sip"}
        and "-e git" not in line
    ]
    packages = [package_name(p) for p in requirements]

    python_version = (
        check_output(f"{env_python} --version").decode("latin1").split(" ")[-1].strip()
    )

    pynsist_cfg_payload = PYNSIST_CFG_TEMPLATE.format(
        version="5.2.6",  # TODO: find dynamically
        icon_file=Path(__file__).parent / "iris.ico",
        license_file=REPO_ROOT / "LICENSE.txt",
        python_version=python_version,
        publisher="Laurent P. René de Cotret",  # TODO: find dynamically
        packages="\n    ".join([importable_name(p) for p in packages]),
        installer_name="iris_64bit.exe",
        template=None,
    )
    with open(filename, mode="wt", encoding="latin1") as f:
        f.write(pynsist_cfg_payload)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="installer-iris-ued-") as work_dir:
        work_dir = Path(work_dir)
        run(f"python -m venv {work_dir} --upgrade --upgrade-deps --copies")

        env_python = work_dir / "Scripts" / "python.exe"
        run(f"{env_python} -m pip install --upgrade wheel --no-warn-script-location")
        run(f"{env_python} -m pip install {REPO_ROOT} --no-warn-script-location")

        # Generating configuration file BEFORE installing pynsist ensures that
        # we bundle the requirements for iris
        cfg_path = work_dir / "pynsist.cfg"
        generate_pynsist_config(python_exe=env_python, filename=cfg_path)
        print(open(cfg_path, "rt").read())

        run(
            f"{env_python} -m pip install -r {INSTALLER_REQUIREMENTS_FILE} --no-warn-script-location"
        )

        run(f"{env_python} -m nsist {cfg_path}")

        DESTINATION.mkdir(exist_ok=True)
        shutil.copy(work_dir / "build" / "nsis" / INSTALLER_NAME, DESTINATION)
