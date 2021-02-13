# -*- coding: utf-8 -*-
"""
Update checks management
========================
"""
import json
from urllib.error import URLError
from urllib.request import urlopen

from PyQt5 import QtCore
from packaging.version import Version

from .. import __version__


def update_available():
    """
    Checks whether the currently-installed iris-ued is outdated.

    Returns
    -------
    is_outdated : bool
        Whether or not a new version is available
    latest : str
        Latest available version, currently installed or not.

    Raises
    ------
    ConnectionError
        if connection to PyPI could not be made.
    """
    url = "https://pypi.org/pypi/iris-ued/json"

    try:
        response = urlopen(url).read().decode("utf-8")
    except URLError:
        raise ConnectionError("No connection available.")

    latest_version = Version(json.loads(response)["info"]["version"])

    is_outdated = latest_version > Version(__version__)
    return is_outdated, str(latest_version)


class UpdateChecker(QtCore.QThread):
    """
    Worker that checks for iris-ued updates in a separate thread. Using a separate QThread
    prevents long start times.
    """

    update_available_signal = QtCore.pyqtSignal(bool)
    update_status_signal = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        QtCore.QThread.__init__(self)

    def run(self):
        try:
            outdated, latest = update_available()
            if outdated:
                msg = f"An update is available: latest version is {latest}, and you are currently running version {__version__}."
            else:
                msg = f"You are running the latest version, {__version__}."
        except ConnectionError:
            outdated = False
            msg = "Could not determine if an update is available. No connections available."

        self.update_available_signal.emit(outdated)
        self.update_status_signal.emit(msg)
