# -*- coding: utf-8 -*-
"""
Logging utilities
"""
import sys
import logging
import logging.handlers
from tempfile import gettempdir

from pathlib import Path
from PyQt5 import QtCore

from .. import __version__

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

LOG_DIRECTORY = Path(gettempdir()) / f"iris-{__version__}-logs"
LOG_DIRECTORY.mkdir(exist_ok=True)


class QLogger(QtCore.QObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=LOG_DIRECTORY / "iris.log", when="midnight", backupCount=5
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(FORMATTER)

        self.stream_handler = logging.StreamHandler(stream=sys.stdout)
        self.stream_handler.setFormatter(logging.INFO)
        self.stream_handler.setFormatter(FORMATTER)

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)

    def log(self, msg, level):
        """General method for logging. Ideal for dynamic logging levels."""
        self.logger.log(level, msg)

    # The methods below mirror those from the logging module
    @QtCore.pyqtSlot(str)
    def debug(self, msg):
        return self.logger.debug(msg)

    @QtCore.pyqtSlot(str)
    def info(self, msg):
        return self.logger.info(msg)

    @QtCore.pyqtSlot(str)
    def warning(self, msg):
        return self.logger.warning(msg)

    @QtCore.pyqtSlot(str)
    def error(self, msg):
        return self.logger.error(msg)

    @QtCore.pyqtSlot(str)
    def critical(self, msg):
        return self.logger.critical(msg)

    @QtCore.pyqtSlot(str)
    def exception(self, msg):
        return self.logger.exception(msg)
