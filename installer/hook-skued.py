import os, glob

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller import log as logging

logger = logging.getLogger(__name__)

datas = []
datas += collect_data_files("skued")
