# -*- coding: utf-8 -*-
__author__ = "Laurent P. René de Cotret"
__email__ = "laurent.decotret@outlook.com"
__license__ = "GPLv3"
__version__ = "5.4.0"

from . import plugins
from .dataset import DiffractionDataset, MigrationError, MigrationWarning
from .meta import ExperimentalParameter
from .plugins import install_plugin, load_plugin
from .powder import PowderDiffractionDataset
from .raw import AbstractRawDataset, check_raw_bounds, open_raw
