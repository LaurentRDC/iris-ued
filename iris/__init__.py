# -*- coding: utf-8 -*-
__author__ = "Laurent P. René de Cotret"
__email__ = "laurent.renedecotret@mail.mcgill.ca"
__license__ = "MIT"
__version__ = "5.1.4"

from . import plugins
from .dataset import DiffractionDataset, PowderDiffractionDataset
from .mcgill import LegacyMcGillRawDataset, McGillRawDataset
from .meta import ExperimentalParameter
from .plugins import install_plugin
from .raw import AbstractRawDataset, check_raw_bounds, open_raw
