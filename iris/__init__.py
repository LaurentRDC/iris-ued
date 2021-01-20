# -*- coding: utf-8 -*-
__author__ = "Laurent P. René de Cotret"
__email__ = "laurent.renedecotret@mail.mcgill.ca"
__license__ = "GPLv3"
__version__ = "6.0.0"

from .raw import AbstractRawDataset, check_raw_bounds, open_raw
from .dataset import DiffractionDataset
from .meta import ExperimentalParameter
from .plugins import install_plugin, load_plugin

from . import plugins
