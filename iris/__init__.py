# -*- coding: utf-8 -*-
__author__  = 'Laurent P. René de Cotret'
__email__   = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'MIT'
__version__ = '5.0a1'

# Versioning should be compliant with PyPI guide 
# https://packaging.python.org/tutorials/distributing-packages/#choosing-a-versioning-scheme

from .raw import AbstractRawDataset
from .mcgill import McGillRawDataset, LegacyMcGillRawDataset
from .dataset import DiffractionDataset, PowderDiffractionDataset

from . import plugins