# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'MIT'
__version__ = '5.0' # TODO: automatic versioning?

from .dataset import (AbstractRawDataset, McGillRawDataset, DiffractionDataset, 
                      PowderDiffractionDataset)
from .beam_properties import beam_properties
from .knife_edge import cdf, knife_edge
from .merlin import mibheader, mibread, imibread
