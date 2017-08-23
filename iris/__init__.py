# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'MIT'
__version__ = '5.0' # TODO: automatic versioning?

from .beam_properties import beam_properties
from .dataset import DiffractionDataset, PowderDiffractionDataset, VALID_DATASET_METADATA, VALID_POWDER_METADATA
from .knife_edge import cdf, knife_edge
from .raw import parse_tagfile, McGillRawDataset, RawDatasetBase, FSURawDataset
