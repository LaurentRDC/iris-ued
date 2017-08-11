# -*- coding: utf-8 -*-
__author__ = 'Laurent P. René de Cotret'
__email__ = 'laurent.renedecotret@mail.mcgill.ca'
__license__ = 'MIT'
__version__ = '4.2' # TODO: automatic versioning?

from .beam_properties import beam_properties
from .dataset import DiffractionDataset, PowderDiffractionDataset, explore_dir
from .knife_edge import cdf, knife_edge
from .optimizations import cached_property
from .raw import parse_tagfile, McGillRawDataset, RawDatasetBase
from .utils import scattering_length
from .processing import process
