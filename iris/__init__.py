# -*- coding: utf-8 -*-

__author__ = 'Laurent P. René de Cotret'
__version__ = '2.0.4'

from .dataset import DiffractionDataset, PowderDiffractionDataset
from .optimizations import cached_property, parallel_map, parallel_sum
from .utils import angular_average, scattering_length
from .raw import RawDataset
from .laplace import laplace