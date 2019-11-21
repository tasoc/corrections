#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .BaseCorrector import BaseCorrector, STATUS
from .KASOCFilterCorrector import KASOCFilterCorrector
from .cbv_corrector import CBVCorrector
from .ensemble import EnsembleCorrector
from .tesscorr import corrclass
from .taskmanager import TaskManager

from .version import get_version
__version__ = get_version(pep440=False)
