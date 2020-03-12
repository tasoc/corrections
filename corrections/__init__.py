#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from .BaseCorrector import BaseCorrector, STATUS
from .KASOCFilterCorrector import KASOCFilterCorrector
from .cbv_corrector import create_cbv, CBVCorrector, CBVCreator, CBV
from .ensemble import EnsembleCorrector
from .tesscorr import corrclass
from .taskmanager import TaskManager

from .version import get_version
__version__ = get_version(pep440=False)
