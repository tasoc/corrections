#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Ensemble.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import ensemble

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#----------------------------------------------------------------------
def test_basic():
	assert 3 = 1+2
#----------------------------------------------------------------------
