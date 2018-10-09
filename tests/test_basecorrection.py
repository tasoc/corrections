#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BaseCorrection. Based off "test_baseclassifier.py" by Rasmus Handberg <rasmush@phys.au.dk>

.. codeauthor:: Lindsey Carboneau
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import BaseCorrection

#----------------------------------------------------------------------
def test_basecorrection():
	with BaseCorrection() as cl:
		assert(cl.__class__.__name__ == 'BaseCorrection')

		
#----------------------------------------------------------------------
if __name__ == '__main__':
	test_basecorrection()