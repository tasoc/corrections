#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BaseCorrection.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import BaseCorrector

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#----------------------------------------------------------------------
def test_import():
	"""
	Tests if the module can even be imported.

	Doesn't really do anything else..."""

	with BaseCorrector(INPUT_DIR) as bc:
		assert bc.__class__.__name__ == 'BaseCorrector', "Did not get the correct class name back"
		assert bc.input_folder == INPUT_DIR, "Incorrect input folder"

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_import()
