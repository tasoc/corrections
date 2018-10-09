#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BaseCorrection.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
from corrections import BaseCorrector

#----------------------------------------------------------------------
def test_import():
	"""
	Tests if the module can even be imported.
	
	Doesn't really do anything else..."""
	
	with BaseCorrector() as bc:
		pass

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_import()
