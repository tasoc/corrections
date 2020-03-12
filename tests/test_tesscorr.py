#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of corrclass.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import pytest
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import corrclass

#--------------------------------------------------------------------------------------------------
def test_corrclass_type():
	"""Check that tesscorr.py returns the correct class"""

	CorrClass = corrclass()
	assert repr(CorrClass) == "<class 'corrections.ensemble.EnsembleCorrector'>"

	CorrClass = corrclass('ensemble')
	assert repr(CorrClass) == "<class 'corrections.ensemble.EnsembleCorrector'>"

	CorrClass = corrclass('cbv')
	assert repr(CorrClass) == "<class 'corrections.cbv_corrector.CBVCorrector.CBVCorrector'>"

	CorrClass = corrclass('kasoc_filter')
	assert repr(CorrClass) == "<class 'corrections.KASOCFilterCorrector.KASOCFilterCorrector'>"

	method = 'not-a-method'
	with pytest.raises(ValueError) as err:
		CorrClass = corrclass(method)
	assert err.value.args[0] == "Invalid method: '{0}'".format(method)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
