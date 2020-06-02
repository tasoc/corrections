#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of corrclass.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import pytest
import conftest # noqa: F401
import corrections

#--------------------------------------------------------------------------------------------------
def test_corrclass_type():
	"""Check that tesscorr.py returns the correct class"""

	CorrClass = corrections.corrclass()
	print(CorrClass)
	assert CorrClass is corrections.EnsembleCorrector

	CorrClass = corrections.corrclass('ensemble')
	print(CorrClass)
	assert CorrClass is corrections.EnsembleCorrector

	CorrClass = corrections.corrclass('cbv')
	print(CorrClass)
	assert CorrClass is corrections.CBVCorrector

	CorrClass = corrections.corrclass('kasoc_filter')
	print(CorrClass)
	assert CorrClass is corrections.KASOCFilterCorrector

	method = 'not-a-method'
	with pytest.raises(ValueError) as err:
		CorrClass = corrections.corrclass(method)
	assert err.value.args[0] == "Invalid method: '{0}'".format(method)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
