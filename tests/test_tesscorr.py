#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Ensemble.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

import sys
import pytest
import os
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

    with pytest.raises(ValueError) as err:
        method = 'not-a-method'
        CorrClass = corrclass('method')
        assert err.value.args[0] == "Invalid method: '{0}'".format(method)
