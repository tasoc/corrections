# -*- coding: utf-8 -*-
"""
TESS Correction - tesscorr.py
Structure from `tessphot by Rasmus Handberg <https://github.com/tasoc/photometry/blob/devel/photometry/tessphot.py>`_


.. codeauthor:: Lindsey Carboneau

"""

from __future__ import absolute_import
from . import KASOCFilterCorrector, EnsembleCorrector #, CBVCorrector

#------------------------------------------------------------------------------
def corrclass(method=None):

	if method is None:
		# assume general, coarse correction
		return EnsembleCorrector

	#elif method == 'cbv':
	#	return CBVCorrector

	elif method == 'ensemble':
		return EnsembleCorrector

	elif method == 'kasoc_filter':
		return KASOCFilterCorrector

	else:
		raise ValueError("Invalid method: '{0}'".format(method))
