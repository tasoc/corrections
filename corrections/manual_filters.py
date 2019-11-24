#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np

#------------------------------------------------------------------------------
def manual_exclude(lc):
	"""
	Return indicies for lightkurve object where "Manual Exclude" flag should be set.

	Parameters:
		lc (``TessLightCurve`` object): Lightcurve to set flags for.

	Returns:
		ndarray: Boolean array with `True` where "Manual Exclude" quality flag should be applied.
	"""

	manexcl = np.zeros_like(lc.time, dtype='bool')

#	if lc.sector == 1 and lc.camera == 1:
#		manexcl[lc.cadenceno >= 5050] = True
#
#	if lc.sector == 1:
#		manexcl[(lc.time >= 1347) & (lc.time <= 1350)] = True

	return manexcl
