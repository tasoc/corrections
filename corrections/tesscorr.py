# -*- coding: utf-8 -*-
"""
TESS Correction - tesscorr.py
Structure from `tessphot by Rasmus Handberg <https://github.com/tasoc/photometry/blob/devel/photometry/tessphot.py>`_


.. codeauthor:: Lindsey Carboneau

"""

from __future__ import absolute_import
import logging
import traceback
from . import STATUS
from .ensemble import EnsembleCorrector # TODO: figure out why the __init__.py doesn't work for this?
# from . import EnsembleCorrector, CBVCorrector, etc...

#------------------------------------------------------------------------------
class _CorrErrorDummy(object):
	def __init__(self, traceback, *args, **kwargs):
		self.status = STATUS.ERROR
		self._details = {'errors':traceback} if traceback else {}

#------------------------------------------------------------------------------
def _try_correction(CorrClass, *args, **kwargs):
	logger = logging.getLogger(__name__)
	tbcollect = []
	# try/except for doing correction
	try:
		with CorrClass(*args, **kwargs) as corr:
			corr.correct()

			if corr._status in (STATUS.OK, STATUS.WARNING):
				logger.info("Correction finished with status: '%s'", str(corr._status))
				# TODO: debate if save_lightcurve() should be called here or within correct()

	except (KeyboardInterrupt, SystemExit):
		logger.info("Stopped by user or system")
		try:
			corr._status = STATUS.ABORT
		except:
			pass
	except:
		logger.exception("Something happened")
		tb = traceback.format_exc().strip()
		try:
			corr._status = STATUS.ERROR
			corr.report_details(error=tb)
		except:
			tbcollect.append(tb)

	try:
		return corr
	except UnboundLocalError:
		return _CorrErrorDummy(tbcollect, *args, **kwargs)

#------------------------------------------------------------------------------
def tesscorr(method=None, *args, **kwargs):
	"""
	Run the corrector

	This function will run the specified corrector for a given target ID, 
	creating CBVs for a given area or CCD if they do not already exist.

	Parameters:
	    # TODO
	
	Returns: 
		# TODO
	"""

	logger = logging.getLogger(__name__)

	if method is None:
		# assume general, coarse correction
		corr = _try_correction(EnsembleCorrector, *args, **kwargs)

		if corr.status == STATUS.WARNING:
			logger.warning("A helpful warning message here") # TODO

	elif method == 'placeholder':
        # TODO: add other correctors
		pass
	else:
		raise ValueError("Invalid method: '{0}'".format(method))

	return corr