# -*- coding: utf-8 -*-
"""
TESS Correction - tesscorr.py
Structure from `tessphot by Rasmus Handberg <https://github.com/tasoc/photometry/blob/devel/photometry/tessphot.py>`_


.. codeauthor:: Lindsey Carboneau

"""

from __future__ import absolute_import
import logging
# from . import STATUS, etc...

#------------------------------------------------------------------------------
class _CorrErrorDummy(object):
	def __init__(self, *args, **kwargs):
		self.status = STATUS.ERROR
		self._details = {}

#------------------------------------------------------------------------------
def _try_correction(CorrClass, *args, **kwargs):
	logger = logging.getLogger(__name__)
	# try/except for doing correction
	try:
		with CorrClass(*args, **kwargs) as corr:
			corr.correction()

			if corr.status in (STATUS.OK, STATUS.WARNING):
				corr.save_lightcurve()

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
			pass

    try:
		return corr
	except UnboundLocalError:
		return _CorrErrorDummy(*args, **kwargs)
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

	elif method == 'placeholder':
        # TODO: add other correctors
        pass
    
    else:
        raise ValueError("Invalid method: '{0}'".format(method))