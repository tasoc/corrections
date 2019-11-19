#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correct lightcurves using Cotrending Basis Vectors.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os
import logging
from ..plots import plt
from .. import BaseCorrector, STATUS
from . import CBV

#--------------------------------------------------------------------------------------------------
class CBVCorrector(BaseCorrector):
	"""
	The CBV (Co-trending Basis Vectors) correction method for the TASOC
	photometry pipeline.

	The CBVCorrector inherits functionality of :py:class:`BaseCorrector`.

	Attributes:
		cbvs (dict): Dictionary of CBV objects.

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, *args, **kwargs):
		"""
		Initialise the CBVCorrector.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Dictionary that will hold CBV objects:
		self.cbvs = {}

	#----------------------------------------------------------------------------------------------
	def do_correction(self, lc, use_prior=False):
		"""
		Function where the correction is called, and where
		additional headers for the FITS are defined

		Parameters:
			lc (`TessLightCurve`): Lightcurve to correct.
			use_prior (boolean, optional): Use prior in fitting of CBVs. Default=False.

		Returns:
			`TessLightcurve`: Corrected lightcurve.
			`STATUS`: Status of the correction.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		logger.info('Co-trending star with TIC ID: %d', lc.targetid)

		# Load the CBV (and Prior) from memory and if it is not already loaded,
		# load it in from file and keep it in memory for next time:
		datasource = lc.meta['task']['datasource']
		cbv_area = lc.meta['task']['cbv_area']

		# Convert datasource into query-string for the database:
		# This will change once more different cadences (i.e. 20s) is defined
		if datasource == 'ffi':
			datasource = "ffi"
		else:
			datasource = "tpf"

		cbv_key = (datasource, cbv_area)
		cbv = self.cbvs.get(cbv_key)
		if cbv is None:
			logger.debug("Loading CBV for area %d into memory", cbv_area)
			cbv = CBV(self.data_folder, cbv_area, datasource)
			self.cbvs[cbv_key] = cbv

			if use_prior and cbv.priors is None:
				raise IOError('Trying to co-trend without a defined prior')

		# Update maximum number of components
		n_components = cbv.cbv.shape[1]

		logger.info('Fitting using number of components: %d', n_components)

		flux_filter, res, diagnostics = cbv.cotrend_single(lc, use_bic=True, use_prior=use_prior)
		#logger.debug('New variability', residual)

		# Corrected light curve in ppm
		lc_corr = 1e6*(lc.copy()/flux_filter - 1)

		# Defining FITS headers:
		no_cbvs_fitted = int((len(res)-1)/2)
		res = np.array([res,]).flatten()

		lc_corr.meta['additional_headers']['CBV_AREA'] = (cbv_area, 'CBV area of star')
		#lc_corr.meta['additional_headers']['CBV_MET'] = (diagnostics['method'], 'Method used to fit CBVs')
		lc_corr.meta['additional_headers']['CBV_BIC'] = (diagnostics['use_bic'], 'Was BIC used to select no of CBVs')
		lc_corr.meta['additional_headers']['CBV_PRI'] = (diagnostics['use_prior'], 'Was prior used')
		lc_corr.meta['additional_headers']['CBV_COMP'] = (no_cbvs_fitted, 'Number of fitted CBVs')
		lc_corr.meta['additional_headers']['CBV_MAX'] = (n_components, 'Number of possible CBVs to fit')
		lc_corr.meta['additional_headers']['CBV_C0'] = (res[-1], 'Fitted offset')

		for ii in range(no_cbvs_fitted):
			lc_corr.meta['additional_headers']['CBV_C%d' % (ii+1)] = (res[ii], 'CBV%d coefficient' % (ii+1))

		for jj in range(no_cbvs_fitted):
			lc_corr.meta['additional_headers']['CBVS_C%d' % (jj+1)] = (res[jj+no_cbvs_fitted], 'Spike-CBV%d coefficient' % (jj+1))

		# Set the status of the correction:
		status = STATUS.OK
		if len(res) < 4: # fitting and using only one CBV
			status = STATUS.WARNING

		if len(res) > 21: # fitting and using more than 10 CBVs
			status = STATUS.WARNING

		if self.plot:
			fig = plt.figure()
			ax1 = fig.add_subplot(211)
			ax1.plot(lc.time, lc.flux)
			ax1.plot(lc.time, flux_filter)
			if 'pc' in diagnostics:
				ax1.plot(lc.time, diagnostics['pc'], 'm--')
			ax1.set_xlabel('Time (BJD)')
			ax1.set_ylabel('Flux (counts)')
			ax1.set_xticks([])
			ax2 = fig.add_subplot(212)
			ax2.plot(lc_corr.time, lc_corr.flux)
			ax2.set_xlabel('Time (BJD)')
			ax2.set_ylabel('Relative flux (ppm)')
			plt.tight_layout()
			filename = 'lc_corr_TIC%d.png' %lc.targetid
			if not os.path.exists(os.path.join(self.plot_folder(lc))):
				os.makedirs(os.path.join(self.plot_folder(lc)))
			fig.savefig(os.path.join(self.plot_folder(lc), filename))
			plt.close(fig)

		return lc_corr, status
