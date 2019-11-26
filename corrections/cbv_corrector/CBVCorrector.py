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
from astropy.io import fits
from bottleneck import nanmedian, nanstd
#from scipy.stats import norm
from ..plots import plt
from .. import BaseCorrector, STATUS
from .cbv import CBV

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

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('Co-trending star with TIC ID: %d', lc.targetid)

		# Load the CBV (and Prior) from memory and if it is not already loaded,
		# load it in from file and keep it in memory for next time:
		datasource = lc.meta['task']['datasource']
		cbv_area = lc.meta['task']['cbv_area']

		# Convert datasource into query-string for the database:
		# This will change once more different cadences (i.e. 20s) is defined
		if datasource != 'ffi':
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

		# Special treatment when not having a CBV created for this cadence:
		# Use the fit of the FFI as a "prior" and do a weighted fit, where
		# the FFI coefficients and high-cadence coefficients are combined.
		# TODO: This is still experimental!
		if use_prior and datasource == 'tpf':
			# Find the corrected lightcurve for the same target, observed in FFI:
			star_ffi = self.search_database(
				select='diagnostics_corr.lightcurve,corr_status',
				join='LEFT JOIN diagnostics_corr ON diagnostics_corr.priority=todolist.priority',
				search=['todolist.starid=%d' % lc.targetid, "datasource='ffi'"], #  "corr_status=1"
				limit=1)[0]
			print(star_ffi)

			if star_ffi['corr_status'] is None:
				raise ValueError("Star has not been processed with FFI data yet")

			#if star_ffi['corr_status'] == STATUS.WARNING:
			#	status = STATUS.WARNING

 			#if star_ffi['corr_status'] not in (STATUS.OK, STATUS.WARNING) or star_ffi['lightcurve'] is None:
			#	logger.warning()
			#	status = STATUS.WARNING
			#	alpha = 0

			# Load CBV coefficients fitted to the FFI data:
			ffi_path = os.path.join(self.input_folder, star_ffi['lightcurve'])
			with fits.open(ffi_path, mode='readonly', memmap=True) as hdu:
				Ncbvs_ffi = hdu[1].header.get('CBV_NUM', hdu[1].header['CBV_COMP'])
				coeff_ffi = np.array([hdu[1].header['CBV_C%d' % (k+1)] for k in range(Ncbvs_ffi)])
				offset_ffi = hdu[1].header['CBV_C0']

			median_flux = nanmedian(lc.flux)

			flux_filter_ffi = cbv.mdl(np.append(coeff_ffi, offset_ffi)) * median_flux

			# Fit the interpolated CBV to the high cadence data using LS:
			flux_filter_interp, coeff_interp, diagnostics = cbv.fit(lc, cbvs=Ncbvs_ffi, use_bic=False, use_prior=False)

			# Separate coefficients and
			offset = coeff_interp[-1]
			coeff_interp = coeff_interp[:Ncbvs_ffi]
			logger.debug("OFFSET = %f", offset)

			# Weighting function of FFI vs SC:
			# For alpha=0, the fit to high-cadence data is weighted up.
			# For alpha=1, the fit to FFI is weighted high.
			#alpha = 1 - max(1, lc.meta['task']['variability'])**-2
			#alpha = min(1, max( 2*lc.meta['task']['rms_hour']/lc.meta['task']['ptp'] - 1, 0))

			flux = lc.flux / lc.meta['task']['mean_flux']
			indx = np.isfinite(flux)
			p = np.polyfit(lc.time[indx], flux[indx], 3)
			fpol = flux - np.polyval(p, lc.time)
			ptp = nanmedian(np.abs(np.diff(fpol)))
			variability = nanstd(fpol) / ptp
			alpha = 1 - 1/max(1, variability)**2

			#alpha = norm.cdf(variability, 2.5, 0.5)

			logger.debug("ALPHA = %f", alpha)
			res = (1 - alpha)*coeff_interp + alpha*coeff_ffi

			# The constant offset is determined solely from the high cadence fit:
			res = np.append(res, offset)

			# Final weighted filter-lightcurve:
			flux_filter = cbv.mdl(res) * median_flux

			# Set that the
			diagnostics['method'] = 'WLS'

			if self.plot:
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.scatter(lc.time, lc.flux, c='k', alpha=0.3, s=2, label='Raw lightcurve')
				ax.plot(lc.time, flux_filter_ffi, label='FFI fit')
				ax.plot(lc.time, flux_filter_interp, label='SC fit')
				ax.plot(lc.time, flux_filter, label='Final fit')
				ax.set_xlabel('Time (TBJD)')
				ax.set_ylabel('Flux (%s)' % lc.flux_unit)
				ax.legend()
				ax.set_title(r'TIC %d - $\alpha = %.3f$' % (lc.targetid, alpha))
				filename = 'tess%011d-cbv_corr-tpf.png' % lc.targetid
				fig.savefig(os.path.join(self.plot_folder(lc), filename))
				plt.close(fig)

		else:
			# Run the standard fitting, using BIC, and optionally also using prior:
			flux_filter, res, diagnostics = cbv.fit(lc, use_bic=True, use_prior=use_prior)

		# Corrected light curve in ppm
		lc_corr = 1e6*(lc.copy()/flux_filter - 1)

		# Defining FITS headers:
		no_cbvs_fitted = int((len(res)-1)/2)
		res = np.array([res,]).flatten()

		lc_corr.meta['additional_headers']['CBV_AREA'] = (cbv_area, 'CBV area of star')
		lc_corr.meta['additional_headers']['CBV_MET'] = (diagnostics['method'], 'Method used to fit CBVs')
		lc_corr.meta['additional_headers']['CBV_BIC'] = (diagnostics['use_bic'], 'Was BIC used to select no of CBVs')
		lc_corr.meta['additional_headers']['CBV_PRI'] = (diagnostics['use_prior'], 'Was prior used')
		lc_corr.meta['additional_headers']['CBV_MAX'] = (n_components, 'Number of possible CBVs to fit')
		lc_corr.meta['additional_headers']['CBV_NUM'] = (no_cbvs_fitted, 'Number of fitted CBVs')
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
			ax1.set_xlabel('Time (TBJD)')
			ax1.set_ylabel('Flux (counts)')
			ax1.set_xticks([])
			ax2 = fig.add_subplot(212)
			ax2.plot(lc_corr.time, lc_corr.flux)
			ax2.set_xlabel('Time (TBJD)')
			ax2.set_ylabel('Relative flux (ppm)')
			plt.tight_layout()
			filename = 'tess%011d-cbv_corr.png' % lc.targetid
			fig.savefig(os.path.join(self.plot_folder(lc), filename))
			plt.close(fig)

		return lc_corr, status
