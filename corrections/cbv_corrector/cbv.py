#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

import numpy as np
import os
import logging
import h5py
from astropy.io import fits
from astropy.time import Time
import datetime
from bottleneck import nansum, nanmedian
from scipy.optimize import minimize, fmin_powell
from scipy import stats
import functools
from ..utilities import loadPickle, fix_fits_table_headers
from ..quality import CorrectorQualityFlags
from .cbv_utilities import MAD_model, MAD_model2

#------------------------------------------------------------------------------
def cbv_snr_test(cbv_ini, threshold_snrtest=5.0):
	logger = logging.getLogger(__name__)

#	A_signal = MAD_model(cbv_ini, axis=0)
#	A_noise = MAD_model(np.diff(cbv_ini, axis=0), axis=0)

	A_signal = np.nanstd(cbv_ini, axis=0)
	A_noise = np.nanstd(np.diff(cbv_ini, axis=0), axis=0)

	snr = 10 * np.log10( A_signal**2 / A_noise**2 )
	logger.info("SNR threshold used %s", threshold_snrtest)
	logger.info("SNR (dB) of CBVs %s", snr)

	indx_lowsnr = (snr < threshold_snrtest)

	# Never throw away first CBV
	indx_lowsnr[0] = False

	return indx_lowsnr

#--------------------------------------------------------------------------------------------------
class CBV(object):
	"""
	Cotrending Basis Vector object.

	Attributes:
		cbv (numpy.array)
		cbv_s (numpy.array)
		priors
		inifit (numpy.array)

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	#--------------------------------------------------------------------------
	def __init__(self, data_folder, cbv_area, datasource):
		logger = logging.getLogger(__name__)

		if datasource not in ('ffi', 'tpf'):
			raise ValueError("Invalid datasource: '%s'" % datasource)

		self.data_folder = data_folder
		self.cbv_area = cbv_area
		self.datasource = datasource

		filepath = os.path.join(data_folder, 'cbv-%s-%d.hdf5' % (datasource, cbv_area))
		if not os.path.exists(filepath):
			raise FileNotFoundError("Could not find CBV file: %s" % filepath)

		self.inifit = None
		with h5py.File(filepath, 'r') as hdf:
			self.sector = int(hdf.attrs['sector'])
			self.cadence = int(hdf.attrs['cadence'])
			self.camera = int(hdf.attrs['camera'])
			self.ccd = int(hdf.attrs['ccd'])
			self.threshold_correlation = float(hdf.attrs['threshold_correlation'])
			self.threshold_variability = float(hdf.attrs['threshold_variability'])
			self.threshold_snrtest = float(hdf.attrs['threshold_snrtest'])
			self.threshold_entropy = float(hdf.attrs['threshold_entropy'])
			self.version = str(hdf.attrs['version'])

			self.time = np.asarray(hdf['time'])
			self.cadenceno = np.asarray(hdf['cadenceno'])

			self.cbv = np.asarray(hdf['cbv-single-scale'])
			self.cbv_s = np.asarray(hdf['cbv-spike'])

			if 'inifit' in hdf:
				self.inifit = np.asarray(hdf['inifit'])

		# Signal-to-Noise test (without actually removing any CBVs):
		indx_lowsnr = cbv_snr_test(self.cbv, self.threshold_snrtest)
		self.remove_cols(indx_lowsnr)

		self.priors = None
		priorpath = os.path.join(data_folder, 'D_%s-area%d.pkl' % (datasource, cbv_area))
		if os.path.exists(priorpath):
			self.priors = loadPickle(priorpath)
		else:
			logger.info('Path to prior distance file does not exist: %s', priorpath)

	#----------------------------------------------------------------------------------------------
	def remove_cols(self, indx_lowsnr):
		self.cbv = self.cbv[:, ~indx_lowsnr]
		self.cbv_s = self.cbv_s[:, ~indx_lowsnr]

	#----------------------------------------------------------------------------------------------
	def lsfit(self, flux, Ncbvs):
		"""
		Computes the least-squares solution to a linear matrix equation.
		"""
		idx = np.isfinite(self.cbv[:,0]) & np.isfinite(flux)
		A0 = np.column_stack((self.cbv[idx,:Ncbvs], self.cbv_s[idx,:Ncbvs]))

		X = np.column_stack((A0, np.ones(A0.shape[0])))
		F = flux[idx]

#		C = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(F)
		try:
			C = (np.linalg.pinv(X.T.dot(X)).dot(X.T)).dot(F)
		except (np.linalg.LinAlgError, ValueError):
			# Another (but slover) implementation
			C = np.linalg.lstsq(X, F, rcond=None)[0]

		return C

	#----------------------------------------------------------------------------------------------
	def lsfit_spike(self, flux, Ncbvs):
		"""
		Computes the least-squares solution to a linear matrix equation.
		"""
		idx = np.isfinite(self.cbv_s[:,0]) & np.isfinite(flux)

		A0 = self.cbv_s[idx,:Ncbvs]
		X = np.column_stack((A0, np.ones(A0.shape[0])))
		F = flux[idx]

		C = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(F)

		return C

	#----------------------------------------------------------------------------------------------
	def mdl(self, coeffs):
		coeffs = np.atleast_1d(coeffs)
		m = np.ones(self.cbv.shape[0], dtype='float64')
		Ncbvs = int((len(coeffs)-1)/2)

		for k in range(Ncbvs):
			m += (coeffs[k] * self.cbv[:, k]) + (coeffs[k+Ncbvs] * self.cbv_s[:, k])

		return m + coeffs[-1]

	#----------------------------------------------------------------------------------------------
	def mdl_spike(self, coeffs):
		coeffs = np.atleast_1d(coeffs)
		m = np.ones(self.cbv.shape[0], dtype='float64')
		for k in range(len(coeffs)-1):
			m += (coeffs[k] * self.cbv_s[:, k])

		return m + coeffs[-1]

	#----------------------------------------------------------------------------------------------
	def mdl_off(self, coeff, fitted, Ncbvs):
		fitted = np.atleast_1d(fitted)

		# Start with ones as the flux is median normalised
		m = np.ones(self.cbv.shape[0], dtype='float64')
		for k in range(Ncbvs):
			m += (fitted[k] * self.cbv[:, k]) + (fitted[k+Ncbvs] * self.cbv_s[:, k])
		return m + coeff

	#----------------------------------------------------------------------------------------------
	def mdl1d(self, coeff, ncbv):
		cbv_comb = np.column_stack((self.cbv, self.cbv_s))

		m = 1 + coeff * cbv_comb[:, ncbv]
		return m

	#----------------------------------------------------------------------------------------------
#	def _lhood(self, coeffs, flux, err):
#		return 0.5*nansum(((flux - self.mdl(coeffs))/err)**2)

	#----------------------------------------------------------------------------------------------
	def _lhood_off(self, coeffs, flux, fitted, Ncbvs):
		return 0.5*nansum((flux - self.mdl_off(coeffs, fitted, Ncbvs))**2)

	#----------------------------------------------------------------------------------------------
#	def _lhood_off_2(self, coeffs, flux, err, fitted):
#		return 0.5*nansum(((flux - self.mdl_off(coeffs, fitted))/err)**2) + 0.5*np.log(err**2)

	#----------------------------------------------------------------------------------------------
	def _lhood1d(self, coeff, flux, ncbv):
		return 0.5*nansum((flux - self.mdl1d(coeff, ncbv))**2)

	#----------------------------------------------------------------------------------------------
#	def _lhood1d_2(self, coeff, flux, err, ncbv):
#		return 0.5*nansum(((flux - self.mdl1d(coeff, ncbv))/err)**2) + 0.5*np.log(err**2)

	#----------------------------------------------------------------------------------------------
	def _posterior1d(self, coeff, flux, ncbv, pos, wscale, KDE):
		Post = self._lhood1d(coeff, flux, ncbv) - wscale*np.log(self._prior1d(coeff, KDE))
		return Post

	#----------------------------------------------------------------------------------------------
#	def _posterior1d_2(self, coeff, flux, err, ncbv, pos, wscale, KDE):
#		Post = self._lhood1d_2(coeff, flux, err, ncbv) + self._prior1d(coeff, wscale, KDE)
#		return Post

	#----------------------------------------------------------------------------------------------
	def _priorcurve(self, x, Ncbvs, N_neigh):
		"""
		Get "most likely" correction from peak in prior distributions
		of fitting coefficients

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
		res = np.zeros_like(self.cbv[:, 0], dtype='float64')
		opts = np.zeros(int(Ncbvs*2)) # entries for both CBV and CBV_S

		no_cbv_coeff = self.cbv.shape[1]

		tree = self.priors
		dist, ind = tree.query(np.array([x]), k=N_neigh+1)
		W = 1/dist[0][1::]**2

		for ncbv in range(Ncbvs):

			V = self.inifit[ind,1+ncbv][0][1::]
			VS = self.inifit[ind,1+ncbv + no_cbv_coeff][0][1::]

			KDE = stats.gaussian_kde(V, weights=W.flatten(), bw_method='scott')
			KDES = stats.gaussian_kde(VS, weights=W.flatten(), bw_method='scott')

			def kernel_opt(x):
				return -1*KDE.logpdf(x)
			opt = fmin_powell(kernel_opt, 0, disp=0)
			opts[ncbv] = opt

			def kernel_opts(x):
				return -1*KDES.logpdf(x)
			opt_s = fmin_powell(kernel_opts, 0, disp=0)
			opts[ncbv + Ncbvs] = opt_s

			res += (self.mdl1d(opt, ncbv) - 1) + (self.mdl1d(opt_s, ncbv + no_cbv_coeff) - 1)
		return res+1, opts

	#----------------------------------------------------------------------------------------------
	def _prior1d(self, c, KDE=None):
		if KDE is None:
			return 0
		else:
			return float(KDE(c))

	#----------------------------------------------------------------------------------------------
	def fitting_lh(self, flux, Ncbvs, err=None, start_guess=None):
		res = self.lsfit(flux, Ncbvs)
		res[-1] -= 1
		return res

	#----------------------------------------------------------------------------------------------
	def fitting_lh_spike(self, flux, Ncbvs, err=None):
		res = self.lsfit_spike(flux, Ncbvs)
		res[-1] -= 1
		return res

	#----------------------------------------------------------------------------------------------
	def fitting_pos_2(self, flux, Ncbvs, err, pos, wscale, N_neigh, start_guess=None):

		# Initial guesses for coefficients:
		if start_guess is not None:
			start_guess = np.append(start_guess, 0)
		else:
			start_guess = np.zeros(Ncbvs*2+1, dtype='float64')

		res = np.zeros(int(Ncbvs*2), dtype='float64')
		tree = self.priors
		dist, ind = tree.query(np.array([pos]), k=N_neigh+1)
		W = 1/dist[0][1::]**2

		# Define posterior function to be minimized:
		def logposterior(coeff):
			return self._lhood1d(coeff, flux, Ncbvs) - prior(coeff)

#
#
#			# TEST with independent spike fit
#			ff = pchip_interpolate(np.arange(len(flux))[np.isfinite(flux)], flux[np.isfinite(flux)], np.arange(len(flux)))
#			F = sig.medfilt(ff, 15)
#			flux2 = np.copy(flux)/F
#			res_s = self.fitting_lh_spike(flux2, Ncbvs)
#			res[Ncbvs:] = res_s[:-1]
#			spike_filt = self.mdl_spike(res_s)
#
##			plt.figure()
##			plt.plot(flux2)
##			plt.plot(spike_filt-0.5)
#
#
#			flux1 = F*(flux2 - spike_filt + 1)

#			plt.figure()
#			plt.plot(flux, 'b')
#			plt.plot(F-0.1, 'k')
#			plt.plot(flux1+0.1, 'r')
#			plt.plot(flux2 - spike_filt + 1.5, 'g')
#
#
#			plt.show()
#			sys.exit()

		for jj in range(Ncbvs):
			V = self.inifit[ind,1+jj][0][1::]
			KDE = stats.gaussian_kde(V, weights=W.flatten(), bw_method='scott')

#				VS = self.inifit[ind,1+jj + no_cbv_coeff][0][1::]
#				KDES = stats.gaussian_kde(VS, weights=W.flatten(), bw_method='scott')

#				res[jj] = minimize(self._posterior1d_2, coeffs0[jj], args=(flux, err, jj, pos, wscale, KDE), method='Powell').x

			res[jj] = minimize(self._posterior1d, start_guess[jj], args=(flux, jj, pos, wscale, KDE), method='Powell').x
			# Using KDE prior:
#				res[jj + Ncbvs] = minimize(self._posterior1d, coeffs0[jj + Ncbvs], args=(flux, jj + no_cbv_coeff, pos, wscale, KDES), method=method).x
			# Using flat prior
#				res[jj + Ncbvs] = minimize(self._posterior1d, coeffs0[jj + Ncbvs], args=(flux1, jj + no_cbv_coeff, pos, wscale, None), method=method).x

#			offset = minimize(self._lhood_off_2, coeffs0[-1], args=(flux, err, res), method='Powell').x
		offset = minimize(self._lhood_off, start_guess[-1], args=(flux, res, Ncbvs), method='Powell').x

		res = np.append(res, offset)
		return res

	#--------------------------------------------------------------------------
	def _fit(self, flux, err=None, Numcbvs=None, sigma_clip=4.0, maxiter=50, use_bic=True, prior=None, start_guess=None):
		"""

		"""

		logger = logging.getLogger(__name__)

		# If no number of CBVs are specified,
		# use the full set of CBVs:
		if Numcbvs is None:
			Numcbvs = self.cbv.shape[1]

		# Function to use for fitting.
		if prior:
			fitfunc = functools.partial(self.fitting_pos_2, err=err, prior=prior)
		else:
			fitfunc = functools.partial(self.fitting_lh, err=err)

		# Find the median flux to normalise light curve
		median_flux = nanmedian(flux)

		# Figure out how many CBVs to loop over:
		if use_bic:
			# Prepare variables for storing results:
			bic = np.array([])
			solutions = []

			# Test a range of CBVs from 1 to Numcbvs
			# Fitting at least 1 CBV and an offset
			Nstart = 1
		else:
			# Test only fit with Numcbvs
			Nstart = Numcbvs

		# Loop over the different number of CBVs to attempt:
		for Ncbvs in range(Nstart, Numcbvs+1):

			fluxi = np.copy(flux) / median_flux

			for iters in range(maxiter):
				# Do the fit:
				res = fitfunc(fluxi, Ncbvs, start_guess=start_guess)

				# Break if nothing changes
				if iters == 0:
					res0 = res
				else:
					d = np.sum(res0 - res)
					res0 = res
					if d == 0:
						break

				flux_filter = self.mdl(res)

				# Do robust sigma clipping:
				absdev = np.abs(fluxi - flux_filter)
				mad = MAD_model(absdev)
				indx = np.greater(absdev, sigma_clip*mad, where=np.isfinite(absdev))

				if np.any(indx):
					fluxi[indx] = np.nan
				else:
					break

			if use_bic:
				# Calculate the Bayesian Information Criterion (BIC) and store the solution:
				filt = self.mdl(res) * median_flux
				bic = np.append(bic, np.log(np.sum(np.isfinite(fluxi)))*len(res) + nansum( (flux - filt)**2 )) # not using errors
				solutions.append(res)

		if use_bic:
			logger.info('bic %s', bic)
			logger.info('iters %s', iters+1)

			# Use the solution which minimizes the BIC:
			indx = np.argmin(bic)
			logger.info('optimum number of CBVs %s', indx+1)
			res_final = solutions[indx]
			flux_filter = self.mdl(res_final) * median_flux

		else:
			res_final = res
			flux_filter = self.mdl(res_final) * median_flux

		return flux_filter, res_final

	#----------------------------------------------------------------------------------------------
	def fit(self, lc, use_bic=True, use_prior=False, cbvs=None, alpha=1.3, WS_lim=0.5, N_neigh=1000):
		"""
		Fit the CBV object to a lightcurve, and return the fitted cotrending-lightcurve
		and the fitting coefficients.

		Parameters:
			lc (:class:`LightCurve`): Lightcurve to be cotrended.
			use_bic (bool, optional): Use the Bayesian Information Criterion to find the
				optimal number of CBVs to fit. Default=True.
			use_prior (bool, optional):
			cbvs (int, optional): Number of CBVs to fit to lightcurve. If `use_bic=True`, this
				indicated the maximum number of CBVs to fit.

		Returns:
			- `numpy.array`: Fitted lightcurve with the same length as `lc`.
			- list: Coefficients for each CBV.
			- dict: Diagnostics information about the fitting.

		"""

		logger = logging.getLogger(__name__)

		# Remove bad data based on quality
		flag_good = CorrectorQualityFlags.filter(lc.quality)
		lc.flux[~flag_good] = np.nan

		# Diagnostics to return at the end about what was
		# actually used in the fitting:
		diagnostics = {
			'method': None,
			'use_bic': use_bic,
			'use_prior': use_prior
		}

		# Fit the CBV to the flux:
		if use_prior:
			"""
			Do fits including prior information from the initial fits -
			allow switching to a simple LSSQ fit depending on
			variability measures (not fully implemented yet!)
			"""

			# Position of target in multidimentional prior space:
			row = lc.meta['task']['pos_row']
			col = lc.meta['task']['pos_column']
			tmag = np.clip(lc.meta['task']['tmag'], 2, 20)
			pos = np.array([row, col, tmag])

			# Prior curve
			n_components = self.cbs.shape[1]
			pc0, opts = self._priorcurve(pos, n_components, N_neigh)
			pc = pc0 * lc.meta['task']['mean_flux']

			# Compute new variability measure
			idx = np.isfinite(lc.flux)
			polyfit = np.polyval(np.polyfit(lc.time[idx], lc.flux[idx], 3), lc.time)
			residual = MAD_model(lc.flux - pc)
			#residual_ratio = MAD_model(lc.flux-lc.meta['task']['mean_flux'])/residual
			#WS = np.min([1, residual_ratio])

			AA = 2
			GRAW = np.std((pc - polyfit) / MAD_model2(lc.flux - polyfit) - 1)
			GPR = 0 + (1 - (GRAW/AA)**2) * (GRAW < AA)

			beta1 = 1
			beta2 = 1
			VAR = np.nanstd(lc.flux - polyfit)
			WS = np.min([1, (VAR**beta1)*(GPR**beta2)])

			if WS > WS_lim:
				logger.debug('Fitting using LLSQ')
				flux_filter, res = self._fit(lc.flux, Numcbvs=5, use_bic=use_bic) # use smaller number of CBVs
				diagnostics['method'] = 'LS'
				diagnostics['use_prior'] = False
				diagnostics['use_bic'] = False

			else:
				logger.debug('Fitting using Priors')

				#dist, ind = self.priors.query(pos, k=N_neigh+1)
				#W = 1/dist[0][1:]**2
				#V = self.inifit[ind, 1+jj][0][1:]
				#KDE = stats.gaussian_kde(V, weights=W.flatten(), bw_method='scott')
				#def logprior(coeff):
				#	return wscale * KDE.logpdf(coeff)

				flux_filter, res = self._fit(lc.flux, err=residual, use_bic=use_bic, prior=logprior, start_guess=opts)

				diagnostics.update({
					'method': 'MAP',
					'residual': residual,
					'WS': WS,
					'pc': pc
				})

		else:
			"""
			Do "simple" LSSQ fits using BIC to decide on number of CBVs
			to include
			"""
			logger.debug('Fitting TIC %d using LLSQ', lc.targetid)
			flux_filter, res = self._fit(lc.flux, Numcbvs=cbvs, use_bic=use_bic)
			diagnostics['method'] = 'LS'

		return flux_filter, res, diagnostics

	#----------------------------------------------------------------------------------------------
	def save_to_fits(self, output_folder, datarel=5):
		"""
		Save CBVs to FITS file.

		Parameters:
			output_folder (str): Path to directory where FITS file should be saved.
			datarel (int): Data release number to add to file header.

		Returns:
			str: Path to the generated FITS file.

		Raises:
			FileNotFoundError: If `output_folder` is invalid.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		.. codeauthor:: Nicholas Saunders <nksaun@hawaii.edu>
		"""

		# Checks of input:
		if not os.path.isdir(output_folder):
			raise FileNotFoundError("Invalid output directory")

		# Store the date that the file is created
		now = datetime.datetime.now()

		# Timestamps of start and end of timeseries:
		tdel = self.cadence/86400
		tstart = self.time[0] - tdel/2
		tstop = self.time[-1] + tdel/2
		tstart_tm = Time(tstart, 2457000, format='jd', scale='tdb')
		tstop_tm = Time(tstop, 2457000, format='jd', scale='tdb')
		telapse = tstop - tstart

		# write fits header information
		hdr = fits.Header()
		hdr['EXTNAME'] = ('PRIMARY', 'extension name')
		hdr['ORIGIN'] = ('TASOC/Aarhus', 'institution responsible for creating this file')
		hdr['DATE'] = (now.strftime("%Y-%m-%d"), 'file creation date')
		hdr['TELESCOP'] = ('TESS', 'telescope')
		hdr['INSTRUME'] = ('TESS Photometer', 'detector type')
		hdr['SECTOR'] = (self.sector, 'Observing sector')
		hdr['DATA_REL'] = (datarel, 'data release version number')
		hdr['PROCVER'] = (self.version, 'Version of corrections pipeline')
		hdr['FILEVER'] = ('1.0', 'File format version')
		hdr['TSTART'] = (tstart, 'observation start time in TJD')
		hdr['TSTOP'] = (tstop, 'observation stop time in TJD')
		hdr['DATE-OBS'] = (tstart_tm.utc.isot, 'TSTART as UTC calendar date')
		hdr['DATE-END'] = (tstop_tm.utc.isot, 'TSTOP as UTC calendar date')
		hdr['CAMERA'] = (self.camera, 'CCD camera')
		hdr['CCD'] = (self.ccd, 'CCD chip')
		hdr['CBV_AREA'] = (self.cbv_area, 'CCD area')

		# Create primary hdu:
		phdu = fits.PrimaryHDU(header=hdr)

		# Create common table headers:
		table_header = fits.Header()
		table_header['TIMEREF'] = ('SOLARSYSTEM', 'barycentric correction applied to times')
		table_header['TIMESYS'] = ('TDB', 'time system is Barycentric Dynamical Time (TDB)')
		table_header['JDREFI'] = (2457000, 'integer part of BTJD reference date')
		table_header['JDREFF'] = (0.0, 'fraction of the day in BTJD reference date')
		table_header['TIMEUNIT'] = ('d', 'time unit for TIME, TSTART and TSTOP')
		table_header['TSTART'] = (tstart, 'observation start time in TJD')
		table_header['TSTOP'] = (tstop, 'observation stop time in TJD')
		table_header['DATE-OBS'] = (tstart_tm.utc.isot, 'TSTART as UTC calendar date')
		table_header['DATE-END'] = (tstop_tm.utc.isot, 'TSTOP as UTC calendar date')
		table_header['TELAPSE'] = (telapse, '[d] LC_STOP - LC_START')
		table_header['TIMEPIXR'] = (0.5, 'bin time beginning=0 middle=0.5 end=1')
		table_header['CAMERA'] = (self.camera, 'CCD camera')
		table_header['CCD'] = (self.ccd, 'CCD chip')
		table_header['CBV_AREA'] = (self.cbv_area, 'CCD area')

		# Settings used when generating the CBV:
		table_header['THR_COR'] = (self.threshold_correlation, 'Fraction of stars used for CBVs')
		table_header['THR_VAR'] = (self.threshold_variability, 'Threshold for variability rejection')
		table_header['THR_SNR'] = (self.threshold_snrtest, 'Threshold for SNR test')
		table_header['THR_ENT'] = (self.threshold_entropy, 'Threshold for entropy cleaning')

		# Columns that are common between all tables:
		col_time = fits.Column(name='TIME', format='D', unit='JD - 2457000, days', disp='D17.7', array=self.time)
		col_cadno = fits.Column(name='CADENCENO', format='J', disp='I10', array=self.cadenceno)

		# Single-scale CBVs:
		cols = [col_time, col_cadno]
		col_titles = {}
		# store all CBVs for each camera and chip
		for n in range(self.cbv.shape[1]):
			col_name = 'VECTOR_%d' % (n+1)
			col = fits.Column(name=col_name, format='E', disp='F8.5', array=self.cbv[:, n])
			cols.append(col)
			col_titles[col_name] = 'column title: co-trending basis vector %d' % (n+1)

		# append CBVs as hdu columns
		table_hdu1 = fits.BinTableHDU.from_columns(cols, header=table_header, name='CBV.SINGLE-SCALE.%d' % self.cbv_area)

		# Fix table headers:
		fix_fits_table_headers(table_hdu1, column_titles=col_titles)
		table_hdu1.header.comments['TTYPE1'] = 'column title: data time stamps'
		table_hdu1.header.comments['TUNIT1'] = 'column units: TESS modified Julian date (TJD)'
		table_hdu1.header.comments['TTYPE2'] = 'column title: unique cadence number'

		# Spike CBVs:
		cols = [col_time, col_cadno]
		col_titles = {}
		# store all CBVs for each camera and chip
		for n in range(self.cbv_s.shape[1]):
			col_name = 'VECTOR_%d' % (n+1)
			col = fits.Column(name=col_name, format='E', disp='F8.5', array=self.cbv_s[:, n])
			col_titles[col_name] = 'column title: co-trending basis vector %d' % (n+1)
			cols.append(col)

		# append CBVs as hdu columns
		table_hdu2 = fits.BinTableHDU.from_columns(cols, header=table_header, name='CBV.SPIKE.%d' % self.cbv_area)

		# Fix table headers:
		fix_fits_table_headers(table_hdu2, column_titles=col_titles)
		table_hdu2.header.comments['TTYPE1'] = 'column title: data time stamps'
		table_hdu2.header.comments['TUNIT1'] = 'column units: TESS modified Julian date (TJD)'
		table_hdu2.header.comments['TTYPE2'] = 'column title: unique cadence number'

		# Name of the
		fname = 'tess-s{sector:04d}-c{cadence:04d}-a{cbvarea:d}-v{datarel:d}-tasoc_cbv.fits'.format(
			sector=self.sector,
			cadence=self.cadence,
			cbvarea=self.cbv_area,
			datarel=datarel
		)
		filepath = os.path.join(output_folder, fname)

		# store as HDU list and write to fits file
		with fits.HDUList([phdu, table_hdu1, table_hdu2]) as hdul:
			hdul.writeto(filepath, overwrite=True, checksum=True)

		return filepath
