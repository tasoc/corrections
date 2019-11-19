#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Creation of Cotrending Basis Vectors.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os
import logging
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric, BallTree
from bottleneck import allnan, nanmedian
from scipy.interpolate import pchip_interpolate
from scipy.signal import savgol_filter, find_peaks
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from tqdm import tqdm
from ..plots import plt
from .. import BaseCorrector
from ..utilities import savePickle, mad_to_sigma
from ..quality import CorrectorQualityFlags, TESSQualityFlags
from .cbv import CBV, cbv_snr_test
from .cbv_utilities import MAD_model2, compute_scores, AlmightyCorrcoefEinsumOptimized, compute_entropy

#--------------------------------------------------------------------------------------------------
def clean_cbv(matrix, n_components, entropy_limit=-1.5, targ_limit=50):
	"""
	Entropy-cleaning of lightcurve matrix using the SVD U-matrix.
	"""
	logger = logging.getLogger(__name__)

	# Calculate the principle components:
	logger.info("Doing Principle Component Analysis...")
	pca = PCA(n_components)
	U, _, _ = pca._fit(matrix)

	Ent = compute_entropy(U)
	logger.info('Entropy start: %f', Ent)

	targets_removed = 0
	components = np.arange(n_components)

	with np.errstate(invalid='ignore'):
		while np.any(Ent < entropy_limit):
			com = components[(Ent < entropy_limit)][0]

			# Remove highest relative weight target
			m = nanmedian(U[:, com])
			s = mad_to_sigma*nanmedian(np.abs(U[:, com] - m))
			dev = np.abs(U[:, com] - m) / s

			idx0 = np.argmax(dev)

			star_no = np.ones(U.shape[0], dtype=bool)
			star_no[idx0] = False

			matrix = matrix[star_no, :]
			U, _, _ = pca._fit(matrix)

			targets_removed += 1
			if targets_removed > targ_limit:
				break

			Ent = compute_entropy(U)

	logger.info('Entropy end: %f', Ent)
	logger.info('Targets removed: %d', targets_removed)
	return matrix

#--------------------------------------------------------------------------------------------------
def lc_matrix_calc(Nstars, mat0):
	logger = logging.getLogger(__name__)

	logger.info("Calculating correlations...")

	indx_nancol = allnan(mat0, axis=0)
	mat1 = mat0[:, ~indx_nancol]

	mat1[np.isnan(mat1)] = 0
	correlations = np.abs(AlmightyCorrcoefEinsumOptimized(mat1.T, mat1.T))
	np.fill_diagonal(correlations, np.nan)

	return correlations

#--------------------------------------------------------------------------------------------------
class CBVCreator(BaseCorrector):

	def __init__(self, *args, datasource='ffi', Numcbvs='all', ncomponents=None,
		threshold_correlation=0.5, threshold_snrtest=5, threshold_variability=1.3, **kwargs):
		"""
		Initialise the corrector

		The CBV init inherets init and functionality of :py:class:`BaseCorrector`

		The CBV init has three import steps run in addition to defining
		various high-level variables:
			1: The CBVs for the specific todo list are computed using the :py:func:`CBVCorrector.compute_cbv` function.
			2: An initial fitting are performed for all targets using linear least squares using the :py:func:`CBVCorrector.cotrend_ini` function.
			This is done to obtain fitting coefficients for the CBVs that will be used to form priors for the final fit.
			3: Prior from step 2 are constructed using the :py:func:`CBVCorrector.compute_weight_interpolations` function. This
			function saves interpolation functions for each of the CBV coefficient priors.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		self.Numcbvs = Numcbvs
		self.threshold_snrtest = threshold_snrtest
		self.threshold_correlation = threshold_correlation
		self.threshold_variability = threshold_variability
		self.ncomponents = ncomponents
		self.datasource = datasource

	#--------------------------------------------------------------------------
	def lc_matrix(self, cbv_area):
		"""
		Computes correlation matrix for light curves in a given cbv-area.

		Only targets with a variability below a user-defined threshold are included
		in the calculation.

		Returns matrix of the *self.threshold_correlation*% most correlated light curves; the threshold is defined in the class init function.

		Parameters:
            cbv_area: the cbv area to calculate matrix for
				additional parameters are contained in *self* and defined in the init function.

        Returns:
            mat: matrix of *self.threshold_correlation*% most correlated light curves, to be used in CBV calculation
			varis: variances of light curves in "mat"

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		#----------------------------------------------------------------------
		# CALCULATE LIGHT CURVE CORRELATIONS
		#----------------------------------------------------------------------

		logger.info("We are running CBV_AREA=%d" % cbv_area)

		tmpfile = os.path.join(self.data_folder, 'mat-%s-%d.npz' % (self.datasource, cbv_area))
		if logger.isEnabledFor(logging.DEBUG) and os.path.exists(tmpfile):
			logger.info("Loading existing file...")
			data = np.load(tmpfile)
			mat = data['mat']
			varis = data['varis']

		else:
			# Convert datasource into query-string for the database:
			# This will change once more different cadences (i.e. 20s) is defined
			if self.datasource == 'ffi':
				search_cadence = "datasource='ffi'"
			else:
				search_cadence = "datasource!='ffi'"

			# Find the median of the variabilities:
			variability = np.array([float(row['variability']) for row in self.search_database(search=[search_cadence, 'cbv_area=%d' % cbv_area], select='variability')], dtype='float64')
			median_variability = nanmedian(variability)

			# Plot the distribution of variability for all stars:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.hist(variability/median_variability, bins=np.logspace(np.log10(0.1), np.log10(1000.0), 50))
			ax.axvline(self.threshold_variability, color='r')
			ax.set_xscale('log')
			ax.set_xlabel('Variability')
			fig.savefig(os.path.join(self.data_folder, 'variability-%s-area%d.png' % (self.datasource, cbv_area)))
			plt.close(fig)

			# Get the list of star that we are going to load in the lightcurves for:
			stars = self.search_database(search=[search_cadence, 'cbv_area=%d' % cbv_area, 'variability < %f' % (self.threshold_variability*median_variability)])

			# Number of stars returned:
			Nstars = len(stars)

			# Load the very first timeseries only to find the number of timestamps.
			lc = self.load_lightcurve(stars[0])
			Ntimes = len(lc.time)

			logger.info("Matrix size: %d x %d", Nstars, Ntimes)

			# Make the matrix that will hold all the lightcurves:
			logger.info("Loading in lightcurves...")
			mat0 = np.empty((Nstars, Ntimes), dtype='float64')
			mat0.fill(np.nan)
			varis0 = np.empty(Nstars, dtype='float64')

			# Loop over stars
			for k, star in tqdm(enumerate(stars), total=Nstars, disable=not logger.isEnabledFor(logging.INFO)):

				# Load lightkurve object
				lc = self.load_lightcurve(star)

				# Remove bad data based on quality
				flag_good = TESSQualityFlags.filter(lc.pixel_quality, TESSQualityFlags.CBV_BITMASK) & CorrectorQualityFlags.filter(lc.quality, CorrectorQualityFlags.CBV_BITMASK)
				lc.flux[~flag_good] = np.nan

				# Normalize the data and store it in the rows of the matrix:
				mat0[k, :] = lc.flux / star['mean_flux'] - 1.0

				# Store the standard deviations of each lightcurve:
				varis0[k] = np.NaN if star['variance'] is None else star['variance']

			# Only start calculating correlations if we are actually filtering using them:
			if self.threshold_correlation < 1.0:
				# Calculate the correlation matrix between all lightcurves:
				correlations = lc_matrix_calc(Nstars, mat0)

				# If running in DEBUG mode, save the correlations matrix to file:
				if logger.isEnabledFor(logging.DEBUG):
					file_correlations = os.path.join(self.data_folder, 'correlations-%s-%d.npy' % (self.datasource, cbv_area))
					np.save(file_correlations, correlations)

				# Find the median absolute correlation between each lightcurve and all other lightcurves:
				c = nanmedian(correlations, axis=0)

				# Indicies that would sort the lightcurves by correlations in descending order:
				indx = np.argsort(c)[::-1]
				indx = indx[:int(self.threshold_correlation*Nstars)]
				#TODO: remove based on threshold value? rather than just % of stars

				# Only keep the top "threshold_correlation"% of the lightcurves that are most correlated:
				mat = mat0[indx, :]
				varis = varis0[indx]

				# Clean up a bit:
				del correlations, c, indx

			# Save something for debugging:
			if logger.isEnabledFor(logging.DEBUG):
				np.savez(tmpfile, mat=mat, varis=varis)

		return mat, varis

	#----------------------------------------------------------------------------------------------
	def lc_matrix_clean(self, cbv_area):
		"""
		Performs gap-filling of light curves returned by :py:func:`CBVCorrector.lc_matrix`, and
		removes time stamps where all flux values are nan

		Parameters:
			cbv_area: the cbv area to calculate light curve matrix for

		Returns:
			mat: matrix from :py:func:`CBVCorrector.lc_matrix` that has been gap-filled and with nans removed, to be used in CBV calculation
			varis: variances of light curves in "mat"
			indx_nancol: the indices for the timestamps with nans in all light curves
			Ntimes: Number of timestamps in light curves contained in mat before removing nans

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		logger.info('Running matrix clean')
		tmpfile = os.path.join(self.data_folder, 'mat-%s-%d_clean.npz' % (self.datasource, cbv_area))
		if logger.isEnabledFor(logging.DEBUG) and os.path.exists(tmpfile):
			logger.info("Loading existing file...")
			data = np.load(tmpfile)
			mat = data['mat']
			varis = data['varis']

			Ntimes = data['Ntimes']
			indx_nancol = data['indx_nancol']

		else:
			# Compute light curve correlation matrix
			mat0, varis = self.lc_matrix(cbv_area)

			# Print the final shape of the matrix:
			logger.info("Matrix size: %d x %d" % mat0.shape)

			# Find columns where all stars have NaNs and remove them:
			indx_nancol = allnan(mat0, axis=0)
			Ntimes = mat0.shape[1]
			mat = mat0[:, ~indx_nancol]
			cadenceno = np.arange(mat.shape[1])

			logger.info("Gap-filling lightcurves...")
			for k in tqdm(range(mat.shape[0]), total=mat.shape[0], disable=not logger.isEnabledFor(logging.INFO)):

				mat[k, :] /= varis[k]
				# Fill out missing values by interpolating the lightcurve:
				indx = np.isfinite(mat[k, :])
				mat[k, ~indx] = pchip_interpolate(cadenceno[indx], mat[k, indx], cadenceno[~indx])

			# Save something for debugging:
			if logger.isEnabledFor(logging.DEBUG):
				np.savez(tmpfile, mat=mat, varis=varis, indx_nancol=indx_nancol, Ntimes=Ntimes)

		return mat, varis, indx_nancol, Ntimes

	#----------------------------------------------------------------------------------------------
	def compute_cbvs(self, cbv_area, ent_limit=-1.5, targ_limit=150):
		"""
		Main function for computing CBVs.

		The steps taken in the function are:
			1: run :py:func:`CBVCorrector.lc_matrix_clean` to obtain matrix with gap-filled, nan-removed light curves
			for the most correlated stars in a given cbv-area
			2: compute principal components and remove significant single-star contributers based on entropy
			3: reun SNR test on CBVs, and only retain CBVs that pass the test
			4: save CBVs and make diagnostics plots

		Parameters:
			*self*: all parameters defined in class init

		Returns:
			Saves CBVs per cbv-area in ".npy" files

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('running CBV')
		logger.info('------------------------------------')

		if os.path.exists(os.path.join(self.data_folder, 'cbv_ini-%s-%d.npy' % (self.datasource, cbv_area))):
			logger.info('CBV for area %d already calculated' % cbv_area)
			return

		logger.info('Computing CBV for %s area %d' % (self.datasource, cbv_area))

		# Extract or compute cleaned and gapfilled light curve matrix
		mat0, _, indx_nancol, Ntimes = self.lc_matrix_clean(cbv_area)

		# Calculate initial CBVs
		logger.info('Computing %d CBVs', self.ncomponents)
		pca0 = PCA(self.ncomponents)
		U0, _, _ = pca0._fit(mat0)

		cbv0 = np.empty((Ntimes, self.ncomponents), dtype='float64')
		cbv0.fill(np.nan)
		cbv0[~indx_nancol, :] = np.transpose(pca0.components_)

		logger.info('Cleaning matrix for CBV - remove single dominant contributions')

		# Clean away targets that contribute significantly as a single star to a given CBV (based on entropy)
		mat = clean_cbv(mat0, self.ncomponents, ent_limit, targ_limit)

		# Calculate the principle components of cleaned matrix
		logger.info("Doing Principle Component Analysis...")
		pca = PCA(self.ncomponents)
		U, _, _ = pca._fit(mat)

		cbv = np.empty((Ntimes, self.ncomponents), dtype='float64')
		cbv.fill(np.nan)
		cbv[~indx_nancol, :] = np.transpose(pca.components_)

		# Signal-to-Noise test (here only for plotting)
		#indx_lowsnr = cbv_snr_test(cbv, self.threshold_snrtest)

		# Save the CBV to file:
		np.save(os.path.join(self.data_folder, 'cbv_ini-%s-%d.npy' % (self.datasource, cbv_area)), cbv)

		####################### PLOTS #################################
		# Plot the "effectiveness" of each CBV:
		max_components = 20
		n_cbv_components = np.arange(max_components, dtype=int)
		pca_scores = compute_scores(mat, n_cbv_components)

		fig0 = plt.figure(figsize=(12, 8))
		ax0 = fig0.add_subplot(121)
		ax0.plot(n_cbv_components, pca_scores, 'b', label='PCA scores')
		ax0.set_xlabel('nb of components')
		ax0.set_ylabel('CV scores')
		ax0.legend(loc='lower right')
		ax02 = fig0.add_subplot(122)
		ax02.plot(np.arange(1, cbv0.shape[1]+1), pca.explained_variance_ratio_, '.-')
		ax02.axvline(x=cbv.shape[1]+0.5, ls='--', color='k')
		ax02.set_xlabel('CBV number')
		ax02.set_ylabel('Variance explained ratio')
		fig0.savefig(os.path.join(self.data_folder, 'cbv-perf-%s-area%d.png' % (self.datasource, cbv_area)))
		plt.close(fig0)

		# Plot all the CBVs:
		fig, axes = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig2, axes2 = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)
		fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)

		for k, ax in enumerate(axes.flatten()):
			if k < cbv0.shape[1]:
				ax.plot(cbv0[:, k]+0.1, 'r-')
				#if not indx_lowsnr is None:
				#	if indx_lowsnr[k]:
				#		col = 'c'
				#	else:
				#		col = 'k'
				#else:
				#	col = 'k'
				ax.plot(cbv[:, k], ls='-', color='k')
				ax.set_title('Basis Vector %d' % (k+1))

		for k, ax in enumerate(axes2.flatten()):
			if k < U0.shape[1]:
				ax.plot(-np.abs(U0[:, k]), 'r-')
				ax.plot(np.abs(U[:, k]), 'k-')
				ax.set_title('Basis Vector %d' % (k+1))

		fig.savefig(os.path.join(self.data_folder, 'cbvs_ini-%s-area%d.png' % (self.datasource, cbv_area)))
		fig2.savefig(os.path.join(self.data_folder, 'U_cbvs-%s-area%d.png' % (self.datasource, cbv_area)))
		plt.close(fig)
		plt.close(fig2)

	#--------------------------------------------------------------------------
	def spike_sep(self, cbv_area):
		"""
		Function that separates CBVs into a "slow" and a "spiky" component

		This is done by filtering the deta and identifying outlier
		with a peak-finding algorithm

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('running CBV spike separation')
		logger.info('------------------------------------')

		if os.path.exists(os.path.join(self.data_folder, 'cbv-%s-%d.npy' % (self.datasource, cbv_area))) \
			and os.path.exists(os.path.join(self.data_folder, 'cbv-s-%s-%d.npy' % (self.datasource, cbv_area))):
			logger.info('Separated CBVs for %s area %d already calculated' % (self.datasource, cbv_area))
			return

		logger.info('Computing CBV spike separation for %s area %d' % (self.datasource, cbv_area))

		# Load initial CBV from "compute_CBV"
		filepath = os.path.join(self.data_folder, 'cbv_ini-%s-%d.npy' % (self.datasource, cbv_area))
		cbv = np.load(filepath)

		# padding window, just needs to be bigger than savgol filtering window
		wmir = 50

		# Initiate arrays for cleaned and spike CBVs
		cbv_new = np.zeros_like(cbv)
		cbv_spike = np.zeros_like(cbv)

		# Iterate over basis vectors
		for j in range(cbv.shape[1]):

			# Pad ends for better peak detection at boundaries of data
			data0 = cbv[:, j]
			data0 = np.append(np.flip(data0[0:wmir])[0:-1], data0)
			data0 = np.append(data0, np.flip(data0[-wmir::])[1::])

			xs = np.arange(0, len(data0))

			data = data0.copy()

			# Iterate peak detection
			for i in range(5):

				# For savgol filter data must be continuous
				data2 = pchip_interpolate(xs[np.isfinite(data)], data[np.isfinite(data)], xs)

				# run low pass filter, set window width
				w = 31 - 2*i
				if w%2 == 0:
					w += 1

				# Smooth, filtered version of data, to use to identify "outliers", i.e., spikes
				y = savgol_filter(data2, w, 2, mode='constant')
				y2 = data2 - y

				# Run peak detection
				sigma = 1.4826 * nanmedian(np.abs(y2))
				peaks, properties = find_peaks(np.abs(y2), prominence=(3*sigma, None), wlen=500)

				data[peaks] = np.nan


			# Interpolate CBVs where spike has been identified
			data = pchip_interpolate(xs[np.isfinite(data)], data[np.isfinite(data)], xs)

			# Remove padded ends and store in CBV matrices
			# Spike signal is difference between original data and data with masked spikes
			S = (data0[wmir-1:-wmir+1] - data[wmir-1:-wmir+1])
			S[np.isnan(S)] = 0

			cbv_spike[:, j] = S
			cbv_new[:, j] = data[wmir-1:-wmir+1]

		# Save files
		np.save(os.path.join(self.data_folder, 'cbv-%s-%d.npy' % (self.datasource, cbv_area)), cbv_new)
		np.save(os.path.join(self.data_folder, 'cbv-s-%s-%d.npy' % (self.datasource, cbv_area)), cbv_spike)

		# Signal-to-Noise test (here only for plotting)
		indx_lowsnr = cbv_snr_test(cbv_new, self.threshold_snrtest)

		# Plot all the CBVs:
		fig, axes = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig2, axes2 = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)
		fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)

		for k, ax in enumerate(axes.flatten()):
			if k < cbv_new.shape[1]:
				if not indx_lowsnr is None:
					if indx_lowsnr[k]:
						col = 'c'
					else:
						col = 'k'
				else:
					col = 'k'
				ax.plot(cbv_new[:, k], ls='-', color=col)
				ax.set_title('Basis Vector %d' % (k+1))

		for k, ax in enumerate(axes2.flatten()):
			if k < cbv_spike.shape[1]:
				if not indx_lowsnr is None:
					if indx_lowsnr[k]:
						col = 'c'
					else:
						col = 'k'
				else:
					col = 'k'
				ax.plot(cbv_spike[:, k], ls='-', color=col)
				ax.set_title('Spike Basis Vector %d' % (k+1))

		fig.savefig(os.path.join(self.data_folder, 'cbvs-%s-area%d.png' % (self.datasource, cbv_area)))
		fig2.savefig(os.path.join(self.data_folder, 'spike-cbvs-%s-area%d.png' % (self.datasource, cbv_area)))
		plt.close(fig)
		plt.close(fig2)

	#--------------------------------------------------------------------------
	def cotrend_ini(self, cbv_area, do_ini_plots=False):
		"""
		Function for running the initial co-trending to obtain CBV coefficients for the construction of priors.

		The steps taken in the function are:
			1: for each cbv-area load calculated CBVs
			2: co-trend all light curves in area using fit of all CBVs using linear least squares
			3: save CBV coefficients

		Parameters:
			*self*: all parameters defined in class init

		Returns:
			Saves CBV coefficients per cbv-area in ".npz" files
			adds loaded CBVs to *self*

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		#------------------------------------------------------------------
		# CORRECTING STARS
		#------------------------------------------------------------------

		logger.info("--------------------------------------------------------------")
		if os.path.exists(os.path.join(self.data_folder, 'mat-%s-%d_free_weights.npz' % (self.datasource, cbv_area))):
			logger.info("Initial co-trending for light curves in %s CBV area%d already done" % (self.datasource, cbv_area))
			return
		else:
			logger.info("Initial co-trending for light curves in %s CBV area%d" % (self.datasource, cbv_area))

		# Convert datasource into query-string for the database:
		# This will change once more different cadences (i.e. 20s) is defined
		if self.datasource == 'ffi':
			search_cadence = "datasource='ffi'"
		else:
			search_cadence = "datasource!='ffi'"

		# Load stars from database
		stars = self.search_database(search=[search_cadence, 'cbv_area=%d' % cbv_area])

		# Load the cbv from file:
		cbv = CBV(self.data_folder, cbv_area, self.datasource, self.threshold_snrtest)

		# Signal-to-Noise test (without actually removing any CBVs):
		#indx_lowsnr = cbv_snr_test(cbv.cbv, self.threshold_snrtest)
		#cbv.remove_cols(indx_lowsnr)

		# Update maximum number of components
		n_components0 = cbv.cbv.shape[1]
		logger.info('New max number of components: %d', n_components0)

		if self.Numcbvs == 'all':
			n_components = n_components0
		else:
			n_components = np.min([self.Numcbvs, n_components0])

		logger.info('Fitting using number of components: %d', n_components)
		# initialize results array, including TIC, CBV components, and an residual offset
		Nres = int(2*n_components+2)
		results = np.zeros([len(stars), Nres])

		# Loop through stars
		for kk, star in tqdm(enumerate(stars), total=len(stars), disable=not logger.isEnabledFor(logging.INFO)):

			lc = self.load_lightcurve(star)

			logger.debug("Correcting star %d", lc.targetid)

			flux_filter, res, _ = cbv.fit(lc, use_bic=False, use_prior=False)

			# TODO: compute diagnostics requiring the light curve
			# SAVE TO DIAGNOSTICS FILE::
			#wn_ratio = GOC_wn(flux, flux-flux_filter)

			res = np.array([res,]).flatten()
			results[kk, 0] = lc.targetid
			results[kk, 1:len(res)+1] = res

			if do_ini_plots:
				lc_corr = (lc.flux/flux_filter-1)*1e6

				fig = plt.figure()
				ax1 = fig.add_subplot(211)
				ax1.plot(lc.time, lc.flux)
				ax1.plot(lc.time, flux_filter)
				ax1.set_xlabel('Time (BJD)')
				ax1.set_ylabel('Flux (counts)')
				ax1.set_xticks([])
				ax2 = fig.add_subplot(212)
				ax2.plot(lc.time, lc_corr)
				ax2.set_xlabel('Time (BJD)')
				ax2.set_ylabel('Relative flux (ppm)')
				filename = 'lc_corr_ini_TIC%d.png' %lc.targetid

				if not os.path.exists(os.path.join(self.plot_folder(lc))):
					os.makedirs(os.path.join(self.plot_folder(lc)))
				fig.savefig(os.path.join(self.plot_folder(lc), filename))
				plt.close(fig)

		# Save weights for priors if it is an initial run
		np.savez(os.path.join(self.data_folder, 'mat-%s-%d_free_weights.npz' % (self.datasource, cbv_area)), res=results)

		# Plot CBV weights
		fig = plt.figure(figsize=(15, 15))
		ax = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		for kk in range(1, int(2*n_components+1)):

			if kk > n_components:
				LS = '--'
			else:
				LS = '-'
			idx = np.nonzero(results[:, kk])
			r = results[idx, kk]
			idx2 = (r > np.percentile(r, 10)) & (r < np.percentile(r, 90))
			kde = KDE(r[idx2])
			kde.fit(gridsize=5000)
			err = nanmedian(np.abs(r[idx2] - nanmedian(r[idx2]))) * 1e5

			if kk > n_components:
				ax3.plot(kde.support*1e5, kde.density/np.max(kde.density), label='CBV ' + str(kk), ls=LS)
				ax4.errorbar(kk, kde.support[np.argmax(kde.density)]*1e5, yerr=err, marker='o', color='k')
			else:
				ax.plot(kde.support*1e5, kde.density/np.max(kde.density), label='CBV ' + str(kk), ls=LS)
				ax2.errorbar(kk, kde.support[np.argmax(kde.density)]*1e5, yerr=err, marker='o', color='k')

		ax.set_xlabel('CBV weight')
		ax2.set_ylabel('CBV weight')
		ax2.set_xlabel('CBV')
		ax.legend()
		fig.savefig(os.path.join(self.data_folder, 'weights-sector-%s-%d.png' % (self.datasource, cbv_area)))
		plt.close(fig)

	#--------------------------------------------------------------------------
	def compute_distance_map(self, cbv_area):
		"""
		3D distance map for weighting initial-fit coefficients
		into a prior

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
		logger = logging.getLogger(__name__)
		logger.info("--------------------------------------------------------------")

		inipath = os.path.join(self.data_folder, 'mat-%s-%d_free_weights.npz' % (self.datasource, cbv_area))
		if not os.path.exists(inipath):
			raise IOError('Trying to make priors without initial corrections')

		results = np.load(inipath)['res']
		n_stars = results.shape[0]

		# Convert datasource into query-string for the database:
		# This will change once more different cadences (i.e. 20s) is defined
		if self.datasource == 'ffi':
			search_cadence = "datasource='ffi'"
		else:
			search_cadence = "datasource!='ffi'"

		# Load in positions and tmags, in same order as results are saved from ini_fit
		pos_mag0 = np.zeros([n_stars, 3])
		for jj, star in enumerate(results[:, 0]):
			star_single = self.search_database(search=[search_cadence, 'cbv_area=%d' % cbv_area, 'todolist.starid=%d' % int(star)])
			pos_mag0[jj, 0] = star_single[0]['pos_row']
			pos_mag0[jj, 1] = star_single[0]['pos_column']
			pos_mag0[jj, 2] = np.clip(star_single[0]['tmag'], 2, 20)

		# Relative importance of dimensions
		#S = np.array([1, 1, 2])
		S = np.array([MAD_model2(pos_mag0[:, 0]), MAD_model2(pos_mag0[:, 1]), 0.5*MAD_model2(pos_mag0[:, 2])])
		LL = np.diag(S)

		#pos_mag0[:, 0] /= np.std(pos_mag0[:, 0])
		#pos_mag0[:, 1] /= np.std(pos_mag0[:, 1])
		#pos_mag0[:, 2] /= np.std(pos_mag0[:, 2])

		# Construct and save distance tree
		dist = DistanceMetric.get_metric('mahalanobis', VI=LL)
		tree = BallTree(pos_mag0, metric=dist)

		savePickle(os.path.join(self.data_folder, 'D_%s-area%d.pkl' % (self.datasource, cbv_area)), tree)
