#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
from ..plots import plt, matplotlib
import os
from sklearn.decomposition import PCA
from bottleneck import allnan, nanmedian
from scipy.interpolate import pchip_interpolate
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from tqdm import tqdm
from .cbv_main import CBV, cbv_snr_test, clean_cbv, lc_matrix_calc
from .cbv_util import compute_scores, ndim_med_filt, reduce_mode, reduce_std
from .. import BaseCorrector, STATUS
from ..utilities import savePickle, loadPickle
import matplotlib.colors as colors
#from mpl_toolkits.mplot3d import Axes3D
import logging
from scipy.spatial.distance import squareform, pdist
from scipy.signal import savgol_filter, find_peaks
import sys
from sklearn.neighbors import DistanceMetric, BallTree
from ..quality import CorrectorQualityFlags, TESSQualityFlags
from .cbv_util import MAD_model, MAD_model2, MAD_scatter

#------------------------------------------------------------------------------


class CBVCorrector(BaseCorrector):
	"""
	The CBV (Co-trending Basis Vectors) correction method for the TASOC
	photometry pipeline

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""

	def __init__(self, *args, Numcbvs='all', ncomponents=None, simple_fit=True, WS_lim=0.8, alpha=1.3, N_neigh=1000, method='Powell', use_bic=True, \
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
		self.use_bic = use_bic
		self.method = method
		self.threshold_snrtest = threshold_snrtest
		self.threshold_correlation = threshold_correlation
		self.threshold_variability = threshold_variability
		self.ncomponents = ncomponents
		self.alpha = alpha
		self.WS_lim = WS_lim
		self.N_neigh = N_neigh
		self.simple_fit = simple_fit
		# Dictionary that will hold CBV objects:
		self.cbvs = {}


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

		tmpfile = os.path.join(self.data_folder, 'mat-%d.npz' %cbv_area)
		if logger.isEnabledFor(logging.DEBUG) and os.path.exists(tmpfile):
			logger.info("Loading existing file...")
			data = np.load(tmpfile)
			mat = data['mat']
			varis = data['varis']

		else:
			# Find the median of the variabilities:
			variability = np.array([float(row['variability']) for row in self.search_database(search=['datasource="ffi"', 'cbv_area=%i' %cbv_area], select='variability')], dtype='float64')
			median_variability = nanmedian(variability)

			# Plot the distribution of variability for all stars:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.hist(variability/median_variability, bins=np.logspace(np.log10(0.1), np.log10(1000.0), 50))
			ax.axvline(self.threshold_variability, color='r')
			ax.set_xscale('log')
			ax.set_xlabel('Variability')
			fig.savefig(os.path.join(self.data_folder, 'variability-area%d.png' % cbv_area))
			plt.close(fig)

			# Get the list of star that we are going to load in the lightcurves for:
			stars = self.search_database(search=['datasource="ffi"', 'cbv_area=%i' %cbv_area, 'variability < %f' %(self.threshold_variability*median_variability)])

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
					file_correlations = os.path.join(self.data_folder, 'correlations-%d.npy' % cbv_area)
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

	#--------------------------------------------------------------------------
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

		logger=logging.getLogger(__name__)

		logger.info('Running matrix clean')
		tmpfile = os.path.join(self.data_folder, 'mat-%d_clean.npz' % cbv_area)
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

	#--------------------------------------------------------------------------
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

		if os.path.exists(os.path.join(self.data_folder, 'cbv_ini-%d.npy' % cbv_area)):
			logger.info('CBV for area %d already calculated' % cbv_area)
			return

		else:
			logger.info('Computing CBV for area %d' % cbv_area)

			# Extract or compute cleaned and gapfilled light curve matrix
			mat0, _, indx_nancol, Ntimes = self.lc_matrix_clean(cbv_area)

			# Calculate initial CBVs
			logger.info('Computing %d CBVs' %self.ncomponents)
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

#			# Signal-to-Noise test (here only for plotting)
#			indx_lowsnr = cbv_snr_test(cbv, self.threshold_snrtest)

			# Save the CBV to file:
			np.save(os.path.join(self.data_folder, 'cbv_ini-%d.npy' % cbv_area), cbv)

			####################### PLOTS #################################
			# Plot the "effectiveness" of each CBV:
			max_components=20
			n_cbv_components = np.arange(max_components, dtype=int)
			pca_scores = compute_scores(mat, n_cbv_components)

			fig0 = plt.figure(figsize=(12,8))
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
			fig0.savefig(os.path.join(self.data_folder, 'cbv-perf-area%d.png' %cbv_area))
			plt.close(fig0)

			# Plot all the CBVs:
			fig, axes = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
			fig2, axes2 = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
			fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)
			fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)

			for k, ax in enumerate(axes.flatten()):
				try:
					ax.plot(cbv0[:, k]+0.1, 'r-')
#					if not indx_lowsnr is None:
#						if indx_lowsnr[k]:
#							col = 'c'
#						else:
#							col = 'k'
#					else:
#						col = 'k'
					ax.plot(cbv[:, k], ls='-', color='k')
					ax.set_title('Basis Vector %d' % (k+1))
				except:
					pass

			for k, ax in enumerate(axes2.flatten()):
				try:
					ax.plot(-np.abs(U0[:, k]), 'r-')
					ax.plot(np.abs(U[:, k]), 'k-')
					ax.set_title('Basis Vector %d' % (k+1))
				except:
					pass
			fig.savefig(os.path.join(self.data_folder, 'cbvs_ini-area%d.png' % cbv_area))
			fig2.savefig(os.path.join(self.data_folder, 'U_cbvs-area%d.png' % cbv_area))
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
		
		
		if os.path.exists(os.path.join(self.data_folder, 'cbv-%d.npy' % cbv_area)) & os.path.exists(os.path.join(self.data_folder, 'cbv-s-%d.npy' % cbv_area)):
			logger.info('Separated CBVs for area %d already calculated' % cbv_area)
			return

		else:
			logger.info('Computing CBV spike separation for area %d' % cbv_area)
			
			# Load initial CBV from "compute_CBV"
			filepath = os.path.join(self.data_folder, 'cbv_ini-%d.npy' % cbv_area)
			cbv = np.load(filepath)
			
			# padding window, just needs to be bigger than savgol filtering window
			wmir = 50
			
			# Initiate arrays for cleaned and spike CBVs
			cbv_new = np.zeros_like(cbv)
			cbv_spike = np.zeros_like(cbv)
	
			# Iterate over basis vectors
			for j in range(cbv.shape[1]):
				
				# Pad ends for better peak detection at boundaries of data
				data0 = cbv[:,j]
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
					if w%2==0:
						w+=1
						
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
				
				cbv_spike[:,j] = S
				cbv_new[:,j] = data[wmir-1:-wmir+1]
		
			# Save files
			np.save(os.path.join(self.data_folder, 'cbv-%d.npy' % cbv_area), cbv_new)
			np.save(os.path.join(self.data_folder, 'cbv-s-%d.npy' % cbv_area), cbv_spike)
			
			# Signal-to-Noise test (here only for plotting)
			indx_lowsnr = cbv_snr_test(cbv_new, self.threshold_snrtest)
			
			# Plot all the CBVs:
			fig, axes = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
			fig2, axes2 = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
			fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)
			fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)
	
			for k, ax in enumerate(axes.flatten()):
				try:
					if not indx_lowsnr is None:
						if indx_lowsnr[k]:
							col = 'c'
						else:
							col = 'k'
					else:
						col = 'k'
					ax.plot(cbv_new[:, k], ls='-', color=col)
					ax.set_title('Basis Vector %d' % (k+1))
				except:
					pass
	
			
			for k, ax in enumerate(axes2.flatten()):
				try:
					if not indx_lowsnr is None:
						if indx_lowsnr[k]:
							col = 'c'
						else:
							col = 'k'
					else:
						col = 'k'
					ax.plot(cbv_spike[:, k], ls='-', color=col)
					ax.set_title('Spike Basis Vector %d' % (k+1))
				except:
					pass		
			
			
			fig.savefig(os.path.join(self.data_folder, 'cbvs-area%d.png' % cbv_area))
			fig2.savefig(os.path.join(self.data_folder, 'spike-cbvs-area%d.png' % cbv_area))
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
		if os.path.exists(os.path.join(self.data_folder, 'mat-%d_free_weights.npz' % cbv_area)):
			logger.info("Initial co-trending for light curves in CBV area%d already done" %cbv_area)
			return
		else:
			logger.info("Initial co-trending for light curves in CBV area%d" % cbv_area)

		# Load stars from data base
		stars = self.search_database(search=['datasource="ffi"', 'cbv_area=%i' % cbv_area])

		# Load the cbv from file:
		cbv = CBV(self.data_folder, cbv_area, self.threshold_snrtest)

#		# Signal-to-Noise test (without actually removing any CBVs):
#		indx_lowsnr = cbv_snr_test(cbv.cbv, self.threshold_snrtest)
#		cbv.remove_cols(indx_lowsnr)

		# Update maximum number of components
		n_components0 = cbv.cbv.shape[1]
		logger.info('New max number of components: %i' %int(n_components0))

		if self.Numcbvs == 'all':
			n_components = n_components0
		else:
			n_components = np.min([self.Numcbvs, n_components0])

		logger.info('Fitting using number of components: %i' %int(n_components))
		# initialize results array, including TIC, CBV components, and an residual offset
		Nres = int(2*n_components+2)
		results = np.zeros([len(stars), Nres])

		# Loop through stars
		for kk, star in tqdm(enumerate(stars), total=len(stars), disable=not logger.isEnabledFor(logging.INFO)):

			lc = self.load_lightcurve(star)

			logger.debug("Correcting star %d", lc.targetid)

			flux_filter, res = cbv.cotrend_single(lc, n_components, self.data_folder, ini=True)
			lc_corr = (lc.flux/flux_filter-1)*1e6

			# TODO: compute diagnostics requiring the light curve
#			# SAVE TO DIAGNOSTICS FILE::
#			wn_ratio = GOC_wn(flux, flux-flux_filter)

			res = np.array([res,]).flatten()
			results[kk, 0] = lc.targetid
			results[kk, 1:len(res)+1] = res

			if do_ini_plots:
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
		np.savez(os.path.join(self.data_folder, 'mat-%d_free_weights.npz' % cbv_area), res=results)

		# Plot CBV weights
		fig = plt.figure(figsize=(15,15))
		ax = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		for kk in range(1,int(2*n_components+1)):
			
			if kk>n_components:
				LS = '--'
			else:
				LS = '-'
			idx = np.nonzero(results[:, kk])
			r = results[idx, kk]
			idx2 = (r>np.percentile(r, 10)) & (r<np.percentile(r, 90))
			kde = KDE(r[idx2])
			kde.fit(gridsize=5000)
			err = nanmedian(np.abs(r[idx2] - nanmedian(r[idx2]))) * 1e5
			
			if kk>n_components:
				ax3.plot(kde.support*1e5, kde.density/np.max(kde.density), label='CBV ' + str(kk), ls=LS)
				ax4.errorbar(kk, kde.support[np.argmax(kde.density)]*1e5, yerr=err, marker='o', color='k')
			else:
				ax.plot(kde.support*1e5, kde.density/np.max(kde.density), label='CBV ' + str(kk), ls=LS)
				ax2.errorbar(kk, kde.support[np.argmax(kde.density)]*1e5, yerr=err, marker='o', color='k')
				
		ax.set_xlabel('CBV weight')
		ax2.set_ylabel('CBV weight')
		ax2.set_xlabel('CBV')
		ax.legend()
		fig.savefig(os.path.join(self.data_folder, 'weights-sector-%d.png' % (cbv_area)))
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
		
		inipath = os.path.join(self.data_folder, 'mat-%d_free_weights.npz' % (cbv_area))
		if not os.path.exists(inipath):
			raise IOError('Trying to make priors without initial corrections')
			
			
		results = np.load(inipath)['res']
		n_stars = results.shape[0]

		# Load in positions and tmags, in same order as results are saved from ini_fit
		pos_mag0 = np.zeros([n_stars, 3])
		for jj, star in enumerate(results[:,0]):
			star_single = self.search_database(search=['datasource="ffi"', 'cbv_area=%i' %cbv_area, 'todolist.starid=%i' %int(star)])
			pos_mag0[jj, 0] = star_single[0]['pos_row']
			pos_mag0[jj, 1] = star_single[0]['pos_column']
			pos_mag0[jj, 2] = np.clip(star_single[0]['tmag'], 2, 20)

		# Relative importance of dimensions
#		S = np.array([1, 1, 2])
		S = np.array([MAD_model2(pos_mag0[:, 0]), MAD_model2(pos_mag0[:, 1]), 0.5*MAD_model2(pos_mag0[:, 2])])
		LL = np.diag(S)

#		pos_mag0[:, 0] /= np.std(pos_mag0[:, 0])
#		pos_mag0[:, 1] /= np.std(pos_mag0[:, 1])
#		pos_mag0[:, 2] /= np.std(pos_mag0[:, 2])
		
		# Construct and save distance tree
		dist = DistanceMetric.get_metric('mahalanobis', VI=LL)
		tree = BallTree(pos_mag0, metric=dist)

		savePickle(os.path.join(self.data_folder, 'D_area%d.pkl' %(cbv_area)), tree)


	#--------------------------------------------------------------------------
	def do_correction(self, lc):
		
		"""
		Function where the correction is called, and where
		additional headers for the FITS are defined
		
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>		
		"""

		logger = logging.getLogger(__name__)

		# Load the CBV (and Prior) from memory and if it is not already loaded,
		# load it in from file and keep it in memory for next time:
		cbv_area = lc.meta['task']['cbv_area']
		
		cbv = self.cbvs.get(cbv_area)
		if cbv is None:
			logger.debug("Loading CBV for area %d into memory", cbv_area)
			cbv = CBV(self.data_folder, cbv_area, self.threshold_snrtest)
			self.cbvs[cbv_area] = cbv
			
			if cbv.priors is None:
				raise IOError('Trying to co-trend without a defined prior')
				
		# Update maximum number of components
		n_components0 = cbv.cbv.shape[1] 

		logger.info('Max number of components above SNR: %i' %n_components0)
		if self.Numcbvs == 'all':
			n_components = n_components0
		else:
			n_components = np.min([self.Numcbvs, n_components0])

		logger.info('Co-trending star with TIC ID: %i' %lc.meta['task']['starid'])
		logger.info('Fitting using number of components: %i' %n_components)

		if self.simple_fit:
			met = 'llsq'
			flux_filter, res = cbv.cotrend_single(lc, n_components, ini=False, simple_fit=self.simple_fit, use_bic=self.use_bic, method=met, alpha=self.alpha, WS_lim=self.WS_lim, N_neigh=self.N_neigh)
		else:
			met = self.method
			flux_filter, res, residual, WS, pc = cbv.cotrend_single(lc, n_components, ini=False, use_bic=self.use_bic, method=met, alpha=self.alpha, WS_lim=self.WS_lim, N_neigh=self.N_neigh)
			logger.debug('New variability', residual)
			
			
		#corrected light curve in ppm
		lc_corr = 1e6*(lc.copy()/flux_filter - 1)

		res = np.array([res,]).flatten()
		
		status = STATUS.OK
		
		no_cbvs_fitted = int((len(res)-1)/2)
		for ii in range(no_cbvs_fitted):
			lc_corr.meta['additional_headers']['CBV_C%i'%int(ii+1)] = (res[ii], 'CBV%i coefficient' %int(ii+1))
			
		for jj in range(no_cbvs_fitted):
			lc_corr.meta['additional_headers']['CBVS_C%i'%int(jj+1)] = (res[jj+no_cbvs_fitted], 'Spike-CBV%i coefficient' %int(jj+1))	
			
			if len(res)<4: #fitting and using only one CBV 
				status = STATUS.WARNING
			
			if len(res)>21: #fitting and using more than 10 CBVs
				status = STATUS.WARNING
		
		lc_corr.meta['additional_headers']['CBV_C0'] = (res[-1], 'fitted offset')

		lc_corr.meta['additional_headers']['CBV_AREA'] = (cbv_area, 'CBV area of star')
		lc_corr.meta['additional_headers']['CBV_BIC'] = (self.use_bic, 'was BIC used to select no of CBVs')
		lc_corr.meta['additional_headers']['CBV_FIT'] = (self.simple_fit, 'CBV fitted with LLSQ?')
		lc_corr.meta['additional_headers']['CBV_COMP'] = (no_cbvs_fitted, 'number of fitted CBVs')
		lc_corr.meta['additional_headers']['CBV_MAX'] = (n_components, 'number of possible CBVs to fit')

		
		if self.plot:
			fig = plt.figure()
			ax1 = fig.add_subplot(211)
			ax1.plot(lc.time, lc.flux)
			ax1.plot(lc.time, flux_filter)
			ax1.plot(lc.time, pc, 'm--')
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
#			plt.show()

		return lc_corr, status