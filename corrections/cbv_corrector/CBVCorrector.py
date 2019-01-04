#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.decomposition import PCA
import matplotlib.colors as colors

from bottleneck import allnan, nanmedian
from scipy.interpolate import pchip_interpolate
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="scipy.stats") # they are simply annoying!
from tqdm import tqdm

from scipy.interpolate import Rbf, SmoothBivariateSpline

from .cbv_main import CBV, cbv_snr_reject, clean_cbv, lc_matrix_calc
from .cbv_util import compute_scores, ndim_med_filt, reduce_mode, reduce_std
import math
from .. import BaseCorrector, STATUS
import dill
import logging

plt.ioff()

	
#------------------------------------------------------------------------------

# TODO: Move plots to dedicated module

class CBVCorrector(BaseCorrector):
	"""
	The CBV (Co-trending Basis Vectors) correction method for the TASOC
	photometry pipeline
	
	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	
	"""
	
	
	def __init__(self, *args, do_ini_plots=True, Numcbvs='all', ncomponents=8, ent_limit=-0.7, WS_lim=20, alpha=1.3, targ_limit=150, method='powell', single_area=None, use_bic=True, \
			  threshold_correlation=0.5, threshold_snrtest=5, threshold_variability=1.3, **kwargs):	
		
		
		"""
		Initialise the corrector
		
		The CBV init inherets init  and functionality of :py:class:`BaseCorrector`
		
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
		self.do_ini_plots = do_ini_plots
		self.single_area = single_area
		self.threshold_snrtest = threshold_snrtest
		self.threshold_correlation = threshold_correlation
		self.threshold_variability = threshold_variability
		self.ncomponents = ncomponents
		self.ent_limit = ent_limit
		self.targ_limit = targ_limit
		self.alpha = alpha
		self.WS_lim = WS_lim
		
		self.compute_cbvs()
		self.cotrend_ini() 
		self.compute_weight_interpolations()

	#-------------------------------------------------------------------------

	def lc_matrix(self, cbv_area):
		
		"""
		Computes correlation matrix for light curves in a given cbv-area. 
		
		Only targets with a variability below a user-defined threshold are included
		in the calculation.
		
		Returns matrix of the *self.threshold_correlation*% most correlated light curves; the threshold is defined in the class init function.
		
		Parameters:
            cbv_area: the cbv area to calculate matrix for
			
			additional parameters are contained in *self* and defined in the init function
        Returns:
            mat: matrix of *self.threshold_correlation*% most correlated light curves, to be used in CBV calculation
			stds: standard deviations of light curves in "mat"
		
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
		
		logger=logging.getLogger(__name__)
		
		#---------------------------------------------------------------------
		# CALCULATE LIGHT CURVE CORRELATIONS
		#---------------------------------------------------------------------
	
		logger.info("We are running CBV_AREA=%d" % cbv_area)
	
	
		tmpfile = os.path.join(self.data_folder, 'mat-%d.npz' %cbv_area)
		if os.path.exists(tmpfile):
			logger.info("Loading existing file...")
			data = np.load(tmpfile)
			mat = data['mat']
			stds = data['stds']
	
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
			fig.savefig(os.path.join(self.data_folder, 'variability-area%d.png' %cbv_area))
			plt.close(fig)
	
			# Get the list of star that we are going to load in the lightcurves for:
			stars = self.search_database(search=['datasource="ffi"', 'cbv_area=%i' %cbv_area, 'variability < %f' %(self.threshold_variability*median_variability)])
	
			# Number of stars returned:
			Nstars = len(stars)
	
			# Load the very first timeseries only to find the number of timestamps.
			lc = self.load_lightcurve(stars[0])	
			Ntimes = len(lc.time)
	
			logger.info("Matrix size: %d x %d" % (Nstars, Ntimes))
	
			# Make the matrix that will hold all the lightcurves:
			logger.info("Loading in lightcurves...")
			mat0 = np.empty((Nstars, Ntimes), dtype='float64')
			mat0.fill(np.nan)
			stds0 = np.empty(Nstars, dtype='float64')

			# Loop over stars
			for k, star in tqdm(enumerate(stars), total=Nstars, disable=logger.isEnabledFor(logging.INFO)):
				
				# Load lightkurve object
				lc = self.load_lightcurve(star)	
				
				# Remove bad data based on quality
				quality_remove = 1 #+...
				flag_removed = (lc.quality & quality_remove != 0)
				lc.flux[flag_removed] = np.nan
				
				# Normalize the data and store it in the rows of the matrix:
				mat0[k, :] = lc.flux / star['mean_flux'] - 1.0
				
				try:
					stds0[k] = np.sqrt(star['variance'])
				except Exception as e:
					stds0[k] = np.nan
					print(e)
	
			# Only start calculating correlations if we are actually filtering using them:
			if self.threshold_correlation < 1.0:
				file_correlations = os.path.join(self.data_folder, 'correlations-%d.npy' %cbv_area)
				if os.path.exists(file_correlations):
					correlations = np.load(file_correlations)
				else:
					# Calculate the correlation matrix between all lightcurves:
					correlations = lc_matrix_calc(Nstars, mat0)#, stds0)
					np.save(file_correlations, correlations)
	
				# Find the median absolute correlation between each lightcurve and all other lightcurves:
				c = nanmedian(correlations, axis=0)
	
				# Indicies that would sort the lightcurves by correlations in descending order:
				indx = np.argsort(c)[::-1]
				indx = indx[:int(self.threshold_correlation*Nstars)]
				#TODO: remove based on threshold value? rather than just % of stars
	
				# Only keep the top "threshold_correlation"% of the lightcurves that are most correlated:
				mat = mat0[indx, :]
				stds = stds0[indx]
	
				# Clean up a bit:
				del correlations, c, indx
	
			# Save something for debugging:
			np.savez(tmpfile, mat=mat, stds=stds) 
	
		return mat, stds

	#-------------------------------------------------------------------------
	
	def lc_matrix_clean(self, cbv_area):
		
		"""
		Performs gap-filling of light curves returned by :py:func:`CBVCorrector.lc_matrix`, and
		removes time stamps where all flux values are nan
		
		Parameters:
			cbv_area: the cbv area to calculate light curve matrix for
			
		Returns:
			mat: matrix from :py:func:`CBVCorrector.lc_matrix` that has been gap-filled and with nans removed, to be used in CBV calculation
			stds: standard deviations of light curves in "mat"
			indx_nancol: the indices for the timestamps with nans in all light curves 
			Ntimes: Number of timestamps in light curves contained in mat before removing nans
		
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
		
		logger=logging.getLogger(__name__)
		
		logger.info('Running matrix clean')
		tmpfile = os.path.join(self.data_folder, 'mat-%d_clean.npz' %cbv_area)
		if os.path.exists(tmpfile):
			logger.info("Loading existing file...")
			data = np.load(tmpfile)
			mat = data['mat']
			stds = data['stds']
			
			Ntimes = data['Ntimes']
			indx_nancol = data['indx_nancol']
	
		else:
			# Compute light curve correlation matrix
			mat0, stds = self.lc_matrix(cbv_area)
			
			# Print the final shape of the matrix:
			logger.info("Matrix size: %d x %d" % mat0.shape)
	
			# Find columns where all stars have NaNs and remove them:
			indx_nancol = allnan(mat0, axis=0)
			Ntimes = mat0.shape[1]
			mat = mat0[:, ~indx_nancol]
			cadenceno = np.arange(mat.shape[1])
	
			logger.info("Gap-filling lightcurves...")
			for k in tqdm(range(mat.shape[0]), total=mat.shape[0], disable=logger.isEnabledFor(logging.INFO)):
	
				mat[k, :] /= stds[k]	
				# Fill out missing values by interpolating the lightcurve:
				indx = np.isfinite(mat[k, :])
				mat[k, ~indx] = pchip_interpolate(cadenceno[indx], mat[k, indx], cadenceno[~indx])
	
			# Save something for debugging:
			np.savez(tmpfile, mat=mat, stds=stds, indx_nancol=indx_nancol, Ntimes=Ntimes)
	
		return mat, stds, indx_nancol, Ntimes
		
	#-------------------------------------------------------------------------
	
	def compute_cbvs(self):
		
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
		
		logger=logging.getLogger(__name__)
		
		cbv_areas = [int(row['cbv_area']) for row in self.search_database(select='cbv_area', distinct=True)]
		

		# Loop through the CBV areas:
		# - or run them in parallel - whatever you like!
		for ii, cbv_area in enumerate(cbv_areas):
			
			if not self.single_area is None:
				if not cbv_area == self.single_area:
					continue
				
				
			logger.info('------------------------------------')
			
			
			if os.path.exists(os.path.join(self.data_folder, 'cbv-%d.npy' % (cbv_area))):
				logger.info('CBV for area%d already calculated' %cbv_area)
				continue
			
			else:
				logger.info('Computing CBV for area%d' %cbv_area)
				# Extract or compute cleaned and gapfilled light curve matrix
				mat0, stds, indx_nancol, Ntimes = self.lc_matrix_clean(cbv_area)
				# Calculate initial CBVs
				pca0 = PCA(self.ncomponents)
				U0, _, _ = pca0._fit(mat0)
	
				cbv0 = np.empty((Ntimes, self.ncomponents), dtype='float64')
				cbv0.fill(np.nan)
				cbv0[~indx_nancol, :] = np.transpose(pca0.components_)
				
				
				logger.info('Cleaning matrix for CBV - remove single dominant contributions')
				# Clean away targets that contribute significantly as a single star to a given CBV (based on entropy)
				mat = clean_cbv(mat0, self.ncomponents, self.ent_limit, self.targ_limit)
				
		
				# Calculate the principle components of cleaned matrix
				logger.info("Doing Principle Component Analysis...")
				pca = PCA(self.ncomponents)
				U, _, _ = pca._fit(mat)
				
				
				cbv00 = np.empty((Ntimes, self.ncomponents), dtype='float64')
				cbv00.fill(np.nan)
				cbv00[~indx_nancol, :] = np.transpose(pca.components_)
				
				# Signal-to-Noise test:
				cbv, indx_lowsnr = cbv_snr_reject(cbv00, self.threshold_snrtest)
			
				# Save the CBV to file:
				np.save(os.path.join(self.data_folder, 'cbv-%d.npy' % (cbv_area)), cbv)
				
				
				# Plot the "effectiveness" of each CBV:
				max_components=20
				n_cbv_components = np.arange(max_components, dtype=int)
				pca_scores = compute_scores(mat, n_cbv_components)
				
				fig0 = plt.figure(figsize=(12,8))
				ax0 = fig0.add_subplot(121)
				ax02 = fig0.add_subplot(122)
				ax0.plot(n_cbv_components, pca_scores, 'b', label='PCA scores')
				ax0.set_xlabel('nb of components')
				ax0.set_ylabel('CV scores')
				ax0.legend(loc='lower right')
				
				ax02.plot(np.arange(1, cbv0.shape[1]+1), pca.explained_variance_ratio_, '.-')
				ax02.axvline(x=cbv.shape[1]+0.5, ls='--', color='k')
				ax02.set_xlabel('CBV number')
				ax02.set_ylabel('Variance explained ratio')
				
				fig0.savefig(os.path.join(self.data_folder, 'cbv-perf-area%d.png' %cbv_area))
				plt.close(fig0)
			
	
				# Plot all the CBVs:
				fig, axes = plt.subplots(4, 2, figsize=(12, 8))
				fig2, axes2 = plt.subplots(4, 2, figsize=(12, 8))
				fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)  
				fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)  
	
				for k, ax in enumerate(axes.flatten()):
					ax.plot(cbv0[:, k]+0.1, 'r-')	
					if not indx_lowsnr is None:
						if indx_lowsnr[k]:
							col = 'c'
						else:
							col = 'k'
					else:
						col = 'k'
					ax.plot(cbv00[:, k], ls='-', color=col)	
					ax.set_title('Basis Vector %d' % (k+1))
					
					
				for k, ax in enumerate(axes2.flatten()):	
					ax.plot(-np.abs(U0[:, k]), 'r-')
					ax.plot(np.abs(U[:, k]), 'k-')
					ax.set_title('Basis Vector %d' % (k+1))
				fig.savefig(os.path.join(self.data_folder, 'cbvs-area%d.png' %cbv_area))
				fig2.savefig(os.path.join(self.data_folder, 'U_cbvs-area%d.png' %cbv_area))
				plt.close('all')
			
	#---------------------------------------------------------------------------------

	def cotrend_ini(self): 
		
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
		
		logger=logging.getLogger(__name__)
		
		cbv_areas = [int(row['cbv_area']) for row in self.search_database(select='cbv_area', distinct=True)]
		self.cbvs = {}
	
	
		# Loop through the CBV areas:
		for ii, cbv_area in enumerate(cbv_areas):
			
			if not self.single_area is None:
				if not cbv_area==self.single_area:
					continue
				
			#---------------------------------------------------------------------------------------------------------
			# CORRECTING STARS
			#---------------------------------------------------------------------------------------------------------
			logger.info("--------------------------------------------------------------")
			if os.path.exists(os.path.join(self.data_folder, 'mat-%d_free_weights.npz' %cbv_area)):
				logger.info("Initial co-trending for light curves in CBV area%d already done" %cbv_area)
				
				# Load the cbv from file:
				cbv = CBV(os.path.join(self.data_folder, 'cbv-%d.npy' % (cbv_area)))
				self.cbvs[cbv_area] = cbv
				
				continue
			else:
				logger.info("Initial co-trending for light curves in CBV area%d" %cbv_area)

			# Load stars from data base			
			stars = self.search_database(search=['datasource="ffi"', 'cbv_area=%i' %cbv_area])#, select='cbv_area')
			
		
			# Load the cbv from file:
			cbv = CBV(os.path.join(self.data_folder, 'cbv-%d.npy' % (cbv_area)))
			self.cbvs[cbv_area] = cbv
	
			# Update maximum number of components	
			n_components0 = cbv.cbv.shape[1]
				
			
			logger.info('New max number of components: %i' %int(n_components0))
			
			if self.Numcbvs=='all':
				n_components = n_components0
			else:	
				n_components = np.min([self.Numcbvs, n_components0])
				
			logger.info('Fitting using number of components: %i' %int(n_components))	
			results = np.zeros([len(stars), n_components+2])
			
	
			# Loop through stars
			for kk, star in tqdm(enumerate(stars), total=len(stars), disable=logger.isEnabledFor(logging.INFO)):
				
				
				
				lc = self.load_lightcurve(star)
				
				logger.info("Correcting star %d" %lc.targetid)

				flux_filter, res = cbv.cotrend_single(lc, n_components, self.data_folder, ini=True)
				lc_corr = (lc.flux/flux_filter-1)*1e6
				
#				# SAVE TO DIAGNOSTICS FILE::
#				wn_ratio = GOC_wn(flux, flux-flux_filter)
				
				res = np.array([res,]).flatten()
				results[kk, 0] = lc.targetid
				results[kk, 1:len(res)+1] = res
				
		
				if self.do_ini_plots:
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
			np.savez(os.path.join(self.data_folder, 'mat-%d_free_weights.npz' %cbv_area), res=results)
			
			
			# Plot CBV weights
			fig = plt.figure(figsize=(15,6))
			ax = fig.add_subplot(121)
			ax2 = fig.add_subplot(122)
			for kk in range(1,n_components+1):
				idx = np.nonzero(results[:, kk])
				r = results[idx, kk]
				idx2 = (r>np.percentile(r, 10)) & (r<np.percentile(r, 90))
				kde = KDE(r[idx2])
				kde.fit(gridsize=5000)
				
				ax.plot(kde.support*1e5, kde.density/np.max(kde.density), label='CBV ' + str(kk))
				
				err = nanmedian(np.abs(r[idx2] - nanmedian(r[idx2]))) * 1e5
				ax2.errorbar(kk, kde.support[np.argmax(kde.density)]*1e5, yerr=err, marker='o', color='k')
			ax.set_xlabel('CBV weight')
			ax2.set_ylabel('CBV weight')
			ax2.set_xlabel('CBV')
			ax.legend()
			fig.savefig(os.path.join(self.data_folder, 'weights-sector-%d.png' % (cbv_area)))
			plt.close(fig)


	#--------------------------------------------------------------------------
	
	def compute_weight_interpolations(self, dimensions=['row', 'col', 'tmag']):
	
		cbv_areas = [int(row['cbv_area']) for row in self.search_database(select='cbv_area', distinct=True)]
		n_cbvs_max = self.ncomponents
		n_cbvs_max_new = 0
		
		figures1 = [];
		figures2 = [];
		for i in range(n_cbvs_max):
			fig, ax = plt.subplots(2,2, num='cbv%i' %i, figsize=(15,6), )
			figures1.append(fig)
			figures2.append(ax)
	
			
		colormap = plt.cm.PuOr #or any other colormap
		min_max_vals = np.zeros([n_cbvs_max, 4])
			
		#TODO: obtain from sector information
		midx = 40.54 # Something wrong! field is 27 deg wide, not 24
		midy = 18
		
		
		pos_mag={}
		
		# Loop through the CBV areas:
		# - or run them in parallel - whatever you like!
		for ii, cbv_area in enumerate(cbv_areas):	
			
			if os.path.exists(os.path.join(self.data_folder, 'Rbf_area%d_cbv1.pkl' %cbv_area)):
				print('Weights for area%d already done' %cbv_area)
				continue
			
			print('Computing weights for area%d' %cbv_area)
			results = np.load(os.path.join(self.data_folder, 'mat-%d_free_weights.npz' % (cbv_area)))['res']
			n_stars = results.shape[0]
			n_cbvs = results.shape[1]-2 #results also include star name and offset
			
			if n_cbvs>n_cbvs_max_new:
				n_cbvs_max_new = n_cbvs
			
			pos_mag0 = np.zeros([n_stars, 7])
			
			pos_mag[cbv_area] = {}
			
			for jj, star in enumerate(results[:,0]):

				star_single = self.search_database(search=['datasource="ffi"', 'cbv_area=%i' %cbv_area, 'todolist.starid=%i' %int(star)])#, select='cbv_area')

				# Fix small glitch /by Rasmus) in assignment of lat/lon
				if cbv_area==122:
					if star_single[0]['eclat']>15:
						pos_mag0[jj, 0] = 50
						pos_mag0[jj, 1] = 10
						pos_mag0[jj, 2] = 10
						continue
					
					
				pos_mag[cbv_area]['eclon'] = np.array([])
					
				pos_mag0[jj, 0] = star_single[0]['eclon']
				pos_mag0[jj, 1] = star_single[0]['eclat']
				pos_mag0[jj, 2] = star_single[0]['pos_row']+(star_single[0]['ccd']>2)*2048
				pos_mag0[jj, 3] = star_single[0]['pos_column']+(star_single[0]['ccd']%2==0)*2048
				
				pos_mag0[jj, 4] = star_single[0]['tmag']
				
				# Convert to polar coordinates
				angle = math.atan2(star_single[0]['eclat']-midy, star_single[0]['eclon']-midx)
				angle = angle * 360 / (2*np.pi)
				if (angle < 0):
					angle += 360
				pos_mag0[jj, 5] = np.sqrt((star_single[0]['eclon']-midx)**2 + (star_single[0]['eclat']-midy)**2)
				pos_mag0[jj, 6] = angle
				
			
			pos_mag[cbv_area]['eclon'] = pos_mag0[:, 0]
			pos_mag[cbv_area]['eclat'] = pos_mag0[:, 1]
			pos_mag[cbv_area]['row'] = pos_mag0[:, 2]
			pos_mag[cbv_area]['col'] = pos_mag0[:, 3]
			pos_mag[cbv_area]['tmag'] = pos_mag0[:, 4]
			pos_mag[cbv_area]['rad'] = pos_mag0[:, 4]
			pos_mag[cbv_area]['theta'] = pos_mag0[:, 4]
			pos_mag[cbv_area]['results'] = results
			
			
			
			for j in range(n_cbvs):
				
				VALS = results[:,1+j]
				# Perform binning
				
				axm = figures2[j][0,0]
				axs = figures2[j][0,1]
				
				axm2 = figures2[j][1,0]
				axs2 = figures2[j][1,1]
				
				
				if np.percentile(VALS,10)<min_max_vals[j,0]:
					min_max_vals[j,0] = np.percentile(VALS,10)
				if np.percentile(VALS,90)>min_max_vals[j,1]:
					min_max_vals[j,1] = np.percentile(VALS,90)
				
				normalize = colors.Normalize(vmin=min_max_vals[j,0], vmax=min_max_vals[j,1])	
	
				
				# CBV values
				hbm = axm.hexbin(pos_mag[cbv_area][dimensions[0]], pos_mag[cbv_area][dimensions[1]], C=VALS, gridsize=10, reduce_C_function=reduce_mode, cmap=colormap, norm=normalize)
				# CBV values scatter
				hbs = axs.hexbin(pos_mag[cbv_area][dimensions[0]], pos_mag[cbv_area][dimensions[1]], C=VALS, gridsize=10, reduce_C_function=reduce_std, cmap=colormap, norm=normalize)
			
				# Get values and vertices of hexbinning
				zvalsm = hbm.get_array();		vertsm = hbm.get_offsets()
				zvalss = hbs.get_array();		vertss = hbs.get_offsets()
			
				# Bins to keed for interpolation
				idxm = ndim_med_filt(zvalsm, vertsm, 6)
				idxs = ndim_med_filt(zvalss, vertss, 6)
				
				# Plot removed bins
				axm.plot(vertsm[~idxm,0], vertsm[~idxm,1], marker='.', ms=1, ls='', color='r')
				axs.plot(vertss[~idxs,0], vertss[~idxs,1], marker='.', ms=1, ls='', color='r')
			
				# Trim binned values before interpolation
				zvalsm, vertsm = zvalsm[idxm], vertsm[idxm] 
				zvalss, vertss = zvalss[idxs], vertss[idxs] 
			
				rbfim = Rbf(vertsm[:,0], vertsm[:,1], zvalsm, smooth=1)
				rbfis = Rbf(vertss[:,0], vertss[:,1], zvalss, smooth=1)
				
				with open(os.path.join(self.data_folder, 'Rbf_area%d_cbv%i.pkl' %(cbv_area,int(j+1))), 'wb') as file:
					dill.dump(rbfim, file)
				with open(os.path.join(self.data_folder, 'Rbf_area%d_cbv%i_std.pkl' %(cbv_area,int(j+1))), 'wb') as file:
					dill.dump(rbfis, file)	
					
					
				# Plot resulting interpolation
				x1 = np.linspace(vertsm[:,0].min(), vertsm[:,0].max(), 100); y1 = np.linspace(vertsm[:,1].min(), vertsm[:,1].max(), 100); xv1, yv1 = np.meshgrid(x1, y1)
				x2 = np.linspace(vertss[:,0].min(), vertss[:,0].max(), 100); y2 = np.linspace(vertss[:,1].min(), vertss[:,1].max(), 100); xv2, yv2 = np.meshgrid(x2, y2)
			
			
				rm = np.abs(rbfim(xv1, yv1))
				rs = np.abs(rbfis(xv2, yv2))
				
				if np.percentile(rm,10)<min_max_vals[j,2]:
					min_max_vals[j,2] = np.percentile(rm,10)
				if np.percentile(rm,90)>min_max_vals[j,3]:
					min_max_vals[j,3] = np.percentile(VALS,90)
					
				normalize = colors.Normalize(vmin=min_max_vals[j,2], vmax=min_max_vals[j,3])
				axm2.contourf(xv1, yv1, rm, cmap=colormap, norm=normalize)
				axs2.contourf(xv2, yv2, rs, cmap=colormap, norm=normalize)
				
		for k, figs in enumerate(figures1):
			if k>=n_cbvs_max_new:
				break			
			filename = 'cbv%i.png' %k
			figs.savefig(os.path.join(self.data_folder, filename))
		
		
	#--------------------------------------------------------------------------		
		
	def do_correction(self, lc):
		
		logger=logging.getLogger(__name__)
		
		# Load the cbv from file:
		cbv = self.cbvs[lc.meta['task']['cbv_area']]
		
		# Update maximum number of components	
		n_components0 = cbv.cbv.shape[1]
			
		logger.info('New max number of components: %i' %n_components0)
		if self.Numcbvs=='all':
			n_components = n_components0
		else:	
			n_components = np.min([self.Numcbvs, n_components0])
			
		logger.info('Fitting using number of components: %i' %n_components)		
		
		flux_filter, res, residual, WS, pc = cbv.cotrend_single(lc, n_components, self.data_folder, ini=False, use_bic=self.use_bic, method=self.method, alpha=self.alpha, WS_lim=self.WS_lim)
		
		#corrected light curve in ppm
		lc_corr = (lc.flux/flux_filter-1)
				
		res = np.array([res,]).flatten()
		
		for ii in range(len(res)-1):
			lc.meta['additional_headers']['CBV_c%i'%int(ii+1)] = (res[ii], 'CBV%i coefficient' %int(ii+1)) 
		lc.meta['additional_headers']['offset'] = (res[-1], 'fitted offset') 
		
		lc.meta['additional_headers']['use_BIC'] = (self.use_bic, 'was BIC used to select no of CBVs') 
		lc.meta['additional_headers']['fit_met'] = (self.method, 'method used to fit CBV') 
		lc.meta['additional_headers']['no_comp'] = (len(res)-1, 'number of fitted CBVs') 
		
				
		logger.debug('New variability', residual)
		
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
			ax2.plot(lc.time, lc_corr*1e6)
			ax2.set_xlabel('Time (BJD)')
			ax2.set_ylabel('Relative flux (ppm)')
			plt.tight_layout()
			filename = 'lc_corr_TIC%d.png' %lc.targetid
			if not os.path.exists(os.path.join(self.plot_folder(lc))):
				os.makedirs(os.path.join(self.plot_folder(lc)))
			fig.savefig(os.path.join(self.plot_folder(lc), filename))
			plt.close('all')

		#TODO: update status
		lc.flux = lc_corr
		lc *= 1e6
		return lc, STATUS.OK