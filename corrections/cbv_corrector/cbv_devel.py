#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import os
import sys
import glob
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from bottleneck import allnan, nansum, move_median, nanmedian, nanstd, nanmean
from scipy.optimize import minimize
from scipy.stats import pearsonr, entropy
from scipy.interpolate import pchip_interpolate
from scipy.signal import correlate
import itertools
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from scipy.special import xlogy
import scipy.linalg as slin
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="scipy.stats") # they are simply annoying!
import dill
import json
from tqdm import tqdm
import time as TIME
from cbv_util import compute_entopy, _move_median_central_1d, move_median_central, compute_scores, rms, MAD_model
from cbv_weights import compute_weight_interpolations
plt.ioff()

	
#------------------------------------------------------------------------------
def cbv_snr_reject(cbv_ini, threshold_snrtest=5.0):
	A_signal = rms(cbv_ini, axis=0)
	A_noise = rms(np.diff(cbv_ini, axis=0), axis=0)
	snr = 10 * np.log10( A_signal**2 / A_noise**2 )
	indx_lowsnr = (snr < threshold_snrtest)
	if np.any(indx_lowsnr):
		print("Rejecting %d CBVs based on SNR test" % np.sum(indx_lowsnr))
		cbv = cbv_ini[:, ~indx_lowsnr]
		return cbv, indx_lowsnr	
	else:
		return cbv_ini, None
	

#------------------------------------------------------------------------------
def clean_cbv(Matrix, n_components, ent_limit=-1.5, targ_limit=50):
	
	# Calculate the principle components:
	print("Doing Principle Component Analysis...")
	pca = PCA(n_components)
	U, _, _ = pca._fit(Matrix)
	
	Ent = compute_entopy(U)
	print('Entropy start:', Ent)
	
	targets_removed = 0
	components = np.arange(n_components)
	
	with np.errstate(invalid='ignore'):	
		while np.any(Ent<ent_limit):
			com = components[(Ent<ent_limit)][0]
			
			# Remove highest relative weight target
			m = nanmedian(U[:, com])
			s = 1.46*nanmedian(np.abs(U[:, com] - m))
			dev = np.abs(U[:, com] - m) / s
	
			idx0 = np.argmax(dev)
		
			star_no = np.ones(U.shape[0], dtype=bool)
			star_no[idx0] = False
			#print('removing star ', idx0)
			
			Matrix = Matrix[star_no, :]
			U, _, _ = pca._fit(Matrix)
	
			targets_removed += 1
			
			if targets_removed>targ_limit:
				break
			
			Ent = compute_entopy(U)
		
	print('Entropy end:', Ent)
	print('Targets removed ', targets_removed)
	return Matrix

#------------------------------------------------------------------------------
def AlmightyCorrcoefEinsumOptimized(O, P):
	
    (n, t) = O.shape      # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (np.einsum("nt->t", O, optimize='optimal') / np.double(n)) # compute O - mean(O)
    DP = P - (np.einsum("nm->m", P, optimize='optimal') / np.double(n)) # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO, optimize='optimal')

    varP = np.einsum("nm,nm->m", DP, DP, optimize='optimal')
    varO = np.einsum("nt,nt->t", DO, DO, optimize='optimal')
    tmp = np.einsum("m,t->mt", varP, varO, optimize='optimal')

    return cov / np.sqrt(tmp)


#------------------------------------------------------------------------------
def lc_matrix_calc(Nstars, mat0, stds):
	
	print("Calculating correlations...")
	
	indx_nancol = allnan(mat0, axis=0)
	mat1 = mat0[:, ~indx_nancol]
	
	
	mat1[np.isnan(mat1)] = 0
	correlations = np.abs(AlmightyCorrcoefEinsumOptimized(mat1.T, mat1.T))
	np.fill_diagonal(correlations, np.nan)
	
	
	
#	correlations = np.empty((Nstars, Nstars), dtype='float64')
#	np.fill_diagonal(correlations, np.nan) # Put NaNs on the diagonal

#	print(mat.shape, Nstars)
	
#	for i, j in tqdm(itertools.combinations(range(Nstars), 2), total=0.5*Nstars**2-Nstars):
#		r = pearsonr(mat[i, :]/stds[i], mat[j, :]/stds[j])[0]
#		correlations[i,j] = correlations[j,i] = np.abs(r)


#	np.testing.assert_allclose(correlations, correlations2)
#	if np.allclose(correlations, correlations2, equal_nan=True):
#		print("Test passed")
#	else:
#		print("Test failed")
#
#	plt.figure()
#	plt.matshow(correlations-correlations2)
#	plt.show()
		
	return correlations	


#------------------------------------------------------------------------------
def lc_matrix(sector, cbv_area, cursor):
	
	#---------------------------------------------------------------------------------------------------------
	# CALCULATE LIGHT CURVE CORRELATIONS
	#---------------------------------------------------------------------------------------------------------

	print("We are running CBV_AREA=%d" % cbv_area)


	tmpfile = 'mat-sector%02d-%d.npz' % (sector, cbv_area)
	if os.path.exists(tmpfile):
		print("Loading existing file...")
		data = np.load(tmpfile)
		mat = data['mat']
		priorities = data['priorities']
		stds = data['stds']

	else:
		# Find the median of the variabilities:
		# SQLite does not have a median function so we are going to
		# load all the values into an array and make Python do the
		# heavy lifting.
		cursor.execute("""SELECT variability FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE
			datasource='ffi'
			AND status=1
			AND cbv_area=?;""", (cbv_area, ))
		variability = np.array([row[0] for row in cursor.fetchall()], dtype='float64')
		median_variability = nanmedian(variability)

		# Plot the distribution of variability for all stars:
		# TODO: Move to dedicated plotting module
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(variability/median_variability, bins=np.logspace(np.log10(0.1), np.log10(1000.0), 50))
		ax.axvline(threshold_variability, color='r')
		ax.set_xscale('log')
		ax.set_xlabel('Variability')
		fig.savefig('plots/sector%02d/variability-area%d.png' % (sector, cbv_area))
		plt.close(fig)

		# Get the list of star that we are going to load in the lightcurves for:
		cursor.execute("""SELECT todolist.starid,todolist.priority,mean_flux,variance FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE
			datasource='ffi'
			AND status=1
			AND cbv_area=?
			AND variability < ?
		ORDER BY variability ASC LIMIT 30000;""", (cbv_area, threshold_variability*median_variability))
		stars = cursor.fetchall()

		# Number of stars returned:
		Nstars = len(stars)

		# Load the very first timeseries only to find the number of timestamps.
		# TODO: When using FITS files in the future, this could simply be loaded from the header.
		time = np.loadtxt(os.path.join('sysnoise', 'Star%d.sysnoise' % (stars[0]['starid'],)), usecols=(0, ), unpack=True)
		Ntimes = len(time)

		print("Matrix size: %d x %d" % (Nstars, Ntimes))

		# Make the matrix that will hold all the lightcurves:
		print("Loading in lightcurves...")
		mat0 = np.empty((Nstars, Ntimes), dtype='float64')
		mat0.fill(np.nan)
		stds0 = np.empty(Nstars, dtype='float64')
		priorities0 = np.empty(Nstars, dtype='int64')
		
		for k, star in tqdm(enumerate(stars), total=Nstars):
			priorities0[k] = star['priority']
			starid = star['starid']

			flux = np.loadtxt(os.path.join('sysnoise', 'Star%d.sysnoise' % (starid,)), usecols=(1, ), unpack=True)

			# Normalize the data and store it in the rows of the matrix:
			mat0[k, :] = flux / star['mean_flux'] - 1.0
			stds0[k] = np.sqrt(star['variance'])

		# Only start calculating correlations if we are actually filtering using them:
		if threshold_correlation < 1.0:
			file_correlations = 'correlations-sector%02d-%d.npy' % (sector, cbv_area)
			if os.path.exists(file_correlations):
				correlations = np.load(file_correlations)
			else:
				# Calculate the correlation matrix between all lightcurves:
				correlations = lc_matrix_calc(Nstars, mat0, stds0)
				np.save(file_correlations, correlations)

			# Find the median absolute correlation between each lightcurve and all other lightcurves:
			c = nanmedian(correlations, axis=0)

			# Indicies that would sort the lightcurves by correlations in descending order:
			indx = np.argsort(c)[::-1]
			indx = indx[:int(threshold_correlation*Nstars)]
			#TODO: remove based on threshold value? rather than just % of stars

			# Only keep the top 50% of the lightcurves that are most correlated:
			priorities = priorities0[indx]
			mat = mat0[indx, :]
			stds = stds0[indx]

			# Clean up a bit:
			del correlations, c, indx

		# Save something for debugging:
		np.savez('mat-sector%02d-%d.npz' % (sector, cbv_area), mat=mat, priorities=priorities, stds=stds)

	return mat, priorities, stds

# =============================================================================
# 
# =============================================================================
	
def lc_matrix_clean(sector, cbv_area, cursor):
	
	
	print('Running matrix clean')
	tmpfile = 'mat-sector%02d-%d_clean.npz' % (sector, cbv_area)
	if os.path.exists(tmpfile):
		print("Loading existing file...")
		data = np.load(tmpfile)
		mat = data['mat']
		priorities = data['priorities']
		stds = data['stds']
		
		Ntimes = data['Ntimes']
		indx_nancol = data['indx_nancol']

	else:
		# Compute light curve correlation matrix
		mat0, priorities, stds = lc_matrix(sector, cbv_area, cursor)
		
		# Print the final shape of the matrix:
		print("Matrix size: %d x %d" % mat0.shape)

		# Simple low-pass filter of the individual targets:
		#mat = move_median_central(mat, 48, axis=1)

		# Find columns where all stars have NaNs and remove them:
		indx_nancol = allnan(mat0, axis=0)
		Ntimes = mat0.shape[1]
		mat = mat0[:, ~indx_nancol]
		cadenceno = np.arange(mat.shape[1])

		# TODO: Is this even needed? Or should it be done earlier?
		print("Gap-filling lightcurves...")
		for k in tqdm(range(mat.shape[0]), total=mat.shape[0]):

			mat[k, :] /= stds[k]

			# Fill out missing values by interpolating the lightcurve:
			indx = np.isfinite(mat[k, :])
			# Do inpainting??
			mat[k, ~indx] = pchip_interpolate(cadenceno[indx], mat[k, indx], cadenceno[~indx])

		# Save something for debugging:
		np.savez('mat-sector%02d-%d_clean.npz' % (sector, cbv_area), mat=mat, priorities=priorities, stds=stds, indx_nancol=indx_nancol, Ntimes=Ntimes)

	return mat, priorities, stds, indx_nancol, Ntimes


# =============================================================================
# 
# =============================================================================

def GOC_corrmatrix(sector, cbv_area, cursor, exponent=3):
	
	#---------------------------------------------------------------------------------------------------------
	# CALCULATE CORRECTED LIGHT CURVE CORRELATIONS
	#---------------------------------------------------------------------------------------------------------
	print("We are running CBV_AREA=%d" % cbv_area)

	file_median_correlations = 'correlations-median_postcorr-sector%02d-%d.json' % (sector, cbv_area)
	# Find the median absolute correlation between each lightcurve and all other lightcurves:
	
	if os.path.exists(file_median_correlations):
		with open(file_median_correlations) as infile:
			C = json.load(infile)
	else:	

		# Query for all stars, no matter what variability and so on
		cursor.execute("""SELECT todolist.starid,todolist.priority, eclon, eclat FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE
			datasource='ffi'
			AND cbv_area=?
			AND status=1;""", (cbv_area, ))
		
		stars = cursor.fetchall()
	
		# Number of stars returned:
		Nstars = len(stars)
	
		# Load the very first timeseries only to find the number of timestamps.
		# TODO: When using FITS files in the future, this could simply be loaded from the header.
		time = np.loadtxt(os.path.join('sysnoise', 'Star%d.sysnoise' % (stars[0]['starid'],)), usecols=(0, ), unpack=True)
		Ntimes = len(time)
	
		print("Matrix size: %d x %d" % (Nstars, Ntimes))
	
		# Make the matrix that will hold all the lightcurves:
		print("Loading in lightcurves...")
		mat = np.empty((Nstars, Ntimes), dtype='float64')
		mat.fill(np.nan)
		stds = np.empty(Nstars, dtype='float64')
		
		for k, star in tqdm(enumerate(stars), total=Nstars):
			starid = star['starid']
	
			flux = np.loadtxt(os.path.join('cbv_corrected', 'sector%02d', 'area%d', 'Star%d.corr' % (sector, cbv_area, starid,)), usecols=(1, ), unpack=True)
	
			# Normalize the data and store it in the rows of the matrix:
			mat[k, :] = flux
			stds[k] = np.nanstd(flux) #np.sqrt(star['variance'])
	
		# Only start calculating correlations if we are actually filtering using them:
		file_correlations = 'correlations-postcorr-sector%02d-%d.npy' % (sector, cbv_area)
		if os.path.exists(file_correlations):
			correlations = np.load(file_correlations)
		else:
			# Calculate the correlation matrix between all lightcurves:
			correlations = lc_matrix_calc(Nstars, mat, stds)
			np.save(file_correlations, correlations)
			
			
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.matshow(correlations)
			fig.savefig('correlations-postcorr-sector%02d-%d.png' % (sector, cbv_area))
			plt.close(fig)
	
	
	
		C = {}
		C['Nstars'] = Nstars
		C['Exponent'] = exponent
		expected_mean = 1/np.sqrt(Nstars)
		for k, star in enumerate(stars):
			starid = star['starid']
			C[starid] = nanmean(correlations[k,:]**exponent) - expected_mean
		
		with open(file_median_correlations, 'w') as outfile:
			json.dump(C, outfile)

	return C
	
	
def GOC_wn(ori_flux, corrected_flux):
	# Added white noise
	do = MAD_model(np.diff(ori_flux-np.nanmedian(ori_flux)))
	dc = MAD_model(np.diff(corrected_flux))
	wn_ratio = dc/do
	
	return wn_ratio
	
	
def GOC_corr(filepath_todo):

	# Open the TODO file for that sector:
	conn = sqlite3.connect(filepath_todo)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	# Get list of CBV areas:
	cursor.execute("SELECT DISTINCT cbv_area FROM todolist ORDER BY cbv_area;")
	cbv_areas = [int(row[0]) for row in cursor.fetchall()]
	print(cbv_areas)

	# Loop through the CBV areas:
	# - or run them in parallel - whatever you like!
	for ii, cbv_area in enumerate(cbv_areas):
		GOC_corrmatrix(sector, cbv_area, cursor, exponent=3)
				
		


# =============================================================================
# 
# =============================================================================


def compute_cbvs(filepath_todo, do_plots=True, n_components0=8, single_area=None):
	
	
	# Open the TODO file for that sector:
	conn = sqlite3.connect(filepath_todo)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	# Get list of CBV areas:
	cursor.execute("SELECT DISTINCT cbv_area FROM todolist ORDER BY cbv_area;")
	cbv_areas = [int(row[0]) for row in cursor.fetchall()]
	print(cbv_areas)

	# Loop through the CBV areas:
	# - or run them in parallel - whatever you like!
	for ii, cbv_area in enumerate(cbv_areas):
		print('------------------------------------')
		print('Computing CBV for area%d' %cbv_area)
		
		if not single_area is None:
			if not cbv_area == single_area:
				continue
		
		if do_plots:
			# Remove old plots:
			if os.path.exists("plots/sector%02d/" % sector):
				for f in glob.iglob("plots/sector%02d/*%d.png" % (sector, cbv_area)):
					os.remove(f)		
			else:    
				os.makedirs("plots/sector%02d/" % sector)
				
			# Remove old plots:
			if os.path.exists("plots/sector%02d/stars_area%d/" % (sector, cbv_area)):
				for f in glob.iglob("plots/sector%02d/stars_area%d/*.png" % (sector, cbv_area)):
					os.remove(f)		
			else:    
				os.makedirs("plots/sector%02d/stars_area%d/" % (sector, cbv_area))



		# Extract or compute cleaned and gapfilled light curve matrix
		mat0, priorities, stds, indx_nancol, Ntimes = lc_matrix_clean(sector, cbv_area, cursor)
		# Calculate initial CBVs
		pca0 = PCA(n_components0)
		U0, _, _ = pca0._fit(mat0)
		# Not very clever, but puts NaNs back into the CBVs:
		# For some reason I also choose to transpose the CBV matrix
		cbv0 = np.empty((Ntimes, n_components0), dtype='float64')
		cbv0.fill(np.nan)
		cbv0[~indx_nancol, :] = np.transpose(pca0.components_)
		
		
		print('Cleaning matrix for CBV - remove single dominant contributions')
		# Clean away targets that contribute significantly as a single star to a given CBV (based on entropy)
		mat = clean_cbv(mat0, n_components0, ent_limit=-2, targ_limit=150)
		

		# Calculate the principle components of cleaned matrix
		print("Doing Principle Component Analysis...")
		pca = PCA(n_components0)
		U, _, _ = pca._fit(mat)
		
		
		# Not very clever, but puts NaNs back into the CBVs:
		# For some reason I also choose to transpose the CBV matrix
		cbv = np.empty((Ntimes, n_components0), dtype='float64')
		cbv.fill(np.nan)
		cbv[~indx_nancol, :] = np.transpose(pca.components_)
		
		
		#DO NOT REMOVE CBVS HERE - DO IT BEFORE FITTING
		
#		# Signal-to-Noise test:
		_, indx_lowsnr = cbv_snr_reject(cbv, threshold_snrtest)
#			
#		# Update maximum number of components	
#		n_components = cbv.shape[1]
#		print('New max number of components: ', n_components)
		
		
		
		# Save the CBV to file:
		np.save('cbv-sector%02d-%d.npy' % (sector, cbv_area), cbv)
		
		max_components=20
		n_cbv_components = np.arange(max_components, dtype=int)
		pca_scores = compute_scores(mat, n_cbv_components)
		
		if do_plots:
			# Plot the "effectiveness" of each CBV:
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
			
			fig0.savefig('plots/sector%02d/cbv-perf-area%d.png' % (sector, cbv_area))
			plt.close(fig0)
		

			# Plot all the CBVs:
			fig, axes = plt.subplots(4, 2, figsize=(12, 8))
			fig2, axes2 = plt.subplots(4, 2, figsize=(12, 8))
			fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)  
			fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)  

			for k, ax in enumerate(axes.flatten()):
				ax.plot(cbv0[:, k]+0.1, 'r-')		
				if indx_lowsnr[k]:
					col = 'c'
				else:
					col = 'k'
				ax.plot(cbv[:, k], ls='-', color=col)	
				ax.set_title('Basis Vector %d' % (k+1))
				
				
			for k, ax in enumerate(axes2.flatten()):	
				ax.plot(-np.abs(U0[:, k]), 'r-')
				ax.plot(np.abs(U[:, k]), 'k-')
				ax.set_title('Basis Vector %d' % (k+1))
			#plt.tight_layout()
			fig.savefig('plots/sector%02d/cbvs-area%d.png' % (sector, cbv_area))
			fig2.savefig('plots/sector%02d/U_cbvs-area%d.png' % (sector, cbv_area))
			plt.close('all')
	
	
	
# =============================================================================
# 	
# =============================================================================
	
def cotrend(filepath_todo, do_plots=True, Numcbvs='all', ini=True, use_bic=True, method='powell', single_area=None):
	
	# Open the TODO file for that sector:
	conn = sqlite3.connect(filepath_todo)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	# Get list of CBV areas:
	cursor.execute("SELECT DISTINCT cbv_area FROM todolist ORDER BY cbv_area;")
	cbv_areas = [int(row[0]) for row in cursor.fetchall()]
	print(cbv_areas)

	# Loop through the CBV areas:
	# - or run them in parallel - whatever you like!
	for ii, cbv_area in enumerate(cbv_areas):
		
		if not single_area is None:
			if not cbv_area==single_area:
				continue
			
	#---------------------------------------------------------------------------------------------------------
	# CORRECTING STARS
	#---------------------------------------------------------------------------------------------------------

		print("CORRECTING STARS...")
	
		# Query for all stars, no matter what variability and so on
		cursor.execute("""SELECT todolist.starid,todolist.priority, eclon, eclat, pos_row, pos_column FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE
			datasource='ffi'
			AND cbv_area=?
			AND status=1;""", (cbv_area, ))
		
		stars = cursor.fetchall()
	
		# Load the cbv from file:
		cbv0 = CBV('cbv-sector%02d-%d.npy' % (sector, cbv_area))
		
		# Signal-to-Noise test:
		cbv_snr, indx_lowsnr = cbv_snr_reject(cbv0.cbv, threshold_snrtest)
				
		if ini:
			cbv = cbv0	
		else:
			cbv0.cbv = cbv_snr
			cbv = cbv0
		
		
		# Update maximum number of components	
		n_components0 = cbv.cbv.shape[1]
			
		
		print('New max number of components: ', n_components0)
		
		if Numcbvs=='all':
			n_components = n_components0
		else:	
			n_components = np.min([Numcbvs, n_components0])
			
		print('Fitting using number of components: ', n_components)	
		results = np.zeros([len(stars), n_components+2])
		

		#remove old files
		if ini:
			if os.path.exists("cbv_corrected/sector%02d/area%d/ini/" % (sector, cbv_area)):
				for f in glob.iglob("cbv_corrected/sector%02d/area%d/ini/*.corr" % (sector, cbv_area)):
					os.remove(f)		
			else:    
				os.makedirs("cbv_corrected/sector%02d/area%d/ini/" % (sector, cbv_area))
		else:	
			if os.path.exists("cbv_corrected/sector%02d/area%d" % (sector, cbv_area)):
				for f in glob.iglob("cbv_corrected/sector%02d/area%d/*.corr" % (sector, cbv_area)):
					os.remove(f)		
			else:    
				os.makedirs("cbv_corrected/sector%02d/area%d" % (sector, cbv_area))


		# Remove old plots:
		if ini:
			if os.path.exists("plots/sector%02d/stars_area%d/ini/" % (sector, cbv_area)):
				for f in glob.iglob("plots/sector%02d/stars_area%d/ini/*.png" % (sector, cbv_area)):
					os.remove(f)		
			else:    
				os.makedirs("plots/sector%02d/stars_area%d/ini/" % (sector, cbv_area))
		else:
			# Remove old plots:
			if os.path.exists("plots/sector%02d/stars_area%d/" % (sector, cbv_area)):
				for f in glob.iglob("plots/sector%02d/stars_area%d/*.png" % (sector, cbv_area)):
					os.remove(f)		
			else:    
				os.makedirs("plots/sector%02d/stars_area%d/" % (sector, cbv_area))		

		
	
		for kk, star in tqdm(enumerate(stars), total=len(stars)):
			
			starid = star['starid']
			time, flux, flux_filter, res, residual, WS, pc = cbv.cotrend_single(star, cbv_area, cbv_areas, n_components, ini=ini, use_bic=use_bic, method=method)
			
			
			
			# SAVE TO DIAGNOSTICS FILE::
			wn_ratio = GOC_wn(flux, flux-flux_filter)
			
			#print(residual, wn_ratio)
			res = np.array([res,]).flatten()
			results[kk, 0] = starid
			results[kk, 1:len(res)+1] = res
			
			
			# Plot comparison between clean and corrected data
			data_clean = pd.read_csv(os.path.join('noisy', 'Star%d.noisy' % (starid,)),  usecols=(0, 1), skiprows=6, sep=' ', header=None, names=['Time', 'Flux'])
			time_clean, flux_clean = data_clean['Time'].values, data_clean['Flux'].values
			
	
			if do_plots:
				fig = plt.figure()
				ax1 = fig.add_subplot(211)
				ax1.plot(time, flux)
				ax1.plot(time, flux_filter)
				ax1.plot(time, pc, 'm--')
				ax1.set_xticks([])
				ax2 = fig.add_subplot(212)
				ax2.plot(time, flux/flux_filter-1)
				ax2.plot(time_clean, flux_clean/nanmedian(flux_clean)-1, alpha=0.5)
				ax2.set_xlabel('Time')
				ax2.set_title(str(WS))
				plt.tight_layout()
				if ini:
					fig.savefig('plots/sector%02d/stars_area%d/ini/star%d.png' % (sector, cbv_area, starid))
				else:
					fig.savefig('plots/sector%02d/stars_area%d/star%d.png' % (sector, cbv_area, starid))
				plt.close('all')
				
				
		# Save weights for priors if it is an initial run
		if ini:
			np.savez('mat-sector%02d-%d_free_weights.npz' % (sector, cbv_area), res=results)
		
			if do_plots:
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
				fig.savefig('plots/sector%02d/weights-sector%02d-%d.png' % (sector, sector, cbv_area))
				plt.close('all')
		else:
			np.savez('mat-sector%02d-%d_post_weights.npz' % (sector, cbv_area), res=results)	
			
			
			
#------------------------------------------------------------------------------
	
	

class CBV(object):

	def __init__(self, filepath):
		self.cbv = np.load(filepath)
		
	
	def lsfit(self, flux):
		
		idx = np.isfinite(self.cbv[:,0]) & np.isfinite(flux)
		""" Computes the least-squares solution to a linear matrix equation. """
		A0 = self.cbv[idx,:]
		X = np.column_stack((A0, np.ones(A0.shape[0])))
		F = flux[idx]
		
		C = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(F)
		
		# Another (but slover) implementation
#		C = slin.lstsq(X, flux[idx])[0]
		return C
	
		
	def mdl(self, coeffs):
		coeffs = np.atleast_1d(coeffs)
		m = np.ones(self.cbv.shape[0], dtype='float64')
		for k in range(len(coeffs)-1):
			m += coeffs[k] * self.cbv[:, k]
		return m + coeffs[-1]

	def mdl_off(self, coeff, fitted):
		fitted = np.atleast_1d(fitted)
		m = np.ones(self.cbv.shape[0], dtype='float64')
		for k in range(len(fitted)):
			m += fitted[k] * self.cbv[:, k]
		return m + coeff
	
	def mdl1d(self, coeff, ncbv):
		m = 1 + coeff * self.cbv[:, ncbv]
		return m
	
	def _lhood(self, coeffs, flux):
		return 0.5*nansum((flux - self.mdl(coeffs))**2)
	
	def _lhood_off(self, coeffs, flux, fitted):
		return 0.5*nansum((flux - self.mdl_off(coeffs, fitted))**2)

	def _lhood_off_2(self, coeffs, flux, err, fitted):
		return 0.5*nansum(((flux - self.mdl_off(coeffs, fitted))/err)**2) + 0.5*np.log(err)

	def _lhood1d(self, coeff, flux, ncbv):
		return 0.5*nansum((flux - self.mdl1d(coeff, ncbv))**2)
	
	def _lhood1d_2(self, coeff, flux, err, ncbv):
		return 0.5*nansum(((flux - self.mdl1d(coeff, ncbv))/err)**2) + 0.5*np.log(err)
	
	def _posterior1d(self, coeff, flux, ncbv, cbv_area, Pdict, pos, wscale=5):
		Post = self._lhood1d(coeff, flux, ncbv) + self._prior1d(Pdict, coeff, pos, cbv_area, ncbv, wscale)
		return Post
	
	def _posterior1d_2(self, coeff, flux, err, ncbv, cbv_area, Pdict, pos, wscale=5):
		Post = self._lhood1d_2(coeff, flux, err, ncbv) + self._prior1d(Pdict, coeff, pos, cbv_area, ncbv, wscale)
		return Post
	
	
	def _prior_load(self, cbv_areas, path='', ncbvs=3):
		P = {}
		for ii, cbv_area in enumerate(cbv_areas):
			for jj, ncbv in enumerate(np.arange(1,ncbvs+1)):
				with open('Rbf_area%d_cbv%i.pkl' %(cbv_area,ncbv), 'rb') as file:
					I = dill.load(file)
					P['cbv_area%d_cbv%i' %(cbv_area, ncbv)] = I
				with open('Rbf_area%d_cbv%i_std.pkl' %(cbv_area,ncbv), 'rb') as file:
					Is = dill.load(file)	
					P['cbv_area%d_cbv%i_std' %(cbv_area, ncbv)] = Is
		return P

	def _priorcurve(self, P, x, cbv_area, Ncbvs):
		X = np.array(x)
		res = np.zeros_like(self.cbv[:, 0], dtype='float64')				
		for ncbv in range(Ncbvs):	
			I = P['cbv_area%d_cbv%i' %(cbv_area, ncbv+1)]
			mid = I(X[0],X[1])
			res += self.mdl1d(mid, ncbv) - 1
		return res + 1

	def _prior1d(self, P, c, x, cbv_area, ncbv, wscale=5):
		X = np.array(x)
		I = P['cbv_area%d_cbv%i' %(cbv_area, ncbv+1)]
		Is = P['cbv_area%d_cbv%i_std' %(cbv_area, ncbv+1)]
		# negative log prior
		
		mid = I(X[0],X[1])
		wid = wscale*Is(X[0],X[1])
		Ptot = 0.5*( (c-mid)/ wid)**2 + 0.5*np.log(wid)
		return Ptot
	
	
	def fitting_lh(self, flux, Ncbvs, method='powell'):
		if method=='powell':
			# Initial guesses for coefficients:
			coeffs0 = np.zeros(Ncbvs+1, dtype='float64')
			coeffs0[0] = 1
			
			res = np.zeros(Ncbvs, dtype='float64')				
			for jj in range(Ncbvs):
				res[jj] = minimize(self._lhood1d, coeffs0[jj], args=(flux, jj), method='Powell').x
				
			offset = minimize(self._lhood_off, coeffs0[-1], args=(flux, res), method='Powell').x
			res = np.append(res, offset)	

			return res

		elif method=='llsq':
			res = self.lsfit(flux)
			res[-1] -= 1
			return res
		
	def fitting_pos(self, flux, Ncbvs, cbv_area, Prior_dict, pos, method='powell', wscale=5):
		if method=='powell':
			# Initial guesses for coefficients:
			coeffs0 = np.zeros(Ncbvs+1, dtype='float64')
			coeffs0[0] = 1

			res = np.zeros(Ncbvs, dtype='float64')				
			for jj in range(Ncbvs):	
				res[jj] = minimize(self._posterior1d, coeffs0[jj], args=(flux, jj, cbv_area, Prior_dict, pos, wscale), method='Powell').x
				
			offset = minimize(self._lhood_off, coeffs0[-1], args=(flux, res), method='Powell').x

			res = np.append(res, offset)
			return res

	def fitting_pos_2(self, flux, err, Ncbvs, cbv_area, Prior_dict, pos, method='powell', wscale=5):
		if method=='powell':
			# Initial guesses for coefficients:
			coeffs0 = np.zeros(Ncbvs+1, dtype='float64')
			coeffs0[0] = 1

			res = np.zeros(Ncbvs, dtype='float64')				
			for jj in range(Ncbvs):	
				res[jj] = minimize(self._posterior1d_2, coeffs0[jj], args=(flux, err, jj, cbv_area, Prior_dict, pos, wscale), method='Powell').x
				
			offset = minimize(self._lhood_off_2, coeffs0[-1], args=(flux, err, res), method='Powell').x

			res = np.append(res, offset)
			return res
	

	def fit(self, flux, err=None, pos=None, cbv_area=None, Prior_dict=None, Numcbvs=3, sigma_clip=4.0, maxiter=3, use_bic=True, method='powel', func='pos', wscale=5):

		# Find the median flux to normalise light curve
		median_flux = nanmedian(flux)
		
		if Numcbvs is None:
			Numcbvs = self.cbv.shape[1]   
			
		if use_bic:	
			# Start looping over the number of CBVs to include:
			bic = np.empty(Numcbvs+1, dtype='float64')
			solutions = []
			
			# Test a range of CBVs from 0 to Numcbvs
			Nstart = 0 
		else:
			# Test only fit with Numcbvs
			Nstart = Numcbvs
	
	
		for Ncbvs in range(Nstart, Numcbvs+1):
			
			iters = 0
			fluxi = np.copy(flux) / median_flux 
			while iters <= maxiter:
				iters += 1
				
				# Do the fit:
				if func=='pos':
#					res = self.fitting_pos(fluxi, Ncbvs, cbv_area, Prior_dict, pos, method=method, wscale=5)
					res = self.fitting_pos_2(fluxi, err, Ncbvs, cbv_area, Prior_dict, pos, method=method, wscale=wscale)
				else:
					res = self.fitting_lh(fluxi, Ncbvs, method=method)
					
				
				# Break if nothing changes
				if iters==1:
					d = 1
					res0 = res
				else:
					d = np.sum(res0 - res)
					res0 = res
					if d==0:
						break
					

				flux_filter = self.mdl(res)

				# Do robust sigma clipping:
				absdev = np.abs(fluxi - flux_filter)
				mad = MAD_model(absdev)
				indx = np.greater(absdev, sigma_clip*mad, where=np.isfinite(fluxi))
								
				if np.any(indx):
					fluxi[indx] = np.nan
				else:
					break

			if use_bic:
				# Calculate the Bayesian Information Criterion (BIC) and store the solution:
				bic[Ncbvs] = np.log(np.sum(np.isfinite(fluxi)))*len(res) + self._lhood(res, fluxi)
				solutions.append(res)
			
							

		if use_bic:
			# Use the solution which minimizes the BIC:
			indx = np.argmin(bic)
			res_final = solutions[indx]
			flux_filter = self.mdl(res_final)  * median_flux

		else:
			res_final = res
			flux_filter = self.mdl(res_final)  * median_flux


		return flux_filter, res_final


	def cotrend_single(self, star, cbv_area, cbv_areas, n_components, ini=True, use_bic=False, method='powell'):
	
		starid = star['starid']
		
		data = pd.read_csv(os.path.join('sysnoise', 'Star%d.sysnoise' % (starid,)),  usecols=(0, 1), skiprows=6, sep=' ', header=None, names=['Time', 'Flux'])
		time, flux = data['Time'].values, data['Flux'].values
	
		# Fit the CBV to the flux:
	
		if ini:
			flux_filter, res = self.fit(flux, Numcbvs=n_components, use_bic=False, method='llsq', func='lh')
			
			filename = 'cbv_corrected/sector%02d/area%d/ini/Star%d.corr' % (sector, cbv_area, starid,)
			residual, WS = 0, 0, 0
			
		else:	
	
			Prior_dict = self._prior_load(cbv_areas, ncbvs=n_components)
#			pos = np.array([star['eclon'], star['eclat']])
			pos = np.array([star['pos_row'], star['pos_column']])
			
			
			# Prior curve
			pc = self._priorcurve(Prior_dict, pos, cbv_area, 3) * np.nanmedian(flux)
			
			# Compute new variability measure
			residual = MAD_model(flux-pc)
			residual_ratio = residual/MAD_model(flux-np.nanmedian(flux)) 
			
			WS = np.max([1, 1/residual_ratio])

			if WS>20:
				flux_filter, res = self.fit(flux, Numcbvs=3, use_bic=False, method=method, func='lh')
			else:
				alpha = 1.3
				flux_filter, res = self.fit(flux, err=residual, pos=pos, cbv_area=cbv_area, Prior_dict=Prior_dict, Numcbvs=n_components, use_bic=use_bic, method=method, func='pos', wscale=WS**alpha)
			
			
			
			filename = 'cbv_corrected/sector%02d/area%d/Star%d.corr' % (sector, cbv_area, starid,)
			
			
		np.savetxt(filename, np.column_stack((time, flux/flux_filter - 1, flux_filter)))
		return time, flux, flux_filter, res, residual, WS, pc





#------------------------------------------------------------------------------
if __name__ == '__main__':

	import pandas as pd
	# Pick a sector, any sector....
	sector = 0
	n_components = 8

	# Other settings:
	threshold_variability = 1.3
	threshold_correlation = 0.5
	threshold_snrtest = 5.0
	do_plots = True
	filepath_todo = 'todo.sqlite'
	area = None
	
#	compute_cbvs(filepath_todo, do_plots, single_area=area)
	
#	cotrend(filepath_todo, do_plots=False, Numcbvs='all', use_bic=False, ini=True, single_area=None)
	
#	compute_weight_interpolations(filepath_todo, sector)
	
	cotrend(filepath_todo, do_plots, Numcbvs=3, ini=False, use_bic=False, method='powell', single_area=111)

#	GOC_corr(filepath_todo)

#		sys.exit()


