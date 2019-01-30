#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from bottleneck import allnan, nansum, move_median, nanmedian, nanstd, nanmean
from scipy.optimize import minimize
from scipy.stats import pearsonr, entropy
from scipy.interpolate import pchip_interpolate
from scipy.signal import correlate
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from scipy.special import xlogy
import scipy.linalg as slin
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="scipy.stats") # they are simply annoying!
from tqdm import tqdm
import time as TIME
from ..utilities import loadPickle
from .cbv_util import compute_entopy, _move_median_central_1d, move_median_central, compute_scores, rms, MAD_model
#from .cbv_weights import compute_weight_interpolations

plt.ioff()


#------------------------------------------------------------------------------
def cbv_snr_test(cbv_ini, threshold_snrtest=5.0):
	logger=logging.getLogger(__name__)

#	A_signal = rms(cbv_ini, axis=0)
#	A_noise = rms(np.diff(cbv_ini, axis=0), axis=0)

	A_signal = MAD_model(cbv_ini, axis=0)
	A_noise = MAD_model(np.diff(cbv_ini, axis=0), axis=0)

	snr = 10 * np.log10( A_signal**2 / A_noise**2 )
	logger.info("SNR threshold used %s", threshold_snrtest)
	logger.info("SNR (dB) of CBVs %s", snr)

	indx_lowsnr = (snr < threshold_snrtest)

	# Never throw away first CBV
	indx_lowsnr[0] = False

	return indx_lowsnr
#	if np.any(indx_lowsnr):
#		logger.info("Rejecting %d CBVs based on SNR test" % np.sum(indx_lowsnr))
#		cbv = cbv_ini[:, ~indx_lowsnr]
#		return cbv, indx_lowsnr
#	else:
#		return cbv_ini, None


#------------------------------------------------------------------------------
def clean_cbv(Matrix, n_components, ent_limit=-1.5, targ_limit=50):
	logger=logging.getLogger(__name__)


	# Calculate the principle components:
	logger.info("Doing Principle Component Analysis...")
	pca = PCA(n_components)
	U, _, _ = pca._fit(Matrix)

	Ent = compute_entopy(U)
	logger.info('Entropy start: ' + str(Ent))

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

	logger.info('Entropy end:'  + str(Ent))
	logger.info('Targets removed ' + str(int(targets_removed)))
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
def lc_matrix_calc(Nstars, mat0):#, stds):
	logger=logging.getLogger(__name__)

	logger.info("Calculating correlations...")

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


class CBV(object):

	def __init__(self, filepath):
		self.cbv = np.load(filepath)

	def remove_cols(self, indx_lowsnr):
		self.cbv = self.cbv[:, ~indx_lowsnr]

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

	def _lhood(self, coeffs, flux, err):
		return 0.5*nansum(((flux - self.mdl(coeffs))/err)**2)

	def _lhood_off(self, coeffs, flux, fitted):
		return 0.5*nansum((flux - self.mdl_off(coeffs, fitted))**2)

	def _lhood_off_2(self, coeffs, flux, err, fitted):
		return 0.5*nansum(((flux - self.mdl_off(coeffs, fitted))/err)**2) + 0.5*np.log(err**2)

	def _lhood1d(self, coeff, flux, ncbv):
		return 0.5*nansum((flux - self.mdl1d(coeff, ncbv))**2)

	def _lhood1d_2(self, coeff, flux, err, ncbv):
		return 0.5*nansum(((flux - self.mdl1d(coeff, ncbv))/err)**2) + 0.5*np.log(err**2)

	def _posterior1d(self, coeff, flux, ncbv, cbv_area, Pdict, pos, wscale=5):
		Post = self._lhood1d(coeff, flux, ncbv) + self._prior1d(Pdict, coeff, pos, cbv_area, ncbv, wscale)
		return Post

	def _posterior1d_2(self, coeff, flux, err, ncbv, cbv_area, Pdict, pos, wscale=5):
		Post = self._lhood1d_2(coeff, flux, err, ncbv) + self._prior1d(Pdict, coeff, pos, cbv_area, ncbv, wscale)
		return Post


	def _prior_load(self, cbv_area, data_path, ncbvs=3):
		P = {}
		for jj, ncbv in enumerate(np.arange(1,ncbvs+1)):
			P['cbv_area%d_cbv%i' %(cbv_area, ncbv)] = loadPickle(os.path.join(data_path, 'Rbf_area%d_cbv%i.pkl' %(cbv_area,ncbv)))
			P['cbv_area%d_cbv%i_std' %(cbv_area, ncbv)] = loadPickle(os.path.join(data_path, 'Rbf_area%d_cbv%i_std.pkl' %(cbv_area,ncbv)))
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

			# Test a range of CBVs from 1 to Numcbvs
			# Fitting at least 1 CBV and an offset
			Nstart = 1
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
#				print(sigma_clip*mad, absdev, flux_filter)
				indx = np.greater(absdev, sigma_clip*mad, where=np.isfinite(absdev))

				if np.any(indx):
					fluxi[indx] = np.nan
				else:
					break

			if use_bic:
				# Calculate the Bayesian Information Criterion (BIC) and store the solution:
				filt = self.mdl(res)  * median_flux
				bic[Ncbvs] = np.log(np.sum(np.isfinite(fluxi)))*len(res) + nansum( ((flux - filt)/err)**2 )
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


	#--------------------------------------------------------------------------

	def cotrend_single(self, lc, n_components, data_path, alpha=1.3, WS_lim=20, ini=True, use_bic=False, method='powell'):

		cbv_area = lc.meta['task']['cbv_area']

		# Remove bad data based on quality
		quality_remove = 1 #+...
		flag_removed = (lc.quality & quality_remove != 0)
		lc.flux[flag_removed] = np.nan


		# Fit the CBV to the flux:
		if ini:
			flux_filter, res = self.fit(lc.flux, Numcbvs=n_components, use_bic=False, method='llsq', func='lh')
			return flux_filter, res

		else:

			Prior_dict = self._prior_load(cbv_area, data_path, ncbvs=n_components)

			#TODO: add option to use other coordinates
			row = lc.meta['task']['pos_row']+(lc.meta['task']['ccd']>2)*2048
			col = lc.meta['task']['pos_column']+(lc.meta['task']['ccd']%2==0)*2048
			pos = np.array([row, col])
#			pos = np.array([lc.centroid_row, lc.centroid_col])

			# Prior curve
			pc = self._priorcurve(Prior_dict, pos, cbv_area, n_components) * np.nanmedian(lc.flux)

			# Compute new variability measure
			residual = MAD_model(lc.flux-pc)
			residual_ratio = residual/MAD_model(lc.flux-np.nanmedian(lc.flux))

			WS = np.max([1, 1/residual_ratio])

			lc.meta['additional_headers']['rratio'] = (residual_ratio, 'residual ratio')
			lc.meta['additional_headers']['var_new'] = (residual, 'new variability')


			if WS>WS_lim:
				flux_filter, res = self.fit(lc.flux, Numcbvs=np.min([n_components, 3]), use_bic=False, method=method, func='lh')
				lc.meta['additional_headers']['pri_use'] = ('No', 'Was prior used')
			else:
				flux_filter, res = self.fit(lc.flux, err=residual, pos=pos, cbv_area=cbv_area, Prior_dict=Prior_dict, Numcbvs=n_components, use_bic=use_bic, method=method, func='pos', wscale=WS**alpha)
				lc.meta['additional_headers']['pri_use'] = ('Yes', 'Was prior used')

			return flux_filter, res, residual, WS, pc





#------------------------------------------------------------------------------
if __name__ == '__main__':

#	import pandas as pd
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

#	cotrend(filepath_todo, do_plots, Numcbvs=3, ini=False, use_bic=False, method='powell', single_area=111)

#	GOC_corr(filepath_todo)

#		sys.exit()


