#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
import os
import logging
from sklearn.decomposition import PCA
from bottleneck import allnan, nansum, nanmedian
from scipy.optimize import minimize, fmin_powell
from scipy.interpolate import pchip_interpolate

import scipy.signal as sig
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="scipy.stats") # they are simply annoying!
from ..utilities import loadPickle
from .cbv_util import compute_entopy, MAD_model,MAD_model2
from ..quality import CorrectorQualityFlags

from ..plots import plt, matplotlib

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

#------------------------------------------------------------------------------
def clean_cbv(Matrix, n_components, ent_limit=-1.5, targ_limit=50):
	logger = logging.getLogger(__name__)

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

	return correlations

#------------------------------------------------------------------------------
class CBV(object):
	"""
	Cotrending Basis Vector.

	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	#--------------------------------------------------------------------------
	def __init__(self, data_folder, cbv_area, threshold_snrtest=5):
		logger = logging.getLogger(__name__)
		
		filepath = os.path.join(data_folder, 'cbv-%d.npy' % cbv_area)
		filepath_s = os.path.join(data_folder, 'cbv-s-%d.npy' % cbv_area)
		self.cbv = np.load(filepath)
		self.cbv_s = np.load(filepath_s)
		
		# Signal-to-Noise test (without actually removing any CBVs):
		indx_lowsnr = cbv_snr_test(self.cbv, threshold_snrtest)
		self.remove_cols(indx_lowsnr)
				
		self.priors = None
		priorpath = os.path.join(data_folder, 'D_area%d.pkl' %cbv_area)
		if os.path.exists(priorpath):
			self.priors = loadPickle(priorpath)
		else:
			logger.info('Path to prior distance file does not exist', priorpath)
		
		self.inires = None
		inipath = os.path.join(data_folder, 'mat-%d_free_weights.npz' %cbv_area)
		if os.path.exists(inipath):
			self.inires = np.load(inipath)['res']
			
			
		

	#--------------------------------------------------------------------------
	def remove_cols(self, indx_lowsnr):
		self.cbv = self.cbv[:, ~indx_lowsnr]
		self.cbv_s = self.cbv_s[:, ~indx_lowsnr]

	#--------------------------------------------------------------------------
	def lsfit(self, flux, Ncbvs):
		"""
		Computes the least-squares solution to a linear matrix equation.
		"""
		idx = np.isfinite(self.cbv[:,0]) & np.isfinite(flux)
#		A0 = self.cbv[idx,:] 
#		A0 = np.column_stack((self.cbv[idx,:], self.cbv_s[idx,:])) 
		A0 = np.column_stack((self.cbv[idx,:Ncbvs], self.cbv_s[idx,:Ncbvs])) 
		
		X = np.column_stack((A0, np.ones(A0.shape[0])))
		F = flux[idx]

		C = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(F)


		# Another (but slover) implementation
#		C = slin.lstsq(X, flux[idx])[0]
		return C
	
	#--------------------------------------------------------------------------
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

	#--------------------------------------------------------------------------
	def mdl(self, coeffs):
		coeffs = np.atleast_1d(coeffs)
		m = np.ones(self.cbv.shape[0], dtype='float64')
		Ncbvs = int((len(coeffs)-1)/2)
		
		for k in range(Ncbvs):
			m += (coeffs[k] * self.cbv[:, k]) + (coeffs[k+Ncbvs] * self.cbv_s[:, k])
			
		return m + coeffs[-1]

	#--------------------------------------------------------------------------
	def mdl_spike(self, coeffs):
		coeffs = np.atleast_1d(coeffs)
		m = np.ones(self.cbv.shape[0], dtype='float64')		
		for k in range(len(coeffs)-1):
			m += (coeffs[k] * self.cbv_s[:, k])
			
		return m + coeffs[-1]
	
	#--------------------------------------------------------------------------
	def mdl_off(self, coeff, fitted, Ncbvs):
		fitted = np.atleast_1d(fitted)
		
		# Start with ones as the flux is median normalised
		m = np.ones(self.cbv.shape[0], dtype='float64')
		for k in range(Ncbvs):
			m += (fitted[k] * self.cbv[:, k]) + (fitted[k+Ncbvs] * self.cbv_s[:, k])
		return m + coeff

	#--------------------------------------------------------------------------
	def mdl1d(self, coeff, ncbv):
		cbv_comb = np.column_stack((self.cbv, self.cbv_s))

		m = 1 + coeff * cbv_comb[:, ncbv]
		return m

	#--------------------------------------------------------------------------
#	def _lhood(self, coeffs, flux, err):
#		return 0.5*nansum(((flux - self.mdl(coeffs))/err)**2)

	#--------------------------------------------------------------------------
	def _lhood_off(self, coeffs, flux, fitted, Ncbvs):
		return 0.5*nansum((flux - self.mdl_off(coeffs, fitted, Ncbvs))**2)

	#--------------------------------------------------------------------------
#	def _lhood_off_2(self, coeffs, flux, err, fitted):
#		return 0.5*nansum(((flux - self.mdl_off(coeffs, fitted))/err)**2) + 0.5*np.log(err**2)

	#--------------------------------------------------------------------------
	def _lhood1d(self, coeff, flux, ncbv):
		return 0.5*nansum((flux - self.mdl1d(coeff, ncbv))**2)

	#--------------------------------------------------------------------------
#	def _lhood1d_2(self, coeff, flux, err, ncbv):
#		return 0.5*nansum(((flux - self.mdl1d(coeff, ncbv))/err)**2) + 0.5*np.log(err**2)

	#--------------------------------------------------------------------------
	def _posterior1d(self, coeff, flux, ncbv, pos, wscale, KDE):
		Post = self._lhood1d(coeff, flux, ncbv) - wscale*np.log(self._prior1d(coeff, KDE))
		return Post


	#--------------------------------------------------------------------------
#	def _posterior1d_2(self, coeff, flux, err, ncbv, pos, wscale, KDE):
#		Post = self._lhood1d_2(coeff, flux, err, ncbv) + self._prior1d(coeff, wscale, KDE)
#		return Post


	#--------------------------------------------------------------------------
	def _priorcurve(self, x, Ncbvs, N_neigh):
#		cbv_comb = np.column_stack((self.cbv, self.cbv_s))

#		res = np.zeros_like(cbv_comb[:, 0], dtype='float64')
		res = np.zeros_like(self.cbv[:, 0], dtype='float64')
		opts = np.zeros(int(Ncbvs*2)) #entries for both CBV and CBV_S
		
		no_cbv_coeff = self.cbv.shape[1]
		plt.figure()
		
		tree = self.priors
		dist, ind = tree.query(np.array([x]), k=N_neigh+1)
		W = 1/dist[0][1::]**2
		
		for ncbv in range(Ncbvs):
			
#			V = self.inires[:,1+ncbv][ind][0][1::]
			V = self.inires[ind,1+ncbv][0][1::] #/10
			VS = self.inires[ind,1+ncbv + no_cbv_coeff][0][1::] #/10
#			print(V[0:5])
			
			KDE = stats.gaussian_kde(V, weights=W.flatten(), bw_method='scott')
			KDES = stats.gaussian_kde(VS, weights=W.flatten(), bw_method='scott')
			
			VP = np.linspace(V.min()-0.5, V.max()+0.5, 100)
#			plt.plot(VP, KDE.evaluate(VP))
			plt.plot(VP, [-1*KDE.logpdf(c)*0.4 for c in VP])

			
			VPS = np.linspace(VS.min()-0.5, VS.max()+0.5, 100)
			plt.plot(VPS, [-1*KDES.logpdf(c)*0.4 for c in VPS], '--')
			
			plt.show()
			
			def kernel_opt(x): return -1*KDE.logpdf(x)
			opt = fmin_powell(kernel_opt, 0, disp=0)
			opts[ncbv] = opt
			
			def kernel_opts(x): return -1*KDES.logpdf(x)
			opt_s = fmin_powell(kernel_opts, 0, disp=0)
			opts[ncbv + Ncbvs] = opt_s
			
#			print(ncbv+1, opt)
#			I = self.priors['cbv%i' %(ncbv+1)]
#			mid = I(X[0],X[1])
			res += (self.mdl1d(opt, ncbv) - 1) + (self.mdl1d(opt_s, ncbv + no_cbv_coeff) - 1)
		return res+1, opts 

	#--------------------------------------------------------------------------
	def _prior1d(self, c, KDE=None):
#		X = np.array(x)
#		tree = self.priors['cbv%i' %(ncbv+1)]
##		Is = self.priors['cbv%i_std' %(ncbv+1)]
#		# negative log prior
#		
#		dist, ind = tree.query(np.array([X]), k=N_neigh+1)
#		V = VALS[ind][0][1::]
#		W = 1/dist[0][1::]
#		
#		KDE = stats.gaussian_kde(V, weights=W.flatten(), bw_method='scott')
		
		if KDE is None:
			return 0
		else:
#			print(KDE.logpdf(c), wscale, KDE.logpdf(c)**wscale)
#			Ptot = -1*(KDE.logpdf(c)**wscale)
#			Ptot = (KDE.logpdf(c)**wscale)
			return float(KDE(c))
#			print(Ptot, KDE.logpdf(c))
		
#		mid = I(X[0],X[1])
#		wid = wscale*Is(X[0],X[1])
#		Ptot = 0.5*( (c-mid)/ wid)**2 + 0.5*np.log(wid)

	#--------------------------------------------------------------------------
	def fitting_lh(self, flux, Ncbvs, method='llsq'): #, Ncbvs, 
#		if method=='powell':
#			# Initial guesses for coefficients:
#			coeffs0 = np.zeros(Ncbvs+1, dtype='float64')
#			coeffs0[0] = 1
#
#			res = np.zeros(Ncbvs, dtype='float64')
#			for jj in range(Ncbvs):
#				res[jj] = minimize(self._lhood1d, coeffs0[jj], args=(flux, jj), method='Powell').x
#
#			offset = minimize(self._lhood_off, coeffs0[-1], args=(flux, res), method='Powell').x
#			res = np.append(res, offset)
#
#			return res

		if method=='llsq':
			res = self.lsfit(flux, Ncbvs)
			res[-1] -= 1
			return res
		
	#--------------------------------------------------------------------------
	def fitting_lh_spike(self, flux, Ncbvs, method='llsq'): #, Ncbvs, 

		if method=='llsq':
			res = self.lsfit_spike(flux, Ncbvs)
			res[-1] -= 1
			return res	


	#--------------------------------------------------------------------------
	def fitting_pos_2(self, flux, err, Ncbvs, pos, wscale, N_neigh, method='Powell', start=None):
		if (method=='Powell') or (method=='Nelder-Mead'):
			# Initial guesses for coefficients:
			if not start is None:
				coeffs0 = start
				coeffs0 = np.append(coeffs0, 0)
#				coeffs0 = np.zeros(Ncbvs*2+1, dtype='float64')

			else:	
				coeffs0 = np.zeros(Ncbvs*2+1, dtype='float64')
#				coeffs0[-1] = 0


			no_cbv_coeff = self.cbv.shape[1]
				
			res = np.zeros(int(Ncbvs*2), dtype='float64')
			tree = self.priors
			dist, ind = tree.query(np.array([pos]), k=N_neigh+1)
			W = 1/dist[0][1::]**2
			

			# TEST with independent spike fit
			ff = pchip_interpolate(np.arange(len(flux))[np.isfinite(flux)], flux[np.isfinite(flux)], np.arange(len(flux)))
			F = sig.medfilt(ff, 15)
			flux2 = np.copy(flux)/F
			res_s = self.fitting_lh_spike(flux2, Ncbvs, method='llsq')
			print('res_s', res_s)
			res[Ncbvs:] = res_s[:-1]
			spike_filt = self.mdl_spike(res_s)
			
#			plt.figure()
#			plt.plot(flux2)
#			plt.plot(spike_filt-0.5)
			
			
			flux1 = F*(flux2 - spike_filt + 1)
			
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
				V = self.inires[ind,1+jj][0][1::] 
				KDE = stats.gaussian_kde(V, weights=W.flatten(), bw_method='scott')
				
#				VS = self.inires[ind,1+jj + no_cbv_coeff][0][1::] 
#				KDES = stats.gaussian_kde(VS, weights=W.flatten(), bw_method='scott')

#				res[jj] = minimize(self._posterior1d_2, coeffs0[jj], args=(flux, err, jj, pos, wscale, KDE), method='Powell').x
				
				res[jj] = minimize(self._posterior1d, coeffs0[jj], args=(flux1, jj, pos, wscale, KDE), method=method).x				
				# Using KDE prior:
#				res[jj + Ncbvs] = minimize(self._posterior1d, coeffs0[jj + Ncbvs], args=(flux, jj + no_cbv_coeff, pos, wscale, KDES), method=method).x
				# Using flat prior
#				res[jj + Ncbvs] = minimize(self._posterior1d, coeffs0[jj + Ncbvs], args=(flux1, jj + no_cbv_coeff, pos, wscale, None), method=method).x

#			offset = minimize(self._lhood_off_2, coeffs0[-1], args=(flux, err, res), method='Powell').x
			offset = minimize(self._lhood_off, coeffs0[-1], args=(flux, res, Ncbvs), method=method).x

			res = np.append(res, offset)
#			print('res', res)
			return res

	#--------------------------------------------------------------------------
	def fit(self, flux, err=None, pos=None, Numcbvs=3, sigma_clip=4.0, maxiter=50, use_bic=True, method='Powell', func='pos', wscale=5, N_neigh=1000, start=None):

		# Find the median flux to normalise light curve
		median_flux = nanmedian(flux)

		if Numcbvs is None:
#			cbv_comb = np.column_stack((, self.cbv_s))
			Numcbvs = self.cbv.shape[1]
#			Numcbvs = self.cbv.shape[1] + self.cbv_s.shape[1]

		if use_bic:
			# Start looping over the number of CBVs to include:
#			bic = np.empty(Numcbvs+1, dtype='float64')
			bic = np.array([]) #np.full(Numcbvs+1, np.inf, dtype='float64')
			solutions = []#np.array([])
#			print('BIC start', bic)

			# Test a range of CBVs from 1 to Numcbvs
			# Fitting at least 1 CBV and an offset
			Nstart = 1
		else:
			# Test only fit with Numcbvs
			Nstart = Numcbvs
			
			
#		print('wscale', wscale)
#		print('start', start)
			
		for Ncbvs in range(Nstart, Numcbvs+1):
			
			iters = 0
			fluxi = np.copy(flux) / median_flux
			
			while iters <= maxiter:
				iters += 1

				# Do the fit:
				if func=='pos':					
					# Reuse results from iterations with fewer CBVs??? 
					res = self.fitting_pos_2(fluxi, err, Ncbvs, pos, wscale, N_neigh, method=method, start=start)
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
				indx = np.greater(absdev, sigma_clip*mad, where=np.isfinite(absdev))

				if np.any(indx):
					fluxi[indx] = np.nan
				else:
					break

			if use_bic:
				# Calculate the Bayesian Information Criterion (BIC) and store the solution:
				filt = self.mdl(res)  * median_flux
#				bic[Ncbvs] = np.log(np.sum(np.isfinite(fluxi)))*len(res) + nansum( ((flux - filt)/err)**2 )
#				bic = np.append(bic, np.log(np.sum(np.isfinite(fluxi)))*len(res) + nansum( ((flux - filt)/err)**2 ))
				bic = np.append(bic, np.log(np.sum(np.isfinite(fluxi)))*len(res) + nansum( (flux - filt)**2 )) #not using errors
				solutions.append(res)

		if use_bic:
			print('bic', bic)
			print('iters', iters)
			# Use the solution which minimizes the BIC:
			indx = np.argmin(bic)
			print('optimum number of CBVs', indx+1)
			res_final = solutions[indx]
#			print('final res', res_final)
			flux_filter = self.mdl(res_final)  * median_flux

		else:
			res_final = res
			flux_filter = self.mdl(res_final)  * median_flux


		return flux_filter, res_final

	#--------------------------------------------------------------------------
	def cotrend_single(self, lc, n_components, alpha=1.3, WS_lim=0.5, ini=True, use_bic=False, method='Powell', N_neigh=1000):
		logger = logging.getLogger(__name__)
		# Remove bad data based on quality
		
		flag_good = CorrectorQualityFlags.filter(lc.quality)
		lc.flux[~flag_good] = np.nan


		# Fit the CBV to the flux:
		if ini:
			flux_filter, res = self.fit(lc.flux, Numcbvs=n_components, use_bic=False, method='llsq', func='lh')
			return flux_filter, res

		else:
			row = lc.meta['task']['pos_row']
			col = lc.meta['task']['pos_column']
			tmag = np.clip(lc.meta['task']['tmag'], 2, 20)
			
			pos = np.array([row, col, tmag])

			# Prior curve
			pc0, opts = self._priorcurve(pos, n_components, N_neigh)
			pc = pc0 * lc.meta['task']['mean_flux']

			# Compute new variability measure
			idx = np.isfinite(lc.flux)
			Poly = np.poly1d(np.polyfit(lc.time[idx], lc.flux[idx], 3))
			Polyfit = Poly(lc.time)
			residual = MAD_model(lc.flux-pc)
#			residual_ratio = MAD_model(lc.flux-lc.meta['task']['mean_flux'])/residual
#
#			print('residual', residual)
#			print('residual_ratio', residual_ratio)
#
#			WS = np.min([1, residual_ratio])
#			
			
			AA = 2
			GRAW = np.std((pc - Polyfit) / MAD_model2(lc.flux - Polyfit) - 1)
			GPR = 0 + (1 - (GRAW/AA)**2)*(GRAW<AA)
			
			beta1 = 1
			beta2 = 1
			VAR = np.nanstd(lc.flux - Polyfit)
			WS = np.min([1, (VAR**beta1)*(GPR**beta2)])
			
			print(GRAW, GPR, VAR, WS)

#			lc.meta['additional_headers']['rratio'] = (residual_ratio, 'residual ratio')
#			lc.meta['additional_headers']['var_new'] = (residual, 'new variability')

			tree = self.priors
			dist, ind = tree.query(np.array([pos]), k=N_neigh+1)
			
			print('LLSQ res', self.inires[ind,:][0][0])
			print('dist', dist[0])

			WS=0
			if WS==0:
				logger.info('Fitting using LSSQ')
#				flux_filter, res = self.fit(lc.flux, Numcbvs=n_components, use_bic=True, method='llsq', func='lh') #use smaller number of CBVs
				flux_filter, res = self.fit(lc.flux, Numcbvs=14, use_bic=True, method='llsq', func='lh') #use smaller number of CBVs
				lc.meta['additional_headers']['pri_use'] = (False, 'Was prior used')
			else:
				logger.info('Fitting using Priors')
				flux_filter, res = self.fit(lc.flux, err=residual, pos=pos, Numcbvs=n_components, use_bic=True, method=method, func='pos', wscale=WS, N_neigh=N_neigh, start=opts)
				lc.meta['additional_headers']['pri_use'] = (True, 'Was prior used')

			return flux_filter, res, residual, WS, pc
