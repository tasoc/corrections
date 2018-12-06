#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
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
from bottleneck import allnan, nansum, move_median, nanmedian, nanstd
from scipy.optimize import minimize
from scipy.stats import pearsonr, entropy
from scipy.interpolate import pchip_interpolate
import itertools
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from scipy.special import xlogy

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="scipy.stats") # they are simply annoying!

from tqdm import tqdm
plt.ioff()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def MAD_model(x):
	# x: difference between input
	return 1.4826*np.nanmedian(np.abs(x))

#------------------------------------------------------------------------------
def _move_median_central_1d(x, width_points):
	y = move_median(x, width_points, min_count=1)
	y = np.roll(y, -width_points//2+1)
	for k in range(width_points//2+1):
		y[k] = nanmedian(x[:(k+2)])
		y[-(k+1)] = nanmedian(x[-(k+2):])
	return y

#------------------------------------------------------------------------------
def move_median_central(x, width_points, axis=0):
	return np.apply_along_axis(_move_median_central_1d, axis, x, width_points)

#------------------------------------------------------------------------------
def pearson(x, y):
	indx = np.isfinite(x) & np.isfinite(y)
	r, _ = pearsonr(x[indx], y[indx]) # Second output (p-value) is not used
	return r

#------------------------------------------------------------------------------
def compute_scores(X, n_components):
    pca = PCA(svd_solver='full')

    pca_scores = []
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))

    return pca_scores

#------------------------------------------------------------------------------
def rms(x, **kwargs):
	return np.sqrt(nansum(x**2, **kwargs)/len(x))

#------------------------------------------------------------------------------
def compute_entopy(U):

	HGauss0 = 0.5 + 0.5*np.log(2*np.pi)
		
	nSingVals = U.shape[1]
	H = np.empty(nSingVals, dtype='float64')
	
	for iBasisVector in range(nSingVals):
		
		kde = KDE(np.abs(U[:, iBasisVector]))
		kde.fit(gridsize=1000)
		
		pdf = kde.density
		x = kde.support
		
		dx = x[1]-x[0]
		
		# Calculate the Gaussian entropy
		pdfMean = nansum(x * pdf)*dx
		with np.errstate(invalid='ignore'):
			sigma = np.sqrt( nansum(((x-pdfMean)**2) * pdf) * dx )
		HGauss = HGauss0 + np.log(sigma)

		# Calculate vMatrix entropy
		pdf_pos = (pdf>0)
		HVMatrix = -np.sum(xlogy(pdf[pdf_pos], pdf[pdf_pos])) * dx

		# Returned entropy is difference between V-Matrix entropy and Gaussian entropy of similar width (sigma)
		H[iBasisVector] = HVMatrix - HGauss	
		
	return H