#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""KASOC Filter for Asteroseismic Data Preparation - Utility functions

@version: $Revision$
@author:  Rasmus Handberg & Mikkel N. Lund
@date:    $Date$"""

#==============================================================================
# Required Packages:
#==============================================================================
from __future__ import division, with_statement, print_function
import numpy as np
from numpy import zeros_like, diff, append, NaN
from bottleneck import nanmedian, median, move_median, nanmean, nansum, move_mean
from six.moves import range

#==============================================================================
def smooth(x, window):
	"""Calculate moving average of input with given window (number of points)"""
	window = int(window)
	if window <= 1: return x
	if window%2==0: window+=1
	if window >= len(x): return zeros_like(x) + nanmean(x)
	y = move_mean(x, window, min_count=1)
	yny = append(y[window//2:], [NaN]*(window//2))
	for k in range(window//2):
		yny[k] = nanmean(x[:(2*k+1)])
		yny[-(k+1)] = nanmean(x[-(2*k+1):])
	return yny

#==============================================================================
def smooth_cyclic(x, window):
	"""Calculate cyclic moving average of input with given window (number of points)"""
	window = int(window)
	if window <= 1: return x
	if window%2==0: window+=1
	wh = window//2
	if wh >= len(x): return zeros_like(x) + nanmean(x)
	# Stich ends onto the array:
	N = len(x)
	xny = np.concatenate((x[-wh-1:N-1], x, x[1:wh+1]))
	# Smooth the full array:
	y = smooth(xny, window)
	# Cut out the central part again:
	N = len(xny)
	y = y[wh : N-wh]
	return y

#==============================================================================
def _median_central(x, width_points):
	y = move_median(x, width_points, min_count=1)
	yny = append(y[width_points//2:], [NaN]*(width_points//2))
	for k in range(width_points//2):
		yny[k] = nanmedian(x[:(2*k+1)])
		yny[-(k+1)] = nanmedian(x[-(2*k+1):])
	return yny

#==============================================================================
def moving_median(t, x, w, dt=None):
	"""Calculate moving median of input with given window (in t-units)"""

	return moving_nanmedian(t, x, w, dt)

#==============================================================================
def moving_nanmedian(t, x, w, dt=None):
	"""Calculate moving median of input with given window (in t-units)"""
	assert len(t)==len(x), "t and x must have the same length."
	if dt is None: dt = median(diff(t))
	width_points = int(w/dt)
	if width_points <= 1: return x
	if width_points%2==0: width_points += 1
	if width_points >= len(x): return zeros_like(x) + nanmedian(x)
	return _median_central(x, width_points)

#==============================================================================
def moving_nanmedian_cyclic(t, x, w, dt=None):
	"""Calculate cyclic moving average of input with given window (in t-units) taking into account NaNs in the data."""
	assert len(t)==len(x), "t and x must have the same length."
	if dt is None: dt = median(diff(t))
	# Calculate width of filter:
	width_points = int(w/dt)
	if width_points <= 1: return x
	if width_points%2 == 0: width_points += 1 # Filter is much faster when using an odd number of points!
	wh = width_points//2
	N = len(x)
	if wh >= N: return zeros_like(x) + nanmedian(x)
	# Stich ends onto the array:
	xny = np.concatenate((x[-wh-1:N-1], x, x[1:wh+1]))
	# Run moving median on longer series:
	N = len(xny)
	y = _median_central(xny, width_points)
	# Cut out the central part again:
	y = y[wh : N-wh]
	return y

#==============================================================================
def BIC(data, model, dof):
	"""Calculate the Bayesian Information Criterion given vectors of data, model and degrees of freedom"""
	return nansum((data - model)**2) + dof*np.log(len(data))

#==============================================================================
def autocorr(x, dt):
	ac = np.correlate(x, x, mode='full')
	ac = ac[ac.size//2:]
	ac /= ac[0]
	x = np.arange(0, dt*ac.size, dt)
	return x, ac
