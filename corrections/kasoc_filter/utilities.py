#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KASOC Filter for Asteroseismic Data Preparation - Utility functions

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

import numpy as np
from numpy import zeros_like, diff, append, NaN
from bottleneck import nanmedian, median, move_median, nanmean, nansum, move_mean

#==============================================================================
def smooth(x, window):
	"""Calculate moving average of input with given window (number of points)"""
	window = int(window)
	if window <= 1: return x
	if window % 2 == 0: window += 1
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
	if window % 2 == 0: window += 1
	wh = window//2
	if wh >= len(x): return zeros_like(x) + nanmean(x)
	# Stitch ends onto the array:
	N = len(x)
	xny = np.concatenate((x[-wh-1:N-1], x, x[1:wh+1]))
	# Smooth the full array:
	y = smooth(xny, window)
	# Cut out the central part again:
	N = len(xny)
	y = y[wh:N-wh]
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
	assert len(t) == len(x), "t and x must have the same length."
	if dt is None: dt = median(diff(t))
	width_points = int(w/dt)
	if width_points <= 1: return x
	if width_points % 2 == 0: width_points += 1
	if width_points >= len(x): return zeros_like(x) + nanmedian(x)
	return _median_central(x, width_points)

#==============================================================================
def moving_nanmedian_cyclic(t, x, w, dt=None):
	"""Calculate cyclic moving average of input with given window (in t-units) taking into account NaNs in the data."""
	assert len(t) == len(x), "t and x must have the same length."
	if dt is None: dt = median(diff(t))
	# Calculate width of filter:
	width_points = int(w/dt)
	if width_points <= 1: return x
	if width_points % 2 == 0: width_points += 1 # Filter is much faster when using an odd number of points!
	wh = width_points//2
	N = len(x)
	if wh >= N: return zeros_like(x) + nanmedian(x)
	# Stich ends onto the array:
	xny = np.concatenate((x[-wh-1:N-1], x, x[1:wh+1]))
	# Run moving median on longer series:
	N = len(xny)
	y = _median_central(xny, width_points)
	# Cut out the central part again:
	y = y[wh:N-wh]
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

#==============================================================================
def theil_sen(x, y, n_samples=1e5):
	"""
	Computes the Theil-Sen estimator for 2d data

	This complexity is O(n**2), which can be poor for large n. We will perform a sampling
	of data points to get an unbiased, but larger variance estimator.
	The sampling will be done by picking two points at random, and computing the slope,
	up to n_samples times.

	Parameters:
		x: 1-d np array, the control variate
		y: 1-d np.array, the ind variate.
		n_samples: how many points to sample.
	"""

	assert x.shape[0] == y.shape[0], "x and y must be the same shape."
	n = x.shape[0]

	i1 = np.random.randint(0, n, n_samples)
	i2 = np.random.randint(0, n, n_samples)
	slopes = _slope( x[i1], x[i2], y[i1], y[i2] )

	slope_ = nanmedian( slopes )
	#find the optimal b as the median of y_i - slope*x_i
	intercepts = np.empty(n, dtype=float)
	for i in range(n):
		intercepts[i] = y[i] - slope_*x[i]
	intercept_ = median( intercepts )

	return np.array([slope_, intercept_])

def _slope(x_1, x_2, y_1, y_2):
	return (1 - 2*(x_1 > x_2)) * ((y_2 - y_1)/np.abs((x_2 - x_1)))

#==============================================================================
def gap_fill(t, y,maxgap=np.inf):
	# Declare variables used:
	times_max = 0
	D = np.diff(t)
	time_tot = list([])
	data_tot = list([])
	ori_or_not = list([])

	# Calculate the desired regular step size:
	step = median(D)
	stepcut = 1.5*step

	# test:
	if not np.isinf(maxgap):
		times_max = int((maxgap/2)/step)+1

	for i in range(len(t)-1):
		# Add the original point:
		time_tot.append(t[i])
		data_tot.append(y[i])
		ori_or_not.append(1)

		d = D[i]

		if d > maxgap:
			# Insert half the maximum number of points in the beginning and end of gap:
			for j in range(1, times_max):
				time_tot.append(t[i]+j*step)
				data_tot.append(np.NaN)
				ori_or_not.append(0)
			# Insert half the maximum number of points in the beginning and end of gap:
			for j in range(times_max,0,-1):
				time_tot.append(t[i+1]-j*step)
				data_tot.append(np.NaN)
				ori_or_not.append(0)
		elif d > stepcut:
			# Calculate the number of points to be inserted and insert them:
			times = int(d/step)-1
			for j in range(times):
				time_tot.append(t[i]+(j+1)*step)
				data_tot.append(np.NaN)
				ori_or_not.append(0)

	# Special treatment of last point:
	time_tot.append(t[-1])
	data_tot.append(y[-1])
	ori_or_not.append(1)

	return np.array(time_tot), np.array(data_tot), np.array(ori_or_not, dtype=bool)
