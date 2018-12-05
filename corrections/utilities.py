#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of utility functions that can be used throughout
the corrections package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
from bottleneck import nanmedian, nanmean
from scipy.stats import binned_statistic

# Constants:
mad_to_sigma = 1.482602218505602 # Constant is 1/norm.ppf(3/4)

#------------------------------------------------------------------------------
def sphere_distance(ra1, dec1, ra2, dec2):
	"""
	Calculate the great circle distance between two points using the Vincenty formulae.

	Parameters:
		ra1 (float or ndarray): Longitude of first point in degrees.
		dec1 (float or ndarray): Lattitude of first point in degrees.
		ra2 (float or ndarray): Longitude of second point in degrees.
		dec2 (float or ndarray): Lattitude of second point in degrees.

	Returns:
		ndarray: Distance between points in degrees.

	Note:
		https://en.wikipedia.org/wiki/Great-circle_distance
	"""

	# Convert angles to radians:
	ra1 = np.deg2rad(ra1)
	ra2 = np.deg2rad(ra2)
	dec1 = np.deg2rad(dec1)
	dec2 = np.deg2rad(dec2)

	# Calculate distance using Vincenty formulae:
	return np.rad2deg(np.arctan2(
		np.sqrt( (np.cos(dec2)*np.sin(ra2-ra1))**2 + (np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2 ),
		np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
	))

#------------------------------------------------------------------------------
def rms_timescale(time, flux, timescale=3600/86400):
	"""
	Compute robust RMS on specified timescale. Using MAD scaled to RMS.

	Parameters:
		time (ndarray): Timestamps in days.
		flux (ndarray): Flux to calculate RMS for.
		timescale (float, optional): Timescale to bin timeseries before calculating RMS. Default=1 hour.

	Returns:
		float: Robust RMS on specified timescale.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Construct the bin edges seperated by the timescale:
	bins = np.arange(np.nanmin(time), np.nanmax(time), timescale)
	bins = np.append(bins, np.nanmax(time))

	# Bin the timeseries to one hour:
	flux_bin, _, _ = binned_statistic(time, flux, nanmean, bins=bins)

	# Compute robust RMS value (MAD scaled to RMS)
	return mad_to_sigma * nanmedian(np.abs(flux_bin - nanmedian(flux_bin)))