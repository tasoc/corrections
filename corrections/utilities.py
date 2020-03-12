#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of utility functions that can be used throughout
the corrections package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import pickle
import gzip
import logging
from astropy.io import fits
from bottleneck import nanmedian, nanmean, allnan
from scipy.stats import binned_statistic

# Constants:
mad_to_sigma = 1.482602218505602 #: Conversion constant from MAD to Sigma. Constant is 1/norm.ppf(3/4)

PICKLE_DEFAULT_PROTOCOL = 4 #: Default protocol to use for saving pickle files.

#------------------------------------------------------------------------------
def savePickle(fname, obj):
	"""
	Save an object to file using pickle.

	Parameters:
		fname (string): File name to save to. If the name ends in '.gz' the file
			will be automatically gzipped.
		obj (object): Any pickalble object to be saved to file.
	"""

	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'wb') as fid:
		pickle.dump(obj, fid, protocol=PICKLE_DEFAULT_PROTOCOL)

#------------------------------------------------------------------------------
def loadPickle(fname):
	"""
	Load an object from file using pickle.

	Parameters:
		fname (string): File name to save to. If the name ends in '.gz' the file
			will be automatically unzipped.
		obj (object): Any pickalble object to be saved to file.

	Returns:
		object: The unpickled object from the file.
	"""

	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'rb') as fid:
		return pickle.load(fid)

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
def rms_timescale(lc, timescale=3600/86400):
	"""
	Compute robust RMS on specified timescale. Using MAD scaled to RMS.

	Parameters:
		lc (``lightkurve.TessLightCurve`` object): Timeseries to calculate RMS for.
		timescale (float, optional): Timescale to bin timeseries before calculating RMS. Default=1 hour.

	Returns:
		float: Robust RMS on specified timescale.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if len(lc.flux) == 0 or allnan(lc.flux):
		return np.nan
	if len(lc.time) == 0 or allnan(lc.time):
		raise ValueError("Invalid time-vector specified. No valid timestamps.")

	time_min = np.nanmin(lc.time)
	time_max = np.nanmax(lc.time)
	if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max - time_min <= 0:
		raise ValueError("Invalid time-vector specified")

	# Construct the bin edges seperated by the timescale:
	bins = np.arange(time_min, time_max, timescale)
	bins = np.append(bins, time_max)

	# Bin the timeseries to one hour:
	indx = np.isfinite(lc.time) & np.isfinite(lc.flux)
	flux_bin, _, _ = binned_statistic(lc.time[indx], lc.flux[indx], nanmean, bins=bins)

	# Compute robust RMS value (MAD scaled to RMS)
	return mad_to_sigma * nanmedian(np.abs(flux_bin - nanmedian(flux_bin)))

#--------------------------------------------------------------------------------------------------
def ptp(lc):
	"""
	Compute robust Point-To-Point scatter.

	Parameters:
		lc (``lightkurve.TessLightCurve`` object): Lightcurve to calculate PTP for.

	Returns:
		float: Robust PTP.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	if len(lc.flux) == 0 or allnan(lc.flux):
		return np.nan
	if len(lc.time) == 0 or allnan(lc.time):
		raise ValueError("Invalid time-vector specified. No valid timestamps.")
	return nanmedian(np.abs(np.diff(lc.flux)))

#--------------------------------------------------------------------------------------------------
def fix_fits_table_headers(table, column_titles=None):
	"""
	Fix headers in FITS files, adding appropiate comments where they are missing.

	Changes headers in-place.

	Parameters:
		table (``fits.BinTableHDU``): Table HDU to fix headers for.
		column_titles (dict): Descriptions of columns to add to header comments.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if isinstance(table, fits.BinTableHDU):
		table = table.header

	for k in range(1, table.get('TFIELDS', 0)+1):
		if not table.comments['TDISP%d' % k]:
			table.comments['TDISP%d' % k] = 'column display format'

		if not table.comments['TFORM%d' % k]:
			table.comments['TFORM%d' % k] = {
				'D': 'column format: 64-bit floating point',
				'E': 'column format: 32-bit floating point',
				'K': 'column format: signed 64-bit integer',
				'J': 'column format: signed 32-bit integer',
				'L': 'column format: logical value'
			}[table.get('TFORM%d' % k)]

		if column_titles is not None:
			key = table.get('TTYPE%d' % k)
			if key and key in column_titles:
				table.comments['TTYPE%d' % k] = 'column title: ' + column_titles[key]

#--------------------------------------------------------------------------------------------------
class ListHandler(logging.Handler):
	"""
	A logging.Handler that writes messages into a list object.

	The standard logging.QueueHandler/logging.QueueListener can not be used
	for this because the QueueListener runs in a private thread, not the
	main thread.

	.. warning::
		This handler is not thread-safe. Do not use it in threaded environments.
	"""

	def __init__(self, *args, message_queue, **kwargs):
		"""Initialize by copying the queue and sending everything else to superclass."""
		logging.Handler.__init__(self, *args, **kwargs)
		self.message_queue = message_queue

	def emit(self, record):
		"""Add the formatted log message (sans newlines) to the queue."""
		self.message_queue.append(self.format(record).rstrip('\n'))
