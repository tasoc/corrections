#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The ensemble photometry detrending class.

.. codeauthor:: Derek Buzasi
.. codeauthor:: Oliver J. Hall
.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Filipe Pereira
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
#import scipy.interpolate
#import scipy.optimize as sciopt
from copy import deepcopy
from timeit import default_timer
import logging
from scipy.optimize import minimize
#from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors
from .plots import plt
from . import BaseCorrector, STATUS


class EnsembleCorrector(BaseCorrector):
	"""
	DOCSTRING
	"""

	#----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		"""
		Initialize the correction object
		Parameters:
			*args: Arguments for the BaseCorrector class
			**kwargs: Keyword Arguments for the BaseCorrector class
		"""
		super(self.__class__, self).__init__(*args, **kwargs)

		logger = logging.getLogger(__name__)
		self.debug = logger.isEnabledFor(logging.DEBUG)

		# Cache of NearestNeighbors objects that will be filled by the get_nearest_neighbors method:
		self._nearest_neighbors = {}

	#----------------------------------------------------------------------------------------------
	def get_nearest_neighbors(self, lc, n_neighbors):
		"""
		Find the nearest neighbors to the given target in pixel-space.

		Parameters:
			lc (`TESSLightCurve` object): Lightcurve object for target obtained from :func:`load_lightcurve`.
			n_neighbors (integer): Number of targets to return.

		Returns:
			list: List of `priority` identifiers of the `n_neighbors` nearest stars.
				Thest values can be passed directly to :func:`load_lightcurve` to load the lightcurves of the targets.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		camera = lc.camera
		ccd = lc.ccd
		ds = 'ffi' if lc.meta["task"]["datasource"] == 'ffi' else 'tpf'
		key = (ds, camera, ccd)

		# Check if the NearestNeighbors object already exists for this camera and CCD.
		# If not create it, and store it for later use.
		if key not in self._nearest_neighbors:
			# StarID, pixel positions are retrieved from the database:
			select_params = ["todolist.priority", "pos_row", "pos_column"]
			search_params = ["camera={:d}".format(camera), "ccd={:d}".format(ccd), "mean_flux>0"]
			if ds == 'ffi':
				search_params.append("datasource = 'ffi'")
			else:
				search_params.append("datasource != 'ffi'")
			db_raw = self.search_database(select=select_params, search=search_params)
			priority = np.asarray([row['priority'] for row in db_raw], dtype='int64')
			pixel_coords = np.array([[row['pos_row'], row['pos_column']] for row in db_raw])

			# Create the NearestNeighbors object:
			nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
			nn.fit(pixel_coords)

			# Save the
			self._nearest_neighbors[key] = {'nn': nn, 'priority': priority}

		# Get the NearestNeighbor object for the given Camera and CCD:
		nn = self._nearest_neighbors[key]['nn']
		priority = self._nearest_neighbors[key]['priority']

		# Pixel coordinates of the target:
		X = np.array([[lc.meta['task']['pos_row'], lc.meta['task']['pos_column']]])

		# Use the NearestNeighbor object to find the targets closest to the main target:
		distance_index = nn.kneighbors(X, n_neighbors=n_neighbors+1, return_distance=False)
		nearby_stars = priority[distance_index.flatten()]

		# Remove the main target from the list, if it is included:
		indx = (nearby_stars != lc.meta['task']['priority'])
		nearby_stars = nearby_stars[indx]

		# Return the list of nearby stars:
		return nearby_stars

	#----------------------------------------------------------------------------------------------
	def fast_median(self, lc_ensemble):
		"""
		A small utility function for calculating the ensemble median for use in
		correcting the target light curve
		Parameters:
			lc_ensemble: an array-like collection of each light curve in the ensemble
		Returns:
			lc_medians: an array-like (list) that represents the median value of
					each light curve in the ensemble at each cadence
		"""
		lc_medians = []
		col, row = np.asarray(lc_ensemble).shape
		for i in range(row):
			# find the median of the ensemble
			temp =[]
			for j in range(col):
				temp.append(lc_ensemble[j][i])
			lc_medians.append(np.median(temp))

		return lc_medians

	#----------------------------------------------------------------------------------------------
	def do_correction(self, lc):
		"""
		Function that takes all input stars for a sector and uses them to find a detrending
		function using ensemble photometry for a star 'star_names[ifile]', where ifile is the
		index for the star in the star_array and star_names list.

		Parameters:
			lc (``lightkurve.TessLightCurve``): Raw lightcurve stored in a TessLightCurve object.
		
		Returns:
			``lightkurve.TessLightCurve``: Corrected lightcurve stored in a TessLightCurve object.
			``corrections.STATUS``: The status of the correction.
		"""
		
		logger = logging.getLogger(__name__)
		logger.info("Data Source: %s", lc.meta['task']['datasource'])

		# Clean up the lightcurve by removing nans and ignoring data points with bad quality flags
		og_time = lc.time.copy()
		lc = lc.remove_nans()
		lc_quality_mask = (lc.quality == 0)
		# lc.time = lc.time[lc_quality_mask]
		# lc.flux = lc.flux[lc_quality_mask]
		# lc.flux_err = lc.flux_err[lc_quality_mask]

		# Set up basic statistical parameters for the light curves.
		# frange is the light curve range from the 5th to the 95th percentile,
		# drange is the relative standard deviation of the differenced light curve (to whiten the noise)
		frange = (np.percentile(lc.flux, 95) - np.percentile(lc.flux, 5) )/ np.mean(lc.flux)

		drange = np.std(np.diff(lc.flux)) / np.mean(lc.flux)
		lc.meta.update({ 'fstd' : np.std(np.diff(lc.flux)),
						'frange' : frange,
						'drange' : drange})

		logger.info(lc.meta.get("drange"))
		logger.info(" ")

		# Set up start/end times for stellar time series
		#time_start = np.amin(lc.time)
		#time_end = np.max(lc.time)

		# Set minimum range parameter...this is log10 photometric range, and stars more variable than this will be excluded from the ensemble
		min_range = 0.0
		# min_range can be changed later on, so we establish a min_range0 for when we want to reset min_range back to its initial value
		#min_range0 = min_range

		# Define variables to use in the loop to build the ensemble of stars
		# List of star indexes to be included in the ensemble
		temp_list = []
		# Initial number of closest stars to consider and variable to increase number
		star_count = 10
		# (Alternate param) Initial distance at which to consider stars around the target
		# initial_search_radius = -1

		nearby_stars = self.get_nearest_neighbors(lc, 2*star_count)
		logger.debug("Nearby stars: %s", nearby_stars)

		# Start loop to build ensemble
		ensemble_start = default_timer()
		lc_ensemble = []
		target_flux = deepcopy(lc.flux)
		sum_ensemble = np.zeros(len(target_flux)) # to check for a large enough ensemble for dimmer stars
		mtarget_flux = target_flux - np.median(target_flux)

		logger.info(str(np.median(target_flux)))

		# First get a list of indexes of a specified number of stars to build the ensemble
		i = 0
		while len(temp_list) < star_count:
			# Get lightkurve for next star closest to target:
			try:
				next_star_index = nearby_stars[i]
			except IndexError:
				logger.error("Ran out of targets")
				return None, STATUS.ERROR

			logger.debug(next_star_index)
			next_star_lc = self.load_lightcurve(nearby_stars[i]).remove_nans()

			# next_star_lc_quality_mask = (next_star_lc.quality == 0)
			# next_star_lc.time = next_star_lc.time[next_star_lc_quality_mask]
			# next_star_lc.flux = next_star_lc.flux[next_star_lc_quality_mask]
			# next_star_lc.flux_err = next_star_lc.flux_err[next_star_lc_quality_mask]

			# Compute the rest of the statistical parameters for the next star to be added to the ensemble.
			frange = (np.percentile(next_star_lc.flux, 95) - np.percentile(next_star_lc.flux, 5) )/ np.mean(next_star_lc.flux)
			drange = np.std(np.diff(next_star_lc.flux)) / np.mean(next_star_lc.flux)

			next_star_lc.meta.update({ 'fstd' : np.std(np.diff(next_star_lc.flux)),
										'frange' : frange,
										'drange' : drange})

			logger.debug("drange=%f, frange=%f", drange, frange)

			# Stars are added to ensemble if they fulfill the requirements. These are (1) drange less than min_range, (2) drange less than 10 times the
			# drange of the target (to ensure exclusion of relatively noisy stars), and frange less than 0.03 (to exclude highly variable stars)
			if np.log10(drange) < min_range and drange < 10*lc.meta['drange'] and frange < 0.4:

				###################################################################
				# median subtracted flux of target and ensemble candidate
				temp_lc = deepcopy(next_star_lc)
				time_ens = temp_lc.time
				ens_flux = temp_lc.flux
				if self.debug:
					plt.plot(time_ens, ens_flux)
					plt.show()

				mens_flux = ens_flux - np.median(ens_flux)

				# 2 sigma
				ens2sig = 2 * np.std(mens_flux)
				targ2sig = 2 * np.std(mtarget_flux)

				# absolute balue
				abstarg = np.absolute(mtarget_flux)
				absens = np.absolute(mens_flux)

				logger.info("2 sigma")
				logger.info(str(ens2sig) + " , " + str(targ2sig))

				# sigma clip the flux used to fit, but don't use that flux again
				clip_target_flux = np.where(
					np.where(abstarg < targ2sig, True, False)
					&
					np.where(absens < ens2sig, True, False),
					mtarget_flux, 1)
				clip_ens_flux = np.where(
					np.where(abstarg < targ2sig, True, False)
					&
					np.where(absens < ens2sig, True, False),
					mens_flux, 1)

				logger.info(str(np.median(target_flux)))
				logger.info(str(np.median(ens_flux)))

				#this is where I'll try adding the background correction portion
				#first get scaled target flux
				#scale_target_flux = clip_target_flux/np.median(target_flux)

				args = tuple((clip_ens_flux+np.median(ens_flux),clip_target_flux+np.median(target_flux)))

				def func1(scaleK, *args):
					temp = (((args[0]+scaleK)/np.median(args[0]+scaleK))-1)-((args[1]/np.median(args[1]))-1)
					temp = (args[1]/np.median(args[1])) - ((args[0]+scaleK)/np.median(args[0]+scaleK))
					temp = temp - np.median(temp)
					return np.sum(np.square(temp))

				scale0 = 100
				res = minimize(func1, scale0, args, method='Powell')

				logger.info("Fit param1: {}".format(res.x))
				logger.info(str(np.median(ens_flux)))
				logger.info(str(res.x))

				ens_flux = ens_flux+res.x
				mens_flux = mens_flux+res.x
				clip_ens_flux = clip_ens_flux+res.x

				if self.debug:
					plt.plot(ens_flux/np.median(ens_flux))
					plt.show(block=True)
				next_star_lc.flux = (ens_flux/np.median(ens_flux))
				temp_list.append([next_star_index, next_star_lc.copy()])
				lc_ensemble.append(ens_flux/np.median(ens_flux))
				sum_ensemble = sum_ensemble + np.array(ens_flux)
				logger.info(str(next_star_lc.targetid)+'\n')
				###################################################################
			i += 1

		logger.info("Build ensemble, Time: %f", default_timer()-ensemble_start)

		lc_medians = self.fast_median(lc_ensemble)
		#lc_medians = np.nanmedian(lc_ensemble, axis=2)

		if self.debug:
			plt.plot(lc.time, lc_medians)
			plt.plot(lc.time, (lc.flux/np.median(lc.flux)))
			plt.show(block=True)
			plt.plot(lc.time, lc_medians)
			plt.show(block=True)
		lc_medians = np.asarray(lc_medians)


		args = tuple((lc.flux, lc_medians))
		def func2(scalef,*args):
			num1 = np.sum(np.abs(np.diff(np.divide(args[0],args[1]+scalef))))
			denom1 = np.median(np.divide(args[0],args[1]+scalef))
			return num1/denom1

		scale0 = 1.0
		res = minimize(func2, scale0, args)
		k_corr = res.x

		logger.info("Fit param: {}".format(res.x))

		#fitf2 = 1+((lc_medians-1)*res.x)

		# Correct the lightcurve
		lc_corr = lc.copy()

		median_only_flux = np.divide(lc_corr.flux, lc_medians)
		lc_corr /= k_corr + lc_medians
		lc_corr /= np.median(lc_corr.flux)
		lc_corr *= np.median(lc.flux)

		if self.debug:
			plt.figure()
			plt.scatter(lc_corr.time, median_only_flux, marker='.', label="Median Only")
			plt.scatter(lc_corr.time, lc_corr.flux, marker='.', label="Corrected LC")
			plt.show()
		#######################################################################################################

		# We probably want to return additional information, including the list of stars in the ensemble, and potentially other things as well.
		logger.info(temp_list)

		# Replace removed points with NaN's so the info can be saved to the FITS
		lc_corr.time = lc_corr.time[lc_quality_mask]
		lc_corr.flux = lc_corr.flux[lc_quality_mask]
		lc_corr.flux_err = lc_corr.flux_err[lc_quality_mask]
		if len(lc_corr.flux) != len(og_time):
			fix_flux = np.asarray(lc_corr.flux.copy())
			indices = np.array(np.where(np.isin(og_time, lc_corr.time, assume_unique=True, invert=True)))[0]
			indices.tolist()

			for ind in indices:
				fix_flux = np.insert(fix_flux, ind, np.nan)

			lc_corr.flux = fix_flux.tolist()
			lc_corr.time = og_time



		if self.plot:
			ax = lc.plot(marker='o', label="Original LC")
			lc_corr.plot(ax=ax, color='orange', marker='o', ls='--', label="Corrected LC")
			plt.show()
			if self.debug:
				#plt.savefig("./temp/" + str(lc.targetid) + "_testrun.png")
				logger.info(np.nanstd(lc.flux))
				logger.info(np.nanstd(lc_corr.flux))
				logger.info(np.nanmedian(lc.flux))
				logger.info(np.nanmedian(lc_corr.flux))

		return lc_corr, STATUS.OK
