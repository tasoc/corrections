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
import os.path
from bottleneck import nanmedian, nanstd, nansum, ss
from timeit import default_timer
import logging
from scipy.optimize import minimize # minimize_scalar
from sklearn.neighbors import NearestNeighbors
import copy
from .plots import plt, save_figure
from .quality import TESSQualityFlags
from . import BaseCorrector, STATUS

#--------------------------------------------------------------------------------------------------
class EnsembleCorrector(BaseCorrector):

	#----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		"""
		Initialize the correction object.
		"""
		super().__init__(*args, **kwargs)

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
				These values can be passed directly to :func:`load_lightcurve` to load the
				lightcurves of the targets.

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
		distance_index = nn.kneighbors(X, n_neighbors=min(n_neighbors+1, len(priority)), return_distance=False)
		nearby_stars = priority[distance_index.flatten()]

		# Remove the main target from the list, if it is included:
		indx = (nearby_stars != lc.meta['task']['priority'])
		nearby_stars = nearby_stars[indx]

		# Return the list of nearby stars:
		return nearby_stars

	#----------------------------------------------------------------------------------------------
	def add_ensemble_member(self, lc, next_star_lc):
		"""
		Add a given target to the ensemble list

		Parameters:
			lc (`TESSLightCurve` object): Lightcurve for target obtained
				from :func:`load_lightcurve`.
			next_star_lc (`TESSLightCurve` object): Lightcurve for star to add to ensemble
				obtained from :func:`load_lightcurve`.

		Returns:
			ndarray: Lightcurve (flux) to add to ensemble.

		.. codeauthor:: Lindsey Carboneau
		.. codeauthor:: Derek Buzasi
		"""
		# Median subtracted flux of target and ensemble candidate
		target_flux_median = lc.meta['task']['mean_flux']
		mtarget_flux = lc.flux - target_flux_median
		ens_flux = next_star_lc.flux
		mens_flux = ens_flux - nanmedian(ens_flux)

		# 2 sigma
		ens2sig = 2 * nanstd(mens_flux)
		targ2sig = 2 * nanstd(mtarget_flux)

		# absolute balue
		abstarg = np.absolute(mtarget_flux)
		absens = np.absolute(mens_flux)

		# sigma clip the flux used to fit, but don't use that flux again
		clip_target_flux = np.where(
			np.where(abstarg < targ2sig, True, False) & np.where(absens < ens2sig, True, False),
			mtarget_flux, 1)

		clip_ens_flux = np.where(
			np.where(abstarg < targ2sig, True, False) & np.where(absens < ens2sig, True, False),
			mens_flux, 1)

		args = tuple((clip_ens_flux + nanmedian(ens_flux), clip_target_flux + target_flux_median))

		def func1(scaleK, *args):
			#temp = (((args[0]+scaleK)/np.median(args[0]+scaleK))-1)-((args[1]/np.median(args[1]))-1)
			temp = (args[1]/nanmedian(args[1])) - ((args[0]+scaleK)/nanmedian(args[0]+scaleK))
			temp -= nanmedian(temp)
			return ss(temp)

		res = minimize(func1, 100, args=args, method='Powell')

		ens_flux = ens_flux + res.x

		return ens_flux/nanmedian(ens_flux)

	#----------------------------------------------------------------------------------------------
	def apply_ensemble(self, lc, lc_ensemble, lc_corr):
		"""
		Apply the ensemble correction method to the target light curve

		Parameters:
			lc (`TESSLightCurve` object): Lightcurve object for target obtained
				from :func:`load_lightcurve`.
			lc_ensemble (list): List of ensemble members flux as ndarrays
			lc_corr (`TESSLightCurve` object): Lightcurve object which stores the ensemble
				corrected flux values.

		Returns:
			lc_corr (`TESSLightCurve` object): The updated object with corrected flux values.

		.. codeauthor:: Lindsey Carboneau
		.. codeauthor:: Derek Buzasi
		"""
		# Calculate the median of all the ensemble lightcurves for each timestamp:
		lc_medians = nanmedian(np.asarray(lc_ensemble), axis=0)

		def func2(scalef, *args):
			num1 = nansum(np.abs(np.diff(args[0] / (args[1] + scalef))))
			denom1 = nanmedian(args[0] / (args[1] + scalef))
			return num1/denom1

		res = minimize(func2, 1.0, args=(lc.flux, lc_medians))
		k_corr = res.x

		# Correct the lightcurve:
		lc_corr /= k_corr + lc_medians
		lc_corr /= nanmedian(lc_corr.flux)
		lc_corr *= nanmedian(lc.flux)
		return lc_corr

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

		# Settings:
		drange_lim = 1.0 # Limit on differenced range - not in log10!
		drange_relfactor = 10 # Limit on differenced range, relative to target d.range.
		frange_lim = 0.4 # Limit on flux range.
		min_star_count = 5 # Minimum number of stars to have in the ensemble.
		max_neighbors = 100 # Maximal number of neighbors to check.

		# Clean up the lightcurve by removing nans and ignoring data points with bad quality flags
		# these values need to be removed, or they will affect the ensemble later
		og_time = lc.time.copy()
		lc = lc.remove_nans()
		lc_quality_mask = TESSQualityFlags.filter(lc.quality, TESSQualityFlags.HARDEST_BITMASK)
		lc = lc[lc_quality_mask]
		lc_corr = lc.copy()

		# Set up basic statistical parameters for the light curves.
		# frange is the light curve range from the 5th to the 95th percentile,
		# drange is the relative standard deviation of the differenced light curve (to whiten the noise)
		target_flux_median = lc.meta['task']['mean_flux']
		target_frange = np.diff(np.nanpercentile(lc.flux, [5, 95])) / target_flux_median
		target_drange = nanstd(np.diff(lc.flux)) / target_flux_median

		logger.debug("Main target: drange=%f, frange=%f", target_drange, target_frange)
		logger.debug("lc size: %f", len(lc.flux))

		# Query for a list of the nearest neigbors to test:
		nearby_stars = self.get_nearest_neighbors(lc, max_neighbors)
		logger.debug("Nearby stars: %s", nearby_stars)

		# Define variables to use in the loop to build the ensemble of stars
		# List of star indexes to be included in the ensemble
		temp_list = []
		lc_ensemble = []

		# Test values store current best correction for testing vs. additional ensemble members
		# NOTE: this might be done more efficiently with further functionalization of the loop below and updated loop bounds/conditions!
		test_corr = []
		test_ens = []
		test_list = []
		fom = np.nan

		# Start loop to build ensemble
		ensemble_start = default_timer()

		# Loop through the neighbors to build the ensemble:
		for next_star_index in nearby_stars:
			# Get lightkurve for next star closest to target:
			next_star_lc = self.load_lightcurve(next_star_index).remove_nans()

			next_star_lc_quality_mask = (next_star_lc.quality == 0)
			next_star_lc = next_star_lc[next_star_lc_quality_mask]

			# Compute the rest of the statistical parameters for the next star to be added to the ensemble.
			frange = np.diff(np.nanpercentile(next_star_lc.flux, [5, 95])) / next_star_lc.meta['task']['mean_flux']
			drange = nanstd(np.diff(next_star_lc.flux)) / next_star_lc.meta['task']['mean_flux']

			logger.debug("drange=%f, frange=%f", drange, frange)
			logger.debug("lc size: %f", len(next_star_lc.flux))

			# Stars are added to ensemble if they fulfill the requirements. These are (1) drange less than min_range, (2) drange less than 10 times the
			# drange of the target (to ensure exclusion of relatively noisy stars), and frange less than 0.03 (to exclude highly variable stars)
			if drange < drange_lim and drange < drange_relfactor*target_drange and frange < frange_lim:

				# Add the star to the ensemble:
				lc_add_ensemble = self.add_ensemble_member(lc, next_star_lc)
				temp_list.append({'priority:': next_star_index, 'starid': next_star_lc.targetid})
				lc_ensemble.append(lc_add_ensemble)

				# Pause the loop if we have reached the desired number of stars, and check the correction:
				if len(temp_list) >= min_star_count:

					if np.isnan(fom):
						# the first time we hit the minimum ensemble size, try to correct the target
						# storing all these as 'test' - we'll revert to these values if adding the next star doesn't 'surpass the test'
						# `deepcopy` because otherwise it uses pointers and overwrites the value we need
						test_corr = self.apply_ensemble(lc, lc_ensemble, lc_corr)
						test_ens = copy.deepcopy(lc_ensemble)
						test_list = copy.deepcopy(temp_list)
						test_fom = np.sum(np.abs(np.diff(lc_corr.flux)))
						fom = copy.deepcopy(test_fom)

						########
						# NOTE: I think this can be more efficiently, but I'm scared to 'fix' what is currently working - LC
						########
					else:
						# see if one more member makes the ensemble 'surpass the test'
						lc_corr = self.apply_ensemble(lc, lc_ensemble, lc_corr)
						fom = np.sum(np.abs(np.diff(lc_corr.flux)))

						if fom < test_fom:
							# the correction "got better" (less noisy) so update and try against another additional member
							test_corr = lc_corr
							test_ens = lc_ensemble
							test_list = temp_list
							test_fom = fom
						else:
							# adding the next neighbor did not improve the correction, so we're done
							lc_corr = test_corr
							lc_ensemble = test_ens
							temp_list = test_list
							break

		# Ensure that we reached the minimum number of stars in the ensemble:
		if len(temp_list) < min_star_count:
			logger.error("Not enough stars for ensemble")
			return None, STATUS.ERROR

		logger.info("Build ensemble, Time: %f", default_timer()-ensemble_start)
		logger.debug("len(lc) vs len(ens): %f vs %f", len(lc.flux), len(lc_ensemble[0]))

		# Convert to parts-per-million:
		lc_corr = 1e6*(lc_corr/nanmedian(lc_corr.flux) - 1)

		# We probably want to return additional information, including the list of stars in the ensemble, and potentially other things as well.
		logger.info(temp_list)
		self.ensemble_starlist = {
			'starids': [tl['starid'] for tl in temp_list]
		}

		# Set additional headers for FITS output:
		lc_corr.meta['additional_headers']['ENS_NUM'] = (len(temp_list), 'Number of targets in ensemble')
		lc_corr.meta['additional_headers']['ENS_DLIM'] = (drange_lim, 'Limit on differenced range metric')
		lc_corr.meta['additional_headers']['ENS_DREL'] = (drange_relfactor, 'Limit on relative diff. range')
		lc_corr.meta['additional_headers']['ENS_RLIM'] = (frange_lim, 'Limit on flux range metric')

		# Replace removed points with NaN's so the info can be saved to the FITS
		if len(lc_corr.flux) != len(og_time):
			fix_flux = np.asarray(lc_corr.flux.copy())
			fix_err = np.asarray(lc_corr.flux_err.copy())
			indices = np.array(np.where(np.isin(og_time, lc_corr.time, assume_unique=True, invert=True)))[0]
			indices.tolist()

			for ind in indices:
				fix_flux = np.insert(fix_flux, ind, np.nan)
				fix_err = np.insert(fix_err, ind, np.nan)

			lc_corr.flux = fix_flux.tolist()
			lc_corr.flux_err = fix_err.tolist()
			lc_corr.time = og_time

		#------------------------------------------------------------------------------------------
		if self.plot and self.debug:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.plot(lc.time, nanmedian(np.asarray(lc_ensemble), axis=0), label='Medians')
			save_figure(os.path.join(self.plot_folder(lc), 'ensemble_lc_medians'), fig=fig)
			plt.close(fig)

		return lc_corr, STATUS.OK
