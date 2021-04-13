#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation of Cotrending Basis Vectors.

The CBV creation has three major steps, which are wrapped in the :class:`CBVCreator` class:

1. The CBVs for the specific todo-list are computed using the :func:`CBVCreator.compute_cbv` function.
2. CBVs are split into "single-scale" CBVs and "spike" CBVs using the :func:`CBVCreator.spike_sep` function.
3. An initial fitting is performed for all targets using linear least squares using the :func:`CBVCreator.cotrend_ini` function.
   This is done to obtain fitting coefficients for the CBVs that will be used to form priors for the final fit.
4. Priors are constructed using the output from step 3, using the :func:`CBVCreator.compute_weight_interpolations` function.
   This function saves interpolation functions for each of the CBV coefficient priors.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os
import logging
import h5py
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric, BallTree
from bottleneck import allnan, nanmedian, replace
from scipy.interpolate import pchip_interpolate, interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import gaussian_kde
from tqdm import tqdm
from ..plots import plt
from .. import BaseCorrector, STATUS
from ..utilities import savePickle, mad_to_sigma
from ..quality import CorrectorQualityFlags, TESSQualityFlags
from ..version import get_version
from .cbv import CBV, cbv_snr_test
from .cbv_utilities import MAD_model2, compute_scores, lightcurve_correlation_matrix, compute_entropy

__version__ = get_version(pep440=False)

#--------------------------------------------------------------------------------------------------
def create_cbv(sector, cbv_area, input_folder=None, cadence='ffi', version=6, ncbv=16,
	threshold_correlation=0.5, threshold_snrtest=5.0, threshold_entropy=-0.5, ip=False):
	"""
	Create CBV for given area.

	It is required that the :class:`corrections.TaskManager` has been initialized on the
	``input_dir`` at least ones before the function is called, since this will
	ensure that the proper database columns and indicies have been created.

	Parameters:
		sector (int): TESS Sector.
		cbv_area (int):
		input_folder (str):
		cadence (str, optional): Default='ffi'.
		version (int): Version to add to output files.
		ncbv (int, optional):
		threshold_correlation (float, optional):
		threshold_snrtest (float, optional):
		threshold_entropy (float, optional):
		ip (bool, optional): Default=False.

	Returns:
		:class:`CBV`: CBV object.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.info('Running CBV for SECTOR=%d, AREA=%d', sector, cbv_area)

	with CBVCreator(input_folder, cadence=cadence, sector=sector, cbv_area=cbv_area,
		threshold_correlation=threshold_correlation, threshold_snrtest=threshold_snrtest,
		threshold_entropy=threshold_entropy, ncomponents=ncbv) as C:

		C.compute_cbvs()
		C.spike_sep()
		C.cotrend_ini(do_ini_plots=ip)
		#C.compute_distance_map()

		# Convert to CBV object and save to FITS:
		cbv_ffi = CBV(C.hdf_filepath)
		cbv_ffi.save_to_fits(C.data_folder, version=version)

		# Interpolate FFI CBVs to higher cadences:
		if cadence == 'ffi':
			# Create new HDF5 file with higher cadence,
			# and save to FITS file as well:
			newfile = C.interpolate_to_higher_cadence(120)
			cbv_120s = CBV(newfile)
			cbv_120s.save_to_fits(C.data_folder, version=version)

			# For later sectors, also create 20s cadence CBVs:
			if sector >= 27:
				newfile = C.interpolate_to_higher_cadence(20)
				cbv_20s = CBV(newfile)
				cbv_20s.save_to_fits(C.data_folder, version=version)

		# Return CBV object for the generated area:
		return cbv_ffi

#--------------------------------------------------------------------------------------------------
class CBVCreator(BaseCorrector):
	"""
	Creation of Cotrending Basis Vectors.

	Attributes:
		sector (int): TESS Sector.
		cadence (int): TESS observing cadence in seconds.
		cbv_area (int):
		datasource (str):
		ncomponents (int): Number of CBVs to be created.
		threshold_variability (float):
		threshold_correlation (float):
		threshold_snrtest (float):
		threshold_entropy (float):
		hdf (:class:`h5py.File`):
		hdf_filepath (str): Path to the HDF5 file containing the CBV.
		cbv_plot_folder (str):
		random_state (int):

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
	"""

	def __init__(self, *args, cadence='ffi', sector=None, cbv_area=None, ncomponents=16,
		threshold_correlation=0.5, threshold_snrtest=5.0, threshold_variability=1.3,
		threshold_entropy=-0.5, **kwargs):
		"""
		Initialize the CBV Creator.

		Parameters:
			sector (int, required): TESS Sector.
			cbv_area (int, required):
			cadence (int or str, optional): TESS observing cadence in seconds.
			ncomponents (int, optional): Number of CBVs to be created.
			threshold_variability (float, optional):
			threshold_correlation (float, optional):
			threshold_snrtest (float, optional):
			threshold_entropy (float, optional):

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		# Call the parent initializing:
		# This will set several default settings
		super().__init__(*args, **kwargs)

		overwrite = False

		# Open logger for displaying information:
		logger = logging.getLogger(__name__)

		# Basic input checks:
		if not isinstance(sector, int):
			self.close()
			raise ValueError("Invalid SECTOR")
		if not isinstance(cadence, int) and cadence != 'ffi':
			self.close()
			raise ValueError("Invalid CADENCE")
		if not isinstance(cbv_area, int):
			self.close()
			raise ValueError("Invalid CBV_AREA")
		if not isinstance(ncomponents, int) or ncomponents <= 0:
			self.close()
			raise ValueError("Invalid NCOMPONENTS")
		if threshold_correlation is None:
			threshold_correlation = 1.0
		elif threshold_correlation <= 0 or threshold_correlation > 1:
			self.close()
			raise ValueError("Invalid THRESHOLD_CORRELATION")

		# Store input settings:
		self.sector = sector
		self.cbv_area = cbv_area
		self.ncomponents = ncomponents
		self.threshold_variability = threshold_variability
		self.threshold_correlation = threshold_correlation
		self.threshold_snrtest = threshold_snrtest
		self.threshold_entropy = threshold_entropy
		self.random_state = 2187

		# Lookup FFI cadence:
		self.cursor.execute("SELECT cadence FROM todolist WHERE sector=? AND datasource='ffi' LIMIT 1;", [sector])
		ffi_cadence = self.cursor.fetchone()['cadence']
		if cadence == 'ffi':
			self.cadence = ffi_cadence
		else:
			self.cadence = cadence

		# Only for backward compatibility:
		self.datasource = 'ffi' if self.cadence == ffi_cadence else 'tpf'

		# Path to the HDF5 file which will contain all the information for a set of CBVs:
		self.hdf_filepath = os.path.join(self.data_folder, f'cbv-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.hdf5')

		# If the file already extsts, determine if it was created using the same settings:
		if os.path.exists(self.hdf_filepath):
			with h5py.File(self.hdf_filepath, 'r') as hdf:
				# If any of these are different, we should start from scratch:
				start_over = False
				#if hdf.attrs.get('version') != __version__:
				#	logger.error("Existing CBV created with another VERSION")
				#	start_over = True
				if hdf.attrs.get('method') != 'normal': # pragma: no cover
					logger.error("Existing CBV created with another METHOD")
					start_over = True
				if hdf.attrs['Ncbvs'] != self.ncomponents:
					logger.error("Existing CBV created with different NCOMPONENTS")
					start_over = True
				if hdf.attrs['threshold_variability'] != self.threshold_variability:
					logger.error("Existing CBV created with different THRESHOLD_VARIABILITY")
					start_over = True
				if hdf.attrs['threshold_correlation'] != self.threshold_correlation:
					logger.error("Existing CBV created with different THRESHOLD_CORRELATION")
					start_over = True
				if hdf.attrs['threshold_snrtest'] != self.threshold_snrtest:
					logger.error("Existing CBV created with different THRESHOLD_SNRTEST")
					start_over = True
				if hdf.attrs['threshold_entropy'] != self.threshold_entropy:
					logger.error("Existing CBV created with different THRESHOLD_ENTROPY")
					start_over = True

			# If we need to start over, we simply delete the existing file:
			if start_over and overwrite: # pragma: no cover
				os.remove(self.hdf_filepath)
			elif start_over:
				self.close()
				raise ValueError("Mismatch between existing file and provided settings")

		# Store wheter the file already exists:
		file_is_new = not os.path.exists(self.hdf_filepath)

		# Open the HDF5 file for storing the resulting CBVs:
		self.hdf = h5py.File(self.hdf_filepath, 'a', libver='latest')

		# Save all settings in the attributes of the root of the HDF5 file:
		if file_is_new:
			self.hdf.attrs['method'] = 'normal'
			self.hdf.attrs['datasource'] = self.datasource
			self.hdf.attrs['cbv_area'] = self.cbv_area
			self.hdf.attrs['cadence'] = self.cadence
			self.hdf.attrs['sector'] = self.sector
			self.hdf.attrs['version'] = __version__
			self.hdf.attrs['Ncbvs'] = self.ncomponents
			self.hdf.attrs['threshold_variability'] = self.threshold_variability
			self.hdf.attrs['threshold_correlation'] = self.threshold_correlation
			self.hdf.attrs['threshold_snrtest'] = self.threshold_snrtest
			self.hdf.attrs['threshold_entropy'] = self.threshold_entropy
			self.hdf.flush()

		# Create directory for plots:
		self.cbv_plot_folder = os.path.join(self.data_folder, 'plots')
		os.makedirs(self.cbv_plot_folder, exist_ok=True)

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close the CBV Creator object."""
		self._close_basecorrector()
		if hasattr(self, 'hdf') and self.hdf:
			self.hdf.close()
			self.hdf = None

	#----------------------------------------------------------------------------------------------
	def lightcurve_matrix(self):
		"""
		Load matrix filled with light curves.

		The steps performed are the following:

		#. Only targets with a variability below a threshold are included.

		#. Computes correlation matrix for light curves in a given cbv-area and only includes the
		   :meth:`threshold_correlation` most correlated light curves.

		#. Performs gap-filling of light curves and removes time stamps where all flux values are NaN.

		Returns:
			tuple:

			- :class:`numpy.ndarray`: matrix of light curves to be used in CBV calculation.
			- :class:`numpy.ndarray`: the indices for the timestamps with nans in all light curves.
			- `int`: Number of timestamps.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		tqdm_settings = {'disable': not logger.isEnabledFor(logging.INFO)}

		logger.info('Running matrix clean')
		if logger.isEnabledFor(logging.DEBUG) and 'matrix' in self.hdf: # pragma: no cover
			logger.info("Loading existing file...")
			return self.hdf['matrix'], self.hdf['nancol'], self.hdf.attrs['Ntimes']

		logger.info("We are running CBV_AREA=%d", self.cbv_area)

		# Set up search parameters for database:
		search_params = [
			f'status={STATUS.OK.value:d}', # Only including targets with status=OK from photometry
			"method_used='aperture'", # Only including aperature photometry targets
			f'cadence={self.cadence:d}',
			f'cbv_area={self.cbv_area:d}',
			f'sector={self.sector:d}'
		]

		# Find the median of the variabilities:
		variability = np.array([float(row['variability']) for row in self.search_database(search=search_params, select='variability')], dtype='float64')
		if len(variability) == 0:
			raise ValueError("No lightcurves found for this CBV_AREA that have VARIABILITY defined")
		median_variability = nanmedian(variability)

		# Plot the distribution of variability for all stars:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(variability/median_variability, bins=np.logspace(np.log10(0.1), np.log10(1000.0), 50))
		ax.axvline(self.threshold_variability, color='r')
		ax.set_xscale('log')
		ax.set_xlabel('Variability')
		fig.savefig(os.path.join(self.cbv_plot_folder, f'variability-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area}.png'))
		plt.close(fig)

		# Get the list of star that we are going to load in the lightcurves for:
		search_params.append('variability < %f' % (self.threshold_variability*median_variability))
		stars = self.search_database(
			select=['lightcurve', 'mean_flux', 'variance'],
			search=search_params
		)

		# Number of stars returned:
		Nstars = len(stars)

		# Load the very first timeseries only to find the number of timestamps.
		lc = self.load_lightcurve(stars[0])
		Ntimes = len(lc.time)

		# Save aux information about this CBV to an separate file.
		self.hdf.create_dataset('time', data=lc.time - lc.timecorr)
		self.hdf.create_dataset('cadenceno', data=lc.cadenceno)
		self.hdf.attrs['camera'] = lc.camera
		self.hdf.attrs['ccd'] = lc.ccd
		self.hdf.attrs['data_rel'] = lc.meta['data_rel']
		self.hdf.flush()

		logger.info("Matrix size: %d x %d", Nstars, Ntimes)

		# Make the matrix that will hold all the lightcurves:
		logger.info("Loading in lightcurves...")
		mat = np.full((Nstars, Ntimes), np.nan, dtype='float64')
		varis = np.empty(Nstars, dtype='float64')

		# Loop over stars, fill
		for k, star in tqdm(enumerate(stars), total=Nstars, **tqdm_settings):
			# Load lightkurve object
			lc = self.load_lightcurve(star)

			# Remove bad data based on quality
			flag_good = TESSQualityFlags.filter(lc.pixel_quality, TESSQualityFlags.CBV_BITMASK) & CorrectorQualityFlags.filter(lc.quality, CorrectorQualityFlags.CBV_BITMASK)
			lc.flux[~flag_good] = np.nan

			# Normalize the data and store it in the rows of the matrix:
			mat[k, :] = lc.flux / star['mean_flux'] - 1.0

			# Store the standard deviations of each lightcurve:
			varis[k] = np.NaN if star['variance'] is None else star['variance']

		# Only start calculating correlations if we are actually filtering using them:
		if self.threshold_correlation < 1.0:
			# Calculate the correlation matrix between all lightcurves:
			logger.info("Calculating correlations...")
			correlations = lightcurve_correlation_matrix(mat)

			# If running in DEBUG mode, save the correlations matrix to file:
			if logger.isEnabledFor(logging.DEBUG): # pragma: no cover
				self.hdf.create_dataset('correlations', data=correlations)

			# Find the median absolute correlation between each lightcurve and all other lightcurves:
			c = nanmedian(correlations, axis=0)

			# Indicies that would sort the lightcurves by correlations in descending order:
			indx = np.argsort(c)[::-1]
			indx = indx[:int(self.threshold_correlation*Nstars)]
			#TODO: remove based on threshold value? rather than just % of stars

			# Only keep the top "threshold_correlation"% of the lightcurves that are most correlated:
			mat = mat[indx, :]
			varis = varis[indx]

			# Clean up a bit:
			del correlations, c, indx

		# Print the final shape of the matrix:
		Nstars = mat.shape[0]
		Ntimes = mat.shape[1]

		# Find columns where all stars have NaNs and remove them:
		indx_nancol = allnan(mat, axis=0)
		mat = mat[:, ~indx_nancol]

		logger.info("Matrix size: %d x %d", mat.shape[0], mat.shape[1])

		logger.info("Gap-filling lightcurves...")
		cadenceno = np.arange(mat.shape[1])
		for k in tqdm(range(Nstars), total=Nstars, **tqdm_settings):
			# Normalize the lightcurves by their variances:
			mat[k, :] /= varis[k]

			# Fill out missing values by interpolating the lightcurve:
			indx = np.isfinite(mat[k, :])
			mat[k, ~indx] = pchip_interpolate(cadenceno[indx], mat[k, indx], cadenceno[~indx])

		# Save something for debugging:
		self.hdf.attrs['Ntimes'] = Ntimes
		self.hdf.attrs['Nstars'] = Nstars
		if logger.isEnabledFor(logging.DEBUG): # pragma: no cover
			self.hdf.create_dataset('matrix', data=mat)
			self.hdf.create_dataset('nancols', data=indx_nancol)

		return mat, indx_nancol, Ntimes

	#----------------------------------------------------------------------------------------------
	def entropy_cleaning(self, matrix, targ_limit=150):
		"""
		Entropy-cleaning of lightcurve matrix using the SVD U-matrix.

		Parameters:
			matrix (:class:`numpy.ndarray`):
			targ_limit (int, optional): Maximum number of targets to remove during cleaning.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
		logger = logging.getLogger(__name__)

		# Calculate the principle components:
		pca = PCA(self.ncomponents, random_state=self.random_state)
		U, _, _ = pca._fit(matrix)

		ent = compute_entropy(U)
		logger.info('Entropy start: %s', ent)

		targets_removed = 0
		components = np.arange(self.ncomponents)

		with np.errstate(invalid='ignore'):
			while np.any(ent < self.threshold_entropy):
				com = components[ent < self.threshold_entropy][0]

				# Remove highest relative weight target
				m = nanmedian(U[:, com])
				s = mad_to_sigma*nanmedian(np.abs(U[:, com] - m))
				dev = np.abs(U[:, com] - m) / s

				idx0 = np.argmax(dev)

				# Remove the star from the lightcurve matrix:
				star_no = np.ones(U.shape[0], dtype=bool)
				star_no[idx0] = False
				matrix = matrix[star_no, :]

				targets_removed += 1
				if targets_removed >= targ_limit:
					break

				U, _, _ = pca._fit(matrix)
				ent = compute_entropy(U)

		logger.info('Entropy end: %s', ent)
		logger.info('Targets removed: %d', targets_removed)
		return matrix

	#----------------------------------------------------------------------------------------------
	def compute_cbvs(self, targ_limit=150):
		"""
		Main function for computing CBVs.

		The steps taken in the function are:

		#. Run :meth:`lightcurve_matrix` to obtain matrix with gap-filled,
		   nan-removed light curves for the most correlated stars in a given cbv-area.

		#. Compute principal components.

		#. Run :meth:`entropy_cleaning` to remove significant single-star
		   contributers based on entropy.

		#. Rerun SNR test on CBVs, and only retain CBVs that pass the test.

		#. Recalculate principal components using cleaned star list.

		#. Save CBVs and make diagnostics plots.

		Parameters:
			targ_limit (int, optional): Maximum number of targets to remove during entropy-cleaning.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('running CBV')
		logger.info('------------------------------------')

		if 'cbv-ini' in self.hdf:
			logger.info('CBV for SECTOR=%d, CADENCE=%d, AREA=%d already calculated.', self.sector, self.cadence, self.cbv_area)
			return
		logger.info('Computing CBV for SECTOR=%d, CADENCE=%d, AREA=%d...', self.sector, self.cadence, self.cbv_area)

		# Extract or compute cleaned and gapfilled light curve matrix
		mat, indx_nancol, Ntimes = self.lightcurve_matrix()

		# Calculate initial CBVs
		logger.info('Computing %d CBVs', self.ncomponents)
		pca = PCA(self.ncomponents, random_state=self.random_state)
		U0, _, _ = pca._fit(mat)

		cbv0 = np.full((Ntimes, self.ncomponents), np.nan, dtype='float64')
		cbv0[~indx_nancol, :] = np.transpose(pca.components_)

		# Clean away targets that contribute significantly
		# as a single star to a given CBV (based on entropy)
		logger.info('Doing Entropy Cleaning...')
		mat = self.entropy_cleaning(mat, targ_limit=targ_limit)

		# Calculate the principle components of cleaned matrix
		logger.info("Doing Principle Component Analysis...")
		U, _, _ = pca._fit(mat)

		cbv = np.full((Ntimes, self.ncomponents), np.nan, dtype='float64')
		cbv[~indx_nancol, :] = np.transpose(pca.components_)

		# Signal-to-Noise test (here only for plotting)
		#indx_lowsnr = cbv_snr_test(cbv, self.threshold_snrtest)

		# Save the CBV to file:
		self.hdf.create_dataset('cbv-ini', data=cbv)

		#------------------------ PLOTS ---------------------------
		# Plot the "effectiveness" of each CBV:
		max_components = 20
		n_cbv_components = np.arange(max_components, dtype=int)
		pca_scores = compute_scores(mat, n_cbv_components)

		fig0 = plt.figure(figsize=(12, 8))
		ax0 = fig0.add_subplot(121)
		ax0.plot(n_cbv_components, pca_scores, 'b', label='PCA scores')
		ax0.set_xlabel('nb of components')
		ax0.set_ylabel('CV scores')
		ax0.legend(loc='lower right')
		ax02 = fig0.add_subplot(122)
		ax02.plot(np.arange(1, cbv0.shape[1]+1), pca.explained_variance_ratio_, '.-')
		ax02.axvline(x=cbv.shape[1]+0.5, ls='--', color='k')
		ax02.set_xlabel('CBV number')
		ax02.set_ylabel('Variance explained ratio')
		fig0.savefig(os.path.join(self.cbv_plot_folder, f'cbv-perf-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.png'))
		plt.close(fig0)

		# Plot all the CBVs:
		fig, axes = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig2, axes2 = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)
		fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)

		for k, ax in enumerate(axes.flatten()):
			if k < cbv0.shape[1]:
				#if indx_lowsnr is not None and indx_lowsnr[k]:
				#	col = 'c'
				#else:
				#	col = 'k'

				ax.plot(cbv0[:, k]+0.1, 'r-')
				ax.plot(cbv[:, k], ls='-', color='k')
				ax.set_title(f'Basis Vector {k+1:d}')

		for k, ax in enumerate(axes2.flatten()):
			if k < U0.shape[1]:
				ax.plot(-np.abs(U0[:, k]), 'r-')
				ax.plot(np.abs(U[:, k]), 'k-')
				ax.set_title(f'Basis Vector {k+1:d}')

		fig.savefig(os.path.join(self.cbv_plot_folder, f'cbvs_ini-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.png'))
		fig2.savefig(os.path.join(self.cbv_plot_folder, f'U_cbvs-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.png'))
		plt.close(fig)
		plt.close(fig2)

	#----------------------------------------------------------------------------------------------
	def spike_sep(self):
		"""
		Separate CBVs into a "slow" and a "spiky" component.

		This is done by filtering the deta and identifying outlier
		with a peak-finding algorithm.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info('running CBV spike separation')
		logger.info('------------------------------------')

		if 'cbv-single-scale' in self.hdf and 'cbv-spike' in self.hdf:
			logger.info('Separated CBVs for SECTOR=%d, CADENCE=%d, AREA=%d already calculated.', self.sector, self.cadence, self.cbv_area)
			return
		logger.info('Computing CBV spike separation for SECTOR=%d, CADENCE=%d, AREA=%d...', self.sector, self.cadence, self.cbv_area)

		# Load initial CBV from "compute_CBV"
		cbv = self.hdf['cbv-ini']

		# padding window, just needs to be bigger than savgol filtering window
		wmir = 50

		# Initiate arrays for cleaned and spike CBVs
		cbv_new = np.zeros_like(cbv)
		cbv_spike = np.zeros_like(cbv)

		# Iterate over basis vectors
		xs = np.arange(0, cbv.shape[0] + 2*wmir-2)
		for j in range(cbv.shape[1]):

			# Pad ends for better peak detection at boundaries of data
			data0 = cbv[:, j]
			data0 = np.append(np.flip(data0[0:wmir])[:-1], data0)
			data0 = np.append(data0, np.flip(data0[-wmir::])[1:])
			data = data0.copy()

			# Iterate peak detection, with different savgol filter widths:
			for w in (31, 29, 27, 25, 23):
				# For savgol filter data must be continuous
				data2 = pchip_interpolate(xs[np.isfinite(data)], data[np.isfinite(data)], xs)

				# Smooth, filtered version of data, to use to identify "outliers", i.e., spikes
				y = savgol_filter(data2, w, 2, mode='constant')
				y2 = data2 - y

				# Run peak detection
				sigma = mad_to_sigma * nanmedian(np.abs(y2))
				peaks, properties = find_peaks(np.abs(y2), prominence=(3*sigma, None), wlen=500)

				data[peaks] = np.nan

			# Interpolate CBVs where spike has been identified
			data = pchip_interpolate(xs[np.isfinite(data)], data[np.isfinite(data)], xs)

			# Remove padded ends and store in CBV matrices
			# Spike signal is difference between original data and data with masked spikes
			cbv_spike[:, j] = data0[wmir-1:-wmir+1] - data[wmir-1:-wmir+1]
			replace(cbv_spike[:, j], np.nan, 0)

			cbv_new[:, j] = data[wmir-1:-wmir+1]

		# Save files
		self.hdf.create_dataset('cbv-single-scale', data=cbv_new)
		self.hdf.create_dataset('cbv-spike', data=cbv_spike)

		# Signal-to-Noise test (here only for plotting)
		indx_lowsnr = cbv_snr_test(cbv_new, self.threshold_snrtest)

		# Plot all the CBVs:
		fig, axes = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig2, axes2 = plt.subplots(int(np.ceil(self.ncomponents/2)), 2, figsize=(12, 16))
		fig.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)
		fig2.subplots_adjust(wspace=0.23, hspace=0.46, left=0.08, right=0.96, top=0.94, bottom=0.055)

		axes = axes.flatten()
		axes2 = axes2.flatten()
		for k in range(cbv_new.shape[1]):
			if indx_lowsnr is not None and indx_lowsnr[k]:
				col = 'c'
			else:
				col = 'k'

			axes[k].plot(cbv_new[:, k], ls='-', color=col)
			axes[k].set_title(f'Basis Vector {k+1:d}')

			axes2[k].plot(cbv_spike[:, k], ls='-', color=col)
			axes2[k].set_title(f'Spike Basis Vector {k+1:d}')

		fig.savefig(os.path.join(self.cbv_plot_folder, f'cbvs-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.png'))
		fig2.savefig(os.path.join(self.cbv_plot_folder, f'spike-cbvs-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.png'))
		plt.close(fig)
		plt.close(fig2)

	#----------------------------------------------------------------------------------------------
	def cotrend_ini(self, do_ini_plots=False):
		"""
		Function for running the initial co-trending to obtain CBV coefficients for
		the construction of priors.

		The function will load the calculated CBVs and co-trend all light curves in area using
		fit of all CBVs using linear least squares. The CBV coefficients from the fit are saved
		into the HDF5 CBV file.

		Parameters:
			do_ini_plots (bool): Plot the LS fit for each light curve? Default=False.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)
		logger.info("--------------------------------------------------------------")
		if 'inifit' in self.hdf:
			logger.info("Initial co-trending in SECTOR=%d, CADENCE=%d, AREA=%d already done.", self.sector, self.cadence, self.cbv_area)
			return
		logger.info("Initial co-trending in SECTOR=%d, CADENCE=%d, AREA=%d...", self.sector, self.cadence, self.cbv_area)

		# Create search parameters for the database:
		search_params = [
			f'status={STATUS.OK.value:d}', # Only including targets with status=OK from photometry
			"method_used='aperture'", # Only including aperature photometry targets
			f'cadence={self.cadence:d}',
			f"cbv_area={self.cbv_area:d}",
			f"sector={self.sector:d}"
		]

		# Load stars from database
		stars = self.search_database(
			select=['lightcurve', 'pos_column', 'pos_row', 'tmag'],
			search=search_params)

		# Load the cbv from file:
		cbv = CBV(os.path.join(self.data_folder, f'cbv-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.hdf5'))

		# Update maximum number of components
		Ncbvs = cbv.cbv.shape[1]
		logger.info('Fitting using number of components: %d', Ncbvs)

		# Initialize results array including CBV coefficients,
		# Spike-CBV coefficients and an residual offset
		Nres = int(2*Ncbvs+1)
		coeffs = np.full((len(stars), Nres), np.NaN, dtype='float64')
		pos = np.full((len(stars), 3), np.NaN, dtype='float64')

		# Loop through stars
		for k, star in tqdm(enumerate(stars), total=len(stars), disable=not logger.isEnabledFor(logging.INFO)):

			lc = self.load_lightcurve(star)

			logger.debug("Correcting star %d", lc.targetid)

			try:
				flux_filter, res, _ = cbv.fit(lc, cbvs=Ncbvs, use_bic=False, use_prior=False)
			except ValueError:
				logger.exception("%d: Ini-fit failed with ValueError", lc.targetid)
				continue

			# TODO: compute diagnostics requiring the light curve
			# SAVE TO DIAGNOSTICS FILE::
			#wn_ratio = GOC_wn(flux, flux-flux_filter)

			coeffs[k, :] = np.array([res,]).flatten()

			#targets[k] = lc.targetid
			pos[k, 0] = star['pos_row']
			pos[k, 1] = star['pos_column']
			pos[k, 2] = star['tmag']

			if do_ini_plots:
				lc_corr = (lc.flux/flux_filter-1)*1e6

				fig = plt.figure()
				ax1 = fig.add_subplot(211)
				ax1.plot(lc.time, lc.flux)
				ax1.plot(lc.time, flux_filter)
				ax1.set_xlabel('Time (TBJD)')
				ax1.set_ylabel('Flux (counts)')
				ax1.set_xticks([])
				ax2 = fig.add_subplot(212)
				ax2.plot(lc.time, lc_corr)
				ax2.set_xlabel('Time (TBJD)')
				ax2.set_ylabel('Relative flux (ppm)')
				fig.savefig(os.path.join(self.plot_folder(lc), f'lc_corr_ini_tic{lc.targetid:011d}.png'))
				plt.close(fig)

		# Filter away any targets that could not be fitted:
		indx = ~np.isnan(pos[:, 0])
		pos = pos[indx, :]
		coeffs = coeffs[indx, :]

		# Save weights for priors if it is an initial run
		self.hdf.create_dataset('inifit', data=coeffs)
		self.hdf.create_dataset('inifit_targets', data=pos)

		# Plot CBV weights
		fig = plt.figure(figsize=(15, 15))
		ax = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		for kk in range(1, Nres):
			idx = np.nonzero(coeffs[:, kk])
			r = coeffs[idx, kk]
			idx2 = (r > np.percentile(r, 10)) & (r < np.percentile(r, 90))

			kde = gaussian_kde(r[idx2])
			kde_support = np.linspace(np.min(r[idx2]), np.max(r[idx2]), 5000)
			kde_density = kde.pdf(kde_support)

			err = nanmedian(np.abs(r[idx2] - nanmedian(r[idx2]))) * 1e5

			imax = np.argmax(kde_density)

			if kk > Ncbvs:
				ax3.plot(kde_support*1e5, kde_density/kde_density[imax], label='CBV ' + str(kk), ls='--')
				ax4.errorbar(kk, kde_support[imax]*1e5, yerr=err, marker='o', color='k')
			else:
				ax.plot(kde_support*1e5, kde_density/kde_density[imax], label='CBV ' + str(kk), ls='-')
				ax2.errorbar(kk, kde_support[imax]*1e5, yerr=err, marker='o', color='k')

		ax.set_xlabel('CBV weight')
		ax2.set_ylabel('CBV weight')
		ax2.set_xlabel('CBV')
		ax.legend()
		fig.savefig(os.path.join(self.cbv_plot_folder, f'weights-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.png'))
		plt.close(fig)

	#----------------------------------------------------------------------------------------------
	def compute_distance_map(self):
		"""
		3D distance map for weighting initial-fit coefficients into a prior.

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""
		logger = logging.getLogger(__name__)
		logger.info("--------------------------------------------------------------")

		if 'inifit' not in self.hdf:
			raise RuntimeError('Trying to make priors without initial corrections.')

		pos_mag0 = np.asarray(self.hdf['inifit_targets'])

		# Load in positions and tmags, in same order as results are saved from ini_fit
		pos_mag0[:, 2] = np.clip(pos_mag0[:, 2], 2, 20)

		# Relative importance of dimensions
		#S = np.array([1, 1, 2])
		S = np.array([MAD_model2(pos_mag0[:, 0]), MAD_model2(pos_mag0[:, 1]), 0.5*MAD_model2(pos_mag0[:, 2])])
		LL = np.diag(S)

		#pos_mag0[:, 0] /= np.std(pos_mag0[:, 0])
		#pos_mag0[:, 1] /= np.std(pos_mag0[:, 1])
		#pos_mag0[:, 2] /= np.std(pos_mag0[:, 2])

		# Construct and save distance tree
		dist = DistanceMetric.get_metric('mahalanobis', VI=LL)
		tree = BallTree(pos_mag0, metric=dist)

		# Save the tree to a pickle file to be easily loaded by the CBV class:
		savePickle(os.path.join(self.data_folder, f'cbv-prior-s{self.sector:04d}-c{self.cadence:04d}-a{self.cbv_area:d}.pickle'), tree)

	#----------------------------------------------------------------------------------------------
	def interpolate_to_higher_cadence(self, cadence=120):
		"""
		Interpolate CBVs generated from FFIs to higher cadence (120 seconds).

		New HDF5 files will be generated, containing the CBVs interpolated using a cubic spline
		to the higher cadence. All spike-CBVs are set to zero, since there is no good way to
		interpolate them.

		Parameters:
			cadence (int):

		Returns:
			str: Path to the new CBV file.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		if self.datasource != 'ffi':
			raise RuntimeError("Can not interpolate this CBV, since it is doesn't come from a FFI.")

		logger = logging.getLogger(__name__)
		logger.info("Interpolating to higher cadence")

		newfile = os.path.join(self.data_folder, f'cbv-s{self.sector:04d}-c{cadence:04d}-a{self.cbv_area:d}.hdf5')
		if os.path.exists(newfile):
			logger.warning("File already exists: %s", newfile)
			return

		# Get single star to load timestamps etc from:
		stars = self.search_database(
			select=['lightcurve'],
			search=[
				f'status={STATUS.OK.value:d}', # Only including targets with status=OK from photometry
				"method_used='aperture'", # Only including aperature photometry targets
				f"sector={self.sector:d}",
				f"cbv_area={self.cbv_area:d}",
				f'cadence={cadence:d}',
			],
			limit=1
		)

		# Load the very first timeseries only to find the number of timestamps.
		lc = self.load_lightcurve(stars[0])
		Ntimes = len(lc.time)

		# Interpolate the FFI CBV into high-cadence timestamps:
		cbv_interp = interp1d(self.hdf['time'], self.hdf['cbv-single-scale'], axis=0, kind='cubic', assume_sorted=True, fill_value='extrapolate')
		cbv = cbv_interp(lc.time - lc.timecorr)
		logger.info("New CBV shape: %s", cbv.shape)

		# Clear Spike-CBVs since we have no reliable way of interpolating them:
		cbv_spike = np.zeros_like(cbv)

		# Modify the file, overwriting the CBVs and Spike-CBVs with the interpolated ones:
		with h5py.File(newfile, 'w', libver='latest') as hdf:
			# Copy all headers:
			for key, value in self.hdf.attrs.items():
				hdf.attrs[key] = value

			# Change the headers that are different now:
			hdf.attrs['cadence'] = cadence
			hdf.attrs['datasource'] = 'tpf'
			hdf.attrs['Ntimes'] = Ntimes
			hdf.attrs['method'] = 'interpolated'

			# Add datasets that are needed for CBVs to load:
			hdf.create_dataset('time', data=lc.time - lc.timecorr)
			hdf.create_dataset('cadenceno', data=lc.cadenceno)
			hdf.create_dataset('cbv-single-scale', data=cbv)
			hdf.create_dataset('cbv-spike', data=cbv_spike)

		return newfile
