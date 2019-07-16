#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:42:49 2018

@author: mikkelnl
"""

import os
import numpy as np
import math as m
import logging
from tqdm import tqdm
from bottleneck import nanmean, nanmedian
import json
import argparse
import sqlite3
import six
from lightkurve import TessLightCurve
from astropy.io import fits
from astropy.stats import LombScargle

from .cbv_main import lc_matrix_calc
from ..plots import plt
from ..quality import CorrectorQualityFlags, TESSQualityFlags
from ..manual_filters import manual_exclude

import matplotlib.pyplot as pl

# =============================================================================
# 
# =============================================================================

def psd_scargle(time, flux, Nsample = 10.):
	"""
	   Calculate the power spectral density using the Lomb-Scargle (L-S) periodogram
	   
	   Parameters:
	        time (numpy array, float): time stamps of the light curve
	        flux (numpy array, float): the flux variations of the light curve
	        Nsample (optional, float): oversampling rate for the periodogram. Default value = 10.
	   
	   Returns:
	        fr (numpy array, float): evaluated frequency values in the domain of the periodogram
	        sc (numpy array, float): the PSD values of the L-S periodogram
	
	.. codeauthor:: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
	"""
	ndata = len(time)                                            # The number of data points
	fnyq = 0.5/np.median(time[1:]-time[:-1])                     # the Nyquist frequency
	fres = 1./(time[-1]-time[0])                                 # the frequency resolution
	fr = np.arange(0.,fnyq,fres/float(Nsample))                  # the frequencies
	sc1 = LombScargle(time, flux).power(fr, normalization='psd')   # The non-normalized Lomb-Scargle "power"
	
	# Computing the appropriate rescaling factors (to convert to astrophysical units)
	fct = m.sqrt(4./ndata)
	T = time.ptp()
	sc = fct**2. * sc1 * T
	
	# Ensuring the output does not contain nans
	if(np.isnan(sc).any()):
	    fr = fr[~np.isnan(sc)]
	    sc = sc[~np.isnan(sc)]
	
	return fr, sc



def wn(ori_lc, corrected_lc, alpha_n = 1.):
	"""
	   Calculate added white noise between two light curves.
	   Based on Eq. 8.4-8.5 in the Kepler PDC 
	   
	   Parameters:
	        ori_lc (light kurve object): the uncorrected TESS light curve
	        corrected_lc (light kurve object): the corrected TESS light curve
	        alpha_n (optional, float): scaling factor. Default value = 1.
	        
	   Returns:
	        Gn (float): goodness metric for the added white noise.
	                    In the limit where ori_lc and corrected_lc are identical, Gn approaches 0.
	                    In the (improbable?) case where noise is removed instead of added, Gn = -1.
	                    
	
	.. codeauthor:: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
	"""
	
	# Excluding nans from the input LCs to avoid problems
	ori_time0 = ori_lc.time[~np.isnan(ori_lc.flux)]
	ori_flux0 = ori_lc.flux[~np.isnan(ori_lc.flux)]
	corr_time0 = corrected_lc.time[~np.isnan(corrected_lc.flux)]
	corr_flux0 = corrected_lc.flux[~np.isnan(corrected_lc.flux)]
	
	# Calculating the Noise floor of both LCs, defined as the differences between adjacent flux values
	ori_time = ori_time0[:-1]
	ori_Nf = ori_flux0[1:] - ori_flux0[:-1]
	
	corr_time = corr_time0[:-1]
	corr_Nf = corr_flux0[1:] - corr_flux0[:-1]
	
	# Computing the PSDs of the noise floors
	corr_fr,corr_psd = psd_scargle(corr_time, corr_Nf - np.mean(corr_Nf))
	ori_fr,ori_psd = psd_scargle(ori_time, ori_Nf - np.mean(ori_Nf))
	
	# Ensuring both PSDs are evaluated for the same frequencies
	int_corr_psd = np.interp(ori_fr, corr_fr, corr_psd)
	
	# Integrate the log of the ratio of PSDs, ensuring the integral exists
	if(np.r_[int_corr_psd < ori_psd].all()):
	    Gn = -1.
	else:
	    integrand = np.log10(int_corr_psd/ori_psd)
	    integrand[np.r_[int_corr_psd < ori_psd]] = 0.
	    Gn = alpha_n * np.trapz(integrand, x=ori_fr)
	
	return Gn





class LCValidation(object):


	def __init__(self, input_folders, output_folder=None, validate=True, method='all', colorbysector=False, ext='png', showplots=False):

		# Store inputs:
		self.input_folders = input_folders
		self.method = method
		self.extension = ext
		self.show = showplots
		self.outfolders = output_folder
		self.doval = validate
		self.color_by_sector = colorbysector

		#load sqlite to-do files
		if len(self.input_folders)==1:
			if self.outfolders is None:
				path = os.path.join(self.input_folders[0], 'data_validation')
				self.outfolders = path
				if not os.path.exists(self.outfolders):
					os.makedirs(self.outfolders)

		for i, f in enumerate(self.input_folders):
			todo_file = os.path.join(f, 'todo.sqlite')
			logger.debug("TODO file: %s", todo_file)
			if not os.path.exists(todo_file):
				raise ValueError("TODO file not found")

			# Open the SQLite file:
			self.conn = sqlite3.connect(todo_file)
			self.conn.row_factory = sqlite3.Row
			self.cursor = self.conn.cursor()

			if self.method == 'all':
				# Create table for diagnostics:
				if self.doval:
					self.cursor.execute('DROP TABLE IF EXISTS datavalidation_raw')
				self.cursor.execute("""CREATE TABLE IF NOT EXISTS datavalidation_raw (
					priority INT PRIMARY KEY NOT NULL,
					dataval INT NOT NULL,
					approved BOOLEAN NOT NULL,
					FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
				);""")

				self.conn.commit()

	def close(self):
		"""Close DataValidation object and all associated objects."""
		self.cursor.close()
		self.conn.close()

	def __exit__(self, *args):
		self.close()

	def __enter__(self):
		return self
	
	
	def load_lightcurve(self, task, ver='RAW'):
		"""
		Load lightcurve from task ID or full task dictionary.

		Parameters:
			task (integer or dict):

		Returns:
			``lightkurve.TessLightCurve``: Lightcurve for the star in question.

		Raises:
			ValueError: On invalid file format.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Find the relevant information in the TODO-list:
		if not isinstance(task, dict) or task.get("lightcurve") is None:
			self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE todolist.priority=? LIMIT 1;", (task, ))
			task = self.cursor.fetchone()
			if task is None:
				raise ValueError("Priority could not be found in the TODO list")
			task = dict(task)

		# Get the path of the FITS file:
		fname = os.path.join(self.input_folder, task.get('lightcurve'))
		logger.debug('Loading lightcurve: %s', fname)

		if fname.endswith('.fits') or fname.endswith('.fits.gz'):
			with fits.open(fname, mode='readonly', memmap=True) as hdu:
				# Quality flags from the pixels:
				pixel_quality = np.asarray(hdu['LIGHTCURVE'].data['PIXEL_QUALITY'], dtype='int32')

				# Create the QUALITY column and fill it with flags of bad data points:
				quality = np.zeros_like(hdu['LIGHTCURVE'].data['TIME'], dtype='int32')
				
				if ver=='RAW':
					LC = hdu['LIGHTCURVE'].data['FLUX_RAW']
					LC_ERR = hdu['LIGHTCURVE'].data['FLUX_RAW_ERR'],
				elif ver=='CORR':
					LC = hdu['LIGHTCURVE'].data['FLUX_CORR']
					LC_ERR = hdu['LIGHTCURVE'].data['FLUX_CORR_ERR'],
					
				bad_data = ~np.isfinite(LC)

				bad_data |= (pixel_quality & TESSQualityFlags.DEFAULT_BITMASK != 0)
				quality[bad_data] |= CorrectorQualityFlags.FlaggedBadData

				# Create lightkurve object:
				lc = TessLightCurve(
					time=hdu['LIGHTCURVE'].data['TIME'],
					flux=LC, 
					flux_err=LC_ERR,
					centroid_col=hdu['LIGHTCURVE'].data['MOM_CENTR1'],
					centroid_row=hdu['LIGHTCURVE'].data['MOM_CENTR2'],
					quality=quality,
					cadenceno=np.asarray(hdu['LIGHTCURVE'].data['CADENCENO'], dtype='int32'),
					time_format='btjd',
					time_scale='tdb',
					targetid=hdu[0].header.get('TICID'),
					label=hdu[0].header.get('OBJECT'),
					camera=hdu[0].header.get('CAMERA'),
					ccd=hdu[0].header.get('CCD'),
					sector=hdu[0].header.get('SECTOR'),
					ra=hdu[0].header.get('RA_OBJ'),
					dec=hdu[0].header.get('DEC_OBJ'),
					quality_bitmask=CorrectorQualityFlags.DEFAULT_BITMASK,
					meta={}
				)

				# Apply manual exclude flag:
				manexcl = manual_exclude(lc)
				lc.quality[manexcl] |= CorrectorQualityFlags.ManualExclude

		else:
			raise ValueError("Invalid file format")

		# Add additional attributes to lightcurve object:
		lc.pixel_quality = pixel_quality

		# Keep the original task in the metadata:
		lc.meta['task'] = task
		lc.meta['additional_headers'] = fits.Header()

		if logger.isEnabledFor(logging.DEBUG):
			lc.show_properties()

		return lc


	def search_database(self, select=None, search=None, order_by=None, limit=None, distinct=False):
		"""
		Search list of lightcurves and return a list of tasks/stars matching the given criteria.

		Parameters:
			search (list of strings or None): Conditions to apply to the selection of stars from the database
			order_by (list, string or None): Column to order the database output by.
			limit (int or None): Maximum number of rows to retrieve from the database. If limit is None, all the rows are retrieved.
			distinct (boolean): Boolean indicating if the query should return unique elements only.

		Returns:
			list of dicts: Returns all stars retrieved by the call to the database as dicts/tasks that can be consumed directly by load_lightcurve

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		if select is None:
			select = '*'
		elif isinstance(select, (list, tuple)):
			select = ",".join(select)

		if search is None:
			search = ''
		elif isinstance(search, (list, tuple)):
			search = "WHERE " + " AND ".join(search)
		else:
			search = 'WHERE ' + search

		if order_by is None:
			order_by = ''
		elif isinstance(order_by, (list, tuple)):
			order_by = " ORDER BY " + ",".join(order_by)
		elif isinstance(order_by, six.string_types):
			order_by = " ORDER BY " + order_by

		limit = '' if limit is None else " LIMIT %d" % limit

		query = "SELECT {distinct:s}{select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority LEFT JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority {search:s}{order_by:s}{limit:s};".format(
			distinct='DISTINCT ' if distinct else '',
			select=select,
			search=search,
			order_by=order_by,
			limit=limit
		)
		logger.debug("Running query: %s", query)

		# Ask the database: status=1
		self.cursor.execute(query)
		return [dict(row) for row in self.cursor.fetchall()]
	
	
	
	
	
	def Validations(self):

		if self.method == 'all':
			self.correlation()
			self.added_noise()
			

#			dv = np.array(list(val.values()), dtype="int32")
#						
#			#Reject: Small/High apertures; Contamination>1;
#			app = np.ones_like(dv, dtype='bool')
#			qf = DatavalQualityFlags.filter(dv)
#			app[~qf] = False
#
#			[self.cursor.execute("INSERT INTO datavalidation_raw (priority, dataval, approved) VALUES (?,?,?);", (int(v1), int(v2), bool(v3))) for v1,v2,v3 in
#					zip(np.array(list(val.keys()), dtype="int32"),dv,app)]
#
#			self.cursor.execute("INSERT INTO datavalidation_raw (priority, dataval, approved) select todolist.priority, 0, 0 FROM todolist WHERE todolist.status not in (1,3);")
#			self.conn.commit()


		elif self.method == 'corr':
			self.correlation()
		elif self.method == 'addnoise':
			self.added_noise()
		
		
		

	def correlations(self, cbv_area):
		
		
		"""
		Function to compute correlation matrix after correction

		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger=logging.getLogger(__name__)

		logger.info('------------------------------------------')
		logger.info('Running correlation check')
		
		
		tmpfile = os.path.join(self.data_folder, 'mat-%d.npz' %cbv_area)
		if logger.isEnabledFor(logging.DEBUG) and os.path.exists(tmpfile):
			logger.info("Loading existing file...")
			data = np.load(tmpfile)
			mat = data['mat']
			varis = data['varis']

		else:
			# Get the list of star that we are going to load in the lightcurves for:
			stars = self.search_database(search=['datasource="ffi"', 'cbv_area=%i' %cbv_area])

			# Number of stars returned:
			Nstars = len(stars)

			# Load the very first timeseries only to find the number of timestamps.
			lc = self.load_lightcurve(stars[0])
			Ntimes = len(lc.time)

			logger.info("Matrix size: %d x %d", Nstars, Ntimes)

			# Make the matrix that will hold all the lightcurves:
			logger.info("Loading in lightcurves...")
			mat0 = np.empty((Nstars, Ntimes), dtype='float64')
			mat0.fill(np.nan)
			varis0 = np.empty(Nstars, dtype='float64')

			# Loop over stars
			for k, star in tqdm(enumerate(stars), total=Nstars, disable=not logger.isEnabledFor(logging.INFO)):

				# Load lightkurve object
				lc = self.load_lightcurve(star)

				# Remove bad data based on quality
				flag_good = TESSQualityFlags.filter(lc.pixel_quality, TESSQualityFlags.CBV_BITMASK) & CorrectorQualityFlags.filter(lc.quality, CorrectorQualityFlags.CBV_BITMASK)
				lc.flux[~flag_good] = np.nan

				# Normalize the data and store it in the rows of the matrix:
				mat0[k, :] = lc.flux / nanmean(lc.flux) - 1.0


			# Calculate the correlation matrix between all lightcurves:
			correlations = lc_matrix_calc(Nstars, mat0)

			# Save the correlations matrix to file:
			file_correlations = os.path.join(self.data_folder, 'post_correlations-%d.npy' % cbv_area)
			np.save(file_correlations, correlations)

			# Find the median absolute correlation between each lightcurve and all other lightcurves:
			


			expected_mean = 1/np.sqrt(Nstars)
			# Goodness
			G = nanmedian(correlations**3, axis=0) - expected_mean
#			G = nanmean(correlations[k,:]**exponent) - expected_mean
			# Save median correlations

				
			# Save something for debugging:
			if logger.isEnabledFor(logging.DEBUG):
				np.savez(tmpfile, mat=mat)
		
		
		
	def added_noise(self):
		
		#call wn on loaded targets
		
		pass
	
	
	
	
	
		
		
	



#------------------------------------------------------------------------------
if __name__ == '__main__':
	
	
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run Data Validation pipeline.')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default='all', choices=('corr', 'addnoise'))
	parser.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png', 'eps'))
	parser.add_argument('-s', '--show', help='Show plots.', action='store_true')
	parser.add_argument('-v', '--validate', help='Compute validation (only run is method is "all").', action='store_true')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('input_folders', type=str, help='Directory to create catalog files in.', nargs='?', default=None)
	parser.add_argument('output_folder', type=str, help='Directory in which to place output if several input folders are given.', nargs='?', default=None)
	args = parser.parse_args()

	# TODO: Remove this before going into production.
	args.show = True
	args.method = 'addnoise'
	args.validate = False
	args.sysnoise = 5
#	args.input_folders = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-combined/'
#	args.input_folders = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-combined/;/media/mikkelnl/Elements/TESS/S02_tests/'
#	args.output_folder = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-combined/'
	args.input_folders = '/media/mikkelnl/Elements/TESS/S02_tests/'

	if args.output_folder is None and len(args.input_folders.split(';'))>1:
		parser.error("Please specify an output directory!")

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('corrections')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)


	logger.info("Loading input data from '%s'", args.input_folders)
	logger.info("Putting output data in '%s'", args.output_folder)

	input_folders = args.input_folders.split(';')
	
	
#	# Use the BaseCorrector to search the database for which CBV_AREAS to run:
#	with BaseCorrector(input_folder) as bc:
#		# Build list of constraints:
#		constraints = []
#		if args.camera:
#			constraints.append('camera IN (%s)' % ",".join([str(c) for c in args.camera]))
#		if args.ccd:
#			constraints.append('ccd IN (%s)' % ",".join([str(c) for c in args.ccd]))
#		if args.area:
#			constraints.append('cbv_area IN (%s)' % ",".join([str(c) for c in args.area]))
#		if not constraints:
#			constraints = None
#
#		# Search for valid areas:
#		cbv_areas = [row['cbv_area'] for row in bc.search_database(select='cbv_area', distinct=True, search=constraints)]
#		logger.debug("CBV areas: %s", cbv_areas)
#
#	# Number of threads to run in parallel:
#	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
#	threads = min(threads, len(cbv_areas))
#	logger.info("Using %d processes.", threads)
#
#	# Create wrapper function which only takes a single cbv_area as input:
#	prepare_cbv_wrapper = partial(prepare_cbv, input_folder=input_folder, threshold=args.snr, ncbv=args.ncbv, el=args.el, ip=args.iniplot)

	# Create DataValidation object:
	with LCValidation(input_folders, output_folder=args.output_folder,
		validate=args.validate, method=args.method, colorbysector=args.colorbysector,
		showplots=args.show, ext=args.ext, sysnoise=args.sysnoise) as dataval:

		# Run validation
		dataval.Validations()
	


			
