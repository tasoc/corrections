#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The basic correction class for the TASOC Photomety pipeline.
All other specific correction classes will inherit from BaseCorrector.

.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Filipe Pereira
"""

import os.path
import shutil
import enum
import logging
import sqlite3
import tempfile
import contextlib
import numpy as np
from timeit import default_timer
from bottleneck import nanmedian, nanvar, allnan
from astropy.io import fits
from lightkurve import TessLightCurve
from .plots import plt, save_figure
from .quality import TESSQualityFlags, CorrectorQualityFlags
from .utilities import rms_timescale, ptp, ListHandler, LoggerWriter, fix_fits_table_headers
from .manual_filters import manual_exclude
from .version import get_version

__version__ = get_version(pep440=False)

__docformat__ = 'restructuredtext'

#--------------------------------------------------------------------------------------------------
class STATUS(enum.Enum):
	"""
	Status indicator of the status of the correction.
	"""
	UNKNOWN = 0 #: The status is unknown. The actual calculation has not started yet.
	STARTED = 6 #: The calculation has started, but not yet finished.
	OK = 1      #: Everything has gone well.
	ERROR = 2   #: Encountered a catastrophic error that I could not recover from.
	WARNING = 3 #: Something is a bit fishy. Maybe we should try again with a different algorithm?
	ABORT = 4   #: The calculation was aborted.
	SKIPPED = 5 #: The target was skipped because the algorithm found that to be the best solution.

#--------------------------------------------------------------------------------------------------
def _filter_fits_hdu(hdu):
	"""
	Filter FITS file for invalid data (undefined timestamps).

	Parameters:
		hdu (:class:`astropy.io.fits.HDUList`): FITS HDUList that needs to be filtered.

	Returns:
		:class:`astropy.io.fits.HDUList`: Modified FITS HDUList with invalid data removed.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Remove non-finite timestamps
	indx = np.isfinite(hdu['LIGHTCURVE'].data['TIME'])

	# Remove where TIME, CADENCENO and FLUX_RAW are all exactly zero:
	indx &= ~((hdu['LIGHTCURVE'].data['CADENCENO'] == 0)
		& (hdu['LIGHTCURVE'].data['TIME'] == 0)
		& (hdu['LIGHTCURVE'].data['FLUX_RAW'] == 0))

	# Remove from in-memory FITS hdu:
	hdu['LIGHTCURVE'].data = hdu['LIGHTCURVE'].data[indx]
	return hdu

#--------------------------------------------------------------------------------------------------
class BaseCorrector(object):
	"""
	The basic correction class for the TASOC Photometry pipeline.
	All other specific correction classes will inherit from BaseCorrector.

	Attributes:
		plot (bool): Boolean indicating if plotting is enabled.
		data_folder (str): Path to directory where axillary data for the corrector
			should be stored.

	.. codeauthor:: Lindsey Carboneau
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, input_folder, plot=False):
		"""
		Initialize the corrector.

		Parameters:
			input_folder (str): Directory with input files.
			plot (bool, optional): Enable plotting.

		Raises:
			FileNotFoundError: TODO-file not found in directory.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Add a ListHandler to the logging of the corrections module.
		# This is needed to catch any errors and warnings made by the correctors
		# for ultimately storing them in the TODO-file.
		# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
		self.message_queue = []
		handler = ListHandler(message_queue=self.message_queue, level=logging.WARNING)
		formatter = logging.Formatter('%(levelname)s: %(message)s')
		handler.setFormatter(formatter)
		logging.getLogger('corrections').addHandler(handler)

		# Save inputs:
		self.plot = plot
		if os.path.isdir(input_folder):
			self.input_folder = input_folder
			todo_file = os.path.join(input_folder, 'todo.sqlite')
		else:
			self.input_folder = os.path.dirname(input_folder)
			todo_file = input_folder

		# The path to the TODO list:
		logger.debug("TODO file: %s", todo_file)
		if not os.path.isfile(todo_file):
			raise FileNotFoundError("TODO file not found")

		self.CorrMethod = {
			'BaseCorrector': 'base',
			'EnsembleCorrector': 'ensemble',
			'CBVCorrector': 'cbv',
			'CBVCreator': 'cbv',
			'KASOCFilterCorrector': 'kasoc_filter'
		}.get(self.__class__.__name__)

		# Find the axillary data directory based on which corrector is running:
		if self.CorrMethod == 'base':
			self.data_folder = os.path.join(os.path.dirname(__file__), 'data')
		else:
			# Create a data folder specific to this corrector:
			if self.CorrMethod == 'cbv':
				self.data_folder = os.path.join(self.input_folder, 'cbv-prepare')
			else:
				self.data_folder = os.path.join(os.path.dirname(__file__), 'data', self.CorrMethod)

			# Make sure that the folder exists:
			os.makedirs(self.data_folder, exist_ok=True)

		# Create readonly copy of the TODO-file:
		with tempfile.NamedTemporaryFile(dir=self.input_folder, suffix='.sqlite', delete=False) as tmpfile:
			self.todo_file_readonly = tmpfile.name
			with open(todo_file, 'rb') as fid:
				shutil.copyfileobj(fid, tmpfile)
			tmpfile.flush()

		# Open the SQLite file in read-only mode:
		self.conn = sqlite3.connect('file:' + self.todo_file_readonly + '?mode=ro', uri=True)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()
		self._close_basecorrector()

	#----------------------------------------------------------------------------------------------
	def __del__(self):
		self.close()
		self._close_basecorrector()

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close correction object."""
		pass

	#----------------------------------------------------------------------------------------------
	def _close_basecorrector(self):
		"""Close BaseCorrection object."""
		if hasattr(self, 'cursor') and hasattr(self, 'conn') and self.cursor is not None:
			try:
				self.conn.rollback()
				self.cursor.close()
				self.cursor = None
			except sqlite3.ProgrammingError:
				pass
		if hasattr(self, 'conn') and self.conn is not None:
			self.conn.close()
			self.conn = None
		if hasattr(self, 'todo_file_readonly') and os.path.isfile(self.todo_file_readonly):
			os.remove(self.todo_file_readonly)

	#----------------------------------------------------------------------------------------------
	def plot_folder(self, lc):
		"""
		Return folder path where plots for a given lightcurve should be saved.

		Parameters:
			lc (:class:`lightkurve.TessLightCurve`): Lightcurve to return plot path for.

		Returns:
			string: Path to directory where plots should be saved.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		lcfile = os.path.join(self.input_folder, lc.meta['task']['lightcurve'])
		plot_folder = os.path.join(os.path.dirname(lcfile), 'plots', '%011d' % lc.targetid)
		if self.plot:
			os.makedirs(plot_folder, exist_ok=True)
		return plot_folder

	#----------------------------------------------------------------------------------------------
	def do_correction(self, lightcurve):
		"""
		Apply corrections to target lightcurve.

		Parameters:
			lightcurve (:class:`lightkurve.TessLightCurve`): Lightcurve of the target star
				to be corrected.

		Returns:
			tuple:
			- :class:`STATUS`: The status of the corrections.
			- :class:`lightkurve.TessLightCurve`: corrected lightcurve object.

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError("A helpful error message goes here")

	#----------------------------------------------------------------------------------------------
	def correct(self, task, output_folder=None):
		"""
		Run correction.

		Parameters:
			task (dict): Dictionary defining a task/lightcurve to process.
			output_folder (str, optional): Path to directory where lightcurve should be saved.

		Returns:
			dict: Result dictionary containing information about the processing.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		t1 = default_timer()

		error_msg = []
		details = {}
		save_file = None
		result = task.copy()
		try:
			# Load the lightcurve
			lc = self.load_lightcurve(task)

			# Run the correction on this lightcurve:
			lc_corr, status = self.do_correction(lc)

		except (KeyboardInterrupt, SystemExit): # pragma: no cover
			status = STATUS.ABORT
			logger.warning("Correction was aborted (priority=%d)", task['priority'])
		except: # noqa: E722 pragma: no cover
			status = STATUS.ERROR
			logger.exception("Correction failed (priority=%d)", task['priority'])

		# Check that the status has been changed:
		if status == STATUS.UNKNOWN: # pragma: no cover
			raise ValueError("STATUS was not set by do_correction")

		# Do sanity checks:
		if status in (STATUS.OK, STATUS.WARNING):
			# Make sure all NaN fluxes have corresponding NaN errors:
			lc_corr.flux_err[np.isnan(lc_corr.flux)] = np.NaN

			# Simple check that entire lightcurve is not NaN:
			if allnan(lc_corr.flux):
				logger.error("Final lightcurve is all NaNs")
				status = STATUS.ERROR
			if allnan(lc_corr.flux_err):
				logger.error("Final lightcurve errors are all NaNs")
				status = STATUS.ERROR
			if np.any(np.isinf(lc_corr.flux)):
				logger.error("Final lightcurve contains Inf")
				status = STATUS.ERROR
			if np.any(np.isinf(lc_corr.flux_err)):
				logger.error("Final lightcurve errors contains Inf")
				status = STATUS.ERROR

		# Calculate diagnostics:
		if status in (STATUS.OK, STATUS.WARNING):
			# Calculate diagnostics:
			details['variance'] = nanvar(lc_corr.flux, ddof=1)
			details['rms_hour'] = rms_timescale(lc_corr, timescale=3600/86400)
			details['ptp'] = ptp(lc_corr)

			# Diagnostics specific to the method:
			if self.CorrMethod == 'cbv':
				details['cbv_num'] = lc_corr.meta['additional_headers']['CBV_NUM']
			elif self.CorrMethod == 'ensemble':
				details['ens_num'] = lc_corr.meta['additional_headers']['ENS_NUM']
				details['ens_fom'] = lc_corr.meta['FOM']

			# Save the lightcurve to file:
			try:
				save_file = self.save_lightcurve(lc_corr, output_folder=output_folder)
			except (KeyboardInterrupt, SystemExit): # pragma: no cover
				status = STATUS.ABORT
				logger.warning("Correction was aborted (priority=%d)", task['priority'])
			except: # noqa: E722 pragma: no cover
				status = STATUS.ERROR
				logger.exception("Could not save lightcurve file (priority=%d)", task['priority'])

			# Plot the final lightcurve:
			if self.plot:
				fig = plt.figure(dpi=200)
				ax = fig.add_subplot(111)
				ax.scatter(lc.time, 1e6*(lc.flux/nanmedian(lc.flux)-1), s=2, alpha=0.3, marker='o', label="Original")
				ax.scatter(lc_corr.time, lc_corr.flux, s=2, alpha=0.3, marker='o', label="Corrected")
				ax.set_xlabel('Time (TBJD)')
				ax.set_ylabel('Relative flux (ppm)')
				ax.legend()
				save_figure(os.path.join(self.plot_folder(lc), self.CorrMethod + '_final'), fig=fig)
				plt.close(fig)

		# Unpack any errors or warnings that were sent to the logger during the correction:
		if self.message_queue:
			error_msg += self.message_queue
			self.message_queue.clear()
		if not error_msg:
			error_msg = None

		# Update results:
		t2 = default_timer()
		details['errors'] = error_msg
		result.update({
			'corrector': self.CorrMethod,
			'status_corr': status,
			'elaptime_corr': t2-t1,
			'lightcurve_corr': save_file,
			'details': details
		})

		return result

	#----------------------------------------------------------------------------------------------
	def search_database(self, select=None, join=None, search=None, order_by=None, limit=None,
		distinct=False):
		"""
		Search list of lightcurves and return a list of tasks/stars matching the given criteria.

		Returned rows are restricted to things not marked as ``STATUS.SKIPPED``, since these have
		been deemed too bad to not require corrections, they are definitely also too bad to use in
		any kind of correction.

		Parameters:
			select (list of strings or None): List of table columns to return.
			search (list of strings or None): Conditions to apply to the selection of stars from the database
			order_by (list, str or None): Column to order the database output by.
			limit (int or None): Maximum number of rows to retrieve from the database.
				If limit is None, all the rows are retrieved.
			distinct (bool): Boolean indicating if the query should return unique elements only.
			join (list): Table join commands to merge several database tables together.

		Returns:
			list: All stars retrieved by the call to the database as dicts/tasks
			that can be consumed directly by load_lightcurve

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		if select is None:
			select = '*'
		elif isinstance(select, (list, tuple)):
			select = ",".join(select)

		joins = ['INNER JOIN diagnostics ON todolist.priority=diagnostics.priority']
		if join is None:
			pass
		elif isinstance(join, (list, tuple)):
			joins += list(join)
		else:
			joins.append(join)
		joins = ' '.join(joins)

		if search is None:
			search = ''
		elif isinstance(search, (list, tuple)):
			search = "AND " + " AND ".join(search)
		else:
			search = 'AND ' + search

		if order_by is None:
			order_by = ''
		elif isinstance(order_by, (list, tuple)):
			order_by = " ORDER BY " + ",".join(order_by)
		elif isinstance(order_by, str):
			order_by = " ORDER BY " + order_by

		limit = '' if limit is None else " LIMIT %d" % limit

		query = "SELECT {distinct:s}{select:s} FROM todolist {join:s} WHERE (corr_status IS NULL OR corr_status!={skipped:d}) {search:s}{order_by:s}{limit:s};".format(
			distinct='DISTINCT ' if distinct else '',
			select=select,
			join=joins,
			skipped=STATUS.SKIPPED.value,
			search=search,
			order_by=order_by,
			limit=limit
		)
		logger.debug("Running query: %s", query)

		# Ask the database:
		self.cursor.execute(query)
		return [dict(row) for row in self.cursor.fetchall()]

	#----------------------------------------------------------------------------------------------
	def load_lightcurve(self, task):
		"""
		Load lightcurve from task ID or full task dictionary.

		Parameters:
			task (integer or dict):

		Returns:
			:class:`lightkurve.TessLightCurve`: Lightcurve for the star in question.

		Raises:
			ValueError: On invalid file format.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Find the relevant information in the TODO-list:
		if not isinstance(task, dict) or task.get("lightcurve") is None:
			if isinstance(task, dict):
				priority = int(task['priority'])
			else:
				priority = int(task)

			self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE todolist.priority=? LIMIT 1;", (priority, ))
			task = self.cursor.fetchone()
			if task is None:
				raise ValueError("Priority could not be found in the TODO list")
			task = dict(task)

		# Get the path of the FITS file:
		fname = os.path.join(self.input_folder, task.get('lightcurve'))
		logger.debug('Loading lightcurve: %s', fname)

		# Load lightcurve file and create a TessLightCurve object:
		if fname.endswith(('.fits.gz', '.fits')):
			with fits.open(fname, mode='readonly', memmap=True) as hdu:
				# Filter out invalid parts of the input lightcurve:
				hdu = _filter_fits_hdu(hdu)

				# Quality flags from the pixels:
				pixel_quality = np.asarray(hdu['LIGHTCURVE'].data['PIXEL_QUALITY'], dtype='int32')

				# Corrections applied to timestamps:
				timecorr = hdu['LIGHTCURVE'].data['TIMECORR']

				# Create the QUALITY column and fill it with flags of bad data points:
				quality = np.zeros_like(hdu['LIGHTCURVE'].data['TIME'], dtype='int32')
				bad_data = ~np.isfinite(hdu['LIGHTCURVE'].data['FLUX_RAW'])
				bad_data |= (pixel_quality & TESSQualityFlags.DEFAULT_BITMASK != 0)
				quality[bad_data] |= CorrectorQualityFlags.FlaggedBadData

				# Create lightkurve object:
				lc = TessLightCurve(
					time=hdu['LIGHTCURVE'].data['TIME'],
					flux=hdu['LIGHTCURVE'].data['FLUX_RAW'],
					flux_err=hdu['LIGHTCURVE'].data['FLUX_RAW_ERR'],
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
					meta={'data_rel': hdu[0].header.get('DATA_REL')}
				)

				# Apply manual exclude flag:
				manexcl = manual_exclude(lc)
				lc.quality[manexcl] |= CorrectorQualityFlags.ManualExclude

		elif fname.endswith(('.noisy', '.sysnoise')): # pragma: no cover
			data = np.loadtxt(fname)

			# Quality flags from the pixels:
			pixel_quality = np.asarray(data[:,3], dtype='int32')

			# Corrections applied to timestamps:
			timecorr = np.zeros(data.shape[0], dtype='float32')

			# Change the Manual Exclude flag, since the simulated data
			# and the real TESS quality flags differ in the definition:
			indx = (pixel_quality & 256 != 0)
			pixel_quality[indx] -= 256
			pixel_quality[indx] |= TESSQualityFlags.ManualExclude

			# Create the QUALITY column and fill it with flags of bad data points:
			quality = np.zeros(data.shape[0], dtype='int32')
			bad_data = ~np.isfinite(data[:,1])
			bad_data |= (pixel_quality & TESSQualityFlags.DEFAULT_BITMASK != 0)
			quality[bad_data] |= CorrectorQualityFlags.FlaggedBadData

			# Create lightkurve object:
			lc = TessLightCurve(
				time=data[:,0],
				flux=data[:,1],
				flux_err=data[:,2],
				quality=quality,
				cadenceno=np.arange(1, data.shape[0]+1, dtype='int32'),
				time_format='jd',
				time_scale='tdb',
				targetid=task['starid'],
				label="Star%d" % task['starid'],
				camera=task['camera'],
				ccd=task['ccd'],
				sector=2,
				#ra=0,
				#dec=0,
				quality_bitmask=CorrectorQualityFlags.DEFAULT_BITMASK,
				meta={}
			)

		else:
			raise ValueError("Invalid file format")

		# Add additional attributes to lightcurve object:
		lc.pixel_quality = pixel_quality
		lc.timecorr = timecorr

		# Modify the "extra_columns" tuple of the lightkurve object:
		# This is used internally in lightkurve to keep track of the columns in the
		# object, and make sure they are propergated.
		lc.extra_columns = tuple(list(lc.extra_columns) + ['timecorr', 'pixel_quality'])

		# Keep the original task in the metadata:
		lc.meta['task'] = task
		lc.meta['additional_headers'] = fits.Header()

		if logger.isEnabledFor(logging.DEBUG):
			with contextlib.redirect_stdout(LoggerWriter(logger, logging.DEBUG)):
				lc.show_properties()

		return lc

	#----------------------------------------------------------------------------------------------
	def save_lightcurve(self, lc, output_folder=None):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (str, optional): Path to directory where to save lightcurve.
				If ``None`` the directory specified in the attribute ``output_folder`` is used.

		Returns:
			str: Path to the generated file.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Find the name of the correction method based on the class name:
		CorrMethod = {
			'EnsembleCorrector': 'Ensemble',
			'CBVCorrector': 'CBV',
			'KASOCFilterCorrector': 'KASOC Filter'
		}.get(self.__class__.__name__)

		# Decide where to save the finished lightcurve:
		if output_folder is None:
			output_folder = self.input_folder

		# Get the filename of the original file from the task:
		fname = lc.meta.get('task').get('lightcurve')

		if fname.endswith(('.fits.gz', '.fits')):
			logger.debug("Saving as FITS file")

			if self.CorrMethod == 'cbv':
				filename = os.path.basename(fname).replace('-tasoc_lc', '-tasoc-cbv_lc')
			if self.CorrMethod == 'ensemble':
				filename = os.path.basename(fname).replace('-tasoc_lc', '-tasoc-ens_lc')
			if self.CorrMethod == 'kasoc_filter':
				filename = os.path.basename(fname).replace('-tasoc_lc', '-tasoc-kf_lc')

			if output_folder != self.input_folder:
				save_file = os.path.join(output_folder, filename)
			else:
				save_file = os.path.join(output_folder, os.path.dirname(fname), filename)

			logger.debug("Saving lightcurve to '%s'", save_file)

			# Open the FITS file to overwrite the corrected flux columns:
			with fits.open(os.path.join(self.input_folder, fname), mode='readonly') as hdu:
				# Filter out invalid parts of the input lightcurve:
				hdu = _filter_fits_hdu(hdu)

				# Overwrite the corrected flux columns:
				hdu['LIGHTCURVE'].data['FLUX_CORR'] = lc.flux
				hdu['LIGHTCURVE'].data['FLUX_CORR_ERR'] = lc.flux_err
				hdu['LIGHTCURVE'].data['QUALITY'] = lc.quality

				# Set headers about the correction:
				hdu['LIGHTCURVE'].header['CORRMET'] = (CorrMethod, 'Lightcurve correction method')
				hdu['LIGHTCURVE'].header['CORRVER'] = (__version__, 'Version of correction pipeline')

				# Set additional headers provided by the individual methods:
				if lc.meta['additional_headers']:
					for key, value in lc.meta['additional_headers'].items():
						hdu['LIGHTCURVE'].header[key] = (value, lc.meta['additional_headers'].comments[key])

				# For Ensemble, also add the ensemble list to the FITS file:
				if self.CorrMethod == 'ensemble' and hasattr(self, 'ensemble_starlist'):
					# Create binary table to hold the list of ensemble stars:
					c1 = fits.Column(name='TIC', format='K', array=self.ensemble_starlist['starids'])
					c2 = fits.Column(name='BZETA', format='E', array=self.ensemble_starlist['bzetas'])

					wm = fits.BinTableHDU.from_columns([c1, c2], name='ENSEMBLE')
					wm.header['TDISP1'] = 'I10'
					wm.header['TDISP2'] = 'E26.17'
					fix_fits_table_headers(wm, {
						'TIC': 'TIC identifier',
						'BZETA': 'background scale'
					})

					# Add the new table to the list of HDUs:
					hdu.append(wm)

				# Write the modified HDUList to the new filename:
				hdu.writeto(save_file, checksum=True, overwrite=True)

		# For the simulated ASCII files, simply create a new ASCII files next to the original one,
		# with an extension ".corr":
		elif fname.endswith(('.noisy', '.sysnoise')): # pragma: no cover
			save_file = os.path.join(output_folder, os.path.dirname(fname), os.path.splitext(os.path.basename(fname))[0] + '.corr')

			# Create new ASCII file:
			with open(save_file, 'w') as fid:
				fid.write("# TESS Asteroseismic Science Operations Center\n")
				fid.write("# TIC identifier:     %d\n" % lc.targetid)
				fid.write("# Sector:             %s\n" % lc.sector)
				fid.write("# Correction method:  %s\n" % CorrMethod)
				fid.write("# Correction Version: %s\n" % __version__)
				if lc.meta['additional_headers']:
					for key, value in lc.meta['additional_headers'].items():
						fid.write("# %-18s: %s\n" % (key, value))
				fid.write("#\n")
				fid.write("# Column 1: Time (days)\n")
				fid.write("# Column 2: Corrected flux (ppm)\n")
				fid.write("# Column 3: Corrected flux error (ppm)\n")
				fid.write("# Column 4: Quality flags\n")
				fid.write("#-------------------------------------------------\n")
				for k in range(len(lc.time)):
					fid.write("%f  %.16e  %.16e  %d\n" % (
						lc.time[k],
						lc.flux[k],
						lc.flux_err[k],
						lc.quality[k]
					))
				fid.write("#-------------------------------------------------\n")

		# Store the output file in the details object for future reference:
		save_file = os.path.relpath(save_file, output_folder).replace('\\', '/')

		return save_file
