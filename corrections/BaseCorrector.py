#!/usr/bin/env python
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
import numpy as np
from timeit import default_timer
from bottleneck import nanmedian, nanvar
from astropy.io import fits
from lightkurve import TessLightCurve
from .plots import plt, save_figure
from .quality import TESSQualityFlags, CorrectorQualityFlags
from .utilities import rms_timescale, ListHandler
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
class BaseCorrector(object):
	"""
	The basic correction class for the TASOC Photometry pipeline.
	All other specific correction classes will inherit from BaseCorrector.

	Attributes:
		plot (boolean): Boolean indicating if plotting is enabled.
		data_folder (string): Path to directory where auxillary data for the corrector
			should be stored.

	.. codeauthor:: Lindsey Carboneau
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, input_folder, plot=False):
		"""
		Initialize the corrector.

		Parameters:
			input_folder (string):
			plot (boolean, optional):

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

		# The path to the TODO list:
		logger.debug("TODO file: %s", todo_file)
		if not os.path.isfile(todo_file):
			raise FileNotFoundError("TODO file not found")

		# Open the SQLite file in read-only mode:
		self.conn = sqlite3.connect('file:' + todo_file + '?mode=ro', uri=True)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()

	#----------------------------------------------------------------------------------------------
	def __del__(self):
		if hasattr(self, 'cursor') and self.cursor: self.cursor.close()
		if hasattr(self, 'conn') and self.conn: self.conn.close()

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close correction object."""
		pass

	#----------------------------------------------------------------------------------------------
	def plot_folder(self, lc):
		"""
		Return folder path where plots for a given lightcurve should be saved.

		Parameters:
			lc (``lightkurve.TessLightCurve``): Lightcurve to return plot path for.

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
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve of the target star to be corrected.

		Returns:
			The status of the corrections and the corrected lightcurve object.

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
			output_folder (string, optional): Path to directory where lightcurve should be saved.

		Returns:
			dict: Result dictionary containing information about the processing.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		t1 = default_timer()

		error_msg = []
		save_file = None
		result = task.copy()
		try:
			# Load the lightcurve
			lc = self.load_lightcurve(task)

			# Run the correction on this lightcurve:
			lc_corr, status = self.do_correction(lc)

		except (KeyboardInterrupt, SystemExit):
			status = STATUS.ABORT
			logger.warning("Correction was aborted (priority=%d)", task['priority'])

		except: # noqa: E722
			status = STATUS.ERROR
			logger.exception("Correction failed (priority=%d)", task['priority'])

		# Check that the status has been changed:
		if status == STATUS.UNKNOWN:
			raise Exception("STATUS was not set by do_correction")

		# Calculate diagnostics:
		details = {}

		# Unpack any errors or warnings that were sent to the logger during the correction:
		if self.message_queue:
			error_msg += self.message_queue
			self.message_queue.clear()
		if not error_msg:
			error_msg = None

		if status in (STATUS.OK, STATUS.WARNING):
			# Calculate diagnostics:
			details['variance'] = nanvar(lc_corr.flux, ddof=1)
			details['rms_hour'] = rms_timescale(lc_corr, timescale=3600/86400)
			details['ptp'] = nanmedian(np.abs(np.diff(lc_corr.flux)))

			# TODO: set outputs; self._details = self.lightcurve, etc.
			save_file = self.save_lightcurve(lc_corr, output_folder=output_folder)

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

			# Construct result dictionary from the original task
			result = lc_corr.meta['task'].copy()

		# Update results:
		t2 = default_timer()
		details['errors'] = error_msg
		result.update({
			'status_corr': status,
			'elaptime_corr': t2-t1,
			'lightcurve_corr': save_file,
			'details': details
		})

		return result

	#----------------------------------------------------------------------------------------------
	def search_database(self, select=None, join=None, search=None, order_by=None, limit=None, distinct=False):
		"""
		Search list of lightcurves and return a list of tasks/stars matching the given criteria.

		Returned rows are restricted to things not marked as ``STATUS.SKIPPED``, since these have
		been deemed too bad to not require corrections, they are definitely also too bad to use in
		any kind of correction.

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
			``lightkurve.TessLightCurve``: Lightcurve for the star in question.

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
		if fname.endswith('.noisy') or fname.endswith('.sysnoise'):
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

		elif fname.endswith('.fits') or fname.endswith('.fits.gz'):
			with fits.open(fname, mode='readonly', memmap=True) as hdu:
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
					meta={}
				)

				# Apply manual exclude flag:
				manexcl = manual_exclude(lc)
				lc.quality[manexcl] |= CorrectorQualityFlags.ManualExclude
		else:
			raise ValueError("Invalid file format")

		# Add additional attributes to lightcurve object:
		lc.pixel_quality = pixel_quality
		lc.timecorr = timecorr

		# Keep the original task in the metadata:
		lc.meta['task'] = task
		lc.meta['additional_headers'] = fits.Header()

		if logger.isEnabledFor(logging.DEBUG):
			lc.show_properties()

		return lc

	#----------------------------------------------------------------------------------------------
	def save_lightcurve(self, lc, output_folder=None):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (string, optional): Path to directory where to save lightcurve. If ``None`` the directory specified in the attribute ``output_folder`` is used.

		Returns:
			string: Path to the generated file.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
		"""

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

		if fname.endswith('.fits') or fname.endswith('.fits.gz'):

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

			shutil.copy(os.path.join(self.input_folder, fname), save_file)

			# Change permission of copied file to allow the addition of the corrected lightcurve
			os.chmod(save_file, 0o640)

			# Open the FITS file to overwrite the corrected flux columns:
			with fits.open(save_file, mode='update') as hdu:
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

					wm = fits.BinTableHDU.from_columns([c1, ], name='ENSEMBLE')

					wm.header['TTYPE1'] = ('TIC', 'column title: TIC identifier')
					wm.header['TFORM1'] = ('K', 'column format: signed 64-bit integer')
					wm.header['TDISP1'] = ('I10', 'column display format')

					# Add the new table to the list of HDUs:
					hdu.append(wm)

				# Save the updated FITS file:
#				hdu.flush()

		# For the simulated ASCII files, simply create a new ASCII files next to the original one,
		# with an extension ".corr":
		elif fname.endswith('.noisy') or fname.endswith('.sysnoise'):
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
