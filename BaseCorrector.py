#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic correction class for the TASOC Photomety pipeline.
All other specific correction classes will inherit from BaseCorrector.

.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Filipe Pereira
"""

from __future__ import division, with_statement, print_function, absolute_import
import six
import os.path
import shutil
import enum
import logging
import sqlite3
import traceback
import numpy as np
from timeit import default_timer
from bottleneck import nanmedian, nanvar
from astropy.io import fits
from lightkurve import TessLightCurve
from .version import get_version
from .quality import TESSQualityFlags, CorrectorQualityFlags
from .utilities import rms_timescale

__version__ = get_version()

__docformat__ = 'restructuredtext'

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

	def __init__(self, input_folder, output_folder, plot=False, debug=False):
		"""
		Initialize the corrector.

		Parameters:
			input_folder (string):
			plot (boolean, optional):

		Raises:
			IOError: If (target ID) could not be found (TODO: other values as well?)
			ValueError: (TODO: on a lot of places)

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Save inputs:
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.data_folder = os.path.join(os.path.dirname(__file__), 'data')
		self.plot = plot
		self.debug = debug

		# The path to the TODO list:
		todo_file = os.path.join(input_folder, 'todo.sqlite')
		logger.debug("TODO file: %s", todo_file)
		if not os.path.exists(todo_file):
			raise ValueError("TODO file not found")

		# Open the SQLite file:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()


	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		"""Close correction object."""
		if self.cursor: self.cursor.close()
		if self.conn: self.conn.close()


	def plot_folder(self, lc):
		"""
		Return folder path where plots for a given lightcurve should be saved.

		Parameters:
			lc (``lightkurve.TessLightCurve``): Lightcurve to return plot path for.

		Returns:
			string: Path to directory where plots should be saved.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		lcfile = os.path.join(self.output_folder, lc.meta['task']['lightcurve'])
		plot_folder = os.path.join(os.path.dirname(lcfile), 'plots', '%011d' % lc.targetid)
		return plot_folder


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
		raise NotImplementedError("A helpful error message goes here") # TODO


	def correct(self, task):
		"""
		Run correction.

		Parameters:
			task (dict): Dictionary defining a task/lightcurve to process.

		Returns:
			dict: Result dictionary containing information about the processing.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		t1 = default_timer()

		error_msg = None
		save_file = None
		result = task.copy()
		try:
			# Load the lightcurve
			lc = self.load_lightcurve(task)

			# Run the correction on this lightcurve:
			lc_corr, status = self.do_correction(lc.copy())

		except (KeyboardInterrupt, SystemExit):
			status = STATUS.ABORT
			logger.warning("Correction was aborted.")

		except:
			status = STATUS.ERROR
			error_msg = traceback.format_exc().strip()
			logger.exception("Correction failed.")

		# Check that the status has been changed:
		if status == STATUS.UNKNOWN:
			raise Exception("STATUS was not set by do_correction")

		# Calculate diagnostics:
		details = {}

		if status in (STATUS.OK, STATUS.WARNING):
			# Calculate diagnostics:
			details['variance'] = nanvar(lc_corr.flux, ddof=1)
			#details['rms_hour'] = rms_timescale(lc_corr, timescale=3600/86400)
			details['ptp'] = nanmedian(np.abs(np.diff(lc_corr.flux)))

			# TODO: set outputs; self._details = self.lightcurve, etc.
			save_file = self.save_lightcurve(lc_corr)

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
			search = "AND " + " AND ".join(search)
		else:
			search = 'AND ' + search

		if order_by is None:
			order_by = ''
		elif isinstance(order_by, (list, tuple)):
			order_by = " ORDER BY " + ",".join(order_by)
		elif isinstance(order_by, six.string_types):
			order_by = " ORDER BY " + order_by

		limit = '' if limit is None else " LIMIT %d" % limit

		query = "SELECT {distinct:s}{select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status=1 {search:s}{order_by:s}{limit:s};".format(
			distinct='DISTINCT ' if distinct else '',
			select=select,
			search=search,
			order_by=order_by,
			limit=limit
		)
		logger.debug("Running query: %s", query)

		# Ask the database:
		self.cursor.execute(query)
		return [dict(row) for row in self.cursor.fetchall()]


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
			self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE todolist.priority=? LIMIT 1;", (task, ))
			task = self.cursor.fetchone()
			if task is None:
				logger.info('Task could not be loaded')
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
				quality_bitmask=CorrectorQualityFlags.DEFAULT_BITMASK
			)

		elif fname.endswith('.fits') or fname.endswith('.fits.gz'):
			with fits.open(fname, mode='readonly', memmap=True) as hdu:
				# Quality flags from the pixels:
				pixel_quality = np.asarray(hdu['LIGHTCURVE'].data['PIXEL_QUALITY'], dtype='int32')

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
					meta={
						'Tmag' : hdu[0].header.get('TESSMAG')
					}
				)

		else:
			raise ValueError("Invalid file format")

		# Add additional attributes to lightcurve object:
		lc.pixel_quality = pixel_quality

		# Keep the original task in the metadata:
		lc.meta['task'] = task
		lc.meta['additional_headers'] = fits.Header()

		if logger.isEnabledFor(logging.DEBUG):
			lc.show_properties()

		return lc.copy()


	def save_lightcurve(self, lc):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (string, optional): Path to directory where to save lightcurve. If ``None`` the directory specified in the attribute ``output_folder`` is used.

		Returns:
			string: Path to the generated file.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		logger = logging.getLogger(__name__)

		# Find the name of the correction method based on the class name:
		CorrMethod = {
			'EnsembleCorrector': 'Ensemble',
			'CBVCorrector': 'CBV',
			'KASOCFilterCorrector': 'KASOC Filter'
		}.get(self.__class__.__name__)

		# Decide where to save the finished lightcurve:
		if self.output_folder is None:
			output_folder = self.input_folder
		else:
			output_folder = self.output_folder

		# Get the filename of the original file from the task:
		fname = lc.meta.get('task').get('lightcurve')

		if fname.endswith('.fits') or fname.endswith('.fits.gz'):
			#if output_folder != self.input_folder:
			save_file = os.path.join(output_folder, 'corr-' + os.path.basename(fname))
			shutil.copy(os.path.join(self.input_folder, fname), save_file)

			# Open the FITS file to overwrite the corrected flux columns:
			with fits.open(os.path.join(self.input_folder,fname), mode='update') as hdu:
				# Overwrite the corrected flux columns:
				hdu['LIGHTCURVE'].data['FLUX_CORR'] = lc.flux
				#hdu['LIGHTCURVE'].data['FLUX_CORR_ERR'] = lc.flux_err
				#hdu['LIGHTCURVE'].data['QUALITY'] = lc.quality

				# Set headers about the correction:
				hdu['LIGHTCURVE'].header['CORRMET'] = (CorrMethod, 'Lightcurve correction method')
				hdu['LIGHTCURVE'].header['CORRVER'] = (__version__, 'Version of correction pipeline')

				# Set additional headers provided by the individual methods:
				if lc.meta['additional_headers']:
					for key, value in lc.meta['additional_headers'].items():
						hdu['LIGHTCURVE'].header[key] = (value, lc.meta['additional_headers'].comments[key])

				# Save the updated FITS file:
				# hdu.flush()

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