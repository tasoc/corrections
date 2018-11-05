#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic correction class for the TASOC Photomety pipeline.
All other specific correction classes will inherit from BaseCorrector.
Structure from `BasePhotometry by Rasmus Handberg <https://github.com/tasoc/photometry/blob/devel/photometry/BasePhotometry.py>`_

- :py:class:`STATUS`: Status flags for pipeline performance logging

.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import os.path
import shutil
import enum
import logging
import sqlite3
import numpy as np
from astropy.io import fits
from lightkurve import TessLightCurve
from .version import get_version

__version__ = get_version()

__docformat__ = 'restructuredtext'

class STATUS(enum.Enum):
	"""
	Status indicator of the status of the correction.

	"""

	UNKNOWN = 0
	OK = 1
	ERROR = 2
	WARNING = 3
	STARTED = 6
	# TODO: various statuses as required

class BaseCorrector(object):
	"""
	The basic correction class for the TASOC Photometry pipeline.
	All other specific correction classes will inherit from BaseCorrector.

	Attributes:
		# TODO

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
			IOError: If (target ID) could not be found (TODO: other values as well?)
			ValueError: (TODO: on a lot of places)

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Save inputs:
		self.input_folder = input_folder
		self.plot = plot

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
		lcfile = os.path.join(self.input_folder, lc.meta['task']['lightcurve'])
		plot_folder = os.path.join(os.path.dirname(lcfile), 'plots', '%011d' % lc.targetid)
		return plot_folder


	def do_correction(self, lightcurve):
		"""
		Apply corrections to target lightcurve.

		Parameters:
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve of the target star to be corrected.

		Returns:
			The status of the corrections.

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError("A helpful error message goes here") # TODO


	def correct(self, task):
		"""
		Run correction.

		Parameters:
			task (dict): Dictionary defining a task/lightcurve to process.

		"""

		logger = logging.getLogger(__name__)

		# Load the lightcurve
		lc = self.load_lightcurve(task)

		# Run the correction on this lightcurve:
		try:
			lc, status = self.do_correction(lc)

		except (KeyboardInterrupt, SystemExit):
			status = STATUS.ABORT
			logger.warning("Correction was aborted.")

		except:
			status = STATUS.ERROR
			logger.exception("Correction failed.")

		# Check that the status has been changed:
		if status == STATUS.UNKNOWN:
			raise Exception("STATUS was not set by do_correction")

		if status in (STATUS.OK, STATUS.WARNING):
			# TODO: set outputs; self._details = self.lightcurve, etc.
			self.save_lightcurve(lc)

		return status


	def search_lightcurves(self, search=None, order_by=None, limit=None):
		"""
		Search list of lightcurves and return a list of tasks/stars matching the given criteria.

		Parameters:
			cbv_area (integer or None): Only return stars from this CBV area.

		Returns:
			list of dicts:

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		if search is None:
			search = ''
		elif isinstance(search, (list, tuple)):
			search = " AND ".join(search)

		order_by = '' if order_by is None else " ORDER BY " + order_by
		limit = '' if limit is None else " LIMIT %d" % limit

		query = "SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status=1 AND {search:s}{order_by:s}{limit:s};".format(
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
		if not isinstance(task, dict):
			self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE todolist.priority=? LIMIT 1;", (task, ))
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
			lc = TessLightCurve(
				time=data[:,0],
				flux=data[:,1],
				flux_err=data[:,2],
				quality=np.asarray(data[:,3], dtype='int32'),
				time_format='jd',
				time_scale='tdb',
				targetid=task['starid'],
				label="Star%d" % task['starid'],
				camera=task['camera'],
				ccd=task['ccd'],
				sector=2,
				#ra=0,
				#dec=0,
				quality_bitmask=2+8+256 # lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK
			)

		elif fname.endswith('.fits') or fname.endswith('.fits.gz'):
			with fits.open(fname, mode='readonly', memmap=True) as hdu:
				lc = TessLightCurve(
					time=hdu['LIGHTCURVE'].data['TIME'],
					flux=hdu['LIGHTCURVE'].data['FLUX_RAW'],
					flux_err=hdu['LIGHTCURVE'].data['FLUX_RAW_ERR'],
					centroid_col=hdu['LIGHTCURVE'].data['MOM_CENTR1'],
					centroid_row=hdu['LIGHTCURVE'].data['MOM_CENTR2'],
					quality=np.asarray(hdu['LIGHTCURVE'].data['QUALITY'], dtype='int32'),
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
					quality_bitmask=2+8+256 # lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK
				)

		else:
			raise ValueError("Invalid file format")

		# Keep the original task in the metadata:
		lc.meta['task'] = task
		lc.meta['additional_headers'] = {}

		if logger.isEnabledFor(logging.DEBUG):
			lc.show_properties()

		return lc


	def save_lightcurve(self, lc, output_folder=None):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (string, optional): Path to directory where to save lightcurve. If ``None`` the directory specified in the attribute ``output_folder`` is used.

		Returns:
			string: Path to the generated file.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Find the name of the correction method based on the class name:
		CorrMethod = {
			'EnsembleCorrector': 'ensemble',
			'CBVCorrector': 'cbv',
			'KASOCFilterCorrector': 'KASOC Filter'
		}.get(self.__class__.__name__)

		# Decide where to save the finished lightcurve:
		if output_folder is None:
			output_folder = self.input_folder

		# Get the filename of the original file from the task:
		fname = lc.meta.get('task').get('lightcurve')

		if fname.endswith('.fits') or fname.endswith('.fits.gz'):
			#if output_folder != self.input_folder:
			save_file = os.path.join(output_folder, os.path.dirname(fname), 'corr-' + os.path.basename(fname))
			shutil.copy(os.path.join(self.input_folder, fname), save_file)

			# Open the FITS file to overwrite the corrected flux columns:
			with fits.open(save_file, mode='update') as hdu:
				# Overwrite the corrected flux columns:
				hdu['LIGHTCURVE'].data['FLUX_CORR'] = lc.flux
				hdu['LIGHTCURVE'].data['FLUX_CORR_ERR'] = lc.flux_err

				# Set headers about the correction:
				hdu['LIGHTCURVE'].header['CORRMET'] = (CorrMethod, 'Lightcurve correction method')
				hdu['LIGHTCURVE'].header['CORRVER'] = (__version__, 'Version of correction pipeline')

				# Set additional headers provided by the individual methods:
				if lc.meta['additional_headers']:
					for key, value in lc.meta['additional_headers'].items():
						hdu['LIGHTCURVE'].header[key] = value

				# Save the updated FITS file:
				hdu.flush()

		# For the simulated ASCII files, simply create a new ASCII files next to the original one,
		# with an extension ".corr":
		elif fname.endswith('.noisy') or fname.endswith('.sysnoise'):
			save_file = os.path.join(output_folder, os.path.dirname(fname), os.path.splitext(fname)[0] + '.corr')

			# Create new ASCII file:
			with open(save_file, 'w') as fid:
				fid.write("# TESS Asteroseismic Science Operations Center\n")
				fid.write("# TIC identifier:     %d\n" % lc.targetid)
				fid.write("# Sector:             %s\n" % lc.sector)
				fid.write("# Correction method:  %s\n" % CorrMethod)
				fid.write("# Correction Version: %s\n" % __version__)
				if lc.meta['additional_headers']:
					for key, value in lc.meta['additional_headers'].items():
						fid.write("# %18s: %s\n" % (key, value[0]))
				fid.write("#\n")
				fid.write("# Column 1: Time (days)\n")
				fid.write("# Column 2: Corrected flux (ppm)\n")
				fid.write("# Column 3: Corrected flux error (ppm)\n")
				fid.write("# Column 4: Quality flags\n")
				fid.write("#-------------------------------------------------\n")
				for k in range(len(lc.time)):
					fid.write("%f  %e  %e  %d\n" % (
						lc.time[k],
						lc.flux[k],
						lc.flux_err[k],
						lc.quality[k]
					))
				fid.write("#-------------------------------------------------\n")

		return save_file
