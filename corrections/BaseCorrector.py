#!/usr/bin/env python
"""
The basic correction class for the TASOC Photomety pipeline
All other specific correction classes will inherit from BaseCorrection.
Structure from `BasePhotometry by Rasmus Handberg <https://github.com/tasoc/photometry/blob/devel/photometry/BasePhotometry.py>`_

- :py:class:`STATUS`: Status flags for pipeline performance logging

.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import os.path
import enum
import logging
import sqlite3
import numpy as np
from astropy.io import fits
from lightkurve import TessLightCurve

__docformat__ = 'restructuredtext'

class STATUS(enum.Enum):
	"""
	Status indicator of the status of the correction.

	"""

	UNKNOWN = 0
	OK = 1
	ERROR = 2
	WARNING = 3
	# TODO: various statuses as required

class BaseCorrector(object):
	"""
	The basic correction class for the TASOC Photometry pipeline.
	All other specific correction classes will inherit from BaseCorrector

	Attributes:
		# TODO
	"""

	def __init__(self, input_folder):
		"""
		Initialize the correction object

		Parameters:
			# TODO

		Returns:
			# TODO

		Raises:
			IOError: If (target ID) could not be found (TODO: other values as well?)
			ValueError: (TODO: on a lot of places)
			NotImplementedError: Everywhere a function has a TODO/FIXME tag preventing execution
		"""

		logger = logging.getLogger(__name__)

		# Save inputs:
		self.input_folder = input_folder
		
		# The path to the TODO list:
		todo_file = os.path.join(input_dir, 'todo.sqlite')
		logger.debug("TODO file: %s", todo_file)
		if not os.path.exists(todo_file):
			raise ValueError("TODO file not found")
		
		# Open the SQLite file:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()
		
		self._status = STATUS.UNKNOWN

	def status(self):
		""" The status of the corrections. From :py:class:`STATUS`."""
		return self._status

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		"""Close correction object."""
		pass

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
		
		# Load the lightcurve 
		lc = self.load_lightcurve(task['priority'])

		self._status = self.do_correction(lc)

		# Check that the status has been changed:
		if self._status == STATUS.UNKNOWN:
			raise Exception("STATUS was not set by do_correction")

		if self._status in (STATUS.OK, STATUS.WARNING):
			# TODO: set outputs; self._details = self.lightcurve, etc.
			pass

	def load_lightcurve(self, task):
		"""
		Load lightcurve from task ID or full task dictionary.
		
		Parameters:
			task (integer or dict):

		Returns:
			``lightkurve.TessLightCurve``: Lightcurve for the star in question.
		
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
		fname = os.path.join(self.input_dir, task.get('lightcurve_filename'))

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
				ticid=task['starid'],
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
					quality=np.asarray(hdu['LIGHTCURVE'].data['QUALITY'], dtype='int32'),
					time_format='jd',
					time_scale='tdb',
					ticid=hdu[0].header.get('TICID'),
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

		return lc
			
	def save_lightcurve(self, lc, output_folder=None):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (string, optional): Path to directory where to save lightcurve. If ``None`` the directory specified in the attribute ``output_folder`` is used.

		Returns:
			string: Path to the generated file.
		"""
		
		# Get the filename of the original file from the task:
		fname = lc.meta.get('task').get('lightcurve_filename')
		
		# Decide where to save the finished lightcurve:
		if output_folder is not None:
			shutil.copy(os.path.join(self.input_folder, fname), output_folder)
		else:
			output_folder = self.input_folder
	
		# Open the FITS file to overwrite the corrected flux columns:
		with fits.open(os.path.join(output_folder, fname)) as hdu:
			# Overwrite the corrected flux columns:
			hdu['LIGHTCURVE'].data['FLUX_CORR'] = lc.flux
			hdu['LIGHTCURVE'].data['FLUX_CORR_ERR'] = lc.flux_err

			# Set headers about the correction:
			
			
			# Set additional headers provided by the individual methods:
			if lc.meta['additional_headers']:
				for key, value in lc.meta['additional_headers'].items():
					hdu[1].header[key] = value
			
			# Save the updated FITS file:
			#hdu.writeto(os.path.join(output_folder, fname), overwrite=True, checksum=True)
			
		return fname
