#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A TaskManager which keeps track of which targets to process.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Filipe Pereira
"""

import os.path
import sqlite3
import logging
import json
from . import STATUS

#--------------------------------------------------------------------------------------------------
def _build_constraints(priority=None, starid=None, sector=None, cadence=None,
	camera=None, ccd=None, cbv_area=None, return_list=False):

	constraints = []
	if priority is not None:
		constraints.append(f'todolist.priority={priority:d}')
	if starid is not None:
		constraints.append(f'todolist.starid={starid:d}')
	if sector is not None:
		constraints.append(f'todolist.sector={sector:d}')
	if cadence == 'ffi':
		constraints.append("todolist.datasource='ffi'")
	elif cadence is not None:
		constraints.append(f'todolist.cadence={cadence:d}')
	if camera is not None:
		constraints.append(f'todolist.camera={camera:d}')
	if ccd is not None:
		constraints.append(f'todolist.ccd={ccd:d}')
	if cbv_area is not None:
		constraints.append(f'todolist.cbv_area={cbv_area:d}')

	#if args.camera:
	#	constraints.append('camera IN (%s)' % ",".join([str(c) for c in args.camera]))
	#if args.ccd:
	#	constraints.append('ccd IN (%s)' % ",".join([str(c) for c in args.ccd]))
	#if args.area:
	#	constraints.append('cbv_area IN (%s)' % ",".join([str(c) for c in args.area]))

	# If asked for it, return the list if constraints otherwise return string
	# which fits into the other queries done by the TaskManager:
	if return_list:
		return constraints
	return ' AND ' + ' AND '.join(constraints) if constraints else ''

#--------------------------------------------------------------------------------------------------
class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, overwrite=False, cleanup_constraints=None,
		summary=None, summary_interval=200):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file (str): Path to the TODO-file.
			cleanup (bool, optional): Perform cleanup/optimization of TODO-file before
				during initialization. Default=False.
			overwrite (bool, optional): Overwrite any previously calculated results. Default=False.
			cleanup_constraints (dict, optional): Dict of constraint for cleanup of the status of
				previous correction runs. If not specified, all bad results are cleaned up.
			summary (str, optional): Path to JSON file which will be periodically updated with
				a status summary of the corrections.
			summary_interval (int, optional): Interval at which summary file is updated.
				Default=100.

		Raises:
			FileNotFoundError: If TODO-file could not be found.
		"""

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.isfile(todo_file):
			raise FileNotFoundError('Could not find TODO-file')

		if cleanup_constraints is not None and not isinstance(cleanup_constraints, (dict, list)):
			raise ValueError("cleanup_constraints should be dict or list")

		# Load the SQLite file:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()
		self.cursor.execute("PRAGMA foreign_keys=ON;")
		self.cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")
		self.cursor.execute("PRAGMA journal_mode=TRUNCATE;")

		self.summary_file = summary
		self.summary_interval = summary_interval
		self.summary_counter = 0
		self.corrector = None

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(logging.INFO)
		if not self.logger.hasHandlers():
			self._loghandler = logging.StreamHandler()
			self._loghandler.setFormatter(formatter)
			self.logger.addHandler(self._loghandler)

		# Add cadence to todolist, if it doesn't already exists:
		# This is only for backwards compatibility.
		self.cursor.execute("PRAGMA table_info(todolist)")
		existing_columns = [r['name'] for r in self.cursor.fetchall()]
		if 'cadence' not in existing_columns:
			self.logger.debug("Adding CADENCE column to todolist")
			self.cursor.execute("BEGIN TRANSACTION;")
			self.cursor.execute("ALTER TABLE todolist ADD COLUMN cadence INTEGER DEFAULT NULL;")
			self.cursor.execute("UPDATE todolist SET cadence=1800 WHERE datasource='ffi' AND sector < 27;")
			self.cursor.execute("UPDATE todolist SET cadence=600 WHERE datasource='ffi' AND sector >= 27 AND sector <= 55;")
			self.cursor.execute("UPDATE todolist SET cadence=120 WHERE datasource!='ffi' AND sector < 27;")
			self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist WHERE cadence IS NULL;")
			if self.cursor.fetchone()['antal'] > 0:
				self.close()
				raise ValueError("TODO-file does not contain CADENCE information and it could not be determined automatically. Please recreate TODO-file.")
			self.conn.commit()

		# Add status indicator for corrections to todolist, if it doesn't already exists:
		if 'corr_status' not in existing_columns:
			self.logger.debug("Adding corr_status column to todolist")
			self.cursor.execute("ALTER TABLE todolist ADD COLUMN corr_status INTEGER DEFAULT NULL")
			self.cursor.execute("CREATE INDEX corr_status_idx ON todolist (corr_status);")
			self.conn.commit()

		# Add method_used to the diagnostics table if it doesn't exist:
		self.cursor.execute("PRAGMA table_info(diagnostics)")
		if 'method_used' not in [r['name'] for r in self.cursor.fetchall()]:
			# Since this one is NOT NULL, we have to do some magic to fill out the
			# new column after creation, by finding ketwords in other columns.
			# This can be a pretty slow process, but it only has to be done once.
			self.logger.debug("Adding method_used column to diagnostics")
			self.cursor.execute("BEGIN TRANSACTION;")
			self.cursor.execute("ALTER TABLE diagnostics ADD COLUMN method_used TEXT NOT NULL DEFAULT 'aperture';")
			for m in ('aperture', 'halo', 'psf', 'linpsf'):
				self.cursor.execute("UPDATE diagnostics SET method_used=? WHERE priority IN (SELECT priority FROM todolist WHERE method=?);", [m, m])
			self.cursor.execute("UPDATE diagnostics SET method_used='halo' WHERE method_used='aperture' AND errors LIKE '%Automatically switched to Halo photometry%';")
			self.conn.commit()

		# Create indicies
		self.cursor.execute("CREATE INDEX IF NOT EXISTS datavalidation_raw_approved_idx ON datavalidation_raw (approved);")
		self.conn.commit()

		# Create table for settings if it doesn't already exits:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS corr_settings (
			corrector TEXT NOT NULL
		);""")
		self.conn.commit()

		# Load settings from setting tables:
		self.cursor.execute("SELECT * FROM corr_settings LIMIT 1;")
		row = self.cursor.fetchone()
		if row is not None:
			self.corrector = row['corrector']

		# Reset the status of everything for a new run:
		if overwrite:
			self.cursor.execute("UPDATE todolist SET corr_status=NULL;")
			self.cursor.execute("DROP TABLE IF EXISTS diagnostics_corr;")
			self.cursor.execute("DELETE FROM corr_settings;")
			self.conn.commit()
			self.corrector = None

		# Create table for diagnostics:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS diagnostics_corr (
			priority INTEGER PRIMARY KEY ASC NOT NULL,
			lightcurve TEXT,
			elaptime REAL,
			worker_wait_time REAL,
			variance DOUBLE PRECISION,
			rms_hour DOUBLE PRECISION,
			ptp DOUBLE PRECISION,
			errors TEXT,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS diagnostics_corr_lightcurve_idx ON diagnostics_corr (lightcurve);")
		self.conn.commit()

		# The corrector is not stored, so try to infer it from the diagnostics information:
		# This is needed on older TODO-files created before the corr_settings table
		# as introduced.
		if self.corrector is None:
			self.cursor.execute("SELECT lightcurve FROM diagnostics_corr WHERE lightcurve IS NOT NULL LIMIT 1;")
			row = self.cursor.fetchone()
			if row is not None:
				if '-tasoc-cbv_lc' in row['lightcurve']:
					self.corrector = 'cbv'
				elif '-tasoc-ens_lc' in row['lightcurve']:
					self.corrector = 'ensemble'
				elif '-tasoc-kf_lc' in row['lightcurve']:
					self.corrector = 'kasoc_filter'

				if self.corrector is not None:
					self.save_settings()

		# Reset calculations with status STARTED, ABORT or ERROR:
		clear_status = str(STATUS.STARTED.value) + ',' + str(STATUS.ABORT.value) + ',' + str(STATUS.ERROR.value) + ',' + str(STATUS.SKIPPED.value)
		constraints = ['corr_status IN (' + clear_status + ')']

		# Add additional constraints from the user input and build SQL query:
		if cleanup_constraints:
			if isinstance(cleanup_constraints, dict):
				constraints += _build_constraints(**cleanup_constraints, return_list=True)
			else:
				constraints += cleanup_constraints

		constraints = ' AND '.join(constraints)
		self.logger.debug(constraints)
		self.cursor.execute("DELETE FROM diagnostics_corr WHERE priority IN (SELECT todolist.priority FROM todolist WHERE " + constraints + ");")
		self.cursor.execute("UPDATE todolist SET corr_status=NULL WHERE " + constraints + ";")
		self.conn.commit()

		# Set all targets that did not return good photometry or were not approved by the Data Validation to SKIPPED:
		self.cursor.execute(f"UPDATE todolist SET corr_status={STATUS.SKIPPED.value:d} WHERE corr_status IS NULL AND (status NOT IN ({STATUS.OK.value:d},{STATUS.WARNING.value:d}) OR todolist.priority IN (SELECT priority FROM datavalidation_raw WHERE approved=0));")
		self.conn.commit()

		# Analyze the tables for better query planning:
		self.logger.debug("Analyzing database...")
		self.cursor.execute("ANALYZE;")

		# Prepare summary object:
		self.summary = {
			'slurm_jobid': os.environ.get('SLURM_JOB_ID', None),
			'numtasks': 0,
			'tasks_run': 0,
			'last_error': None,
			'mean_elaptime': None,
			'mean_worker_waittime': None
		}
		# Make sure to add all the different status to summary:
		for s in STATUS:
			self.summary[s.name] = 0
		# If we are going to output summary, make sure to fill it up:
		if self.summary_file:
			# Extract information from database:
			self.cursor.execute("SELECT corr_status,COUNT(*) AS cnt FROM todolist GROUP BY corr_status;")
			for row in self.cursor.fetchall():
				self.summary['numtasks'] += row['cnt']
				if row['corr_status'] is not None:
					self.summary[STATUS(row['corr_status']).name] = row['cnt']
			# Make sure the containing directory exists:
			if not os.path.isdir(os.path.dirname(self.summary_file)):
				os.makedirs(os.path.dirname(self.summary_file))
			# Write summary to file:
			self.write_summary()

		# Run a cleanup/optimization of the database before we get started:
		if cleanup:
			self.logger.info("Cleaning TODOLIST before run...")
			tmp_isolevel = self.conn.isolation_level
			try:
				self.conn.isolation_level = None
				self.cursor.execute("VACUUM;")
			finally:
				self.conn.isolation_level = tmp_isolevel

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()

	#----------------------------------------------------------------------------------------------
	def __del__(self):
		self.close()

	#----------------------------------------------------------------------------------------------
	def close(self):
		if hasattr(self, 'cursor') and hasattr(self, 'conn') and self.conn:
			try:
				self.conn.rollback()
				self.cursor.execute("PRAGMA journal_mode=DELETE;")
				self.conn.commit()
				self.cursor.close()
			except sqlite3.ProgrammingError: # pragma: no cover
				pass

		if hasattr(self, 'conn') and self.conn:
			self.conn.close()
			self.conn = None

		if hasattr(self, '_loghandler') and hasattr(self, 'logger') and self._loghandler:
			self.logger.removeHandler(self._loghandler)

	#----------------------------------------------------------------------------------------------
	def save_settings(self):
		"""
		Save settings to TODO-file and create method-specific columns in ``diagnostics_corr`` table.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		try:
			self.cursor.execute("DELETE FROM corr_settings;")
			self.cursor.execute("INSERT INTO corr_settings (corrector) VALUES (?);", [self.corrector])

			# Create additional diagnostics columns based on which corrector we are running:
			self.cursor.execute("PRAGMA table_info(diagnostics_corr)")
			diag_columns = [r['name'] for r in self.cursor.fetchall()]
			if self.corrector == 'cbv':
				if 'cbv_num' not in diag_columns:
					self.cursor.execute("ALTER TABLE diagnostics_corr ADD COLUMN cbv_num INTEGER DEFAULT NULL;")

			elif self.corrector == 'ensemble':
				if 'ens_num' not in diag_columns:
					self.cursor.execute("ALTER TABLE diagnostics_corr ADD COLUMN ens_num INTEGER DEFAULT NULL;")
				if 'ens_fom' not in diag_columns:
					self.cursor.execute("ALTER TABLE diagnostics_corr ADD COLUMN ens_fom REAL DEFAULT NULL;")

			self.conn.commit()
		except: # noqa: E722, pragma: no cover
			self.conn.rollback()
			raise

	#----------------------------------------------------------------------------------------------
	def get_number_tasks(self, priority=None, starid=None, sector=None, cadence=None,
		camera=None, ccd=None, cbv_area=None):
		"""
		Get number of tasks due to be processed.

		Parameters:
			priority (int, optional): Only return task matching this priority.
			starid (int, optional): Only return tasks matching this starid.
			sector (int, optional): Only return tasks matching this Sector.
			cadence (int, optional): Only return tasks matching this cadence.
			camera (int, optional): Only return tasks matching this camera.
			ccd (int, optional): Only return tasks matching this CCD.

		Returns:
			int: Number of tasks due to be processed.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		constraints = _build_constraints(
			priority=priority,
			starid=starid,
			sector=sector,
			cadence=cadence,
			camera=camera,
			ccd=ccd,
			cbv_area=cbv_area)

		self.cursor.execute("SELECT COUNT(*) AS num FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE corr_status IS NULL" + constraints + ";")
		return int(self.cursor.fetchone()['num'])

	#----------------------------------------------------------------------------------------------
	def get_task(self, priority=None, starid=None, sector=None, cadence=None,
		camera=None, ccd=None, cbv_area=None, chunk=1):
		"""
		Get next task to be processed.

		Parameters:
			priority (int, optional): Only return task matching this priority.
			starid (int, optional): Only return tasks matching this starid.
			sector (int, optional): Only return tasks matching this Sector.
			cadence (int, optional): Only return tasks matching this cadence.
			camera (int, optional): Only return tasks matching this camera.
			ccd (int, optional): Only return tasks matching this CCD.
			chunk (int, optional): Chunk of tasks to return. Default is to not chunk (=1).

		Returns:
			dict, list or None: Dictionary of settings for task.
				If ``chunk`` is larger than one, a list of dicts is retuned instead.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		constraints = _build_constraints(
			priority=priority,
			starid=starid,
			sector=sector,
			cadence=cadence,
			camera=camera,
			ccd=ccd,
			cbv_area=cbv_area)

		self.cursor.execute(f"SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE corr_status IS NULL {constraints:s} ORDER BY todolist.priority LIMIT {chunk:d};")
		tasks = self.cursor.fetchall()
		if tasks and chunk == 1:
			return dict(tasks[0])
		elif tasks:
			return [dict(task) for task in tasks]
		return None

	#----------------------------------------------------------------------------------------------
	def get_random_task(self, chunk=1):
		"""
		Get random task to be processed.

		Parameters:
			chunk (int, optional): Chunk of tasks to return. Default is to not chunk (=1).

		Returns:
			dict, list or None: Dictionary of settings for task.
				If ``chunk`` is larger than one, a list of dicts is retuned instead.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		self.cursor.execute(f"SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE corr_status IS NULL ORDER BY RANDOM() LIMIT {chunk:d};")
		tasks = self.cursor.fetchall()
		if tasks and chunk == 1:
			return dict(tasks[0])
		elif tasks:
			return [dict(task) for task in tasks]
		return None

	#----------------------------------------------------------------------------------------------
	def start_task(self, tasks):
		"""
		Mark tasks as STARTED in the TODO-list.

		Parameters:
			tasks (list or dict): Task or list of tasks coming from ``get_tasks``
				or ``get_random_task``.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		if isinstance(tasks, dict):
			priorities = [(int(tasks['priority']),)]
		else:
			priorities = [(int(task['priority']),) for task in tasks]

		self.cursor.executemany(f"UPDATE todolist SET corr_status={STATUS.STARTED.value:d} WHERE priority=?;", priorities)
		self.summary['STARTED'] += self.cursor.rowcount
		self.conn.commit()

	#----------------------------------------------------------------------------------------------
	def save_results(self, results):
		"""
		Save result, or list of results, to TaskManager.

		Parameters:
			results (list or dict):

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		if isinstance(results, dict):
			results = [results]

		if self.corrector is None:
			self.corrector = results[0]['corrector']
			self.save_settings()

		additional_diags_keys = ''
		Nadditional = 0
		if self.corrector == 'cbv':
			additional_diags_keys = ',cbv_num'
			Nadditional = 1
		elif self.corrector == 'ensemble':
			additional_diags_keys = ',ens_num,ens_fom'
			Nadditional = 2
		placeholders = ','.join(['?']*(8 + Nadditional))

		for result in results:
			# Extract details dictionary:
			details = result.get('details', {})

			# The status of this target returned by the photometry:
			my_status = result['status_corr']

			# If the corrector has not already been set for this TODO-file,
			# update the settings, and if it has check that we are not
			# mixing results from different correctors in one TODO-file.
			if result['corrector'] != self.corrector:
				raise ValueError("Attempting to mix results from multiple correctors")

			# Calculate mean elapsed time using "streaming weighted mean" with (alpha=0.1):
			# https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl
			if self.summary['mean_elaptime'] is None and result.get('elaptime_corr') is not None:
				self.summary['mean_elaptime'] = result['elaptime_corr']
			elif result.get('elaptime_corr') is not None:
				self.summary['mean_elaptime'] += 0.1 * (result['elaptime_corr'] - self.summary['mean_elaptime'])

			# Save additional diagnostics:
			error_msg = details.get('errors', None)
			if error_msg:
				error_msg = "\n".join(error_msg) if isinstance(error_msg, (list, tuple)) else error_msg.strip()
				self.summary['last_error'] = error_msg

			additional_diags = ()
			if self.corrector == 'cbv':
				additional_diags = (
					details.get('cbv_num', None),
				)
			elif self.corrector == 'ensemble':
				additional_diags = (
					details.get('ens_num', None),
					details.get('ens_fom', None)
				)

			try:
				# Update the status in the TODO list:
				self.cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (
					result['status_corr'].value,
					result['priority']
				))

				# Save additional diagnostics:
				self.cursor.execute("INSERT OR REPLACE INTO diagnostics_corr (priority,lightcurve,elaptime,worker_wait_time,variance,rms_hour,ptp,errors" + additional_diags_keys + ") VALUES (" + placeholders + ");", (
					result['priority'],
					result.get('lightcurve_corr', None),
					result.get('elaptime_corr', None),
					result.get('worker_wait_time', None),
					details.get('variance', None),
					details.get('rms_hour', None),
					details.get('ptp', None),
					error_msg
				) + additional_diags)
				self.conn.commit()
			except: # noqa: E722, pragma: no cover
				self.conn.rollback()
				raise

			self.summary['tasks_run'] += 1
			self.summary[my_status.name] += 1
			self.summary['STARTED'] -= 1

		# All the results should have the same worker_waittime.
		# So only update this once, using just that last result in the list:
		if self.summary['mean_worker_waittime'] is None and result.get('worker_wait_time') is not None:
			self.summary['mean_worker_waittime'] = result['worker_wait_time']
		elif result.get('worker_wait_time') is not None:
			self.summary['mean_worker_waittime'] += 0.1 * (result['worker_wait_time'] - self.summary['mean_worker_waittime'])

		# Write summary file:
		self.summary_counter += len(results)
		if self.summary_file and self.summary_counter >= self.summary_interval:
			self.summary_counter = 0
			self.write_summary()

	#----------------------------------------------------------------------------------------------
	def write_summary(self):
		"""Write summary of progress to file. The summary file will be in JSON format."""
		if self.summary_file:
			try:
				with open(self.summary_file, 'w') as fid:
					json.dump(self.summary, fid)
			except: # noqa: E722, pragma: no cover
				self.logger.exception("Could not write summary file")
