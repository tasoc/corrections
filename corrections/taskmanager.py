#!/usr/bin/env python
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
from numpy import atleast_1d
from . import STATUS

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, overwrite=False, cleanup_constraints=None,
		summary=None, summary_interval=100):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file (string): Path to the TODO-file.
			cleanup (boolean, optional): Perform cleanup/optimization of TODO-file before
				during initialization. Default=False.
			overwrite (boolean, optional): Overwrite any previously calculated results. Default=False.
			cleanup_constraints (dict, optional): Dict of constraint for cleanup of the status of
				previous correction runs. If not specified, all bad results are cleaned up.
			summary (string, optional): Path to JSON file which will be periodically updated with
				a status summary of the corrections.
			summary_interval (integer, optional): Interval at which summary file is updated.
				Default=100.

		Raises:
			FileNotFoundError: If TODO-file could not be found.
		"""

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.isfile(todo_file):
			raise FileNotFoundError('Could not find TODO-file')

		# Load the SQLite file:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()
		self.cursor.execute("PRAGMA foreign_keys=ON;")
		self.cursor.execute("PRAGMA locking_mode=NORMAL;") # Needs to be NORMAL, since we need to have multiple processes read at the same time
		self.cursor.execute("PRAGMA journal_mode=TRUNCATE;")

		self.summary_file = summary
		self.summary_interval = summary_interval
		self.corrector = None

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		self.logger.addHandler(console)
		self.logger.setLevel(logging.INFO)

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

		# Add status indicator for corrections to todolist, if it doesn't already exists:
		self.cursor.execute("PRAGMA table_info(todolist)")
		if 'corr_status' not in [r['name'] for r in self.cursor.fetchall()]:
			self.logger.debug("Adding corr_status column to todolist")
			self.cursor.execute("ALTER TABLE todolist ADD COLUMN corr_status INTEGER DEFAULT NULL")
			self.cursor.execute("CREATE INDEX corr_status_idx ON todolist (corr_status);")
			self.conn.commit()

		# Create indicies
		self.cursor.execute("CREATE INDEX IF NOT EXISTS datavalidation_raw_approved_idx ON datavalidation_raw (approved);")
		self.conn.commit()

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
			cc = cleanup_constraints.copy()
			if isinstance(cc, dict):
				if cc.get('datasource'):
					constraints.append("datasource='ffi'" if cc.pop('datasource') == 'ffi' else "datasource!='ffi'")
				for key, val in cc.items():
					if val is not None:
						constraints.append(key + ' IN (%s)' % ','.join([str(v) for v in atleast_1d(val)]))
			elif isinstance(cc, list):
				constraints += cc
			else:
				raise ValueError("cleanup_constraints should be dict or list")

		constraints = ' AND '.join(constraints)
		self.cursor.execute("DELETE FROM diagnostics_corr WHERE priority IN (SELECT todolist.priority FROM todolist WHERE " + constraints + ");")
		self.cursor.execute("UPDATE todolist SET corr_status=NULL WHERE " + constraints + ";")
		self.conn.commit()

		# Set all targets that did not return good photometry or were not approved by the Data Validation to SKIPPED:
		self.cursor.execute("UPDATE todolist SET corr_status=%d WHERE corr_status IS NULL AND (status NOT IN (%d,%d) OR todolist.priority IN (SELECT priority FROM datavalidation_raw WHERE approved=0));" % (
			STATUS.SKIPPED.value,
			STATUS.OK.value,
			STATUS.WARNING.value
		))
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
			# Write summary to file:
			self.write_summary()

		# Run a cleanup/optimization of the database before we get started:
		if cleanup:
			self.logger.info("Cleaning TODOLIST before run...")
			try:
				self.conn.isolation_level = None
				self.cursor.execute("VACUUM;")
			finally:
				self.conn.isolation_level = ''

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()

	#----------------------------------------------------------------------------------------------
	def close(self):
		if self.cursor and self.conn:
			self.conn.rollback()
			self.cursor.execute("PRAGMA journal_mode=DELETE;")
			self.conn.commit()

		if self.cursor: self.cursor.close()
		if self.conn: self.conn.close()

	#----------------------------------------------------------------------------------------------
	def save_settings(self):

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
		except:
			self.conn.rollback()
			raise

	#----------------------------------------------------------------------------------------------
	def get_number_tasks(self, starid=None, camera=None, ccd=None, datasource=None, priority=None):
		"""
		Get number of tasks due to be processed.

		Returns:
			int: Number of tasks due to be processed.
		"""

		constraints = []
		if priority is not None:
			constraints.append('todolist.priority=%d' % priority)
		if starid is not None:
			constraints.append('todolist.starid=%d' % starid)
		if camera is not None:
			constraints.append('todolist.camera=%d' % camera)
		if ccd is not None:
			constraints.append('todolist.ccd=%d' % ccd)
		if datasource is not None:
			constraints.append("todolist.datasource='ffi'" if datasource == 'ffi' else "todolist.datasource!='ffi'")

		if constraints:
			constraints = ' AND ' + " AND ".join(constraints)
		else:
			constraints = ''

		self.cursor.execute("SELECT COUNT(*) AS num FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE corr_status IS NULL %s;" % (
			constraints,
		))

		num = int(self.cursor.fetchone()['num'])
		return num

	#----------------------------------------------------------------------------------------------
	def get_task(self, starid=None, camera=None, ccd=None, datasource=None, priority=None):
		"""
		Get next task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""

		constraints = []
		if priority is not None:
			constraints.append('todolist.priority=%d' % priority)
		if starid is not None:
			constraints.append('todolist.starid=%d' % starid)
		if camera is not None:
			constraints.append('todolist.camera=%d' % camera)
		if ccd is not None:
			constraints.append('todolist.ccd=%d' % ccd)
		if datasource is not None:
			constraints.append("todolist.datasource='ffi'" if datasource == 'ffi' else "todolist.datasource!='ffi'")

		if constraints:
			constraints = ' AND ' + " AND ".join(constraints)
		else:
			constraints = ''

		self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE corr_status IS NULL %s ORDER BY todolist.priority LIMIT 1;" % (
			constraints,
		))
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	#----------------------------------------------------------------------------------------------
	def save_results(self, result):

		# Extract details dictionary:
		details = result.get('details', {})

		# The status of this target returned by the photometry:
		my_status = result['status_corr']

		# If the corrector has not already been set for this TODO-file,
		# update the settings, and if it has check that we are not
		# mixing results from different correctors in one TODO-file.
		if self.corrector is None:
			self.corrector = result['corrector']
			self.save_settings()
		elif result['corrector'] != self.corrector:
			raise ValueError("Attempting to mix results from multiple correctors")

		try:
			# Update the status in the TODO list:
			self.cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (
				result['status_corr'].value,
				result['priority']
			))

			self.summary['tasks_run'] += 1
			self.summary[my_status.name] += 1
			self.summary['STARTED'] -= 1

			# Save additional diagnostics:
			error_msg = details.get('errors', None)
			if error_msg:
				error_msg = "\n".join(error_msg) if isinstance(error_msg, (list, tuple)) else error_msg.strip()
				self.summary['last_error'] = error_msg

			additional_diags_keys = ''
			additional_diags = ()
			if self.corrector == 'cbv':
				additional_diags_keys = ',cbv_num'
				additional_diags = (
					details.get('cbv_num', None),
				)
			elif self.corrector == 'ensemble':
				additional_diags_keys = ',ens_num,ens_fom'
				additional_diags = (
					details.get('ens_num', None),
					details.get('ens_fom', None)
				)

			# Save additional diagnostics:
			placeholders = ','.join(['?']*(8+len(additional_diags)))
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
		except: # noqa: E722
			self.conn.rollback()
			raise

		# Calculate mean elapsed time using "streaming weighted mean" with (alpha=0.1):
		# https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl
		if self.summary['mean_elaptime'] is None and result.get('elaptime_corr') is not None:
			self.summary['mean_elaptime'] = result['elaptime_corr']
		elif result.get('elaptime_corr') is not None:
			self.summary['mean_elaptime'] += 0.1 * (result['elaptime_corr'] - self.summary['mean_elaptime'])

		if self.summary['mean_worker_waittime'] is None and result.get('worker_wait_time') is not None:
			self.summary['mean_worker_waittime'] = result['worker_wait_time']
		elif result.get('worker_wait_time') is not None:
			self.summary['mean_worker_waittime'] += 0.1 * (result['worker_wait_time'] - self.summary['mean_worker_waittime'])

		# Write summary file:
		if self.summary_file and self.summary['tasks_run'] % self.summary_interval == 0:
			self.write_summary()

	#----------------------------------------------------------------------------------------------
	def start_task(self, taskid):
		"""
		Mark a task as STARTED in the TODO-list.
		"""
		self.cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (STATUS.STARTED.value, taskid))
		self.conn.commit()
		self.summary['STARTED'] += 1

	#----------------------------------------------------------------------------------------------
	def get_random_task(self):
		"""
		Get random task to be processed.
		Returns:
			dict or None: Dictionary of settings for task.
		"""
		self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE corr_status IS NULL ORDER BY RANDOM() LIMIT 1;")
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	#----------------------------------------------------------------------------------------------
	def write_summary(self):
		"""Write summary of progress to file. The summary file will be in JSON format."""
		if self.summary_file:
			try:
				with open(self.summary_file, 'w') as fid:
					json.dump(self.summary, fid)
			except: # noqa: E722
				self.logger.exception("Could not write summary file")
