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
from . import STATUS

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, overwrite=False, summary=None, summary_interval=100):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file (string): Path to the TODO-file.
			cleanup (boolean, optional): Perform cleanup/optimization of TODO-file before
				during initialization. Default=False.
			overwrite (boolean, optional): Overwrite any previously calculated results. Default=False.

		Raises:
			FileNotFoundError: If TODO-file could not be found.
		"""

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.exists(todo_file):
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

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		self.logger.addHandler(console)
		self.logger.setLevel(logging.INFO)

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
			self.conn.commit()

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

		# Reset calculations with status STARTED or ABORT:
		clear_status = str(STATUS.STARTED.value) + ',' + str(STATUS.ABORT.value) + ',' + str(STATUS.ERROR.value)
		self.cursor.execute("DELETE FROM diagnostics_corr WHERE priority IN (SELECT todolist.priority FROM todolist WHERE corr_status IN (" + clear_status + "));")
		self.cursor.execute("UPDATE todolist SET corr_status=NULL WHERE corr_status IN (" + clear_status + ");")
		self.conn.commit()

		# Analyze the tables for better query planning:
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
		for s in STATUS: self.summary[s.name] = 0
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
		if self.cursor: self.cursor.close()
		if self.conn: self.conn.close()

	#----------------------------------------------------------------------------------------------
	def get_number_tasks(self, starid=None, camera=None, ccd=None, datasource=None):
		"""
		Get number of tasks due to be processed.

		Returns:
			int: Number of tasks due to be processed.
		"""

		constraints = []
		if starid is not None:
			constraints.append('todolist.starid=%d' % starid)
		if camera is not None:
			constraints.append('todolist.camera=%d' % camera)
		if ccd is not None:
			constraints.append('todolist.ccd=%d' % ccd)
		if datasource is not None:
			constraints.append('todolist.datasource="%s"' % datasource)

		if constraints:
			constraints = ' AND ' + " AND ".join(constraints)
		else:
			constraints = ''

		self.cursor.execute("SELECT COUNT(*) AS num FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority INNER JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority WHERE status IN (%d,%d) AND corr_status IS NULL AND datavalidation_raw.approved=1 %s;" % (
			STATUS.OK.value,
			STATUS.WARNING.value,
			constraints
		))

		num = int(self.cursor.fetchone()['num'])
		return num

	#----------------------------------------------------------------------------------------------
	def get_task(self, starid=None, camera=None, ccd=None, datasource=None):
		"""
		Get next task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""

		constraints = []
		if starid is not None:
			constraints.append('todolist.starid=%d' % starid)
		if camera is not None:
			constraints.append('todolist.camera=%d' % camera)
		if ccd is not None:
			constraints.append('todolist.ccd=%d' % ccd)
		if datasource is not None:
			constraints.append('todolist.datasource="%s"' % datasource)

		if constraints:
			constraints = ' AND ' + " AND ".join(constraints)
		else:
			constraints = ''

		self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority INNER JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority WHERE status IN (%d,%d) AND corr_status IS NULL AND datavalidation_raw.approved=1 %s ORDER BY todolist.priority LIMIT 1;" % (
			STATUS.OK.value,
			STATUS.WARNING.value,
			constraints
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
				self.summary['last_error'] = error_msg

			# Save additional diagnostics:
			self.cursor.execute("INSERT OR REPLACE INTO diagnostics_corr (priority, lightcurve, elaptime, worker_wait_time, variance, rms_hour, ptp, errors) VALUES (?,?,?,?,?,?,?,?);", (
				result['priority'],
				result.get('lightcurve_corr', None),
				result.get('elaptime_corr', None),
				result.get('worker_wait_time', None),
				details.get('variance', None),
				details.get('rms_hour', None),
				details.get('ptp', None),
				details.get('errors', None)
			))
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
		self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status IN (1,3) AND corr_status IS NULL ORDER BY RANDOM() LIMIT 1;")
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
