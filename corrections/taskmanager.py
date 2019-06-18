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
from . import STATUS

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, overwrite=False):
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
			self.cursor.execute("ALTER TABLE todolist ADD COLUMN corr_status INT DEFAULT NULL")
			self.cursor.execute("CREATE INDEX corr_status_idx ON todolist (corr_status);")
			self.conn.commit()

		# Reset the status of everything for a new run:
		if overwrite:
			self.cursor.execute("UPDATE todolist SET corr_status=NULL;")
			self.cursor.execute("DROP TABLE IF EXISTS diagnostics_corr;")
			self.conn.commit()

		# Create table for diagnostics:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS diagnostics_corr (
			priority INT PRIMARY KEY NOT NULL,
			lightcurve TEXT,
			elaptime REAL NOT NULL,
			variance DOUBLE PRECISION,
			rms_hour DOUBLE PRECISION,
			ptp DOUBLE PRECISION,
			errors TEXT,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.conn.commit()

		# Reset calculations with status STARTED or ABORT:
		clear_status = str(STATUS.STARTED.value) + ',' + str(STATUS.ABORT.value)
		self.cursor.execute("DELETE FROM diagnostics_corr WHERE priority IN (SELECT todolist.priority FROM todolist WHERE corr_status IN (" + clear_status + "));")
		self.cursor.execute("UPDATE todolist SET corr_status=NULL WHERE corr_status IN (" + clear_status + ");")
		self.conn.commit()

		# Run a cleanup/optimization of the database before we get started:
		if cleanup:
			self.logger.info("Cleaning TODOLIST before run...")
			try:
				self.conn.isolation_level = None
				self.cursor.execute("VACUUM;")
			except:
				raise
			finally:
				self.conn.isolation_level = ''

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		if self.cursor: self.cursor.close()
		if self.conn: self.conn.close()
		
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

		self.cursor.execute("SELECT COUNT(*) AS num FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status IN (%d,%d) AND corr_status IS NULL %s ORDER BY todolist.priority LIMIT 1;" % (
			STATUS.OK.value,
			STATUS.WARNING.value,
			constraints
		))
		
		num = int(self.cursor.fetchone()['num'])
		return num
	

	def get_task(self, camera=None, ccd=None, datasource=None):
		"""
		Get next task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""

		constraints = []
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

		print(constraints)

		self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status IN (%d,%d) AND corr_status IS NULL %s ORDER BY todolist.priority LIMIT 1;" % (
			STATUS.OK.value,
			STATUS.WARNING.value,
			constraints
		))
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	def save_results(self, result):

		# Extract details dictionary:
		details = result.get('details', {})

		try:
			# Update the status in the TODO list:
			self.cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (
				result['status_corr'].value,
				result['priority']
			))

			# Save additional diagnostics:
			self.cursor.execute("INSERT OR REPLACE INTO diagnostics_corr (priority, lightcurve, elaptime, variance, rms_hour, ptp, errors) VALUES (?,?,?,?,?,?,?);", (
				result['priority'],
				result['lightcurve_corr'],
				result['elaptime_corr'],
				details.get('variance', None),
				details.get('rms_hour', None),
				details.get('ptp', None),
				details.get('errors', None)
			))
			self.conn.commit()
		except:
			self.conn.rollback()
			raise

	def start_task(self, taskid):
		"""
		Mark a task as STARTED in the TODO-list.
		"""
		self.cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (STATUS.STARTED.value, taskid))
		self.conn.commit()

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
