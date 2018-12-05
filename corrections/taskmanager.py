#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A TaskManager which keeps track of which targets to process.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Lindsey Carboneau
"""

from __future__ import division, with_statement, print_function, absolute_import
import os.path
import sqlite3
import logging
from . import STATUS

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file: Path to the TODO-file.

		Raises:
			IOError: If TODO-file could not be found.
		"""

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.exists(todo_file):
			raise IOError('Could not find TODO-file')

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
			errors TEXT
		);""")

		self.cursor.execute("UPDATE todolist SET corr_status=NULL")
		self.conn.commit()


	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		if self.cursor: self.cursor.close()
		if self.conn: self.conn.close()

	def get_task(self, starid=None):
		"""
		Get next task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""

		constraints = []
		if starid is not None:
			constraints.append('todolist.starid=%d' % starid)

		if constraints:
			constraints = ' AND ' + " AND ".join(constraints)
		else:
			constraints = ''

		self.cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE status IN (%d,%d) AND corr_status IS NULL %s ORDER BY priority LIMIT 1;" % (
			STATUS.OK.value,
			STATUS.WARNING.value,
			constraints
		))
		task = self.cursor.fetchone()
		if task is not None: task = dict(task)
		return task

	def save_results(self, result):

		details = result.get('details', {})

		try:
			# Update the status in the TODO list:
			self.cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (
				result['status_corr'].value,
				result['priority']
			))

			# Save additional diagnostics:
			self.cursor.execute("INSERT INTO diagnostics_corr (priority, lightcurve, elaptime, variance, rms_hour, ptp, errors) VALUES (?,?,?,?,?,?,?);", (
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
		raise NotImplementedError("A helpful error message goes here") # TODO
