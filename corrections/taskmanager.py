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
	
	def get_task(self, starid=None):
		"""
		Get next task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""
		raise NotImplementedError("A helpful error message goes here") # TODO

	def get_random_task(self):
		"""
		Get random task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""
		raise NotImplementedError("A helpful error message goes here") # TODO
		