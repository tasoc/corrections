#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BaseCorrection.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import sqlite3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import BaseCorrector, TaskManager, STATUS

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#--------------------------------------------------------------------------------------------------
def test_import_nonexistent():
	"""
	Tests that BaseCorrector handles being called with non-existing input directory.
	"""
	with pytest.raises(FileNotFoundError):
		with BaseCorrector(INPUT_DIR + '/does/not/exist/'):
			pass

#--------------------------------------------------------------------------------------------------
def test_search_database(PRIVATE_TODO_FILE):
	"""
	Tests the search_database method of BaseCorrector.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	test_priority = 17

	with BaseCorrector(PRIVATE_TODO_FILE) as bc:

		res1 = bc.search_database(search='todolist.priority=%d' % test_priority)
		print(res1)

		assert isinstance(res1, list), "Expects to get a list of results"
		assert len(res1) == 1, "There should only be one result for this query"
		assert isinstance(res1[0], dict), "Expects each result to be a dict"

		res2 = bc.search_database(search=['todolist.priority=%d' % test_priority], limit=10)
		print(res2)
		assert res1 == res2, "Should give same result with list"

		res3 = bc.search_database(search=('todolist.priority=%d' % test_priority,), limit=5)
		print(res3)
		assert res1 == res3, "Should give same result with tuple"

		res4 = bc.search_database(search='todolist.priority=%d' % test_priority, distinct=True)
		print(res4)
		assert res1 == res4, "Should give same result with distinct=True"

		# Make sure that we are not able to change anything in the TODO-file:
		with pytest.raises(sqlite3.OperationalError):
			bc.cursor.execute("UPDATE todolist SET priority=-{0:d} WHERE priority={0:d};".format(test_priority))
		bc.conn.rollback()

#--------------------------------------------------------------------------------------------------
def test_search_database_changes(PRIVATE_TODO_FILE):
	"""
	Test wheter changing corr_status will change what is returned by search_database
	"""

	with BaseCorrector(PRIVATE_TODO_FILE) as bc:
		rows1 = bc.search_database(search=['todolist.starid=29281992'])
		print(rows1)

	with TaskManager(PRIVATE_TODO_FILE) as tm:
		task = tm.get_task(priority=17)
		tm.start_task(task)

	with BaseCorrector(PRIVATE_TODO_FILE) as bc:
		rows2 = bc.search_database(search=['todolist.starid=29281992'])
		print(rows2)

	# Only the corr_status column was allowed to change!
	assert len(rows1) == len(rows2)
	assert rows1[0]['corr_status'] is None
	assert rows2[0]['corr_status'] == STATUS.STARTED.value
	for k in range(len(rows1)):
		r1 = rows1[k]
		r1.pop('corr_status')
		r2 = rows2[k]
		r2.pop('corr_status')
		assert r1 == r2

#----------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
