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
from corrections import BaseCorrector

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#----------------------------------------------------------------------
def test_import():
	"""
	Tests if the module can even be imported.

	Doesn't really do anything else..."""

	with BaseCorrector(INPUT_DIR) as bc:
		assert bc.__class__.__name__ == 'BaseCorrector', "Did not get the correct class name back"
		assert bc.input_folder == INPUT_DIR, "Incorrect input folder"

#----------------------------------------------------------------------
def test_import_nonexistent():
	"""
	Tests that BaseCorrector handles being called with non-existing input directory.
	"""

	with pytest.raises(ValueError):
		with BaseCorrector(INPUT_DIR + '/does/not/exist/'):
			pass

#----------------------------------------------------------------------
def test_search_database():
	"""
	Tests the search_database method of BaseCorrector.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	test_priority = 17

	with BaseCorrector(INPUT_DIR) as bc:

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

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_import()
	test_import_nonexistent()
	test_search_database()
