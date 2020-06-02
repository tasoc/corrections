#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BaseCorrection.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import sqlite3
import conftest # noqa: F401
from corrections import BaseCorrector

#--------------------------------------------------------------------------------------------------
def test_import_nonexistent(INPUT_DIR):
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
if __name__ == '__main__':
	pytest.main([__file__])
