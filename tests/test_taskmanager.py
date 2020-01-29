#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of corrections.TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import TaskManager, STATUS

#--------------------------------------------------------------------------------------------------
def test_taskmanager():
	"""Test of TaskManager"""

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	# Find the shape of the original image:
	with TaskManager(INPUT_DIR, overwrite=True) as tm:
		# Get the number of tasks:
		numtasks = tm.get_number_tasks()
		print(numtasks)
		assert(numtasks == 62700)

		# Get the first task in the TODO file:
		task1 = tm.get_task()
		print(task1)

		# Check that it contains what we know it should:
		# The first priority in the TODO file is the following:
		assert(task1['priority'] == 17)
		assert(task1['starid'] == 29281992)
		assert(task1['camera'] == 1)
		assert(task1['ccd'] == 4)
		assert(task1['datasource'] == 'ffi')
		assert(task1['sector'] == 1)
		assert(task1['cbv_area'] == 143)

		# Start task with priority=17:
		tm.start_task(17)

		# Get the next task, which should be the one with priority=2:
		task2 = tm.get_task()
		print(task2)

		assert(task2['priority'] == 18)
		assert(task2['starid'] == 29281992)
		assert(task2['camera'] == 1)
		assert(task2['ccd'] == 4)
		assert(task2['datasource'] == 'tpf')
		assert(task2['sector'] == 1)
		assert(task2['cbv_area'] == 143)

		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT corr_status FROM todolist WHERE priority=17;")
		task1_status = tm.cursor.fetchone()['corr_status']
		print(task1_status)

		assert(task1_status == STATUS.STARTED.value)

#--------------------------------------------------------------------------------------------------
def test_taskmanager_constraints():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	constraints = {'datasource': 'tpf', 'priority': 17}
	with TaskManager(INPUT_DIR, overwrite=True, cleanup_constraints=constraints) as tm:
		task = tm.get_task(**constraints)
		print(task)
		assert task is None, "Task1 should be None"

	constraints = {'datasource': 'ffi', 'priority': 17}
	with TaskManager(INPUT_DIR, overwrite=True, cleanup_constraints=constraints) as tm:
		task = tm.get_task(**constraints)
		print(task)
		assert task['priority'] == 17, "Task2 should be #17"

#--------------------------------------------------------------------------------------------------
def test_taskmanager_cleanup():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	# Reset the TODO-file completely, and mark the first task as STARTED:
	with TaskManager(INPUT_DIR, overwrite=True) as tm:
		task1 = tm.get_task()
		print(task1)
		pri = task1['priority']
		tm.start_task(pri)

	# Cleanup, but with a constraint not matching the one we changed:
	with TaskManager(INPUT_DIR, cleanup_constraints={'priority': 18}) as tm:
		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT corr_status FROM todolist WHERE priority=?;", [pri])
		task1_status = tm.cursor.fetchone()['corr_status']
		print(task1_status)
		assert task1_status == STATUS.STARTED.value

	# Now clean with a constraint that matches:
	with TaskManager(INPUT_DIR, cleanup_constraints={'priority': pri}) as tm:
		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT corr_status FROM todolist WHERE priority=?;", [pri])
		task1_status = tm.cursor.fetchone()['corr_status']
		print(task1_status)
		assert task1_status is None

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	test_taskmanager()
	test_taskmanager_constraints()
	test_taskmanager_cleanup()
