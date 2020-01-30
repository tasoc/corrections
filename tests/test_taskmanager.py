#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of corrections.TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import sys
import os.path
import tempfile
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import TaskManager, STATUS

#--------------------------------------------------------------------------------------------------
def test_taskmanager():
	"""Test of TaskManager"""

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	# Find the shape of the original image:
	with TaskManager(INPUT_DIR, overwrite=True, cleanup=True) as tm:
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
def test_taskmanager_notexist():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input', 'does-not-exist')
	with pytest.raises(FileNotFoundError):
		with TaskManager(INPUT_DIR):
			pass

#--------------------------------------------------------------------------------------------------
def test_taskmanager_constraints():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	constraints = {'datasource': 'tpf', 'priority': 17}
	with TaskManager(INPUT_DIR, overwrite=True, cleanup_constraints=constraints) as tm:
		task = tm.get_task(**constraints)
		print(task)
		assert task is None, "Task1 should be None"

	constraints = {'datasource': 'tpf', 'priority': 17, 'camera': None}
	with TaskManager(INPUT_DIR, overwrite=True, cleanup_constraints=constraints) as tm:
		task2 = tm.get_task(**constraints)
		print(task2)
		assert task2 == task, "Tasks should be identical"

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
def test_taskmanager_summary():

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	with tempfile.TemporaryDirectory() as tmpdir:
		summary_file = os.path.join(tmpdir, 'summary.json')
		with TaskManager(INPUT_DIR, overwrite=True, summary=summary_file) as tm:
			# Load the summary file:
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			# Everytning should be really empty:
			print(j)
			assert j['numtasks'] == 89394
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 0
			assert j['ERROR'] == 0
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 0
			#assert j['SKIPPED'] == 0
			assert j['tasks_run'] == 0
			assert j['slurm_jobid'] is None
			assert j['last_error'] is None
			assert j['mean_elaptime'] is None

			initial_numtasks = j['numtasks']
			initial_skipped = j['SKIPPED']

			# Start task with priority=1:
			task = tm.get_random_task()
			print(task)
			tm.start_task(task['priority'])
			tm.write_summary()

			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			print(j)
			assert j['numtasks'] == initial_numtasks
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 0
			assert j['ERROR'] == 0
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 1
			assert j['SKIPPED'] == initial_skipped
			assert j['tasks_run'] == 0
			assert j['slurm_jobid'] is None
			assert j['last_error'] is None
			assert j['mean_elaptime'] is None

			# Make a fake result we can save;
			result = task.copy()
			result['status_corr'] = STATUS.OK
			result['elaptime_corr'] = 3.14

			# Save the result:
			tm.save_results(result)
			tm.write_summary()

			# Load the summary file after "running the task":
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			print(j)
			assert j['numtasks'] == initial_numtasks
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 1
			assert j['ERROR'] == 0
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 0
			assert j['SKIPPED'] == initial_skipped
			assert j['tasks_run'] == 1
			assert j['slurm_jobid'] is None
			assert j['last_error'] is None
			assert j['mean_elaptime'] == 3.14

			task = tm.get_random_task()
			tm.start_task(task['priority'])

			# Make a fake result we can save;
			result = task.copy()
			result['status_corr'] = STATUS.ERROR
			result['elaptime_corr'] = 6.14
			result['details'] = {
				'errors': ['dummy error 1', 'dummy error 2']
			}

			# Save the result:
			tm.save_results(result)
			tm.write_summary()

			# Load the summary file after "running the task":
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			print(j)
			assert j['numtasks'] == initial_numtasks
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 1
			assert j['ERROR'] == 1
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 0
			assert j['SKIPPED'] == initial_skipped
			assert j['tasks_run'] == 2
			assert j['slurm_jobid'] is None
			assert j['last_error'] == "dummy error 1\ndummy error 2"
			assert j['mean_elaptime'] == 3.44

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	test_taskmanager()
	test_taskmanager_notexist()
	test_taskmanager_constraints()
	test_taskmanager_cleanup()
	test_taskmanager_summary()
