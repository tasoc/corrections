#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of corrections.TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import json
import conftest # noqa: F401
from corrections import TaskManager, STATUS

#--------------------------------------------------------------------------------------------------
def test_taskmanager(PRIVATE_TODO_FILE):
	"""Test of TaskManager"""
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup=True) as tm:
		# Get the number of tasks:
		# Check this number with:
		#  SELECT COUNT(*) FROM datavalidation_raw WHERE approved=1;
		numtasks = tm.get_number_tasks()
		print(numtasks)
		assert numtasks == 57799

		# Get the first task in the TODO file:
		task1 = tm.get_task()
		print(task1)

		# Check that it contains what we know it should:
		# The first priority in the TODO file is the following:
		assert task1['priority'] == 17
		assert task1['starid'] == 29281992
		assert task1['camera'] == 1
		assert task1['ccd'] == 4
		assert task1['datasource'] == 'ffi'
		assert task1['sector'] == 1
		assert task1['cbv_area'] == 143

		# Start task with priority=17:
		tm.start_task(task1)

		# Get the next task, which should be the one with priority=2:
		task2 = tm.get_task()
		print(task2)

		assert task2['priority'] == 18
		assert task2['starid'] == 29281992
		assert task2['camera'] == 1
		assert task2['ccd'] == 4
		assert task2['datasource'] == 'tpf'
		assert task2['sector'] == 1
		assert task2['cbv_area'] == 143

		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT corr_status FROM todolist WHERE priority=17;")
		task1_status = tm.cursor.fetchone()['corr_status']
		print(task1_status)

		assert task1_status == STATUS.STARTED.value

#--------------------------------------------------------------------------------------------------
def test_taskmanager_notexist(INPUT_DIR):
	with pytest.raises(FileNotFoundError):
		with TaskManager(os.path.join(INPUT_DIR, 'does-not-exist')):
			pass

#--------------------------------------------------------------------------------------------------
def test_taskmanager_constraints(PRIVATE_TODO_FILE):

	constraints = {'datasource': 'tpf', 'priority': 17}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task = tm.get_task(**constraints)
		numtasks = tm.get_number_tasks(**constraints)
		print(task)
		assert task is None, "Task1 should be None"
		assert numtasks == 0, "Task1 search should give no results"

	constraints = {'datasource': 'tpf', 'priority': 17, 'camera': None}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task2 = tm.get_task(**constraints)
		numtasks2 = tm.get_number_tasks(**constraints)
		print(task2)
		assert task2 == task, "Tasks should be identical"
		assert numtasks2 == 0, "Task2 search should give no results"

	constraints = {'datasource': 'ffi', 'priority': 17}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task = tm.get_task(**constraints)
		numtasks = tm.get_number_tasks(**constraints)
		print(task)
		assert task['priority'] == 17, "Task2 should be #17"
		assert task['datasource'] == 'ffi'
		assert task['camera'] == 1
		assert task['ccd'] == 4
		assert numtasks == 1, "Priority search should give one results"

	constraints = {'datasource': 'ffi', 'priority': 17, 'camera': 1, 'ccd': 4}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task3 = tm.get_task(**constraints)
		numtasks3 = tm.get_number_tasks(**constraints)
		print(task3)
		assert task3 == task, "Tasks should be identical"
		assert numtasks3 == 1, "Task3 search should give one results"

	constraints = ['priority=17']
	with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints=constraints) as tm:
		task4 = tm.get_task(priority=17)
		numtasks4 = tm.get_number_tasks(priority=17)
		print(task4)
		assert task4['priority'] == 17, "Task4 should be #17"
		assert numtasks4 == 1, "Priority search should give one results"

	constraints = {'starid': 29281992}
	with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints=constraints) as tm:
		numtasks5 = tm.get_number_tasks(**constraints)
		assert numtasks5 == 2
		task5 = tm.get_task(**constraints)
		assert task5['priority'] == 17

#--------------------------------------------------------------------------------------------------
def test_taskmanager_constraints_invalid(PRIVATE_TODO_FILE):
	with pytest.raises(ValueError) as e:
		with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints='invalid'):
			pass
	assert str(e.value) == 'cleanup_constraints should be dict or list'

#--------------------------------------------------------------------------------------------------
def test_taskmanager_cleanup(PRIVATE_TODO_FILE):

	# Reset the TODO-file completely, and mark the first task as STARTED:
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
		task1 = tm.get_task()
		print(task1)
		pri = task1['priority']
		tm.start_task(task1)

	# Cleanup, but with a constraint not matching the one we changed:
	with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints={'priority': 18}) as tm:
		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT corr_status FROM todolist WHERE priority=?;", [pri])
		task1_status = tm.cursor.fetchone()['corr_status']
		print(task1_status)
		assert task1_status == STATUS.STARTED.value

	# Now clean with a constraint that matches:
	with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints={'priority': pri}) as tm:
		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT corr_status FROM todolist WHERE priority=?;", [pri])
		task1_status = tm.cursor.fetchone()['corr_status']
		print(task1_status)
		assert task1_status is None

#--------------------------------------------------------------------------------------------------
def test_taskmanager_chunks(PRIVATE_TODO_FILE):

	# Reset the TODO-file completely, and mark the first task as STARTED:
	with TaskManager(PRIVATE_TODO_FILE) as tm:
		task1 = tm.get_task()
		assert isinstance(task1, dict)

		task10 = tm.get_task(chunk=10)
		assert isinstance(task10, list)
		assert len(task10) == 10
		for task in task10:
			assert isinstance(task, dict)

		task10r = tm.get_random_task(chunk=9)
		assert isinstance(task10r, list)
		assert len(task10r) == 9
		for task in task10r:
			assert isinstance(task, dict)

		tm.start_task(task10r)
		tm.cursor.execute("SELECT COUNT(*) FROM todolist WHERE corr_status=?;", [STATUS.STARTED.value])
		assert tm.cursor.fetchone()[0] == 9

#--------------------------------------------------------------------------------------------------
def test_taskmanager_summary_and_settings(PRIVATE_TODO_FILE):
	with tempfile.TemporaryDirectory() as tmpdir:
		summary_file = os.path.join(tmpdir, 'summary.json')
		with TaskManager(PRIVATE_TODO_FILE, overwrite=True, summary=summary_file, summary_interval=2) as tm:
			# Load the summary file:
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			assert tm.summary_counter == 0  # Counter should start at zero

			# Everytning should be really empty:
			# numtask checked with: SELECT COUNT(*) FROM todolist;
			print(j)
			assert j['numtasks'] == 78736
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

			# Check the settings table:
			assert tm.corrector is None
			tm.cursor.execute("SELECT * FROM corr_settings;")
			settings = tm.cursor.fetchone()
			assert settings is None

			# Start a random task:
			task = tm.get_random_task()
			print(task)
			tm.start_task(task)
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
			result['corrector'] = 'cbv'
			result['status_corr'] = STATUS.OK
			result['elaptime_corr'] = 3.14
			result['worker_wait_time'] = 1.0
			result['details'] = {'cbv_num': 10}

			# Save the result:
			tm.save_results(result)
			assert tm.summary_counter == 1 # We saved once, so counter should have gone up one
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
			assert j['mean_worker_waittime'] == 1.0

			# Check the setting again - it should now have changed:
			assert tm.corrector == 'cbv'
			tm.cursor.execute("SELECT * FROM corr_settings;")
			settings = tm.cursor.fetchone()
			assert settings['corrector'] == 'cbv'

			# Also check that the additional diagnostic was saved correctly:
			tm.cursor.execute("SELECT cbv_num FROM diagnostics_corr WHERE priority=?;", [result['priority']])
			assert tm.cursor.fetchone()['cbv_num'] == 10

			# Start another random task:
			task = tm.get_random_task()
			tm.start_task(task)

			# Make a fake result we can save;
			result = task.copy()
			result['corrector'] = 'cbv'
			result['status_corr'] = STATUS.ERROR
			result['elaptime_corr'] = 6.14
			result['worker_wait_time'] = 2.0
			result['details'] = {'errors': ['dummy error 1', 'dummy error 2']}

			# Save the result:
			tm.save_results(result)
			assert tm.summary_counter == 0 # We saved again, so summary_counter should be zero
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
			assert j['mean_worker_waittime'] == 1.1

			# Make a new fake result we can save;
			# but this time try to change the corrector
			result = task.copy()
			result['corrector'] = 'kasoc_filter'
			result['status_corr'] = STATUS.OK
			result['elaptime_corr'] = 7.14

			# This should fail when we try to save it:
			with pytest.raises(ValueError) as e:
				tm.save_results(result)
			assert str(e.value) == "Attempting to mix results from multiple correctors"

#--------------------------------------------------------------------------------------------------
def test_taskmanager_no_more_tasks(PRIVATE_TODO_FILE):
	with TaskManager(PRIVATE_TODO_FILE) as tm:
		# Set all the tasks as completed:
		tm.cursor.execute("UPDATE todolist SET corr_status=1;")
		tm.conn.commit()

		# When we now ask for a new task, there shouldn't be any:
		assert tm.get_task() is None
		assert tm.get_random_task() is None
		assert tm.get_number_tasks() == 0

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('corrector,fdesc', [('cbv','cbv'), ('ensemble','ens'), ('kasoc_filter','kf')])
def test_taskmanager_corrector_detection(PRIVATE_TODO_FILE, corrector, fdesc):
	# Start out with the blank TODO file, and
	# manualky add a single lightcurve to the diagnostics_corr table,
	# as if we had done a previous run with an older version:
	with TaskManager(PRIVATE_TODO_FILE) as tm:
		assert tm.corrector is None
		tm.cursor.execute("INSERT INTO diagnostics_corr (priority,lightcurve) VALUES (17,'tess0012345678-tasoc-{0:s}_lc.fits.gz');".format(fdesc))
		tm.conn.commit()

	# When re-opening the TODO-file, it should now be detected
	# as a corrected TODO file:
	with TaskManager(PRIVATE_TODO_FILE) as tm:
		assert tm.corrector == corrector
		tm.cursor.execute("SELECT corrector FROM corr_settings;")
		assert tm.cursor.fetchone()[0] == corrector

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
