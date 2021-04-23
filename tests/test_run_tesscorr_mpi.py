#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that run run_tesscorr_mpi with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
from test_known_stars import STAR_LIST
import conftest # noqa: F401
from conftest import capture_run_cli
from corrections import TaskManager

# Skip these tests if mpi4py can not be loaded:
pytest.importorskip("mpi4py")

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_method():
	out, err, exitcode = capture_run_cli('run_tesscorr_mpi.py', "--method=invalid")
	assert exitcode == 2
	assert "error: argument -m/--method: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_sector():
	out, err, exitcode = capture_run_cli('run_tesscorr_mpi.py', "--sector=invalid")
	assert exitcode == 2
	assert "error: argument --sector: invalid int value: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_cadence():
	out, err, exitcode = capture_run_cli('run_tesscorr_mpi.py', "--cadence=15")
	assert exitcode == 2
	assert "error: argument --cadence: invalid choice: 15 (choose from 'ffi', 1800, 600, 120, 20)" in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_camera():
	out, err, exitcode = capture_run_cli('run_tesscorr_mpi.py', "--camera=5")
	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_ccd():
	out, err, exitcode = capture_run_cli('run_tesscorr_mpi.py', "--ccd=14")
	assert exitcode == 2
	assert 'error: argument --ccd: invalid choice: 14 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
@pytest.mark.parametrize('method', ['cbv', 'ensemble', 'kasoc_filter'])
def test_run_tesscorr_mpi(PRIVATE_TODO_FILE, method):

	# We are using all the stars in the STAR_LIST even
	# if data is downloaded or not, since we are actually
	# not using any data at all.
	# All corrections will fail, but we are only checking
	# if things run under MPI here.
	with TaskManager(PRIVATE_TODO_FILE) as tm:
		tm.cursor.execute("UPDATE todolist SET corr_status=1;")
		for s in STAR_LIST:
			if isinstance(s, (tuple, list)):
				meth, starid, cadence = s[0:3]
			else:
				meth, starid, cadence = s.values[0:3]
			if meth != method:
				continue
			tm.cursor.execute("UPDATE todolist SET corr_status=NULL WHERE starid=? AND cadence=?;", [starid, cadence])
		tm.conn.commit()
		tm.cursor.execute("SELECT COUNT(*) FROM todolist WHERE corr_status IS NULL;")
		num = tm.cursor.fetchone()[0]

	print(num)
	assert num > 0, "There should be at least one star to be processed with each method"
	assert num < 10, "There should not be more than 10 stars for each method"

	with tempfile.TemporaryDirectory() as tmpdir:
		params = [
			'--debug',
			f'--method={method:s}',
			os.path.dirname(PRIVATE_TODO_FILE),
			tmpdir
		]
		out, err, exitcode = capture_run_cli('run_tesscorr_mpi.py', params, mpiexec=True)

	assert ' - INFO - %d tasks to be run' % num in err
	assert ' - INFO - Master starting with 1 workers' in err
	assert ' - DEBUG - Got data from worker 1: [{' in err
	assert ' - INFO - Worker 1 exited.' in err
	assert ' - INFO - Master finishing' in err
	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
