#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests that run run_tesscorr with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import subprocess
import shlex
import sys
from test_known_stars import STAR_LIST
import conftest # noqa: F401
from corrections import TaskManager

# Skip these tests if mpi4py can not be loaded:
pytest.importorskip("mpi4py")

#--------------------------------------------------------------------------------------------------
def capture_run_tesscorr_mpi(params, mpiexec=True):

	command = '"%s" run_tesscorr_mpi.py %s' % (sys.executable, params.strip())
	if mpiexec:
		command = 'mpiexec -n 2 ' + command
	print(command)

	cmd = shlex.split(command)
	proc = subprocess.Popen(cmd,
		cwd=os.path.join(os.path.dirname(__file__), '..'),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True
	)
	out, err = proc.communicate()
	exitcode = proc.returncode
	proc.kill()

	print("ExitCode: %d" % exitcode)
	print("StdOut:\n%s" % out)
	print("StdErr:\n%s" % err)
	return out, err, exitcode

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_method():
	out, err, exitcode = capture_run_tesscorr_mpi("--method=invalid", mpiexec=False)
	assert exitcode == 2
	assert "error: argument -m/--method: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_datasource():
	out, err, exitcode = capture_run_tesscorr_mpi("--datasource=invalid", mpiexec=False)
	assert exitcode == 2
	assert "error: argument --datasource: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_camera():
	out, err, exitcode = capture_run_tesscorr_mpi("--camera=5", mpiexec=False)
	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tesscorr_mpi_invalid_ccd():
	out, err, exitcode = capture_run_tesscorr_mpi("--ccd=14", mpiexec=False)
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
			meth, starid, datasource = s.values[0:3]
			if meth != method:
				continue
			tm.cursor.execute("UPDATE todolist SET corr_status=NULL WHERE starid=? AND datasource=?;", [starid, datasource])
		tm.conn.commit()
		tm.cursor.execute("SELECT COUNT(*) FROM todolist WHERE corr_status IS NULL;")
		num = tm.cursor.fetchone()[0]

	print(num)
	assert num < 10

	with tempfile.TemporaryDirectory() as tmpdir:
		params = '-d --method={method:s} "{input_dir:s}" "{output:s}"'.format(
			method=method,
			input_dir=os.path.dirname(PRIVATE_TODO_FILE),
			output=tmpdir
		)
		out, err, exitcode = capture_run_tesscorr_mpi(params)

	assert ' - INFO - %d tasks to be run' % num in err
	assert ' - INFO - Master starting with 1 workers' in err
	assert ' - DEBUG - Got data from worker 1: [{' in err
	assert ' - INFO - Worker 1 exited.' in err
	assert ' - INFO - Master finishing' in err
	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
