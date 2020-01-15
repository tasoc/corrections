#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests that run run_tesscorr with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import subprocess
import shlex

test_data_available = os.path.exists(os.path.join(os.path.dirname(__file__), 'input', 'test_data_available.txt'))

#--------------------------------------------------------------------------------------------------
def capture(command):

	cmd = shlex.split(command)
	proc = subprocess.Popen(cmd,
		cwd=os.path.join(os.path.dirname(__file__), '..'),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True
	)
	out, err = proc.communicate()
	exitcode = proc.returncode

	print(out)
	print(err)
	print(exitcode)

	return out, err, exitcode

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_method():
	command = "python run_tesscorr.py -t --starid=29281992 --method=invalid"
	out, err, exitcode = capture(command)

	assert exitcode == 2
	assert "error: argument -m/--method: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_datasource():
	command = "python run_tesscorr.py -t --starid=29281992 --datasource=invalid"
	out, err, exitcode = capture(command)

	assert exitcode == 2
	assert "error: argument --datasource: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_camera():
	command = "python run_tesscorr.py -t --starid=29281992 --camera=5"
	out, err, exitcode = capture(command)

	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_ccd():
	command = "python run_tesscorr.py -t --starid=29281992 --ccd=14"
	out, err, exitcode = capture(command)

	assert exitcode == 2
	assert 'error: argument --ccd: invalid choice: 14 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("datasource", ['ffi', 'tpf'])
def test_run_tesscorr_cbv(datasource):
	command = "python run_tesscorr.py -t -o -p --starid=29281992 -m cbv --datasource=%s" % (
		datasource
	)
	out, err, exitcode = capture(command)

	assert ' - ERROR - ' not in err
	#assert exitcode == 1

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not test_data_available,
	reason="This requires a sector of data. Only run if available.")
@pytest.mark.parametrize("datasource", ['ffi']) #  'tpf'
def test_run_tesscorr_ensemble(datasource):
	command = "python run_tesscorr.py -t -o -p --starid=29281992 -m ensemble --datasource=%s" % (
		datasource
	)
	out, err, exitcode = capture(command)

	assert ' - ERROR - ' not in err
	#assert exitcode == 1

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	test_run_tesscorr_invalid_method()
	test_run_tesscorr_invalid_datasource()
	test_run_tesscorr_invalid_camera()
	test_run_tesscorr_invalid_ccd()

	test_run_tesscorr_cbv('ffi')
	test_run_tesscorr_cbv('tpf')

	test_run_tesscorr_ensemble('ffi')
	test_run_tesscorr_ensemble('tpf')
