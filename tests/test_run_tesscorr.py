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
from test_known_stars import STAR_LIST

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#--------------------------------------------------------------------------------------------------
def capture(command):

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
	with tempfile.TemporaryDirectory() as tmpdir:
		command = 'python run_tesscorr.py --starid=29281992 --ccd=14 "%s" "%s"' % (
			INPUT_DIR,
			tmpdir
		)
		out, err, exitcode = capture(command)

	assert exitcode == 2
	assert 'error: argument --ccd: invalid choice: 14 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method,starid,datasource,var_goal,rms_goal,ptp_goal", STAR_LIST)
def test_run_tesscorr(SHARED_INPUT_DIR, method, starid, datasource, var_goal,rms_goal, ptp_goal):
	with tempfile.TemporaryDirectory() as tmpdir:
		command = 'python run_tesscorr.py -o -p --starid={starid:d} --method={method:s} --datasource={datasource:s} "{input_dir:s}" "{output:s}"'.format(
			starid=starid,
			method=method,
			datasource=datasource,
			input_dir=SHARED_INPUT_DIR,
			output=tmpdir
		)
		out, err, exitcode = capture(command)

	assert ' - ERROR - ' not in err
	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
