#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that run run_tesscorr with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
from conftest import capture_run_cli
from test_known_stars import STAR_LIST

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_method():
	out, err, exitcode = capture_run_cli('run_tesscorr.py', "-t --starid=29281992 --method=invalid")
	assert exitcode == 2
	assert "error: argument -m/--method: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_datasource():
	out, err, exitcode = capture_run_cli('run_tesscorr.py', "-t --starid=29281992 --datasource=invalid")
	assert exitcode == 2
	assert "error: argument --datasource: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_camera():
	out, err, exitcode = capture_run_cli('run_tesscorr.py', "-t --starid=29281992 --camera=5")
	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
def test_run_tesscorr_invalid_ccd():
	out, err, exitcode = capture_run_cli('run_tesscorr.py', "-t --starid=29281992 --ccd=14")
	assert exitcode == 2
	assert 'error: argument --ccd: invalid choice: 14 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method,starid,datasource,var_goal,rms_goal,ptp_goal", STAR_LIST)
def test_run_tesscorr(SHARED_INPUT_DIR, method, starid, datasource, var_goal,rms_goal, ptp_goal):
	with tempfile.TemporaryDirectory() as tmpdir:
		params = [
			'-o',
			'-d',
			'-p',
			'--starid={starid:d}'.format(starid=starid),
			'--method={method:s}'.format(method=method),
			'--datasource={datasource:s}'.format(datasource=datasource),
			SHARED_INPUT_DIR,
			tmpdir
		]
		out, err, exitcode = capture_run_cli('run_tesscorr.py', params)

	assert ' - ERROR - ' not in err
	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
