#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that run run_cbvprep with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
from conftest import capture_run_cli

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_EXISTS = os.path.exists(os.path.join(INPUT_DIR, 'test_data_available_v2.txt'))

#--------------------------------------------------------------------------------------------------
def test_run_cbvprep_invalid_sector():
	out, err, exitcode = capture_run_cli('run_cbvprep.py', "--sector=invalid")
	assert exitcode == 2
	assert "error: argument --sector: invalid int value: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_cbvprep_invalid_cadence():
	out, err, exitcode = capture_run_cli('run_cbvprep.py', "--cadence=invalid")
	assert exitcode == 2
	assert "error: argument --cadence: invalid int value: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_cbvprep_invalid_camera():
	out, err, exitcode = capture_run_cli('run_cbvprep.py', "--camera=5")
	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
def test_run_cbvprep_invalid_ccd():
	out, err, exitcode = capture_run_cli('run_cbvprep.py', "--ccd=14")
	assert exitcode == 2
	assert 'error: argument --ccd: invalid choice: 14 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
#@pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a full sector of data.")
#def test_run_cbvprep():
#
#	params = [
#		'--debug',
#		'--version=17',
#		'--area=114',
#		INPUT_DIR
#	]
#	out, err, exitcode = capture_run_cli('run_cbvprep.py', params)
#
#	assert ' - ERROR - ' not in err
#	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
