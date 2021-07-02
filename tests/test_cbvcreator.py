#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of CBVCreator.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import conftest # noqa: F401
from corrections import CBVCreator, TaskManager

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_EXISTS = os.path.exists(os.path.join(INPUT_DIR, 'test_data_available_v2.txt'))

#--------------------------------------------------------------------------------------------------
def test_import_input_nonexistent(INPUT_DIR):
	"""
	Tests that CBVCreator handles being called with non-existing input directory.
	"""
	with pytest.raises(FileNotFoundError) as e:
		with CBVCreator(INPUT_DIR + '/does/not/exist/'):
			pass
	assert str(e.value) == "TODO file not found"

#--------------------------------------------------------------------------------------------------
def test_import_output_nonexistent(SHARED_INPUT_DIR):
	"""
	Tests that CBVCreator handles being called with non-existing output directory.
	"""
	with pytest.raises(FileNotFoundError) as e:
		with CBVCreator(input_folder=SHARED_INPUT_DIR, sector=1, cbv_area=114,
			output_folder=SHARED_INPUT_DIR + '/does/not/exist/'):
			pass
	assert str(e.value) == "The output directory does not exist."

#--------------------------------------------------------------------------------------------------
def test_invalid_input(PRIVATE_INPUT_DIR_NOLC):
	"""
	Tests that CBVCreator handles being called with various wrong input.
	"""
	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=None)
	assert str(e.value) == "Invalid CBV_AREA"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area='invalid-value')
	assert str(e.value) == "Invalid CBV_AREA"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector='invalid-value', cbv_area=111)
	assert str(e.value) == "Invalid SECTOR"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=111, cadence='invalid-value')
	assert str(e.value) == "Invalid CADENCE"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=111, ncomponents=-1)
	assert str(e.value) == "Invalid NCOMPONENTS"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=111, threshold_correlation=0)
	assert str(e.value) == "Invalid THRESHOLD_CORRELATION"

#--------------------------------------------------------------------------------------------------
def test_mismatch_existing_settings(PRIVATE_INPUT_DIR_NOLC):
	"""
	Tests that CBVCreator handles being called with various inputs that differ from the one the
	existing file was created with.
	"""
	# Start by initializing the TaskManager because this will fix any
	# inconsistencies in the input todo-lists (like adding cadence column):
	with TaskManager(PRIVATE_INPUT_DIR_NOLC, cleanup=False):
		pass

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=143, cadence='ffi', ncomponents=42)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=143, cadence='ffi', threshold_variability=2.5)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=143, cadence='ffi', threshold_correlation=0.8)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=143, cadence='ffi', threshold_snrtest=2.5)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=143, cadence='ffi', threshold_entropy=-1.6)
	assert str(e.value) == "Mismatch between existing file and provided settings"

#--------------------------------------------------------------------------------------------------
def test_load_existing(PRIVATE_INPUT_DIR_NOLC):
	"""
	Test loading CBVCreator with existing CBV.
	"""
	# Start by initializing the TaskManager because this will fix any
	# inconsistencies in the input todo-lists (like adding cadence column):
	with TaskManager(PRIVATE_INPUT_DIR_NOLC, cleanup=False):
		pass

	# Start CBVCreator on an existing CBV:
	with CBVCreator(PRIVATE_INPUT_DIR_NOLC, sector=1, cbv_area=143) as C:
		assert C.threshold_variability == 1.3
		assert C.threshold_correlation == 0.5
		assert C.threshold_snrtest == 5.0
		assert C.threshold_entropy == -0.5

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
