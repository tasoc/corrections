#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of CBVCreator.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import conftest # noqa: F401
from corrections import CBV, CBVCreator, create_cbv, TaskManager, get_version

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_EXISTS = os.path.exists(os.path.join(INPUT_DIR, 'test_data_available_v2.txt'))

#--------------------------------------------------------------------------------------------------
def test_import_nonexistent(INPUT_DIR):
	"""
	Tests that CBVCreator handles being called with non-existing input directory.
	"""
	with pytest.raises(FileNotFoundError):
		with CBVCreator(INPUT_DIR + '/does/not/exist/'):
			pass

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
@pytest.mark.fulldata
@pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a full sector of data.")
def test_create_cbv(PRIVATE_INPUT_DIR):
	"""
	Test create_cbv.
	"""
	# Invoke the TaskManager to ensure that the input TODO-file has the correct columns
	# and indicies, which is automatically created by the TaskManager init function.
	with TaskManager(PRIVATE_INPUT_DIR, cleanup=False):
		pass

	# Run the creation of a single CBV on an area the doesn't already exist:
	cbv = create_cbv(sector=1, cbv_area=114, input_folder=PRIVATE_INPUT_DIR, version=42)

	# Check the returned object:
	assert isinstance(cbv, CBV), "Not a CBV object"
	assert cbv.cbv_area == 114
	assert cbv.datasource == 'ffi'
	assert cbv.cadence == 1800
	assert cbv.sector == 1
	assert cbv.camera == 1
	assert cbv.ccd == 1
	assert cbv.data_rel == 1
	assert cbv.ncomponents == 16
	assert cbv.threshold_variability == 1.3
	assert cbv.threshold_correlation == 0.5
	assert cbv.threshold_snrtest == 5.0
	assert cbv.threshold_entropy == -0.5

	# Check that the version has been set correctly:
	assert cbv.version == get_version()

	# The file should also exist:
	assert os.path.isfile(cbv.filepath), "HDF5 file does not exist"

	# Check arrays stored in CBV object:
	N = len(cbv.time)
	assert N > 0, "Time column is length 0"
	assert len(cbv.cadenceno) == N
	assert cbv.cbv.shape == (N, 16)
	assert cbv.cbv_s.shape == (N, 16)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
