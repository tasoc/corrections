#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of CBVCreator.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import conftest # noqa: F401
from corrections import CBV, CBVCreator, create_cbv

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
def test_invalid_input(SHARED_INPUT_DIR):
	"""
	Tests that CBVCreator handles being called with various wrong input.
	"""

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, datasource='invalid-value')
	assert str(e.value) == "Invalid DATASOURCE"

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area='invalid-value')
	assert str(e.value) == "Invalid CBV_AREA"

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area=111, ncomponents=-1)
	assert str(e.value) == "Invalid NCOMPONENTS"

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area=111, threshold_correlation=0)
	assert str(e.value) == "Invalid THRESHOLD_CORRELATION"

#--------------------------------------------------------------------------------------------------
def test_mismatch_existing_settings(SHARED_INPUT_DIR):
	"""
	Tests that CBVCreator handles being called with various inputs that differ from the one the
	existing file was created with.
	"""

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area=143, datasource='ffi', ncomponents=42)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area=143, datasource='ffi', threshold_variability=2.5)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area=143, datasource='ffi', threshold_correlation=0.8)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area=143, datasource='ffi', threshold_snrtest=2.5)
	assert str(e.value) == "Mismatch between existing file and provided settings"

	with pytest.raises(ValueError) as e:
		CBVCreator(SHARED_INPUT_DIR, cbv_area=143, datasource='ffi', threshold_entropy=-1.6)
	assert str(e.value) == "Mismatch between existing file and provided settings"

#--------------------------------------------------------------------------------------------------
def test_load_existing(SHARED_INPUT_DIR):
	"""
	Tests that CBVCreator handles being called with various wrong input.
	"""

	with CBVCreator(SHARED_INPUT_DIR, cbv_area=143) as C:
		assert C.threshold_variability == 1.3
		assert C.threshold_correlation == 0.5
		assert C.threshold_snrtest == 5.0
		assert C.threshold_entropy == -0.5

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a full sector of data.")
def test_create_cbv(SHARED_INPUT_DIR):
	"""
	Test create_cbv.
	"""

	# Run the creation of a single CBV on an area the doesnøt already exist:
	cbv = create_cbv(114, SHARED_INPUT_DIR)

	assert isinstance(cbv, CBV), "Not a CBV object"
	assert cbv.cbv_area == 114
	assert cbv.datasource == 'ffi'
	assert cbv.sector == 1
	assert cbv.camera == 1
	assert cbv.ccd == 1

	#assert os.path.isfile(cbv.filepath), "HDF5 file does not exist"

#----------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
