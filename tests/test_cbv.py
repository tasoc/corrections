#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests Cotrending Basis Vector objects.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
from astropy.io import fits
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corrections import CBV

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("datasource", ['ffi', 'tpf'])
def test_cbv(datasource):
	# Folder containing test CBV files:
	data_folder = os.path.join(INPUT_DIR, 'cbv-prepare')

	# Create CBV object:
	cbv = CBV(data_folder, 143, datasource)

	# Check CBV attributes:
	assert cbv.sector == 1
	assert cbv.camera == 1
	assert cbv.ccd == 4
	assert cbv.cbv_area == 143
	assert cbv.cadence == 1800 if datasource == 'ffi' else 120

	# Save the CBV object to a FITS file:
	with tempfile.TemporaryDirectory() as tmpdir:
		# Test saving to FITS file to directory that does not exist:
		with pytest.raises(FileNotFoundError):
			cbv.save_to_fits(os.path.join(tmpdir, 'non-existing'), datarel=5)

		# Save CBV to FITS file:
		fitsfile = cbv.save_to_fits(tmpdir, datarel=5)
		print(fitsfile)

		# Open generated FITS file and check saved values:
		with fits.open(fitsfile) as hdu:
			assert hdu[0].header['SECTOR'] == 1
			assert hdu[0].header['CAMERA'] == 1
			assert hdu[0].header['CCD'] == 4
			assert hdu[0].header['CBV_AREA'] == 143
			assert hdu[0].header['DATA_REL'] == 5

#--------------------------------------------------------------------------------------------------
def test_cbv_invalid():
	# Folder containing test CBV files:
	data_folder = os.path.join(INPUT_DIR, 'cbv-prepare')

	with pytest.raises(ValueError):
		CBV(data_folder, 143, 'invalid-datasource')

	with pytest.raises(FileNotFoundError):
		# Create CBV object:
		CBV(data_folder, 9999, 'ffi')

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
