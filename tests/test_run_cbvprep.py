#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that run run_cbvprep with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
import tempfile
from astropy.io import fits
from conftest import capture_run_cli
import corrections

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_EXISTS = os.path.exists(os.path.join(INPUT_DIR, 'test_data_available_v2.txt'))

#--------------------------------------------------------------------------------------------------
def test_run_cbvprep_invalid_sector():
	out, err, exitcode = capture_run_cli('run_cbvprep.py', "--sector=invalid")
	assert exitcode == 2
	assert "error: argument --sector: invalid int value: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_cbvprep_invalid_cadence():
	out, err, exitcode = capture_run_cli('run_cbvprep.py', "--cadence=15")
	assert exitcode == 2
	assert "error: argument --cadence: invalid choice: 15 (choose from 'ffi', 1800, 600, 120, 20)" in err

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
@pytest.mark.fulldata
@pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a full sector of data.")
def test_run_cbvprep(SHARED_INPUT_DIR):
	with tempfile.TemporaryDirectory() as tmpdir:
		# Run CLI program:
		params = [
			'--version=17',
			'--sector=1',
			'--area=114',
			'--output=' + tmpdir,
			SHARED_INPUT_DIR
		]
		out, err, exitcode = capture_run_cli('run_cbvprep.py', params)
		assert ' - ERROR - ' not in err
		assert exitcode == 0

		# Check that the plots directory was created:
		print(os.listdir(tmpdir))
		assert os.path.isdir(os.path.join(tmpdir, 'plots')), "Plots directory does not exist"

		# This should create CBVs for several cadences:
		for cadence in (1800, 120):

			# The CBV file should now exist:
			cbvfile = os.path.join(tmpdir, f'cbv-s0001-c{cadence:04d}-a114.hdf5')
			assert os.path.isfile(cbvfile), f"HDF5 file does not exist ({cadence:d}s)"

			# Create CBV object:
			cbv = corrections.CBV(cbvfile)

			# Check the returned object:
			assert cbv.cbv_area == 114
			assert cbv.datasource == 'ffi' if cadence == 1800 else 'tpf'
			assert cbv.cadence == cadence
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
			assert cbv.version == corrections.__version__

			# The file point to the one we use as input:
			assert cbv.filepath == cbvfile

			# Check arrays stored in CBV object:
			N = len(cbv.time)
			assert N > 0, "Time column is length 0"
			assert len(cbv.cadenceno) == N
			assert cbv.cbv.shape == (N, cbv.ncomponents)
			assert cbv.cbv_s.shape == (N, cbv.ncomponents)

			# Check the FITS file exists:
			fitsfile = os.path.join(tmpdir, f'tess-s0001-c{cadence:04d}-a114-v17-tasoc_cbv.fits.gz')
			assert os.path.isfile(fitsfile), f"FITS file does not exist ({cadence:d}s)"

			# Open FITS file and check headers and data:
			with fits.open(fitsfile, mode='readonly') as hdu:
				# Header:
				hdr = hdu[0].header
				assert hdr['CADENCE'] == cadence
				assert hdr['SECTOR'] == cbv.sector
				assert hdr['CAMERA'] == cbv.camera
				assert hdr['CCD'] == cbv.ccd
				assert hdr['CBV_AREA'] == cbv.cbv_area
				assert hdr['DATA_REL'] == cbv.data_rel
				assert hdr['VERSION'] == 17
				assert hdr['PROCVER'] == cbv.version

				for k in range(1, len(hdu)):
					hdr1 = hdu[k].header
					assert hdr1['CAMERA'] == cbv.camera
					assert hdr1['CCD'] == cbv.ccd
					assert hdr1['CBV_AREA'] == cbv.cbv_area
					assert hdr1['THR_COR'] == cbv.threshold_correlation
					assert hdr1['THR_VAR'] == cbv.threshold_variability
					assert hdr1['THR_SNR'] == cbv.threshold_snrtest
					assert hdr1['THR_ENT'] == cbv.threshold_entropy

				# Data:
				assert hdu['CBV.SINGLE-SCALE.114'].data.shape[0] == N
				assert hdu['CBV.SPIKE.114'].data.shape[0] == N

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
