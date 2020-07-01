#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests Cotrending Basis Vector objects.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
import numpy as np
from astropy.io import fits
from lightkurve import TessLightCurve
import os.path
import conftest # noqa: F401
from corrections import CBV
from corrections.plots import plt

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("datasource", ['ffi', 'tpf'])
def test_cbv(INPUT_DIR, datasource):
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
def test_cbv_invalid(INPUT_DIR):
	# Folder containing test CBV files:
	data_folder = os.path.join(INPUT_DIR, 'cbv-prepare')

	with pytest.raises(ValueError):
		CBV(data_folder, 143, 'invalid-datasource')

	with pytest.raises(FileNotFoundError):
		# Create CBV object:
		CBV(data_folder, 9999, 'ffi')

#--------------------------------------------------------------------------------------------------
def test_cbv_fit(INPUT_DIR):
	# Folder containing test CBV files:
	data_folder = os.path.join(INPUT_DIR, 'cbv-prepare')

	# Create CBV object:
	cbv = CBV(data_folder, 143, 'ffi')

	coeffs = [10, 500, 50, 100, 0, 10, 0]
	abs_flux = 3500

	# Create model using coefficients, and make fake lightcurve out of it:
	mdl = cbv.mdl(coeffs) * abs_flux

	# Another check of making crazy weights:
	#sigma = np.ones_like(mdl)*100
	#mdl[200] = 50000
	#sigma[200] = 1e-17

	lc = TessLightCurve(
		time=cbv.time,
		flux=mdl
	)

	# Run CBV fitting with fixed number of CBVs:
	flux_filter, res, diagnostics = cbv.fit(lc, cbvs=3, use_bic=False)

	# Plot:
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
	ax1.scatter(cbv.time, mdl, alpha=0.3)
	ax1.plot(cbv.time, flux_filter, alpha=0.5, color='r')
	ax2.scatter(cbv.time, mdl - flux_filter)

	# Check the diagnostics dict:
	print(diagnostics)
	assert diagnostics['method'] == 'LS'
	assert not diagnostics['use_bic']
	assert not diagnostics['use_prior']

	# Check the coefficients coming out of the fit:
	# They should be the same as the ones we put in
	print(res - coeffs)
	np.testing.assert_allclose(res, coeffs, atol=0.5, rtol=0.5)

	# The fitted model should be very close to the model going in:
	np.testing.assert_allclose(mdl, flux_filter)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plt.switch_backend('Qt5Agg')
	plt.close('all')
	pytest.main([__file__])
	plt.show(True)
