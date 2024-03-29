#!/usr/bin/env python3
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
from corrections.plots import plt, plots_interactive

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('cadence', [1800, 120])
def test_cbv(INPUT_DIR, cadence):
	# Create CBV object:
	cbv = CBV(os.path.join(INPUT_DIR, 'cbv-prepare', f'cbv-s0001-c{cadence:04d}-a143.hdf5'))

	# Check CBV attributes:
	assert cbv.sector == 1
	assert cbv.camera == 1
	assert cbv.ccd == 4
	assert cbv.cbv_area == 143
	assert cbv.cadence == cadence
	assert cbv.data_rel == 1
	assert cbv.datasource == 'ffi' if cadence == 1800 else 'tpf'

	# Save the CBV object to a FITS file:
	with tempfile.TemporaryDirectory() as tmpdir:
		# Test saving to FITS file to directory that does not exist:
		with pytest.raises(FileNotFoundError):
			cbv.save_to_fits(os.path.join(tmpdir, 'non-existing'), version=5)

		# Save CBV to FITS file:
		fitsfile = cbv.save_to_fits(tmpdir, version=5)
		print(fitsfile)

		# Open generated FITS file and check saved values:
		with fits.open(fitsfile, mode='readonly') as hdu:
			assert hdu[0].header['SECTOR'] == 1
			assert hdu[0].header['CAMERA'] == 1
			assert hdu[0].header['CCD'] == 4
			assert hdu[0].header['CBV_AREA'] == 143
			assert hdu[0].header['DATA_REL'] == 1
			assert hdu[0].header['VERSION'] == 5
			assert hdu[0].header['CADENCE'] == cbv.cadence

#--------------------------------------------------------------------------------------------------
def test_cbv_invalid(INPUT_DIR):
	with pytest.raises(FileNotFoundError):
		# Create CBV object:
		CBV(os.path.join(INPUT_DIR, 'cbv-prepare', 'does-not-exist.hdf5'))

#--------------------------------------------------------------------------------------------------
def test_cbv_fit(INPUT_DIR):
	# Create CBV object:
	cbv = CBV(os.path.join(INPUT_DIR, 'cbv-prepare', 'cbv-s0001-c1800-a143.hdf5'))

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
	plots_interactive()
	plt.close('all')
	pytest.main([__file__])
	plt.show(True)
