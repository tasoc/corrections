#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of utility functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import numpy as np
import tempfile
import warnings
from lightkurve import LightCurve, LightkurveWarning
import conftest # noqa: F401
from corrections.utilities import savePickle, loadPickle, sphere_distance, rms_timescale, ptp

#--------------------------------------------------------------------------------------------------
def test_pickle():

	# Pretty random object to save and restore:
	test_object = {'test': 42, 'whatever': (1,2,3,4,5432)}

	with tempfile.TemporaryDirectory() as tmpdir:
		fname = os.path.join(tmpdir, 'test.pickle')

		# Save object in file:
		savePickle(fname, test_object)
		assert os.path.exists(fname), "File does not exist"

		# Recover the object again:
		recovered_object = loadPickle(fname)
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

		# Save object in file:
		savePickle(fname + ".gz", test_object)
		assert os.path.exists(fname + ".gz"), "File does not exist"

		# Recover the object again:
		recovered_object = loadPickle(fname + ".gz")
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

#--------------------------------------------------------------------------------------------------
def test_sphere_distance():
	np.testing.assert_allclose(sphere_distance(0, 0, 90, 0), 90)
	np.testing.assert_allclose(sphere_distance(90, 0, 0, 0), 90)
	np.testing.assert_allclose(sphere_distance(0, -90, 0, 90), 180)
	np.testing.assert_allclose(sphere_distance(45, 45, 45, 45), 0)
	np.testing.assert_allclose(sphere_distance(33.2, 45, 33.2, -45), 90)
	np.testing.assert_allclose(sphere_distance(337.5, 0, 22.5, 0), 45)
	np.testing.assert_allclose(sphere_distance(22.5, 0, 337.5, 0), 45)
	np.testing.assert_allclose(sphere_distance(0, 0, np.array([0, 90]), np.array([90, 90])), np.array([90, 90]))

#--------------------------------------------------------------------------------------------------
def test_rms_timescale():

	time = np.linspace(0, 27, 100)
	flux = np.zeros(len(time))

	rms = rms_timescale(LightCurve(time=time, flux=flux))
	print(rms)
	np.testing.assert_allclose(rms, 0)

	rms = rms_timescale(LightCurve(time=time, flux=flux*np.nan))
	print(rms)
	assert np.isnan(rms), "Should return nan on pure nan input"

	rms = rms_timescale(LightCurve(time=[], flux=[]))
	print(rms)
	assert np.isnan(rms), "Should return nan on empty input"

	# Pure nan in the time-column should raise ValueError:
	with np.testing.assert_raises(ValueError):
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=LightkurveWarning, message='LightCurve object contains NaN times')
			rms = rms_timescale(LightCurve(time=time*np.nan, flux=flux))

	# Time with invalid contents (e.g. Inf) should throw an ValueError:
	time_invalid = time.copy()
	time_invalid[2] = np.inf
	with np.testing.assert_raises(ValueError):
		rms = rms_timescale(LightCurve(time=time_invalid, flux=flux))

	time_someinvalid = time.copy()
	time_someinvalid[2] = np.nan
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=LightkurveWarning, message='LightCurve object contains NaN times')
		rms = rms_timescale(LightCurve(time=time_someinvalid, flux=flux))
	print(rms)
	np.testing.assert_allclose(rms, 0)

	# Test with timescale longer than timespan should return zero:
	flux = np.random.randn(1000)
	time = np.linspace(0, 27, len(flux))
	rms = rms_timescale(LightCurve(time=time, flux=flux), timescale=30.0)
	print(rms)
	np.testing.assert_allclose(rms, 0)

#--------------------------------------------------------------------------------------------------
def test_ptp():

	time = np.linspace(0, 27, 1000)
	flux = np.zeros(len(time))

	p = ptp(LightCurve(time=time, flux=flux))
	print(p)
	np.testing.assert_allclose(p, 0)

	p = ptp(LightCurve(time=time, flux=flux*np.nan))
	print(p)
	assert np.isnan(p), "Should return nan on pure nan input"

	p = ptp(LightCurve(time=[], flux=[]))
	print(p)
	assert np.isnan(p), "Should return nan on empty input"

	# Pure nan in the time-column should raise ValueError:
	with np.testing.assert_raises(ValueError):
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=LightkurveWarning, message='LightCurve object contains NaN times')
			p = ptp(LightCurve(time=time*np.nan, flux=flux))

	# Test with constant lightcurve should return zero:
	flux = np.full(100, np.pi)
	time = np.linspace(0, 27, len(flux))
	p = ptp(LightCurve(time=time, flux=flux))
	print(p)
	np.testing.assert_allclose(p, 0)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
