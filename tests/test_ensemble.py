#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Ensemble.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Lindsey Carboneau <lmcarboneau@gmail.com>

"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import corrections
import lightkurve
import numpy as np
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

# TODO: Check the frange and etc algebraically
starid = 281703991
camera = 3
ccd = 1
sector =1
#----------------------------------------------------------------------
def test_ensemble_basics():
	"""Check that the Ensemblecorrector can be initiated at all"""
	with corrections.EnsembleCorrector(INPUT_DIR, plot=True) as ec:
		assert ec.__class__.__name__ == 'EnsembleCorrector', "Did not get the correct class name back"
		assert ec.input_folder == INPUT_DIR, "Incorrect input folder"
		assert ec.plot == True, "Plot parameter passed appropriately"

@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == True,
					reason="This requires a sector of data. Impossible to run with Travis")
def test_ensemble_returned_values():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	tm = corrections.TaskManager(INPUT_DIR)
	task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	corr = CorrClass(INPUT_DIR, plot=False)
	inlc = corr.load_lightcurve(task)
	outlc, status = corr.do_correction(inlc.copy())

	#Check input validation
	with pytest.raises(ValueError) as err:
		outlc, status = corr.do_correction('hello world')
		assert('The input to `do_correction` is not a TessLightCurve object!' in err.value.args[0])

	#Check contents
	assert len(outlc.flux) == len(inlc.flux), "Input flux ix different length to output flux"
	assert all(inlc.time == outlc.time), "Input time is nonidentical to output time"
	assert all(outlc.flux != inlc.flux), "Input and output flux are identical."
	assert len(outlc.flux) == len(outlc.time), "Check time and flux have same length"

	#Check status
	assert status == corrections.STATUS.OK, "STATUS was not set appropriately"

@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == True,
					reason="This requires a sector of data. Impossible to run with Travis")
def test_run_metadata():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	tm = corrections.TaskManager(INPUT_DIR)
	task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	corr = CorrClass(INPUT_DIR, plot=False)
	inlc = corr.load_lightcurve(task)
	outlc, status = corr.do_correction(inlc.copy())

	#Check metadata
	assert 'fmean' in outlc.meta, "Metadata is incomplete"
	assert 'fstd' in outlc.meta, "Metadata is incomplete"
	assert 'frange' in outlc.meta, "Metadata is incomplete"
	assert 'drange' in outlc.meta, "Metadata is incomplete"
	assert outlc.meta['task']['starid'] == inlc.meta['task']['starid'], "Metadata is incomplete"
	assert outlc.meta['task'] == inlc.meta['task'], "Metadata is incomplete"

@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == True,
					reason="This requires a sector of data. Impossible to run with Travis")
def test_ensemble_metadata():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	tm = corrections.TaskManager(INPUT_DIR)
	task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	corr = CorrClass(INPUT_DIR, plot=False)
	inlc = corr.load_lightcurve(task)
	outlc, status = corr.do_correction(inlc.copy())

	#Check ensemble specific metadata
	ensemble = outlc.meta['ensemble']
	assert ensemble['star_count'] > 20, "Fewer than 20 stars in ensemble"

	assert 'ensemble_list' in ensemble, "Ensemble list is not output"
	assert ensemble['star_count'] == len(ensemble['ensemble_list']), "Star counts and length of ensemble list are different"
	assert isinstance(ensemble['ensemble_list'][15][1], lightkurve.lightcurve.TessLightCurve), "Stored object in ensemble_list is not a lightkurve object"

	assert 'ensemble_spline' in ensemble, "Ensemble Spline not output"
	assert len(ensemble['ensemble_spline']) == len(outlc.flux), "Spline and flux are different lenghts"
	assert all(np.isclose(outlc.flux, inlc.flux/ensemble['ensemble_spline'], 1e-3)), "Output flux is not equal to input divided by spline"

	assert 'search_radius' in ensemble, "Search radius not included in ensemble"

if __name__ == "__main__":
	tm = corrections.TaskManager(INPUT_DIR)
	task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	corr = CorrClass(INPUT_DIR, plot=False)
	inlc = corr.load_lightcurve(task)
	outlc, status = corr.do_correction(inlc.copy())
	import matplotlib.pyplot as plt
	ax = inlc.plot()
	outlc.plot(ax=ax)
	plt.show()
