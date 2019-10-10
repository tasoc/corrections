#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Ensemble Corrector.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Lindsey Carboneau <lmcarboneau@gmail.com>
"""

import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import corrections

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
starid = 29281992
camera = 1
ccd = 4
sector = 1

#--------------------------------------------------------------------------------------------------
def test_ensemble_basics():
	"""Check that the Ensemblecorrector can be initiated at all"""
	with corrections.EnsembleCorrector(INPUT_DIR, plot=True) as ec:
		assert ec.__class__.__name__ == 'EnsembleCorrector', "Did not get the correct class name back"
		assert ec.input_folder == INPUT_DIR, "Incorrect input folder"
		assert ec.plot == True, "Plot parameter passed appropriately"

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get('CI') == 'true' and os.environ.get('TRAVIS') == 'true',
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

	# Check input validation
	#with pytest.raises(ValueError) as err:
	#	outlc, status = corr.do_correction('hello world')
	#	assert('The input to `do_correction` is not a TessLightCurve object!' in err.value.args[0])

	#C heck contents
	assert len(outlc.flux) == len(inlc.flux), "Input flux ix different length to output flux"
	assert all(inlc.time == outlc.time), "Input time is nonidentical to output time"
	assert all(outlc.flux != inlc.flux), "Input and output flux are identical."
	assert len(outlc.flux) == len(outlc.time), "Check time and flux have same length"

	# Check status
	assert status == corrections.STATUS.OK, "STATUS was not set appropriately"

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get('CI') == 'true' and os.environ.get('TRAVIS') == 'true',
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

	# Check metadata
	#assert 'fmean' in outlc.meta, "Metadata is incomplete"
	#assert 'fstd' in outlc.meta, "Metadata is incomplete"
	#assert 'frange' in outlc.meta, "Metadata is incomplete"
	#assert 'drange' in outlc.meta, "Metadata is incomplete"
	assert outlc.meta['task']['starid'] == inlc.meta['task']['starid'], "Metadata is incomplete"
	assert outlc.meta['task'] == inlc.meta['task'], "Metadata is incomplete"

#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	test_ensemble_basics()
	test_ensemble_returned_values()
	test_run_metadata()
