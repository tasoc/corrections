#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Ensemble Corrector.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Lindsey Carboneau <lmcarboneau@gmail.com>
"""

import pytest
import numpy as np
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import corrections

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_EXISTS = os.path.exists(os.path.join(INPUT_DIR, 'test_data_available.txt'))
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
		assert ec.plot, "Plot parameter passed appropriately"

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a sector of data.")
def test_ensemble_returned_values():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	with corrections.TaskManager(INPUT_DIR) as tm:
		task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	with CorrClass(INPUT_DIR, plot=False) as corr:
		inlc = corr.load_lightcurve(task)
		outlc, status = corr.do_correction(inlc.copy())

	# Check status
	assert outlc is not None, "Ensemble fails"
	assert status == corrections.STATUS.OK, "STATUS was not set appropriately"

	# Check input validation
	#with pytest.raises(ValueError) as err:
	#	outlc, status = corr.do_correction('hello world')
	#	assert('The input to `do_correction` is not a TessLightCurve object!' in err.value.args[0])

	print( inlc.show_properties() )
	print( outlc.show_properties() )

	# Check contents
	assert len(outlc) == len(inlc), "Input flux ix different length to output flux"
	assert all(inlc.time == outlc.time), "Input time is nonidentical to output time"
	assert all(outlc.flux != inlc.flux), "Input and output flux are identical."
	assert np.sum(np.isnan(outlc.flux)) < 0.5*len(outlc), "More than half the lightcurve is NaN"

	assert len(outlc.flux) == len(outlc.time), "Check TIME and FLUX have same length"
	assert len(outlc.flux_err) == len(outlc.time), "Check TIME and FLUX_ERR have same length"
	assert len(outlc.quality) == len(outlc.time), "Check TIME and QUALITY have same length"
	assert len(outlc.pixel_quality) == len(outlc.time), "Check TIME and QUALITY have same length"
	assert len(outlc.cadenceno) == len(outlc.time), "Check TIME and CADENCENO have same length"
	assert len(outlc.centroid_col) == len(outlc.time), "Check TIME and CENTROID_COL have same length"
	assert len(outlc.centroid_row) == len(outlc.time), "Check TIME and CENTROID_ROW have same length"
	assert len(outlc.timecorr) == len(outlc.time), "Check TIME and TIMECORR have same length"

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a sector of data.")
def test_run_metadata():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	with corrections.TaskManager(INPUT_DIR) as tm:
		task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	with CorrClass(INPUT_DIR, plot=False) as corr:
		inlc = corr.load_lightcurve(task)
		outlc, status = corr.do_correction(inlc.copy())

	assert outlc is not None, "Ensemble fails"

	print( inlc.show_properties() )
	print( outlc.show_properties() )

	# Check metadata
	assert outlc.meta['task']['starid'] == inlc.meta['task']['starid'], "Metadata is incomplete"
	assert outlc.meta['task'] == inlc.meta['task'], "Metadata is incomplete"

#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	test_ensemble_basics()
	test_ensemble_returned_values()
	test_run_metadata()
