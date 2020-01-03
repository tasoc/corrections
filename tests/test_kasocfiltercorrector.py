#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of KASOC Filter Corrector.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

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
def test_kasocfilter_basics():
	"""Check that the Ensemblecorrector can be initiated at all"""
	with corrections.KASOCFilterCorrector(INPUT_DIR, plot=True) as ec:
		assert ec.__class__.__name__ == 'KASOCFilterCorrector', "Did not get the correct class name back"
		assert ec.input_folder == INPUT_DIR, "Incorrect input folder"
		assert ec.plot, "Plot parameter passed appropriately"

#--------------------------------------------------------------------------------------------------
def test_kasocfilter_returned_values():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	with corrections.TaskManager(INPUT_DIR) as tm:
		task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('kasoc_filter')
	with CorrClass(INPUT_DIR, plot=False) as corr:
		inlc = corr.load_lightcurve(task)
		outlc, status = corr.do_correction(inlc.copy())

	# Check status
	assert outlc is not None, "KASOC Filter fails"
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

	assert len(outlc.flux) == len(outlc.time), "Check TIME and FLUX have same length"
	assert len(outlc.flux_err) == len(outlc.time), "Check TIME and FLUX_ERR have same length"
	assert len(outlc.quality) == len(outlc.time), "Check TIME and QUALITY have same length"
	assert len(outlc.pixel_quality) == len(outlc.time), "Check TIME and QUALITY have same length"
	assert len(outlc.cadenceno) == len(outlc.time), "Check TIME and CADENCENO have same length"
	assert len(outlc.centroid_col) == len(outlc.time), "Check TIME and CENTROID_COL have same length"
	assert len(outlc.centroid_row) == len(outlc.time), "Check TIME and CENTROID_ROW have same length"
	assert len(outlc.timecorr) == len(outlc.time), "Check TIME and TIMECORR have same length"

#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	test_kasocfilter_basics()
	test_kasocfilter_returned_values()
