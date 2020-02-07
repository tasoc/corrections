# -*- coding: utf-8 -*-
"""
Tests on known stars.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import sys
import os.path
from bottleneck import nanmedian, nanvar
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import corrections
from corrections.utilities import rms_timescale
#from corrections.plots import plt
#plt.switch_backend('Qt5Agg')

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_EXISTS = os.path.exists(os.path.join(INPUT_DIR, 'test_data_available.txt'))

# List of stars to test:
# The parameters are:
#  1) Corrector (ensemble, cbv)
#  2) TIC-number
#  3) Datasource (ffi or tpf)
#  4) Expected VARIANCE
#  5) Expected RMS (1 hour)
#  6) Expected PTP
STAR_LIST = (
	('cbv', 29281992, 'ffi', 1.48e6, 905.23, 257.2),
	('cbv', 29281992, 'tpf', 1.66e6, 905.23, 68.64),
	('ensemble', 29281992, 'ffi', 1.48e6, 905.23, 257.2),
	#('ensemble', 29281992, 'tpf', 1.48e6, 905.23, 257.2),
)

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a sector of data.")
@pytest.mark.parametrize('corrector,starid,datasource,var_goal,rms_goal,ptp_goal', STAR_LIST)
def test_known_star(corrector, starid, datasource, var_goal, rms_goal, ptp_goal):
	""" Check that the ensemble returns values that are reasonable and within expected bounds """

	print("-------------------------------------------------------------")
	print("CORRECTOR = %s, DATASOURCE = %s, STARID = %d" % (corrector, datasource, starid))

	# All stars are from the same CCD, find the task for it:
	with corrections.TaskManager(INPUT_DIR) as tm:
		task = tm.get_task(starid=starid, camera=1, ccd=4, datasource=datasource)

	#Initiate the class
	CorrClass = corrections.corrclass(corrector)
	with CorrClass(INPUT_DIR, plot=False) as corr:
		inlc = corr.load_lightcurve(task)

		# Plot input lightcurve:
		#print( inlc.show_properties() )
		#inlc.plot(normalize=False)

		outlc, status = corr.do_correction(inlc.copy())

	# Check status
	print(status)
	assert outlc is not None, "Correction fails"
	assert status in (corrections.STATUS.OK, corrections.STATUS.WARNING), "STATUS was not set appropriately"

	# Plot output lightcurves:
	#print( outlc.show_properties() )
	#outlc.plot(ylabel='Relative Flux [ppm]', normalize=False)
	#plt.show(block=True)

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

	# Check metadata
	assert outlc.meta['task']['starid'] == inlc.meta['task']['starid'], "Metadata is incomplete"
	assert outlc.meta['task'] == inlc.meta['task'], "Metadata is incomplete"

	# Check performance metrics:
	var_in = nanvar(inlc.flux)
	var_out = nanvar(outlc.flux)
	var_diff = np.abs(var_out - var_goal) / var_goal
	print(var_in, var_out, var_diff)
	assert var_diff < 0.05, "VARIANCE changed outside interval"

	rms_in = rms_timescale(inlc)
	rms_out = rms_timescale(outlc)
	rms_diff = np.abs(rms_out - rms_goal) / rms_goal
	print(rms_in, rms_out, rms_diff)
	assert rms_diff < 0.05, "RMS changed outside interval"

	ptp_in = nanmedian(np.abs(np.diff(inlc.flux)))
	ptp_out = nanmedian(np.abs(np.diff(outlc.flux)))
	ptp_diff = np.abs(ptp_out - ptp_goal) / ptp_goal
	print(ptp_in, ptp_out, ptp_diff)
	assert ptp_diff < 0.05, "PTP changed outside interval"

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	#pytest.main([__file__])

	for s in STAR_LIST:
		test_known_star(*s)
