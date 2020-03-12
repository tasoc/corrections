# -*- coding: utf-8 -*-
"""
Tests on known stars.

This is escentially a combination of the older tests of ensemble, CBV, and KASOC Filter correctors.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
.. codeauthor:: Lindsey Carboneau <lmcarboneau@gmail.com>
"""

import pytest
import logging
import numpy as np
from numpy.testing import assert_array_equal, assert_array_less
from lightkurve import TessLightCurve
import sys
import os.path
from bottleneck import nanvar, allnan
from astropy.io import fits
from astropy.table import Table
#import gzip
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import corrections
from corrections.utilities import rms_timescale, ptp
from corrections.plots import plt

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
TEST_DATA_EXISTS = os.path.exists(os.path.join(INPUT_DIR, 'test_data_available_v2.txt'))
DATA_ONLY = pytest.mark.skipif(not TEST_DATA_EXISTS, reason="This requires a full sector of data.")

# List of stars to test:
# The parameters are:
#  1) Corrector (ensemble, cbv, kasoc_filter)
#  2) TIC-number
#  3) Datasource (ffi or tpf)
#  4) Expected VARIANCE
#  5) Expected RMS (1 hour)
#  6) Expected PTP
STAR_LIST = (
	('cbv', 29281992, 'ffi', 1.485801e+06, 8.949764e+02, 2.572604e+02),
	('cbv', 29281992, 'tpf', 1.664423e+06, 9.052271e+02, 6.864500e+01),
	('ensemble', 8196567, 'ffi', 2.419432e+07, 6.305472e+03, 6.530170e+02),
	pytest.param('ensemble', 8195216, 'ffi', 4.548763e+06, 2.213988e+03, 1.049298e+03, marks=DATA_ONLY),
	pytest.param('ensemble', 8196502, 'ffi', 6.901682e+07, 5.227739e+03, 2.679721e+03, marks=DATA_ONLY),
	pytest.param('ensemble', 165109591, 'tpf', 4.462307e+05, 5.684376e+02, 1.107972e+02, marks=DATA_ONLY),
	pytest.param('ensemble', 147424478, 'tpf', 4.115776e+06, 1.190016e+03, 3.930971e+02, marks=DATA_ONLY),
	pytest.param('ensemble', 159778915, 'tpf', 3.985035e+06, 5.127329e+02, 1.754027e+03, marks=DATA_ONLY),
	('kasoc_filter', 29281992, 'ffi', None, None, None), # KASOC Filter performs baaaaaaad here
	('kasoc_filter', 336732616, 'ffi', 4.311563e+06, 1.429098e+03, 8.663223e+02),
	('kasoc_filter', 29281992, 'tpf', 3.970388e+04, 1.239991e+02, 6.421494e+01),
	('kasoc_filter', 336732616, 'tpf', 1.262934e+07, 7.343828e+02, 3.362000e+03) # HATS-3: Known planet
)

#--------------------------------------------------------------------------------------------------
# Here we are doing pure pytest black magic!
# The "SHARED_INPUT_DIR" is defined in conftest.py and is automatically detected by pytest.
#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('corrector,starid,datasource,var_goal,rms_goal,ptp_goal', STAR_LIST)
def test_known_star(SHARED_INPUT_DIR, corrector, starid, datasource, var_goal, rms_goal, ptp_goal):
	""" Check that the ensemble returns values that are reasonable and within expected bounds """

	__dir__ = os.path.abspath(os.path.dirname(__file__))
	logger = logging.getLogger(__name__)
	logger.info("-------------------------------------------------------------")
	logger.warning("CORRECTOR = %s, DATASOURCE = %s, STARID = %d" % (corrector, datasource, starid))

	# All stars are from the same CCD, find the task for it:
	with corrections.TaskManager(SHARED_INPUT_DIR) as tm:
		task = tm.get_task(starid=starid, camera=1, datasource=datasource)

	# Check that task was actually found:
	assert task is not None, "Task could not be found"

	# Load lightcurve that will also be plotted together with the result:
	# This lightcurve is of the same objects, at a state where it was deemed that the
	# corrections were doing a good job.
	compare_lc_path = os.path.join(__dir__, 'compare', 'compare-{0}-{1}-{2}.ecsv.gz'.format(corrector, datasource, starid))
	compare_lc = None
	if os.path.isfile(compare_lc_path):
		compare_lc = Table.read(compare_lc_path, format='ascii.ecsv')

	# Initiate the class
	CorrClass = corrections.corrclass(corrector)
	with tempfile.TemporaryDirectory() as tmpdir:
		with CorrClass(SHARED_INPUT_DIR, plot=True) as corr:
			# Check basic parameters of object (from BaseCorrector):
			assert corr.input_folder == SHARED_INPUT_DIR, "Incorrect input folder"
			assert corr.plot, "Plot parameter passed appropriately"
			assert os.path.isdir(corr.data_folder), "DATA_FOLDER doesn't exist"

			# Load the input lightcurve:
			inlc = corr.load_lightcurve(task)

			# Print input lightcurve properties:
			print( inlc.show_properties() )

			# Run correction:
			tmplc = inlc.copy()
			outlc, status = corr.do_correction(tmplc)

			# Check status
			assert outlc is not None, "Correction fails"
			assert isinstance(outlc, TessLightCurve), "Should return TessLightCurve object"
			assert isinstance(status, corrections.STATUS), "Should return a STATUS object"
			assert status in (corrections.STATUS.OK, corrections.STATUS.WARNING), "STATUS was not set appropriately"

			# Print output lightcurve properties:
			print( outlc.show_properties() )

			# Save the lightcurve to FITS file to be tested later on:
			save_file = corr.save_lightcurve(outlc, output_folder=tmpdir)

		# Check contents
		assert len(outlc) == len(inlc), "Input flux ix different length to output flux"
		assert isinstance(outlc.flux, np.ndarray), "FLUX is not a ndarray"
		assert isinstance(outlc.flux_err, np.ndarray), "FLUX_ERR is not a ndarray"
		assert isinstance(outlc.quality, np.ndarray), "QUALITY is not a ndarray"
		assert outlc.flux.dtype.type is inlc.flux.dtype.type, "FLUX changes dtype"
		assert outlc.flux_err.dtype.type is inlc.flux_err.dtype.type, "FLUX_ERR changes dtype"
		assert outlc.quality.dtype.type is inlc.quality.dtype.type, "QUALITY changes dtype"
		assert outlc.flux.shape == inlc.flux.shape, "FLUX changes shape"
		assert outlc.flux_err.shape == inlc.flux_err.shape, "FLUX_ERR changes shape"
		assert outlc.quality.shape == inlc.quality.shape, "QUALITY changes shape"

		# Plot output lightcurves:
		fig, (ax1, ax2, ax3) = plt.subplots(3, 1, squeeze=True, figsize=[10, 10])
		ax1.plot(inlc.time, inlc.flux, lw=0.5)
		ax1.set_title("{0} - {1} - TIC {2}".format(corrector, datasource, starid))
		if compare_lc:
			ax2.plot(compare_lc['time'], compare_lc['flux'], label='Compare', lw=0.5)
			ax3.axhline(0, lw=0.5, ls=':', color='0.7')
			ax3.plot(outlc.time, outlc.flux - compare_lc['flux'], lw=0.5)
		ax2.plot(outlc.time, outlc.flux, label='New', lw=0.5)
		ax1.set_ylabel('Flux [e/s]')
		ax1.minorticks_on()
		ax2.set_ylabel('Relative Flux [ppm]')
		ax2.minorticks_on()
		ax2.legend()
		ax3.set_ylabel('New - Compare [ppm]')
		ax3.set_xlabel('Time [TBJD]')
		ax3.minorticks_on()
		fig.savefig(os.path.join(__dir__, 'test-{0}-{1}-{2}.png'.format(corrector, datasource, starid)), bbox_inches='tight')
		plt.close(fig)

		# Check things that are allowed to change:
		assert all(outlc.flux != inlc.flux), "Input and output flux are identical."
		assert not np.any(np.isinf(outlc.flux)), "FLUX contains Infinite"
		assert not np.any(np.isinf(outlc.flux_err)), "FLUX_ERR contains Infinite"
		assert np.sum(np.isnan(outlc.flux)) < 0.5*len(outlc), "More than half the lightcurve is NaN"
		assert allnan(outlc.flux_err[np.isnan(outlc.flux)]), "FLUX_ERR should be NaN where FLUX is"

		# TODO: Check that quality hasn't changed in ways that are not allowed:
		# - Only values defined in CorrectorQualityFlags
		# - No removal of flags already set
		assert all(outlc.quality >= 0)
		assert all(outlc.quality <= 128)
		assert all(outlc.quality >= inlc.quality)

		# Things that shouldn't chance from the corrections:
		assert outlc.targetid == inlc.targetid, "TARGETID has changed"
		assert outlc.label == inlc.label, "LABEL has changed"
		assert outlc.sector == inlc.sector, "SECTOR has changed"
		assert outlc.camera == inlc.camera, "CAMERA has changed"
		assert outlc.ccd == inlc.ccd, "CCD has changed"
		assert outlc.quality_bitmask == inlc.quality_bitmask, "QUALITY_BITMASK has changed"
		assert outlc.ra == inlc.ra, "RA has changed"
		assert outlc.dec == inlc.dec, "DEC has changed"
		assert outlc.mission == 'TESS', "MISSION has changed"
		assert outlc.time_format == 'btjd', "TIME_FORMAT has changed"
		assert outlc.time_scale == 'tdb', "TIME_SCALE has changed"
		assert_array_equal(outlc.time, inlc.time, "TIME has changed")
		assert_array_equal(outlc.timecorr, inlc.timecorr, "TIMECORR has changed")
		assert_array_equal(outlc.cadenceno, inlc.cadenceno, "CADENCENO has changed")
		assert_array_equal(outlc.pixel_quality, inlc.pixel_quality, "PIXEL_QUALITY has changed")
		assert_array_equal(outlc.centroid_col, inlc.centroid_col, "CENTROID_COL has changed")
		assert_array_equal(outlc.centroid_row, inlc.centroid_row, "CENTROID_ROW has changed")

		# Check metadata
		assert tmplc.meta == inlc.meta, "Correction changed METADATA in-place"
		assert outlc.meta['task'] == inlc.meta['task'], "Metadata is incomplete"
		assert isinstance(outlc.meta['additional_headers'], fits.Header)

		# Check performance metrics:
		#logger.warning("VAR: %e", nanvar(outlc.flux))
		if var_goal is not None:
			var_in = nanvar(inlc.flux)
			var_out = nanvar(outlc.flux)
			var_diff = np.abs(var_out - var_goal) / var_goal
			logger.info("VAR: %f - %f - %f", var_in, var_out, var_diff)
			assert_array_less(var_diff, 0.05, "VARIANCE changed outside interval")

		#logger.warning("RMS: %e", rms_timescale(outlc))
		if rms_goal is not None:
			rms_in = rms_timescale(inlc)
			rms_out = rms_timescale(outlc)
			rms_diff = np.abs(rms_out - rms_goal) / rms_goal
			logger.info("RMS: %f - %f - %f", rms_in, rms_out, rms_diff)
			assert_array_less(rms_diff, 0.05, "RMS changed outside interval")

		#logger.warning("PTP: %e", ptp(outlc))
		if ptp_goal is not None:
			ptp_in = ptp(inlc)
			ptp_out = ptp(outlc)
			ptp_diff = np.abs(ptp_out - ptp_goal) / ptp_goal
			logger.info("PTP: %f - %f - %f", ptp_in, ptp_out, ptp_diff)
			assert_array_less(ptp_diff, 0.05, "PTP changed outside interval")

		# Check FITS file:
		with fits.open(os.path.join(tmpdir, save_file), mode='readonly') as hdu:
			# Lightcurve FITS table:
			fitslc = hdu['LIGHTCURVE'].data
			hdr = hdu['LIGHTCURVE'].header

			# Checks of things in FITS table that should not have changed at all:
			assert_array_equal(fitslc['TIME'], inlc.time, "FITS: TIME has changed")
			assert_array_equal(fitslc['TIMECORR'], inlc.timecorr, "FITS: TIMECORR has changed")
			assert_array_equal(fitslc['CADENCENO'], inlc.cadenceno, "FITS: CADENCENO has changed")
			assert_array_equal(fitslc['FLUX_RAW'], inlc.flux, "FITS: FLUX_RAW has changed")
			assert_array_equal(fitslc['FLUX_RAW_ERR'], inlc.flux_err, "FITS: FLUX_RAW_ERR has changed")
			assert_array_equal(fitslc['MOM_CENTR1'], inlc.centroid_col, "FITS: CENTROID_COL has changed")
			assert_array_equal(fitslc['MOM_CENTR2'], inlc.centroid_row, "FITS: CENTROID_ROW has changed")

			# Some things are allowed to change, but still within some requirements:
			assert all(fitslc['FLUX_CORR'] != inlc.flux), "FITS: Input and output flux are identical."
			assert np.sum(np.isnan(fitslc['FLUX_CORR'])) < 0.5*len(fitslc['TIME']), "FITS: More than half the lightcurve is NaN"
			assert allnan(fitslc['FLUX_CORR_ERR'][np.isnan(fitslc['FLUX_CORR'])]), "FITS: FLUX_ERR should be NaN where FLUX is"

			if corrector == 'ensemble':
				# Check special headers:
				assert np.isfinite(hdr['ENS_MED']) and hdr['ENS_MED'] > 0
				assert isinstance(hdr['ENS_NUM'], int) and hdr['ENS_NUM'] > 0
				assert hdr['ENS_DLIM'] == 1.0
				assert hdr['ENS_DREL'] == 10.0
				assert hdr['ENS_RLIM'] == 0.4

				# Special extension for ensemble:
				tic = hdu['ENSEMBLE'].data['TIC']
				bzeta = hdu['ENSEMBLE'].data['BZETA']
				assert len(tic) == len(bzeta)
				assert len(np.unique(tic)) == len(tic), "TIC numbers in ENSEMBLE table are not unique"
				assert len(tic) == hdr['ENS_NUM'], "Not the same number of targets in ENSEMBLE table as specified in header"

			elif corrector == 'cbv':
				# Check special headers:
				assert isinstance(hdr['CBV_NUM'], int) and hdr['CBV_NUM'] > 0

				# Check coefficients:
				for k in range(0, hdr['CBV_NUM']+1):
					assert np.isfinite(hdr['CBV_C%d' % k])
				for k in range(1, hdr['CBV_NUM']+1):
					assert np.isfinite(hdr['CBVS_C%d' % k])
				# Check that no other coefficients are present
				assert 'CBV_C%d' % (hdr['CBV_NUM']+1) not in hdr
				assert 'CBVS_C%d' % (hdr['CBV_NUM']+1) not in hdr

			elif corrector == 'kasoc_filter':
				# Check special headers:
				assert hdr['KF_POSS'] == 'None'
				assert np.isfinite(hdr['KF_LONG']) and hdr['KF_LONG'] > 0
				assert np.isfinite(hdr['KF_SHORT']) and hdr['KF_SHORT'] > 0
				assert hdr['KF_SCLIP'] == 4.5
				assert hdr['KF_TCLIP'] == 5.0
				assert hdr['KF_TWDTH'] == 1.0
				assert hdr['KF_PSMTH'] == 200

				assert isinstance(hdr['NUM_PER'], int) and hdr['NUM_PER'] >= 0
				for k in range(1, hdr['NUM_PER']+1):
					assert np.isfinite(hdr['PER_%d' % k]) and hdr['PER_%d' % k] > 0
				# Check that no other periods are present
				assert 'PER_%d' % (hdr['NUM_PER'] + 1) not in hdr

		# Save a new version of the lightcurve to compare with in the future:
		# For some reason Table is very picky with the gzip fileobj, hence the newline definition
		#with gzip.open(compare_lc_path, 'wt', newline="\n") as fid:
		#	Table({'time': outlc.time, 'flux': outlc.flux, 'flux_err': outlc.flux_err}).write(fid, format='ascii.ecsv', delimiter=',')

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
