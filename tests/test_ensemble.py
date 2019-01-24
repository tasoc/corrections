#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Ensemble.

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import corrections

# INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
INPUT_DIR = '../../TESS_data/lightcurves-2127753/'

# TODO: Build a dummy structure that we can test failures on
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


def test_ensemble_return_values():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	tm = corrections.TaskManager(INPUT_DIR)
	task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	corr = CorrClass(INPUT_DIR, plot=False)
	inlc = corr.load_lightcurve(task)
	outlc, status = corr.do_correction(inlc.copy())

	#Check contents
	assert len(outlc.flux) == len(inlc.flux), "Input flux ix different length to output flux"
	assert all(inlc.time == outlc.time), "Input time is nonidentical to output time"
	assert all(outlc.flux != inlc.flux), "Input and output flux are identical."
	assert len(outlc.flux) == len(outlc.time), "Check time and flux have same length"

	#Check metadata
	assert 'fmean' in outlc.meta, "Metadata is incomplete"
	assert 'fstd' in outlc.meta, "Metadata is incomplete"
	assert 'frange' in outlc.meta, "Metadata is incomplete"
	assert 'drange' in outlc.meta, "Metadata is incomplete"
	assert outlc.meta['task']['starid'] == inlc.meta['task']['starid'], "Metadata is incomplete"
	assert outlc.meta.task == inlc.meta.task, "Metadata is incomplete"

	#Check status
	assert status == corrections.STATUS.OK, "STATUS was not set appropriately"

def test_ensemble_cbv_comparison():
	pass

if __name__ == "__main__":
	tm = corrections.TaskManager(INPUT_DIR)
	task = tm.get_task(starid=starid, camera=camera, ccd=ccd)

	#Initiate the class
	CorrClass = corrections.corrclass('ensemble')
	corr = CorrClass(INPUT_DIR, plot=False)
	inlc = corr.load_lightcurve(task)
	outlc, status = corr.do_correction(inlc.copy())
