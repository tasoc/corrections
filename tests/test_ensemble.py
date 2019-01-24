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
from corrections import EnsembleCorrector

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

starid = '??'
camera = '??'
ccd = '??'
#----------------------------------------------------------------------
def test_ensemble_basics():
	"""Check that the Ensemblecorrector can be initiated at all"""
	with EnsembleCorrector(INPUT_DIR, plot=True) as ec:
		assert ec.__class__.__name__ == 'EnsembleCorrector', "Did not get the correct class name back"
		assert ec.input_folder == INPUT_DIR, "Incorrect input folder"
		assert ec.plot == True, "Plot parameter passed appropriately"

def test_ensemble_correction():
	"""Check that ensemble performs its most basic correction."""
	CorrClass = corrections.corrclass('ensemble')
	#Initiate the class
	with CorrClass(INPUT_DIR, plot=False) as corr:
		#Create a task
		with corrections.TaskManager(INPUT_DIR) as tm:
			task = tm.get_task(starid=starid, camera=camera, ccd=ccd)
		#Read in the data
		inlc = corr.load_lightcurve(task)

		outlc, status = corr.do_correction(inlc)
	return inlc, outlc, status

def test_ensemble_return_values():
	""" Check that the ensemble returns values that are reasonable and within expected bounds """
	CorrClass = corrections.corrclass('ensemble')
	#Initiate the class
	with CorrClass(INPUT_DIR, plot=False) as corr:
		#Create a task
		with corrections.TaskManager(INPUT_DIR) as tm:
			task = tm.get_task(starid=starid, camera=camera, ccd=ccd)
		#Read in the data
		inlc = corr.load_lightcurve(task)

		outlc, status = corr.do_correction(inlc)

		assert status == STATUS.OK, "STATUS was not set appropriately"		
		assert outlc.meta[]
		assert outlc.meta[]
		assert outlc.meta[]
		assert outlc.meta[]
	

#----------------------------------------------------------------------
if if __name__ == "__main__":
	pass # placeholder