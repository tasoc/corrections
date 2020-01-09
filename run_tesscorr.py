#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line utility to run TESS detrend correction from command-line.

Note:
	This tool allows for small tests of single target inputs from
	a variety of formats, including FITS and text files,
	and provides an option for debugging and maintenance

Structure inspired by `tessphot` by Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import argparse
import logging
import corrections

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Corrector pipeline on single star.')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default=None, choices=('ensemble', 'cbv', 'kasoc_filter'))
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-p', '--plot', help='Save plots when running.', action='store_true')
	parser.add_argument('-r', '--random', help='Run on random target from TODO-list.', action='store_true')
	parser.add_argument('-t', '--test', help='Use test data and ignore TESSCORR_INPUT environment variable.', action='store_true')
	parser.add_argument('-a', '--all', help='Run correction on all targets.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite previous runs and start over.', action='store_true')
	parser.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, help='TESS Camera. Default is to run all cameras.')
	parser.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('--starid', type=int, help='TIC identifier of target.', nargs='?', default=None)
	parser.add_argument('--datasource', type=str, choices=('ffi','tpf'), default=None, help='Data source or cadence. Default is to run all.')
	parser.add_argument('input_folder', type=str, help='Input directory. This directory should contain a TODO-file and corresponding lightcurves.', nargs='?', default=None)
	parser.add_argument('output_folder', type=str, help='Directory to save output in.', nargs='?', default=None)
	args = parser.parse_args()

	# Make sure at least one setting is given:
	if not args.all and args.starid is None and not args.random:
		parser.error("Please select either a specific STARID or RANDOM.")

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	if not logger.hasHandlers():
		logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('corrections')
	if not logger_parent.hasHandlers():
		logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Get input and output folder from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests', 'input'))
		if args.test:
			input_folder = test_folder
		else:
			input_folder = os.environ.get('TESSCORR_INPUT', test_folder)

	output_folder = args.output_folder
	if output_folder is None:
		output_folder = os.environ.get('TESSCORR_OUTPUT', os.path.join(input_folder, 'lightcurves'))

	logger.info("Loading input data from '%s'", input_folder)
	logger.info("Putting output data in '%s'", output_folder)

	# Make sure the output directory exists:
	os.makedirs(output_folder, exist_ok=True)

	# Get the class for the selected method:
	CorrClass = corrections.corrclass(args.method)

	# Initialize the corrector class:
	with CorrClass(input_folder, plot=args.plot) as corr:
		# Start the TaskManager:
		with corrections.TaskManager(input_folder, overwrite=args.overwrite) as tm:
			while True:
				if args.random:
					task = tm.get_random_task()
				else:
					task = tm.get_task(starid=args.starid, camera=args.camera, ccd=args.ccd, datasource=args.datasource)

				if task is None: break

				# Run the correction:
				result = corr.correct(task, output_folder=output_folder)

				# Construct results to return to TaskManager:
				tm.save_results(result)

				if not args.all:
					break

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
