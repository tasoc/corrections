#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run preparation of CBVs for single or several CBV-areas.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import os
import logging
from functools import partial
import multiprocessing
import corrections
from corrections.taskmanager import _build_constraints
from corrections.utilities import CadenceType

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run preparation of CBVs for single or several CBV-areas.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-ip', '--iniplot', help='Make Initial fitting plots.', action='store_true')
	parser.add_argument('--threads', type=int, default=None, help="Number of parallel threads to use. If not specified, all available CPUs will be used.")
	parser.add_argument('--output', type=str, default=None, help="Directory where output CBVs will be saved.")

	group = parser.add_argument_group('Specifying CBVs to calculate')
	group.add_argument('--sector', type=int, default=None, help='Sector to create CBVs for.')
	group.add_argument('--cadence', type=CadenceType, default='ffi', choices=('ffi', 1800, 600, 120, 20), help='Cadence for the creation of CBVs.')
	group.add_argument('--camera', type=int, choices=(1,2,3,4), action='append', default=None, help='TESS Camera. Default is to run all cameras.')
	group.add_argument('--ccd', type=int, choices=(1,2,3,4), action='append', default=None, help='TESS CCD. Default is to run all CCDs.')
	group.add_argument('-a', '--area', type=int, action='append', default=None, help='Single CBV_area for which to prepare photometry. Default is to run all areas.')

	group = parser.add_argument_group('Settings')
	group.add_argument('--ncbv', type=int, default=16, help='Number of CBVs to compute')
	group.add_argument('--corr', type=float, default=0.5, help='Fraction of most correlated stars to use for CBVs.')
	group.add_argument('--snr', type=float, default=5, help='SNR (dB) for selection of CBVs.')
	group.add_argument('--el', type=float, default=-0.5, help='Entropy limit for discarting star contribution to CBV.')

	parser.add_argument('--version', type=int, required=True, help='Data release number to store in output files.')
	parser.add_argument('input_folder', type=str, nargs='?', default=None, help='Directory to create catalog files in.')
	args = parser.parse_args()

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

	# Parse the input folder:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('TESSCORR_INPUT')
	if input_folder is None or not os.path.exists(input_folder):
		parser.error("Invalid input folder")
	logger.info("Loading input data from '%s'", input_folder)

	# Build list of constraints:
	constraints = _build_constraints(
		sector=args.sector,
		cadence=args.cadence,
		camera=args.camera,
		ccd=args.ccd,
		cbv_area=args.area,
		return_list=True)
	if not constraints:
		constraints = None

	# Invoke the TaskManager to ensure that the input TODO-file has the correct columns
	# and indicies, which is automatically created by the TaskManager init function.
	with corrections.TaskManager(input_folder, cleanup=False, cleanup_constraints=constraints):
		pass

	# Use the BaseCorrector to search the database for which CBV_AREAS to run:
	with corrections.BaseCorrector(input_folder) as bc:
		# Search for valid areas:
		cbv_areas = [(row['sector'], row['cbv_area']) for row in bc.search_database(select=['sector','cbv_area'], distinct=True, search=constraints, order_by=['sector', 'cbv_area'])]
		logger.debug("CBV areas: %s", cbv_areas)

	# Stop if there are no CBV-Areas to process:
	if not cbv_areas:
		logger.info("No CBV-areas found to be processed.")
		return

	# Number of threads to run in parallel:
	threads = args.threads
	if not threads:
		threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	threads = min(threads, len(cbv_areas))
	logger.info("Using %d processes.", threads)

	# Create wrapper function which only takes a single cbv_area as input:
	create_cbv_wrapper = partial(corrections.create_cbv,
		input_folder=input_folder,
		output_folder=args.output,
		cadence=args.cadence,
		version=args.version,
		threshold_correlation=args.corr,
		threshold_snrtest=args.snr,
		ncbv=args.ncbv,
		threshold_entropy=args.el,
		ip=args.iniplot)

	# Run the preparation:
	if threads > 1:
		# Disable printing info messages from the parent function.
		# It is going to be all jumbled up anyway.
		logger_parent.setLevel(logging.WARNING)

		# There is more than one area to process, so let's start
		# a process pool and process them in parallel:
		with multiprocessing.Pool(threads) as pool:
			pool.map(create_cbv_wrapper, cbv_areas)

	else:
		# Only a single area to process, so let's not bother with
		# starting subprocesses, but simply run it directly:
		for cbv_area in cbv_areas:
			create_cbv_wrapper(cbv_area)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
