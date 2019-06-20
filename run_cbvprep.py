#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run preparation of CBVs for single or several CBV-areas.

.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import argparse
import os
import logging
from functools import partial
import multiprocessing
from corrections import CBVCorrector, BaseCorrector

#------------------------------------------------------------------------------
def prepare_cbv(cbv_area, input_folder=None, threshold=None, ncbv=None, el=None, ip=False, datasource='ffi'):

	logger = logging.getLogger(__name__)
	logger.info('running CBV for area %s', str(cbv_area))

	with CBVCorrector(input_folder, threshold_snrtest=threshold, ncomponents=ncbv, datasource=datasource) as C:
		C.compute_cbvs(cbv_area, ent_limit=el)
		C.spike_sep(cbv_area)
		
		
#		C.cotrend_ini(cbv_area, do_ini_plots=ip)
#		try:
#			C.compute_distance_map(cbv_area)
#		except:
#			pass

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run preparation of CBVs for single or several CBV-areas.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-ip', '--iniplot', help='Make Initial fitting plots.', action='store_true')

	group = parser.add_argument_group('Specifying CBVs to calculate')
	group.add_argument('-a', '--area', type=int, help='Single CBV_area for which to prepare photometry. Default is to run all areas.', action='append', default=None)
	group.add_argument('--camera', type=int, choices=(1,2,3,4), action='append', default=None, help='TESS Camera. Default is to run all cameras.')
	group.add_argument('--ccd', type=int, choices=(1,2,3,4), action='append', default=None, help='TESS CCD. Default is to run all CCDs.')

	group = parser.add_argument_group('Settings')
	group.add_argument('--snr', type=float, default=5, help='SNR (dB) for selection of CBVs.')
	group.add_argument('--ncbv', type=int, default=16, help='Number of CBVs to compute')
	group.add_argument('--el', type=float, default=-0.5, help='Entropy limit for discarting star contribution to CBV')

	group.add_argument('--datasource', type=str, default='ffi', choices=('ffi', 'tpf'), help='Data source for the creation of CBVs')

	parser.add_argument('input_folder', type=str, help='Directory to create catalog files in.', nargs='?', default=None)

	args = parser.parse_args()
	
	#args.input_folder = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-combined'
#	args.area = [413]
	
	
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
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('corrections')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Parse the input folder:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('TESSCORR_INPUT')
	if input_folder is None or not os.path.isdir(input_folder):
		parser.error("Invalid input folder")
	logger.info("Loading input data from '%s'", input_folder)

	# Use the BaseCorrector to search the database for which CBV_AREAS to run:
	with BaseCorrector(input_folder) as bc:
		# Build list of constraints:
		constraints = []
		if args.camera:
			constraints.append('camera IN (%s)' % ",".join([str(c) for c in args.camera]))
		if args.ccd:
			constraints.append('ccd IN (%s)' % ",".join([str(c) for c in args.ccd]))
		if args.area:
			constraints.append('cbv_area IN (%s)' % ",".join([str(c) for c in args.area]))
		if not constraints:
			constraints = None

		# Search for valid areas:
		cbv_areas = [row['cbv_area'] for row in bc.search_database(select='cbv_area', distinct=True, search=constraints)]
		logger.debug("CBV areas: %s", cbv_areas)

	# Number of threads to run in parallel:
	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	threads = min(threads, len(cbv_areas))
	logger.info("Using %d processes.", threads)

	# Create wrapper function which only takes a single cbv_area as input:
	prepare_cbv_wrapper = partial(prepare_cbv, input_folder=input_folder, threshold=args.snr, ncbv=args.ncbv, el=args.el, ip=args.iniplot, datasource=args.datasource)

	# Run the preparation:
	if threads > 1:
		# Disable printing info messages from the parent function.
		# It is going to be all jumbled up anyway.
		logger_parent.setLevel(logging.WARNING)

		# There is more than one area to process, so let's start
		# a process pool and process them in parallel:
		pool = multiprocessing.Pool(threads)
		pool.map(prepare_cbv_wrapper, cbv_areas)
		pool.close()
		pool.join()

	elif threads == 1:
		# Only a single area to process, so let's not bother with
		# starting subprocesses, but simply run it directly:
		for cbv_area in cbv_areas:
			prepare_cbv_wrapper(cbv_area)

	else:
		logger.info("No cbv-areas found to be processed.")
