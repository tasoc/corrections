#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheduler using MPI for running the TESS lightcurve corrections
pipeline on a large scale multi-core computer.

The setup uses the task-pull paradigm for high-throughput computing
using ``mpi4py``. Task pull is an efficient way to perform a large number of
independent tasks when there are more tasks than processors, especially
when the run times vary for each task.

The basic example was inspired by
https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py

Example
-------
To run the program using four processes (one master and three workers) you can
execute the following command:

>>> mpiexec -n 4 python run_tesscorr_mpi.py

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from mpi4py import MPI
import argparse
import logging
import traceback
import os
import enum
from timeit import default_timer
import corrections
from corrections.utilities import CadenceType

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Corrections in parallel using MPI.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default='cbv', choices=('ensemble', 'cbv', 'kasoc_filter'))
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('-p', '--plot', help='Save plots when running.', action='store_true')
	group = parser.add_argument_group('Filter which targets to process')
	group.add_argument('--sector', type=int, default=None, help='TESS Sector.')
	group.add_argument('--cadence', type=CadenceType, choices=('ffi',1800,600,120,20), default=None, help='Cadence. Default is to run all.')
	group.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, help='TESS Camera. Default is to run all cameras.')
	group.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('input_folder', type=str, help='Input directory. This directory should contain a TODO-file and corresponding lightcurves.', nargs='?', default=None)
	parser.add_argument('output_folder', type=str, help='Directory to save output in.', nargs='?', default=None)
	args = parser.parse_args()

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Get input and output folder from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('TESSCORR_INPUT')
	if not input_folder:
		parser.error("Please specify an INPUT_FOLDER.")

	output_folder = args.output_folder
	if output_folder is None:
		output_folder = os.environ.get('TESSCORR_OUTPUT', os.path.join(os.path.dirname(input_folder), 'lightcurves'))

	# Define MPI message tags
	tags = enum.IntEnum('tags', ('INIT', 'READY', 'DONE', 'EXIT', 'START'))

	# Initializations and preliminaries
	comm = MPI.COMM_WORLD   # get MPI communicator object
	size = comm.size        # total number of processes
	rank = comm.rank        # rank of this process
	status = MPI.Status()   # get MPI status object

	if rank == 0:
		try:
			# Constraints on which targets to process:
			constraints = {
				'sector': args.sector,
				'cadence': args.cadence,
				'camera': args.camera,
				'ccd': args.ccd
			}

			# File path to write summary to:
			summary_file = os.path.join(output_folder, f'summary_corr_{args.method:s}.json')

			# Invoke the TaskManager to ensure that the input TODO-file has the correct columns
			# and indicies, which is automatically created by the TaskManager init function.
			with corrections.TaskManager(input_folder, cleanup=True, overwrite=args.overwrite,
				cleanup_constraints=constraints):
				pass

			# Signal that workers are free to initialize:
			comm.Barrier() # Barrier 1

			# Wait for all workers to initialize:
			comm.Barrier() # Barrier 2

			# Start TaskManager, which keeps track of the task that needs to be performed:
			with corrections.TaskManager(input_folder, overwrite=args.overwrite,
				cleanup_constraints=constraints, summary=summary_file) as tm:

				# Set level of TaskManager logger:
				tm.logger.setLevel(logging_level)

				# Get list of tasks:
				numtasks = tm.get_number_tasks(**constraints)
				tm.logger.info("%d tasks to be run", numtasks)

				# Start the master loop that will assign tasks
				# to the workers:
				num_workers = size - 1
				closed_workers = 0
				tm.logger.info("Master starting with %d workers", num_workers)
				while closed_workers < num_workers:
					# Get information from worker:
					data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
					source = status.Get_source()
					tag = status.Get_tag()

					if tag == tags.DONE:
						# The worker is done with a task
						tm.logger.debug("Got data from worker %d: %s", source, data)
						tm.save_results(data)

					if tag in (tags.DONE, tags.READY):
						# Worker is ready for a new task, so send it a task
						tasks = tm.get_task(**constraints, chunk=10)
						if tasks:
							tm.start_task(tasks)
							tm.logger.debug("Sending %d tasks to worker %d", len(tasks), source)
							comm.send(tasks, dest=source, tag=tags.START)
						else:
							comm.send(None, dest=source, tag=tags.EXIT)

					elif tag == tags.EXIT:
						# The worker has exited
						tm.logger.info("Worker %d exited.", source)
						closed_workers += 1

					else:
						# This should never happen, but just to
						# make sure we don't run into an infinite loop:
						raise RuntimeError(f"Master received an unknown tag: '{tag}'")

				tm.logger.info("Master finishing")

		except: # noqa: E722, pragma: no cover
			# If something fails in the master
			print(traceback.format_exc().strip())
			comm.Abort(1)

	else:
		# Worker processes execute code below
		# Configure logging within photometry:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		logger = logging.getLogger('corrections')
		logger.addHandler(console)
		logger.setLevel(logging.WARNING)

		# Get the class for the selected method:
		CorrClass = corrections.corrclass(args.method)

		try:
			# Wait for signal that we are okay to initialize:
			comm.Barrier() # Barrier 1

			# We can now safely initialize the corrector on the input file:
			with CorrClass(input_folder, plot=args.plot) as corr:

				# Wait for all workers do be done initializing:
				comm.Barrier() # Barrier 2

				# Send signal that we are ready for task:
				comm.send(None, dest=0, tag=tags.READY)

				while True:
					# Receive a task from the master:
					tic = default_timer()
					tasks = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
					tag = status.Get_tag()
					toc = default_timer()

					if tag == tags.START:
						# Make sure we can loop through tasks,
						# even in the case we have only gotten one:
						results = []
						if not isinstance(tasks, (list, tuple)):
							tasks = list(tasks)

						# Loop through the tasks given to us:
						for task in tasks:
							result = task.copy()

							# Run the correction:
							try:
								result = corr.correct(task)
							except: # noqa: E722
								# Something went wrong
								error_msg = traceback.format_exc().strip()
								result.update({
									'status_corr': corrections.STATUS.ERROR,
									'details': {'errors': [error_msg]},
								})

							result.update({'worker_wait_time': toc-tic})
							results.append(result)

						# Send the result back to the master:
						comm.send(results, dest=0, tag=tags.DONE)

					elif tag == tags.EXIT:
						# We were told to EXIT, so lets do that
						break

					else:
						# This should never happen, but just to
						# make sure we don't run into an infinite loop:
						raise RuntimeError(f"Worker received an unknown tag: '{tag}'")

		except: # noqa: E722, pragma: no cover
			logger.exception("Something failed in worker")

		finally:
			comm.send(None, dest=0, tag=tags.EXIT)

if __name__ == '__main__':
	main()
