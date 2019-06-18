#!/usr/bin/env python
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
import corrections

#------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Corrections in parallel using MPI.')
	#parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	#parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-m', '--method', help='Corrector method to use.', default=None, choices=('ensemble', 'cbv', 'kasoc_filter'))
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('-p', '--plot', help='Save plots when running.', action='store_true')
	parser.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, help='TESS Camera. Default is to run all cameras.')
	parser.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('--datasource', type=str, choices=('ffi','tpf'), default='ffi', help='Data source or cadence.')
	parser.add_argument('input_folder', type=str, help='Input directory. This directory should contain a TODO-file and corresponding lightcurves.', nargs='?', default=None)
	args = parser.parse_args()

	# Get input and output folder from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('TESSCORR_INPUT')
	if not input_folder:
		parser.error("Please specify an INPUT_FOLDER.")
	output_folder = os.environ.get('TESSCORR_OUTPUT', os.path.join(input_folder, 'lightcurves'))

	# Define MPI message tags
	tags = enum.IntEnum('tags', ('READY', 'DONE', 'EXIT', 'START'))

	# Initializations and preliminaries
	comm = MPI.COMM_WORLD   # get MPI communicator object
	size = comm.size        # total number of processes
	rank = comm.rank        # rank of this process
	status = MPI.Status()   # get MPI status object

	if rank == 0:
		try:
			with corrections.TaskManager(input_folder, cleanup=True, overwrite=args.overwrite) as tm: #, summary=os.path.join(output_folder, 'summary.json')) as tm:
				# Get list of tasks:
				numtasks = tm.get_number_tasks()
				tm.logger.info("%d tasks to be run", numtasks)

				# Start the master loop that will assign tasks
				# to the workers:
				num_workers = size - 1
				closed_workers = 0
				tm.logger.info("Master starting with %d workers", num_workers)
				while closed_workers < num_workers:
					# Ask workers for information:
					data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
					source = status.Get_source()
					tag = status.Get_tag()

					if tag == tags.DONE:
						# The worker is done with a task
						tm.logger.info("Got data from worker %d: %s", source, data)
						tm.save_result(data)

					if tag in (tags.DONE, tags.READY):
						# Worker is ready, so send it a task
						task = tm.get_task(camera=args.camera, ccd=args.ccd, datasource=args.datasource)
						if task:
							task_index = task['priority']
							tm.start_task(task_index)
							comm.send(task, dest=source, tag=tags.START)
							tm.logger.info("Sending task %d to worker %d", task_index, source)
						else:
							comm.send(None, dest=source, tag=tags.EXIT)

					elif tag == tags.EXIT:
						# The worker has exited
						tm.logger.info("Worker %d exited.", source)
						closed_workers += 1

					else:
						# This should never happen, but just to
						# make sure we don't run into an infinite loop:
						raise Exception("Master received an unknown tag: '{0}'".format(tag))

				tm.logger.info("Master finishing")

		except:
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
			with CorrClass(input_folder, plot=args.plot) as corr:

				# Send signal that we are ready for task:
				comm.send(None, dest=0, tag=tags.READY)

				while True:
					# Receive a task from the master:
					task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
					tag = status.Get_tag()

					if tag == tags.START:
						# Run the correction:
						result = corr.correct(task)

						# Send the result back to the master:
						comm.send(result, dest=0, tag=tags.DONE)

						# Attempt some cleanup:
						# TODO: Is this even needed?
						del task, result

					elif tag == tags.EXIT:
						# We were told to EXIT, so lets do that
						break

					else:
						# This should never happen, but just to
						# make sure we don't run into an infinite loop:
						raise Exception("Worker received an unknown tag: '{0}'".format(tag))

		except:
			logger.exception("Something failed in worker")

		finally:
			comm.send(None, dest=0, tag=tags.EXIT)

if __name__ == '__main__':
	main()