#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.decomposition import PCA
import matplotlib.colors as colors
import argparse
import logging
from bottleneck import allnan, nanmedian
from scipy.interpolate import pchip_interpolate
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="scipy.stats") # they are simply annoying!
from tqdm import tqdm

from scipy.interpolate import Rbf, SmoothBivariateSpline

from functools import partial

from corrections import CBVCorrector
import dill
import six
plt.ioff()
import sqlite3


from multiprocessing import Pool

	
#------------------------------------------------------------------------------

def search_database(cursor, select=None, search=None, order_by=None, limit=None, distinct=False):
	"""
	Search list of lightcurves and return a list of tasks/stars matching the given criteria.

	Parameters:
		search (list of strings or None): Conditions to apply to the selection of stars from the database
		order_by (list, string or None): Column to order the database output by.
		limit (int or None): Maximum number of rows to retrieve from the database. If limit is None, all the rows are retrieved.
		distinct (boolean): Boolean indicating if the query should return unique elements only.

	Returns:
		list of dicts: Returns all stars retrieved by the call to the database as dicts/tasks that can be consumed directly by load_lightcurve

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	if select is None:
		select = '*'
	elif isinstance(select, (list, tuple)):
		select = ",".join(select)

	if search is None:
		search = ''
	elif isinstance(search, (list, tuple)):
		search = "WHERE " + " AND ".join(search)
	else:
		search = 'WHERE ' + search

	if order_by is None:
		order_by = ''
	elif isinstance(order_by, (list, tuple)):
		order_by = " ORDER BY " + ",".join(order_by)
	elif isinstance(order_by, six.string_types):
		order_by = " ORDER BY " + order_by

	limit = '' if limit is None else " LIMIT %d" % limit

	query = "SELECT {distinct:s}{select:s} FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority {search:s}{order_by:s}{limit:s};".format(
		distinct='DISTINCT ' if distinct else '',
		select=select,
		search=search,
		order_by=order_by,
		limit=limit
	)
	logger.debug("Running query: %s", query)

	# Ask the database: status=1 
	cursor.execute(query)
	return [dict(row) for row in cursor.fetchall()]



def prepare_cbv(cbv_area, input_folder=None):

	logger=logging.getLogger(__name__)		
	logger.info('running CBV for area %s', str(cbv_area))
	
	with CBVCorrector(input_folder) as C:
		C.compute_cbvs(cbv_area)
		C.cotrend_ini(cbv_area)	
		

#def prepare_wei(cbv_area):
#
#	logger=logging.getLogger(__name__)		
#	logger.info('running CBV')
#	
#	with CBVCorrector as C:
#		C.compute_weight_interpolations(cbv_area)	
		
		
# =============================================================================
#
# =============================================================================

if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run preparation of CBVs for single or several CBV-areas')
	parser.add_argument('-e', '--ext', help='Extension of plots.', default='png', choices=('png', 'eps'))
	parser.add_argument('-s', '--show', help='Show plots.', default=False, choices=('True', 'False'))
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-a', '--area', help='Single CBV_area for which to prepare photometry.', nargs='?', default=None)
	parser.add_argument('input_folder', type=str, help='Directory to create catalog files in.', nargs='?', default=None)
	parser.add_argument('output_folder', type=str, help='Directory in which to place output if several input folders are given.', nargs='?', default=None)
	args = parser.parse_args()


	args.show = 'True'
	args.input_folder = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-2127753/'
	todo_file = os.path.join(args.input_folder, 'todo.sqlite')
#	args.area='111'

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


	logger.info("Loading input data from '%s'", args.input_folder)
	logger.info("Putting output data in '%s'", args.output_folder)

	# Load the SQLite file:
	print(todo_file)
	conn = sqlite3.connect(todo_file)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()
		
		
	if args.area is None:
		cbv_areas = [int(row['cbv_area']) for row in search_database(cursor, select='cbv_area', distinct=True)]

		p = Pool(8)
		wrap=partial(prepare_cbv, input_folder=args.input_folder)
		p.map(wrap, cbv_areas)
		
	else:
		prepare_cbv(args.input_folder, int(args.area))
	
	
	
	
	