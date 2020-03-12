#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sqlite3
import zipfile
import os.path
import shutil
from contextlib import closing
from tqdm import tqdm
import sys

if __name__ == '__main__':

	#input_todo = '/aadc/tasoc/archive/S01_DR01/lightcurves-combined/todo.sqlite'
	#input_dir = '/aadc/tasoc/archive/S01_DR01/lightcurves-combined/'
	input_todo = r'E:\TASOC_DR05\S01\todo.sqlite'
	input_dir = r'E:\TASOC_DR05\S01'

	# List of priorities of files that we would like to include directly for testing
	# as part of the code (via Git LFS):
	priorities_from_cbv = [17, 18]
	priorities_from_kasoc_filter = [114686, 114688]
	priorities_from_ensemble = [91933, 129658, 314088, 501674, 82079, 885429, 158688, 203350, 286816, 292400, 634875]
	pri_for_lfs = priorities_from_cbv + priorities_from_kasoc_filter + priorities_from_ensemble

	todo_file = os.path.abspath('./todo.sqlite')
	zippath = os.path.abspath('./corrections_tests_input_v2.zip')

	print("Copying TODO-file...")
	shutil.copyfile(input_todo, todo_file)

	# Open the SQLite file:
	with closing(sqlite3.connect(todo_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		print("File that should be added directly via Git LFS:")
		already_packaged = []
		cursor.execute("SELECT lightcurve FROM diagnostics WHERE priority IN (%s)" % (','.join([str(pri) for pri in pri_for_lfs])))
		for row in cursor.fetchall():
			print("git add -f %s" % row['lightcurve'])
			already_packaged.append(row['lightcurve'])
		already_packaged = set(already_packaged)

		print("Setting PRAGMAs if they havent been already...")
		cursor.execute("PRAGMA foreign_keys=ON;")
		cursor.execute("PRAGMA locking_mode=NORMAL;")
		cursor.execute("PRAGMA journal_mode=TRUNCATE;")
		conn.commit()

		# Remove tables that are not needed
		print("Removing not needed tables...")
		cursor.execute("DROP TABLE IF EXISTS photometry_skipped;")
		cursor.execute("DROP TABLE IF EXISTS datavalidation;")
		cursor.execute("DROP TABLE IF EXISTS datavalidation_corr;")
		cursor.execute("DROP TABLE IF EXISTS diagnostics_corr;")
		conn.commit()

		# Only keep targets from a few CCDs
		print("Deleting all targets not from specific CCDs...")
		cursor.execute("DELETE FROM todolist WHERE starid not in (29281992,336732616) and (camera != 1 OR ccd != 1);")
		conn.commit()

		# Clear other things:
		print("Cleaning up other stupid stuff...")
		cursor.execute("UPDATE todolist SET corr_status=NULL;")
		cursor.execute("DELETE FROM diagnostics WHERE diagnostics.priority NOT IN (SELECT todolist.priority FROM todolist);")
		cursor.execute("DELETE FROM datavalidation_raw WHERE datavalidation_raw.priority NOT IN (SELECT todolist.priority FROM todolist);")
		conn.commit()

		# Create indicies
		print("Making sure indicies are there...")
		cursor.execute("CREATE INDEX IF NOT EXISTS datavalidation_raw_approved_idx ON datavalidation_raw (approved);")
		conn.commit()

		# Optimize tables
		print("Optimizing tables...")
		try:
			conn.isolation_level = None
			cursor.execute("VACUUM;")
			cursor.execute("ANALYZE;")
			cursor.execute("VACUUM;")
			conn.commit()
		except:
			raise
		finally:
			conn.isolation_level = ''

		# Crate the ZIP file and add all the files:
		# We do allow for ZIP64 extensions for large files - lets see if anyone complains
		print("Packaging up lightcurves...")
		with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_STORED, True) as myzip:

			cursor.execute("SELECT todolist.priority,lightcurve FROM todolist INNER JOIN diagnostics ON diagnostics.priority=todolist.priority INNER JOIN datavalidation_raw ON todolist.priority=datavalidation_raw.priority WHERE status IN (1,3) AND datavalidation_raw.approved=1;")
			for row in tqdm(cursor.fetchall()):

				filepath = os.path.join(input_dir, row['lightcurve'])
				if not os.path.exists(filepath):
					raise FileNotFoundError("File not found: '" + filepath + "'")

				if row['lightcurve'] in already_packaged:
					continue

				# Add the file to the ZIP archive:
				myzip.write(filepath, row['lightcurve'], zipfile.ZIP_STORED)

		cursor.close()
