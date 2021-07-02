#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import glob
import h5py
import sys
if sys.path[0] != os.path.abspath(os.path.join(os.path.dirname(__file__), '..')):
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from corrections import CBV

#--------------------------------------------------------------------------------------------------
def fix_cbs(data_folder, datarel, version=5):

	for fname in glob.glob(os.path.join(data_folder, 'cbv-*.hdf5')):
		print(fname)

		with h5py.File(fname, 'r+') as hdf:

			cbv_area = int(hdf.attrs['cbv_area'])
			cadence = int(hdf.attrs['cadence'])
			if cadence == 1800:
				datasource = 'ffi'
			else:
				datasource = 'tpf'

			hdf.attrs['data_rel'] = datarel
			hdf.attrs['datasource'] = datasource
			hdf.flush()

		cbv = CBV(data_folder, cbv_area, datasource)
		cbv.save_to_fits(data_folder, version=version)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	#fix_cbs(r'C:\Users\au195407\Documents\GitHub\corrections\tests\input\cbv-prepare', datarel=1)

	fix_cbs('/home/rasmush/tasoc/output-fits/TASOC_DR05/S01/cbv-prepare/', datarel=1)
	fix_cbs('/home/rasmush/tasoc/output-fits/TASOC_DR05/S02/cbv-prepare/', datarel=2)
	fix_cbs('/home/rasmush/tasoc/output-fits/TASOC_DR05/S03/cbv-prepare/', datarel=4)
	fix_cbs('/home/rasmush/tasoc/output-fits/TASOC_DR05/S04/cbv-prepare/', datarel=5)
	fix_cbs('/home/rasmush/tasoc/output-fits/TASOC_DR05/S05/cbv-prepare/', datarel=7)
	fix_cbs('/home/rasmush/tasoc/output-fits/TASOC_DR05/S06/cbv-prepare/', datarel=8)

	print("Done.")
