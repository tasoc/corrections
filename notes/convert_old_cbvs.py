#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert old CBV NPY files to HDF5.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import h5py
import os.path

if __name__ == '__main__':

	cbv_area = 114
	folder = r'E:\TASOC_DR05\S01\cbv-prepare\old'

	hdf5file = os.path.join(folder, 'cbv-ffi-%d.hdf5' % cbv_area)

	cbv_ini = np.load(os.path.join(folder, 'cbv_ini-ffi-%d.npy' % cbv_area))
	cbv = np.load(os.path.join(folder, 'cbv-ffi-%d.npy' % cbv_area))
	cbv_spike = np.load(os.path.join(folder, 'cbv-s-ffi-%d.npy' % cbv_area))

	with h5py.File(hdf5file, 'w', libver='latest') as hdf:
		# Save all settings in the attributes of the root of the HDF5 file:
		hdf.attrs['cbv_area'] = cbv_area
		hdf.attrs['camera'] = int(str(cbv_area)[0])
		hdf.attrs['ccd'] = int(str(cbv_area)[1])
		hdf.attrs['sector'] = 1
		hdf.attrs['cadence'] = 1800
		hdf.attrs['version'] = 'pre'
		hdf.attrs['Ncbvs'] = 16
		hdf.attrs['threshold_variability'] = 1.3
		hdf.attrs['threshold_correlation'] = 0.5
		hdf.attrs['threshold_snrtest'] = 5.0
		hdf.attrs['threshold_entropy'] = -0.5
		#hdf.attrs['Nstars'] = ?
		hdf.attrs['Ntimes'] = cbv_ini.shape[0]

		hdf.create_dataset('cbv-ini', data=cbv_ini)
		hdf.create_dataset('cbv-single-scale', data=cbv)
		hdf.create_dataset('cbv-spike', data=cbv_spike)

		#hdf.create_dataset('time', data=time)
		#hdf.create_dataset('cadenceno', data=cadenceno)
