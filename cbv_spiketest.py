#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:12:05 2019

@author: mikkelnl
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import interp1d
import astropy.units as u
from scipy.interpolate import pchip_interpolate

from corrections.cbv_corrector.cbv_main import CBV
import corrections

cbv_area = 334
#data_folder= '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TASOC_code/corrections/corrections/data/cbv_old'
data_folder= '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TASOC_code/corrections/corrections/data/cbv'

filepath = os.path.join(data_folder, 'cbv-%d.npy' % cbv_area)
cbv = np.load(filepath)



fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

fig2 = plt.figure()
ax2_1 = fig2.add_subplot(111)

wmir = 50

cbv_new = np.zeros_like(cbv)
cbv_spike = np.zeros_like(cbv)

for j in range(cbv.shape[1]):
	
	data0 = cbv[:,j]
	data0 = np.append(np.flip(data0[0:wmir])[0:-1], data0)
	data0 = np.append(data0, np.flip(data0[-wmir::])[1::])
	
	xs = np.arange(0, len(data0))
	
	ax2_1.plot(xs[wmir-1:-wmir+1], data0[wmir-1:-wmir+1])
	
	data = data0.copy()
#	data3 = data0.copy()
	
	
	for i in range(5):
	
		
		data2 = pchip_interpolate(xs[np.isfinite(data)], data[np.isfinite(data)], xs)
#		fint = interp1d(xs[np.isfinite(data)], data[np.isfinite(data)])#, kind='linear')
#		data2 = fint(xs)
		
		# run low pass filter
		w = 31 - 2*i
		if w%2==0:
			w+=1
		y = savgol_filter(data2, w, 2, mode='constant')
#		y = medfilt(data2, 15)
		y2 = data2 - y
		
		sigma = 1.4826 * np.nanmedian(np.abs(y2))
#		peaks = signal.find_peaks_cwt(np.abs(y2), np.arange(1,10), min_snr=100*sigma, noise_perc=10)
		peaks, properties = signal.find_peaks(np.abs(y2), prominence=(3*sigma, None), wlen=500)
#		peaks = peaks[np.abs(y2)[peaks]>3*sigma]
		
		data[peaks] = np.nan
#		data3[peaks] = 0


#	fint = interp1d(xs[np.isfinite(data)], data[np.isfinite(data)])#, kind='linear')
#	data = fint(xs)
	data = pchip_interpolate(xs[np.isfinite(data)], data[np.isfinite(data)], xs)

	#plt.plot(xs, data2, 'r', zorder=-10)
	ax2_1.plot(xs[wmir-1:-wmir+1], data[wmir-1:-wmir+1])
	ax2_1.plot(xs[wmir-1:-wmir+1], y[wmir-1:-wmir+1], 'k')
	
	S = (data0[wmir-1:-wmir+1] - data[wmir-1:-wmir+1])
	S[np.isnan(S)] = 0
	
	ax1.plot(xs[wmir-1:-wmir+1], data0[wmir-1:-wmir+1]+j*0.1)
	ax2.plot(xs[wmir-1:-wmir+1], S+j*0.1)
	ax3.plot(xs[wmir-1:-wmir+1], data[wmir-1:-wmir+1]+j*0.1)
	
	
	cbv_spike[:,j] = S
	cbv_new[:,j] = data[wmir-1:-wmir+1]
	






data_folder = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TASOC_code/corrections/corrections/data/spike_test'
input_folder = '/media/mikkelnl/Elements/TESS/S01_tests/lightcurves-combined/'
np.save(os.path.join(data_folder, 'cbv-%d.npy' % cbv_area), cbv_new)
np.save(os.path.join(data_folder, 'cbv-s-%d.npy' % cbv_area), cbv_spike)


starid = 370250245 #Nice spike example
#starid = 370250142
#starid = 370327326
#starid = 370327507 #LPV
#starid = 370328324 #Coherent oscc
#starid = 370328523
#starid = 370328558 #LPV
#starid = 370327584
#starid = 370328257
#starid = 370327409


#starid = 370327074
#starid = 370327058
#starid = 370250607
#starid = 370327159

cbv = CBV(data_folder, cbv_area, threshold_snrtest=5)

with corrections.TaskManager(input_folder) as tm:
	task = tm.get_task(starid=starid, datasource='ffi')
	
with corrections.BaseCorrector(input_folder) as B:	
	lc = B.load_lightcurve(task)


fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)
ax1.plot(lc.time, lc.flux)

n_components = 5
flux_filter, res = cbv.cotrend_single(lc, n_components, data_folder, ini=True)
lc_corrected = (lc.copy()/flux_filter-1)#*1e6

ax2.plot(lc.time, lc.flux)	
ax2.plot(lc.time, flux_filter)	


#time_cut = (lc.time<1346) | (lc.time>1350)
#lc_corrected = lc_corrected[time_cut]
lc_corrected= lc_corrected.remove_nans()


ax3.plot(lc.time, (lc.flux/np.nanmedian(lc.flux) - 1)*1e6)
ax3.plot(lc_corrected.time, lc_corrected.flux*1e6)
ax4.plot(lc_corrected.time, lc_corrected.flux*1e6)


lc = (lc/np.nanmedian(lc.flux) -1)
lc2 = lc.copy().remove_nans()
#

p = lc_corrected.to_periodogram(freq_unit=u.microHertz, max_frequency=282, min_frequency=0.1)
p2 = lc2.remove_nans().to_periodogram(freq_unit=u.microHertz, max_frequency=282, min_frequency=0.1, oversample_factor=1)
figp = plt.figure(figsize=(15,5))
ax1p = figp.add_subplot(211)
ax2p = figp.add_subplot(212)

p2.plot(ax=ax1p, c='r', scale='log')
p2.plot(ax=ax2p, c='r')
p.plot(ax=ax1p, c='k', scale='log')
p.plot(ax=ax2p, c='k')

#fig.savefig()
plt.show()