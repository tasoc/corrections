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


cbv_area = 222
data_folder= '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TASOC_code/corrections/corrections/data/cbv'

filepath = os.path.join(data_folder, 'cbv-%d.npy' % cbv_area)
cbv = np.load(filepath)



fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

fig2 = plt.figure()
ax2_1 = fig2.add_subplot(111)

wmir = 5

for j in range(cbv.shape[1]):
	
	data0 = cbv[:,j]
	data0 = np.append(np.flip(data0[0:wmir])[0:-1], data0)
	data0 = np.append(data0, np.flip(data0[-wmir::])[1::])
	
#	print(len(data0), len(data0[wmir-1:-wmir+1]))
	
	xs = np.arange(0, len(data0))
	
	ax2_1.plot(xs[wmir-1:-wmir+1], data0[wmir-1:-wmir+1])
	
	data = data0.copy()
	data3 = data0.copy()
	
	
	for i in range(5):
	
		fint = interp1d(xs[np.isfinite(data)], data[np.isfinite(data)])#, kind='linear')
		data2 = fint(xs)
		
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
		data3[peaks] = 0



	#plt.plot(xs, data2, 'r', zorder=-10)
	ax2_1.plot(xs[wmir-1:-wmir+1], data[wmir-1:-wmir+1])
	ax2_1.plot(xs[wmir-1:-wmir+1], y[wmir-1:-wmir+1], 'k')
	
	
	ax1.plot(xs[wmir-1:-wmir+1], data0[wmir-1:-wmir+1]+j*0.1)
	ax2.plot(xs[wmir-1:-wmir+1], (data0[wmir-1:-wmir+1] - data2[wmir-1:-wmir+1])+j*0.1)
	ax3.plot(xs[wmir-1:-wmir+1], data[wmir-1:-wmir+1]+j*0.1)
	
	
#	plt.figure()
#	plt.plot(xs, np.abs(y2))
#	
#	
#	plt.scatter(xs[peaks], np.abs(y2)[peaks])

plt.show()