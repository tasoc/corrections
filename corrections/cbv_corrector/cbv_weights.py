#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mikkel N. Lund <mikkelnl@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import sys
import glob
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from bottleneck import allnan, nansum, move_median, nanmedian, nanstd
from scipy.optimize import minimize
from scipy.stats import pearsonr, entropy
from scipy.interpolate import Rbf, SmoothBivariateSpline
import itertools
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from scipy.special import xlogy
import pandas as pd
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="scipy.stats") # they are simply annoying!
from scipy.spatial import distance
from tqdm import tqdm
import math
from cbv_util import compute_entopy, _move_median_central_1d, move_median_central, compute_scores, rms, MAD_model
plt.ioff()
import corner
import dill
from scipy import integrate

# =============================================================================
# 
# =============================================================================
def reduce_std(x):
	return np.median(np.abs(x-np.median(x)))
	
def reduce_mode(x):
	kde = KDE(x)
	kde.fit(gridsize=2000)
	
	pdf = kde.density
	x = kde.support
	return x[np.argmax(pdf)]

def ndim_med_filt(v, x, n, dist='euclidean', mad_frac=2):
	
	d = distance.cdist(x, x, dist)
	
	
	idx = np.zeros_like(v, dtype=bool)
	for i in range(v.shape[0]):
		idx_sort = np.argsort(d[i,:])
#		xx = x[idx_sort, :][1:n+1, :]
		vv= v[idx_sort][1:n+1] # sort values according to distance from point
		
		vm = np.median(vv) # median value of n nearest points
		mad = MAD_model(vv-vm)
		
#		if i==10:
#			plt.figure()
#			plt.scatter(xx[:,0], xx[:,1])
#			plt.scatter(x[i,0], x[i,1], color='r')
#			
#			plt.figure()
#			plt.scatter(xx[:,0],vv)
#			plt.scatter(x[i,0], v[i], color='r')
#			plt.axhline(y=vm)
#			plt.axhline(y=vm+3*mad)
#			plt.axhline(y=vm-3*mad)
#			plt.axhline(y=vm+2*mad)
#			plt.axhline(y=vm-2*mad)			
#			plt.axhline(y=vm+mad)
#			plt.axhline(y=vm-mad)
#			plt.show()
#			sys.exit()
			
		if (v[i]<vm+mad_frac*mad) & (v[i]>vm-mad_frac*mad):
			idx[i] = True
	return idx		

def compute_weight_interpolations(filepath_todo, sector, dimensions=['row', 'col', 'tmag'], do_plot=True):
	
	# Open the TODO file for that sector:
	conn = sqlite3.connect(filepath_todo)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()
	
	# Get list of CBV areas:
	cursor.execute("SELECT DISTINCT cbv_area FROM todolist ORDER BY cbv_area;")
	cbv_areas = [int(row[0]) for row in cursor.fetchall()]
	print(cbv_areas)
		
	
	#Just to know number of computed cbvs
	results0 = np.load('mat-sector%02d-%d_free_weights.npz' % (sector, cbv_areas[0]))['res']
	n_cbvs = results0.shape[1]-2 #results also include star name and offset
	
	figures1 = [];
	figures2 = [];
	for i in range(n_cbvs):
		fig, ax = plt.subplots(2,2, num='cbv%i' %i, figsize=(15,6), )
		figures1.append(fig)
		figures2.append(ax)

		
	colormap = plt.cm.PuOr #or any other colormap
	min_max_vals = np.zeros([n_cbvs, 4])
		
	#TODO: obtain from sector information
	midx = 40.54 # Something wrong! field is 27 deg wide, not 24
	midy = 18
	
	
	pos_mag={}
	
	# Loop through the CBV areas:
	# - or run them in parallel - whatever you like!
	for ii, cbv_area in enumerate(cbv_areas):	
		
		print('Computing weights for area%d' %cbv_area)
		results = np.load('mat-sector%02d-%d_free_weights.npz' % (sector, cbv_area))['res']
		n_stars = results.shape[0]
		n_cbvs = results.shape[1]-2 #results also include star name and offset
		pos_mag0 = np.zeros([n_stars, 7])
		
		pos_mag[cbv_area] = {}
		
		for jj, star in enumerate(results[:,0]):
			cursor.execute("""SELECT todolist.starid,todolist.priority,todolist.tmag,ccd,eclon,eclat,pos_row,pos_column,mean_flux,variance FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE
				datasource='ffi'
				AND status=1
				AND cbv_area=?
				AND todolist.starid=?
			ORDER BY variability ASC LIMIT 30000;""", (cbv_area, int(star)))		
			star_single = cursor.fetchall()
			
			# Fix small glitch /by Rasmus) in assignment of lat/lon
			if cbv_area==122:
				if star_single[0]['eclat']>15:
					pos_mag0[jj, 0] = 50
					pos_mag0[jj, 1] = 10
					pos_mag0[jj, 2] = 10
					continue
				
				
			pos_mag[cbv_area]['eclon'] = np.array([])
				
			pos_mag0[jj, 0] = star_single[0]['eclon']
			pos_mag0[jj, 1] = star_single[0]['eclat']
			pos_mag0[jj, 2] = star_single[0]['pos_row']+(star_single[0]['ccd']>2)*2048
			pos_mag0[jj, 3] = star_single[0]['pos_column']+(star_single[0]['ccd']%2==0)*2048
			
			pos_mag0[jj, 4] = star_single[0]['tmag']
			
			# Convert to polar coordinates
			angle = math.atan2(star_single[0]['eclat']-midy, star_single[0]['eclon']-midx)
			angle = angle * 360 / (2*np.pi)
			if (angle < 0):
				angle += 360
			pos_mag0[jj, 5] = np.sqrt((star_single[0]['eclon']-midx)**2 + (star_single[0]['eclat']-midy)**2)
			pos_mag0[jj, 6] = angle
			
		
		pos_mag[cbv_area]['eclon'] = pos_mag0[:, 0]
		pos_mag[cbv_area]['eclat'] = pos_mag0[:, 1]
		pos_mag[cbv_area]['row'] = pos_mag0[:, 2]
		pos_mag[cbv_area]['col'] = pos_mag0[:, 3]
		pos_mag[cbv_area]['tmag'] = pos_mag0[:, 4]
		pos_mag[cbv_area]['rad'] = pos_mag0[:, 4]
		pos_mag[cbv_area]['theta'] = pos_mag0[:, 4]
		pos_mag[cbv_area]['results'] = results
		
#		results=np.column_stack((results, pos_mag))
		
		
		for j in range(n_cbvs):
			
			VALS = results[:,1+j]
			# Perform binning
			
			axm = figures2[j][0,0]
			axs = figures2[j][0,1]
			
			axm2 = figures2[j][1,0]
			axs2 = figures2[j][1,1]
			
			
			if np.percentile(VALS,10)<min_max_vals[j,0]:
				min_max_vals[j,0] = np.percentile(VALS,10)
			if np.percentile(VALS,90)>min_max_vals[j,1]:
				min_max_vals[j,1] = np.percentile(VALS,90)
			
			normalize = colors.Normalize(vmin=min_max_vals[j,0], vmax=min_max_vals[j,1])	

			
			# CBV values
			hbm = axm.hexbin(pos_mag[cbv_area][dimensions[0]], pos_mag[cbv_area][dimensions[1]], C=VALS, gridsize=10, reduce_C_function=reduce_mode, cmap=colormap, norm=normalize)
			# CBV values scatter
			hbs = axs.hexbin(pos_mag[cbv_area][dimensions[0]], pos_mag[cbv_area][dimensions[1]], C=VALS, gridsize=10, reduce_C_function=reduce_std, cmap=colormap, norm=normalize)
		
			# Get values and vertices of hexbinning
			zvalsm = hbm.get_array();		vertsm = hbm.get_offsets()
			zvalss = hbs.get_array();		vertss = hbs.get_offsets()
		
			# Bins to keed for interpolation
			idxm = ndim_med_filt(zvalsm, vertsm, 6)
			idxs = ndim_med_filt(zvalss, vertss, 6)
			
			# Plot removed bins
			axm.plot(vertsm[~idxm,0], vertsm[~idxm,1], marker='.', ms=1, ls='', color='r')
			axs.plot(vertss[~idxs,0], vertss[~idxs,1], marker='.', ms=1, ls='', color='r')
		
			# Trim binned values before interpolation
			zvalsm, vertsm = zvalsm[idxm], vertsm[idxm] 
			zvalss, vertss = zvalss[idxs], vertss[idxs] 
		
			rbfim = Rbf(vertsm[:,0], vertsm[:,1], zvalsm, smooth=1)
			rbfis = Rbf(vertss[:,0], vertss[:,1], zvalss, smooth=1)
			
			with open('Rbf_area%d_cbv%i.pkl' %(cbv_area,int(j+1)), 'wb') as file:
				dill.dump(rbfim, file)
			with open('Rbf_area%d_cbv%i_std.pkl' %(cbv_area,int(j+1)), 'wb') as file:
				dill.dump(rbfis, file)	
				
				
			# Plot resulting interpolation
			x1 = np.linspace(vertsm[:,0].min(), vertsm[:,0].max(), 100); y1 = np.linspace(vertsm[:,1].min(), vertsm[:,1].max(), 100); xv1, yv1 = np.meshgrid(x1, y1)
			x2 = np.linspace(vertss[:,0].min(), vertss[:,0].max(), 100); y2 = np.linspace(vertss[:,1].min(), vertss[:,1].max(), 100); xv2, yv2 = np.meshgrid(x2, y2)
		
		
			rm = np.abs(rbfim(xv1, yv1))
			rs = np.abs(rbfis(xv2, yv2))
			
			if np.percentile(rm,10)<min_max_vals[j,2]:
				min_max_vals[j,2] = np.percentile(rm,10)
			if np.percentile(rm,90)>min_max_vals[j,3]:
				min_max_vals[j,3] = np.percentile(VALS,90)
				
			normalize = colors.Normalize(vmin=min_max_vals[j,2], vmax=min_max_vals[j,3])
			axm2.contourf(xv1, yv1, rm, cmap=colormap, norm=normalize)
			axs2.contourf(xv2, yv2, rs, cmap=colormap, norm=normalize)
			
	for k, figs in enumerate(figures1):
		figs.savefig('cbv%i' %k)

				
				
	
#------------------------------------------------------------------------------
	
# TODO: include removal of simple trend in nd median filtering
# TODO: Do nd median filtering before binning
# TODO: Try binned_stats over hexbin (scipy.stats.binned_statistic_dd)
# TODO: Inclusion of tmag dimension - normalise "distance" in 3D space
# TODO: Better/Other CBV areas
# TODO: Tests with polar coordinates in combination with other CBV areas	
# TODO: mode vs. median in binning
# TODO: use klepto to cashe interpolation function (https://stackoverflow.com/questions/23997431/is-there-a-way-to-pickle-a-scipy-interpolate-rbf-object)


if __name__ == '__main__':
	# Pick a sector, any sector....
	sector = 0
	n_components = 8
	
	# Open the TODO file for that sector:
	filepath_todo = 'todo.sqlite'
	
	
	compute_weight_interpolations(filepath_todo, sector)
	
	plt.show()
#	conn = sqlite3.connect(filepath_todo)
#	conn.row_factory = sqlite3.Row
#	cursor = conn.cursor()
#	
#	# Get list of CBV areas:
#	cursor.execute("SELECT DISTINCT cbv_area FROM todolist ORDER BY cbv_area;")
#	cbv_areas = [int(row[0]) for row in cursor.fetchall()]
#	print(cbv_areas)
#	
#	colormap = plt.cm.PuOr #or any other colormap
#	normalize1 = colors.Normalize(vmin=-0.45, vmax=0.45)
#	normalize2 = colors.Normalize(vmin=-0.15, vmax=0.15)
#	normalize3 = colors.Normalize(vmin=-0.05, vmax=0.05)
#
#
#	fig = plt.figure(figsize=(15,6))
#	ax1 = fig.add_subplot(231)
#	ax2 = fig.add_subplot(232)
#	ax3 = fig.add_subplot(233)
#	
#	ax4 = fig.add_subplot(234)
#	ax5 = fig.add_subplot(235)
#	ax6 = fig.add_subplot(236)
#	
#	fig2 = plt.figure(figsize=(15,6))
#	ax1_2 = fig2.add_subplot(231)
#	ax2_2 = fig2.add_subplot(232)
#	ax3_2 = fig2.add_subplot(233)
#	
#	ax4_2 = fig2.add_subplot(234)
#	ax5_2 = fig2.add_subplot(235)
#	ax6_2 = fig2.add_subplot(236)
#	
#	midx = 40.54 # Something wrong! field is 27 deg wide, not 24
#	midy = 18
#	
##	minlat = 100
##	maxlat = 0
#
#	# Loop through the CBV areas:
#	# - or run them in parallel - whatever you like!
#	for ii, cbv_area in enumerate(cbv_areas):	
#		
#
#		results = np.load('mat-sector%02d-%d_free_weights.npz' % (sector, cbv_area))['res']
#		n_stars = results.shape[0]
#		n_cbvs = results.shape[1]-2 #results also include star name and offset
#		pos_mag = np.zeros([n_stars, 5])
#		
#		for jj, star in enumerate(results[:,0]):
#			cursor.execute("""SELECT todolist.starid,todolist.priority,todolist.tmag,ccd,eclon,eclat,pos_row,pos_column,mean_flux,variance FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE
#				datasource='ffi'
#				AND status=1
#				AND cbv_area=?
#				AND todolist.starid=?
#			ORDER BY variability ASC LIMIT 30000;""", (cbv_area, int(star)))		
#			star_single = cursor.fetchall()
#			
#			# Fix small glitch /by Rasmus) in assignment of lat/lon
#			if cbv_area==122:
#				if star_single[0]['eclat']>15:
#					pos_mag[jj, 0] = 50
#					pos_mag[jj, 1] = 10
#					pos_mag[jj, 2] = 10
#					continue
#				
#				
##			pos_mag[jj, 0] = star_single[0]['eclon']
##			pos_mag[jj, 1] = star_single[0]['eclat']
#			pos_mag[jj, 0] = star_single[0]['pos_row']+(star_single[0]['ccd']>2)*2048
#			pos_mag[jj, 1] = star_single[0]['pos_column']+(star_single[0]['ccd']%2==0)*2048
#			
#			pos_mag[jj, 2] = star_single[0]['tmag']
#			
#			# Convert to polar coordinates
#			angle = math.atan2(star_single[0]['eclat']-midy, star_single[0]['eclon']-midx)
#			angle = angle * 360 / (2*np.pi)
#			if (angle < 0):
#				angle += 360
#			pos_mag[jj, 3] = np.sqrt((star_single[0]['eclon']-midx)**2 + (star_single[0]['eclat']-midy)**2)
#			pos_mag[jj, 4] = angle
#			
#			
##			if pos_mag[jj, 1]<minlat:
##				minlat = pos_mag[jj, 1]
##			if pos_mag[jj, 1]>maxlat:
##				maxlat = pos_mag[jj, 1]	
#				
#		results=np.column_stack((results, pos_mag))
#		print(results.shape, np.min(results[:,1]), np.max(results[:,1]), np.median(np.abs(results[:,1])), np.median(np.abs(results[:,2])), np.median(np.abs(results[:,3])))
#		
#		ELON = results[:,-5]
#		ELAT = results[:,-4]
#		
#		
##		for j in range(n_cbvs):
##			
##			VALS = results[:,1+j]
##			# Perform binning
##			
##			# CBV values
##			hbm = plt.hexbin(ELON, ELAT, C=VALS, gridsize=10, reduce_C_function=reduce_mode, cmap=colormap, norm=normalize1, marginals=False)
##			# CBV values scatter
##			hbs = plt.hexbin(ELON, ELAT, C=VALS, gridsize=10, reduce_C_function=reduce_std, cmap=colormap, norm=normalize1, marginals=False)
##		
##			# Get values and vertices of hexbinning
##			zvalsm = hbm.get_array();		vertsm = hbm.get_offsets()
##			zvalss = hbs.get_array();		vertss = hbs.get_offsets()
##		
##			# Bins to keed for interpolation
##			idxm = ndim_med_filt(zvalsm, vertsm, 6)
##			idxs = ndim_med_filt(zvalss, vertss, 6)
##		
##			# Trim binned values before interpolation
##			zvalsm, vertsm = zvalsm[idxm], vertsm[idxm] 
##			zvalss, vertss = zvalss[idxs], vertss[idxs] 
##		
##			rbfim = Rbf(vertsm[:,0], vertsm[:,1], zvalsm, smooth=1)
##			rbfis = Rbf(vertss[:,0], vertss[:,1], zvalss, smooth=1)
##			
##			with open('Rbf_area%d_cbv%i.pkl' %(cbv_area,int(j+1)), 'wb') as file:
##				dill.dump(rbfim, file)
##			with open('Rbf_area%d_cbv%i_std.pkl' %(cbv_area,int(j+1)), 'wb') as file:
##				dill.dump(rbfis, file)	
#				
#				
#				
#		VALS1 = results[:,1]
#		VALS2 = results[:,2]
#		VALS3 = results[:,3]
#		
#		
##		idx = (VALS1>np.percentile(VALS1, 20)) & (VALS1<np.percentile(VALS1, 80))
##		corner.corner(np.column_stack((VALS1[idx], VALS2[idx], VALS3[idx])))
##		plt.show()
##		sys.exit()
#		
#		# Perform binning
#		
#		# CBV values
#		hb1 = ax1.hexbin(ELON, ELAT, C=VALS1, gridsize=10, reduce_C_function=reduce_mode, cmap=colormap, norm=normalize1, marginals=False)
#		hb2 = ax2.hexbin(ELON, ELAT, C=VALS2, gridsize=10, reduce_C_function=reduce_mode, cmap=colormap, norm=normalize2, marginals=False)
#		hb3 = ax3.hexbin(ELON, ELAT, C=VALS3, gridsize=10, reduce_C_function=reduce_mode, cmap=colormap, norm=normalize3, marginals=False)
#		
#		# CBV values scatter
#		hb4 = ax4.hexbin(ELON, ELAT, C=VALS1, gridsize=10, reduce_C_function=reduce_std, cmap=colormap, norm=normalize1, marginals=False)
#		hb5 = ax5.hexbin(ELON, ELAT, C=VALS2, gridsize=10, reduce_C_function=reduce_std, cmap=colormap, norm=normalize2, marginals=False)
#		hb6 = ax6.hexbin(ELON, ELAT, C=VALS3, gridsize=10, reduce_C_function=reduce_std, cmap=colormap, norm=normalize3, marginals=False)
#
#
#		
##		scipy.stats.binned_statistic_dd
#		
#		
#		# Get values and vertices of hexbinning
#		zvals1 = hb1.get_array();		verts1 = hb1.get_offsets()
#		zvals2 = hb2.get_array();		verts2 = hb2.get_offsets()
#		zvals3 = hb3.get_array();		verts3 = hb3.get_offsets()	
#		zvals4 = hb4.get_array();		verts4 = hb4.get_offsets()	
#		zvals5 = hb5.get_array();		verts5 = hb5.get_offsets()	
#		zvals6 = hb6.get_array();		verts6 = hb6.get_offsets()	
#		
#		# Bins to keed for interpolation
#		idx1 = ndim_med_filt(zvals1, verts1, 6)
#		idx2 = ndim_med_filt(zvals2, verts2, 6)
#		idx3 = ndim_med_filt(zvals3, verts3, 6)
#		idx4 = ndim_med_filt(zvals4, verts4, 6)
#		idx5 = ndim_med_filt(zvals5, verts5, 6)
#		idx6 = ndim_med_filt(zvals6, verts6, 6)
#		
#		# Plot removed bins
#		ax1.plot(verts1[~idx1,0], verts1[~idx1,1], marker='.', ms=1, ls='', color='r')
#		ax2.plot(verts2[~idx2,0], verts2[~idx2,1], marker='.', ms=1, ls='', color='r')
#		ax3.plot(verts3[~idx3,0], verts3[~idx3,1], marker='.', ms=1, ls='', color='r')
#		ax4.plot(verts4[~idx4,0], verts4[~idx4,1], marker='.', ms=1, ls='', color='r')
#		ax5.plot(verts5[~idx5,0], verts5[~idx5,1], marker='.', ms=1, ls='', color='r')
#		ax6.plot(verts6[~idx6,0], verts6[~idx6,1], marker='.', ms=1, ls='', color='r')
#		
#		# Trim binned values before interpolation
#		zvals1, verts1 = zvals1[idx1], verts1[idx1] 
#		zvals2, verts2 = zvals2[idx2], verts2[idx2] 
#		zvals3, verts3 = zvals3[idx3], verts3[idx3] 
#		zvals4, verts4 = zvals4[idx4], verts4[idx4] 
#		zvals5, verts5 = zvals5[idx5], verts5[idx5] 
#		zvals6, verts6 = zvals6[idx6], verts6[idx6] 
#		
#
##		# Interpolate
##		rbfi1 = Rbf(verts1[:,0], verts1[:,1], zvals1, smooth=1, function='linear')
##		rbfi2 = Rbf(verts2[:,0], verts2[:,1], zvals2, smooth=1, function='linear')
##		rbfi3 = Rbf(verts3[:,0], verts3[:,1], zvals3, smooth=1, function='linear')
#
#		rbfi1 = Rbf(verts1[:,0], verts1[:,1], zvals1, smooth=1)
#		rbfi2 = Rbf(verts2[:,0], verts2[:,1], zvals2, smooth=1)
#		rbfi3 = Rbf(verts3[:,0], verts3[:,1], zvals3, smooth=1)
#		rbfi4 = Rbf(verts4[:,0], verts4[:,1], zvals4, smooth=1)
#		rbfi5 = Rbf(verts5[:,0], verts5[:,1], zvals5, smooth=1)
#		rbfi6 = Rbf(verts6[:,0], verts6[:,1], zvals6, smooth=1)
#			
##		rbfi1 = SmoothBivariateSpline(verts1[:,0], verts1[:,1], zvals1)
##		rbfi2 = SmoothBivariateSpline(verts2[:,0], verts2[:,1], zvals2)
##		rbfi3 = SmoothBivariateSpline(verts3[:,0], verts3[:,1], zvals3)
#		
##		with open('Rbf_area%d_cbv1.pkl' %cbv_area, 'wb') as file:
##			dill.dump(rbfi1, file)
##		with open('Rbf_area%d_cbv1_std.pkl' %cbv_area, 'wb') as file:
##			dill.dump(rbfi4, file)	
##			
##		with open('Rbf_area%d_cbv2.pkl' %cbv_area, 'wb') as file:
##			dill.dump(rbfi2, file)
##		with open('Rbf_area%d_cbv2_std.pkl' %cbv_area, 'wb') as file:
##			dill.dump(rbfi5, file)	
##			
##		with open('Rbf_area%d_cbv3.pkl' %cbv_area, 'wb') as file:
##			dill.dump(rbfi3, file)
##		with open('Rbf_area%d_cbv3_std.pkl' %cbv_area, 'wb') as file:
##			dill.dump(rbfi6, file)		
##
##
##
##
##
##		with open('Rbf_area%d_cbv1.pkl' %cbv_area, 'rb') as file:
##			B = dill.load(file)
#		
#		
##		I=integrate.nquad(B, [[verts1[:,0].min(), verts1[:,0].max()], [verts1[:,1].min(), verts1[:,1].max()]], full_output=True) #opts=[opts0,{},{},{}], 
##		print('I:', I)
#		
#		# Plot resulting interpolation
#		x1 = np.linspace(verts1[:,0].min(), verts1[:,0].max(), 100); y1 = np.linspace(verts1[:,1].min(), verts1[:,1].max(), 100); xv1, yv1 = np.meshgrid(x1, y1)
#		x2 = np.linspace(verts2[:,0].min(), verts2[:,0].max(), 100); y2 = np.linspace(verts2[:,1].min(), verts2[:,1].max(), 100); xv2, yv2 = np.meshgrid(x2, y2)
#		x3 = np.linspace(verts3[:,0].min(), verts3[:,0].max(), 100); y3 = np.linspace(verts3[:,1].min(), verts3[:,1].max(), 100); xv3, yv3 = np.meshgrid(x3, y3)
#		x4 = np.linspace(verts4[:,0].min(), verts4[:,0].max(), 100); y4 = np.linspace(verts4[:,1].min(), verts4[:,1].max(), 100); xv4, yv4 = np.meshgrid(x4, y4)
#		x5 = np.linspace(verts5[:,0].min(), verts5[:,0].max(), 100); y5 = np.linspace(verts5[:,1].min(), verts5[:,1].max(), 100); xv5, yv5 = np.meshgrid(x5, y5)
#		x6 = np.linspace(verts6[:,0].min(), verts6[:,0].max(), 100); y6 = np.linspace(verts6[:,1].min(), verts6[:,1].max(), 100); xv6, yv6 = np.meshgrid(x6, y6)
#		
##		xv1, yv1 = np.meshgrid(np.sort(list(set(verts1[:,0]))), np.sort(list(set(verts1[:,1]))))
##		from scipy.interpolate import griddata
##		grid_z0 = griddata(verts1, zvals1, (xv1, yv1), method='nearest')
#
##		ax1_2.contourf(xv1, yv1, B(xv1, yv1), cmap=colormap, norm=normalize1)
#		ax1_2.contourf(xv1, yv1, rbfi1(xv1, yv1), cmap=colormap, norm=normalize1)
#		ax2_2.contourf(xv2, yv2, rbfi2(xv2, yv2), cmap=colormap, norm=normalize2)
#		ax3_2.contourf(xv3, yv3, rbfi3(xv3, yv3), cmap=colormap, norm=normalize3)
#		ax4_2.contourf(xv4, yv4, rbfi4(xv4, yv4), cmap=colormap, norm=normalize1)
#		ax5_2.contourf(xv5, yv5, rbfi5(xv5, yv5), cmap=colormap, norm=normalize2)
#		ax6_2.contourf(xv6, yv6, rbfi6(xv6, yv6), cmap=colormap, norm=normalize3)
#
##		ax2.hexbin(results[:,-2], results[:,-1], C=np.abs(results[:,2]), gridsize=10, reduce_C_function=np.median, cmap=colormap, norm=normalize2, marginals=False)
##		ax3.hexbin(results[:,-2], results[:,-1], C=np.abs(results[:,3]), gridsize=10, reduce_C_function=np.median, cmap=colormap, norm=normalize3, marginals=False)
#		
##		break
#
#
##	print(minlat, maxlat)
#	plt.show()
















