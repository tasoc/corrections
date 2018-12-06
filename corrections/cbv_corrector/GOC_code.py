#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:42:49 2018

@author: mikkelnl
"""

def GOC_corrmatrix(sector, cbv_area, cursor, exponent=3):
	
	#---------------------------------------------------------------------------------------------------------
	# CALCULATE CORRECTED LIGHT CURVE CORRELATIONS
	#---------------------------------------------------------------------------------------------------------
	logger.info("We are running CBV_AREA=%d" % cbv_area)

	file_median_correlations = 'correlations-median_postcorr-sector%02d-%d.json' % (sector, cbv_area)
	# Find the median absolute correlation between each lightcurve and all other lightcurves:
	
	if os.path.exists(file_median_correlations):
		with open(file_median_correlations) as infile:
			C = json.load(infile)
	else:	

		# Query for all stars, no matter what variability and so on
		cursor.execute("""SELECT todolist.starid,todolist.priority, eclon, eclat FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE
			datasource='ffi'
			AND cbv_area=?
			AND status=1;""", (cbv_area, ))
		
		stars = cursor.fetchall()
	
		# Number of stars returned:
		Nstars = len(stars)
	
		# Load the very first timeseries only to find the number of timestamps.
		# TODO: When using FITS files in the future, this could simply be loaded from the header.
		time = np.loadtxt(os.path.join('sysnoise', 'Star%d.sysnoise' % (stars[0]['starid'],)), usecols=(0, ), unpack=True)
		Ntimes = len(time)
	
		logger.info("Matrix size: %d x %d" % (Nstars, Ntimes))
	
		# Make the matrix that will hold all the lightcurves:
		logger.info("Loading in lightcurves...")
		mat = np.empty((Nstars, Ntimes), dtype='float64')
		mat.fill(np.nan)
		stds = np.empty(Nstars, dtype='float64')
		
		for k, star in tqdm(enumerate(stars), total=Nstars):
			starid = star['starid']
	
			flux = np.loadtxt(os.path.join('cbv_corrected', 'sector%02d', 'area%d', 'Star%d.corr' % (sector, cbv_area, starid,)), usecols=(1, ), unpack=True)
	
			# Normalize the data and store it in the rows of the matrix:
			mat[k, :] = flux
			stds[k] = np.nanstd(flux) #np.sqrt(star['variance'])
	
		# Only start calculating correlations if we are actually filtering using them:
		file_correlations = 'correlations-postcorr-sector%02d-%d.npy' % (sector, cbv_area)
		if os.path.exists(file_correlations):
			correlations = np.load(file_correlations)
		else:
			# Calculate the correlation matrix between all lightcurves:
			correlations = lc_matrix_calc(Nstars, mat, stds)
			np.save(file_correlations, correlations)
			
			
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.matshow(correlations)
			fig.savefig('correlations-postcorr-sector%02d-%d.png' % (sector, cbv_area))
			plt.close(fig)
	
	
	
		C = {}
		C['Nstars'] = Nstars
		C['Exponent'] = exponent
		expected_mean = 1/np.sqrt(Nstars)
		for k, star in enumerate(stars):
			starid = star['starid']
			C[starid] = nanmean(correlations[k,:]**exponent) - expected_mean
		
		with open(file_median_correlations, 'w') as outfile:
			json.dump(C, outfile)

	return C
	
	
def GOC_wn(ori_flux, corrected_flux):
	# Added white noise
	do = MAD_model(np.diff(ori_flux-np.nanmedian(ori_flux)))
	dc = MAD_model(np.diff(corrected_flux))
	wn_ratio = dc/do
	
	return wn_ratio
	
	
def GOC_corr(filepath_todo):

	# Open the TODO file for that sector:
	conn = sqlite3.connect(filepath_todo)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	# Get list of CBV areas:
	cursor.execute("SELECT DISTINCT cbv_area FROM todolist ORDER BY cbv_area;")
	cbv_areas = [int(row[0]) for row in cursor.fetchall()]
	print(cbv_areas)

	# Loop through the CBV areas:
	# - or run them in parallel - whatever you like!
	for ii, cbv_area in enumerate(cbv_areas):
		GOC_corrmatrix(sector, cbv_area, cursor, exponent=3)
			