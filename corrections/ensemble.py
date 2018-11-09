#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Initial code structure for the ensemble photometry detrending program.

Global Parameters:
    __data_folder__ (str): Location of the data
    __sector__ (str): Sector to look at

Created on Thu Mar 29 09:58:55 2018

.. codeauthor:: Derek Buzasi
.. codeauthor:: Oliver J. Hall
.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Filipe Pereira
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.interpolate
import scipy.optimize as sciopt
import sqlite3
import sys
from tqdm import tqdm
from copy import deepcopy
import time
import pandas
import lightkurve

import pickle
from pathlib import Path

from . import BaseCorrector, STATUS


class EnsembleCorrector(BaseCorrector):
    """
    DOCSTRING
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the correction object

        Parameters:
            *args: Arguments for the BaseCorrector class
            **kwargs: Keyword Arguments for the BaseCorrector class
        """
        super(self.__class__, self).__init__(*args, **kwargs)

    def do_correction(self, lc):
        """
        Function that takes all input stars for a sector and uses them to find a detrending
        function using ensemble photometry for a star 'star_names[ifile]', where ifile is the
        index for the star in the star_array and star_names list.

        Parameters:
            lc (``lightkurve.TessLightCurve``): Raw lightcurve stored in a TessLightCurve object.

        Returns:
            lc_corr (``lightkurve.TessLightCurve``): Corrected lightcurve stored in a TessLightCurve object.
            The status of the correction.
        """
        # TODO: Remove in final version. Used to test execution time
        fstart_time = time.time()

        # Calculate extra data for the target lightcurve
        lc = lc.remove_nans()
        # NOTE: Is frange supposed to be missing the brackets and not join the percentiles ??
        frange = np.percentile(lc.flux, 95) - np.percentile(lc.flux, 5) / np.mean(lc.flux)
        drange = np.std(np.diff(lc.flux)) / np.mean(lc.flux)
        lc.meta.update({ 'fmean' : np.max(lc.flux),
                        'fstd' : np.std(np.diff(lc.flux)),
                        'frange' : frange,
                        'drange' : drange})

        # StarID, pixel positions and lightcurve filenames are retrieved from the database
        select_params = ["todolist.starid", "pos_row", "pos_column"]
        search_params = [f"camera={lc.camera:d}", f"ccd={lc.ccd:d}", "mean_flux>0"]
        db_raw = self.search_lightcurves(select=select_params, search=search_params)
        starid = np.array([row['starid'] for row in db_raw])
        pixel_coords = np.array([[row['pos_row'], row['pos_column']] for row in db_raw])

        # TODO: We can leave the target star for the distance comparison and get its exact index for free. Just need to be careful with the entry 0 of dist 
        # Determine distance of all stars to target. Array of star indexes by distance to target and array of the distance. Pixel distance used
        idx = starid == lc.targetid
        dist = np.sqrt((pixel_coords[:,0] - pixel_coords[idx,0])**2 + (pixel_coords[:,1] - pixel_coords[idx,1])**2)
        distance_index = dist.argsort()
        target_index = distance_index[0]
        distance = dist[distance_index]

        # Set up start/end times for stellar time series
        time_start = np.amin(lc.time)
        time_end = np.max(lc.time)

        # Set minimum range parameter...this is log10 photometric range, and stars more variable than this will be excluded from the ensemble
        min_range = -2.0
        min_range0 = min_range
        flag = 1

        # Define variables to use in the loop to build the ensemble of stars
        # List of star indexes to be included in the ensemble
        temp_list = []
        # Initial number of closest stars to consider and variable to increase number
        initial_num_stars = 20
        star_count = initial_num_stars
        # (Alternate param) Initial distance at which to consider stars around the target
        # initial_search_radius = -1

        # Follows index of dist array to restart search after the last addded star. Starts at 1 as the first star ordered by distance is the target
        i = 1
        # Setup search and select params to use in loop
        select_loop = ["todolist.starid", "camera", "ccd", "lightcurve"]
        search_loop = [f"camera={lc.camera:d}", f"ccd={lc.ccd:d}", "mean_flux>0"]
        # Start loop to build ensemble
        while True:

            # First get a list of indexes of a specified number of stars to build the ensemble
            while len(temp_list) < star_count:
                
                # Get lightkurve for next star closest to target
                # NOTE: This seems needlesly complicated. Probably can just change load_lightcurve
                next_star_index = distance_index[i]
                search_loop.append(f"todolist.starid={starid[next_star_index]:}")
                next_star_task = self.search_lightcurves(search=search_loop, select=select_loop)[0]
                next_star_lc = self.load_lightcurve(next_star_task).remove_nans()
                search_loop.pop(-1)
                
                # Compute the rest of its data. NOTE: Change this to the database or not ???
                frange = np.percentile(next_star_lc.flux, 95) - np.percentile(next_star_lc.flux, 5) / np.mean(next_star_lc.flux)
                drange = np.std(np.diff(next_star_lc.flux)) / np.mean(next_star_lc.flux)
                next_star_lc.meta.update({ 'fmean' : np.max(next_star_lc.flux),
                                            'fstd' : np.std(np.diff(next_star_lc.flux)),
                                            'frange' : frange,
                                            'drange' : drange})

                # Stars are added to ensemble if they fulfill the requirements
                if (np.log10(next_star_lc.meta['drange']) < min_range and next_star_lc.meta['drange'] < 10*lc.meta['drange']):
                    temp_list.append([next_star_index, next_star_lc.copy()])
                i += 1

            ensemble_list = np.array(temp_list)
            # Now populate the arrays of data with the stars in the ensemble
            full_time = np.concatenate([temp_lc.time for temp_lc in ensemble_list[:,1]]).ravel()
            tflux = np.concatenate(np.array([temp_lc.flux / temp_lc.meta['fmean'] for temp_lc in ensemble_list[:,1]])).ravel()
            full_weight = np.concatenate(np.array([np.full(temp_lc.flux.size, temp_lc.meta['fmean'] / temp_lc.meta['fstd']) for temp_lc in ensemble_list[:,1]])).ravel()
            full_flux = np.multiply(tflux, full_weight)

            # TODO: As of now the code begins by ensuring 20 stars are added to the ensemble and then adds one by one. Might have to change to use a search radius
            # Fetch distance of last added star to ensemble to use as search radius to test conditions ahead
            search_radius = distance[i-1]

            # Set up time array with 0.5-day resolution which spans the time range of the time series then histogram the data based on that array
            gx = np.arange(time_start,time_end,0.5)
            n = np.histogram(full_time, gx)[0]
            n2 = np.histogram(lc.time, gx)[0]

            # TODO: Probably should change the condition that adds stars to the algorithm
            # If the least-populated bin has less than 2000 points, increase the size of the ensemble by first
            # increasing the level of acceptable variability until it exceeds the variability of the star. Once that happens,
            # increase the search radius and reset acceptable variability back to initial value. If the search radius exceeds
            # a limiting value (pi/4 at this point), accept that we can't do any better.
            # if np.min(n[0])<400:
            # print np.min(n[n2>0])
            if np.min(n[n2>0]) < 1000:
                min_range = min_range+0.3
                if min_range > np.log10(np.max(lc.meta['drange'])):
                    if (search_radius < 100):
                        # search_radius += 10
                        star_count += 1
                        search_radius = distance[i]
                    else:
                        # search_radius *= 1.1
                        min_range = min_range0
                        star_count += 1
                        search_radius = distance[i]

                # if search_radius > np.pi/4:
                if search_radius > 400:
                    break
            else:
                    break

        
        # Ensemble is now built. Clean up ensemble points by removing NaNs
        not_nan_idx = ~np.isnan(full_flux) # Since index is same for all arrays save it first to use cached version
        full_time = full_time[not_nan_idx]
        full_weight = full_weight[not_nan_idx]
        full_flux = full_flux[not_nan_idx]
        tflux = tflux[not_nan_idx]

        # Sort ensemble into time order
        time_idx = np.argsort(full_time)
        full_time = full_time[time_idx]
        full_flux = full_flux[time_idx]
        full_weight = full_weight[time_idx]

        # Simplify by discarding ensemble points outside the temporal range of the stellar time series
        idx = (full_time>time_start) & (full_time<time_end)
        full_time = full_time[idx]
        full_flux = full_flux[idx]
        full_weight = full_weight[idx]

        temp_time = full_time
        temp_flux = full_flux
        temp_weight = full_weight

        # TODO: Remove in final version. Used to test execution time
        start_time = time.time()
        # Initialize bin size in days. We will fit the ensemble with splines
        bin_size = 4.0
        for ib in range(6):
            #decrease bin size and bin data
            clip_c = 6 - ib*0.75
            gx = np.arange(time_start-0.5*bin_size,time_end+bin_size,bin_size)
            #bidx  = np.digitize(full_time,gx)
            bidx  = np.digitize(temp_time,gx)
            bidx = bidx-1
            #n, bin_edges = np.histogram(full_time,gx) #bin data
            n, bin_edges = np.histogram(temp_time,gx) #bin data
            #if there are too few points in the least-populated bin after the first couple of iterations, break out
            #and stop decreasing the size of the bins
            #if np.nanmin(n) < 10 and ib > 2:
            #    break
            ttflux = []
            ttweight = []
            ttime = []
            #bin by bin build temporary arrays for weight, time, flux
            for ix in range(len(n)):
                ttweight = np.append(ttweight,np.nanmean(temp_weight[bidx==ix]))
                ttime = np.append(ttime,np.nanmean(temp_time[bidx==ix]))
                ttflux = np.append(ttflux,np.nanmedian(np.divide(temp_flux[bidx==ix],temp_weight[bidx==ix])))
            ottime = ttime #keep track of originals since we will modify the tt arrays
            otflux = ttflux
            #clean up any NaNs
            ttime = np.asarray(ttime)
            ttflux = np.asarray(ttflux)
            w1 = ttime[~np.isnan(ttflux)]
            w2 = ttflux[~np.isnan(ttflux)]

            counter = len(ttime)
            while counter > 0:
                pp = scipy.interpolate.pchip(w1,w2)
                diff1 = np.divide(temp_flux,temp_weight)-pp(temp_time)
                sdiff = clip_c*np.nanstd(diff1)
                counter = len(diff1[np.abs(diff1)>sdiff])
                temp_time = temp_time[np.abs(diff1)<sdiff]
                temp_flux = temp_flux[np.abs(diff1)<sdiff]
                temp_weight = temp_weight[np.abs(diff1)<sdiff]

            pp = scipy.interpolate.pchip(w1,w2)

            break_locs = np.where(np.diff(lc.time)>0.1) #find places where there is a break in time
            break_locs = np.array(break_locs)
            if break_locs.size>0: #set up boundaries to correspond with breaks
                break_locs = np.array(break_locs)+1
                break_locs.astype(int)
                if (np.max(break_locs) < len(lc.time)):
                    break_locs = np.append(break_locs, len(lc.time)-1)
                digit_bounds = lc.time
                digit_bounds = np.array(digit_bounds)
                digit_bounds = digit_bounds[break_locs]
                if digit_bounds[0] > np.min(full_time):
                    digit_bounds = np.append(np.min(full_time)-1e-5, digit_bounds)
                if digit_bounds[-1] < np.max(full_time):
                    digit_bounds = np.append(digit_bounds,np.max(full_time)+1e-5)
                if digit_bounds[0] > np.min(lc.time):
                    digit_bounds = np.append(np.min(lc.time)-1e-5, digit_bounds)
                if digit_bounds[-1] < np.max(lc.time):
                    digit_bounds = np.append(digit_bounds,np.max(lc.time)+1e-5)

                bincts, edges = np.histogram(lc.time,digit_bounds)
                bidx = np.digitize(lc.time, digit_bounds) #binning for star
                bidx = bidx-1
                bincts2, edges = np.histogram(full_time,full_time[break_locs])
                bidx2 = np.digitize(full_time, full_time[break_locs]) #binning for ensemble
                bidx2 = bidx2-1
                num_segs = len(break_locs)
            else:
                bincts, edges = np.histogram(lc.time,[lc.time[0],lc.time[-1]])
                bidx = np.digitize(lc.time, [lc.time[0],lc.time[-1]]) #binning for star
                bidx = bidx-1
                bincts2, edges = np.histogram(full_time,[full_time[0],full_time[-1]])
                bidx2 = np.digitize(full_time, [full_time[0],full_time[-1]]) #binning for ensemble
                bidx2 = bidx2-1
                num_segs = 1

            tscale = []
            for iseg in range(num_segs):
                influx = np.array(lc.flux)
                intime = np.array(lc.time)
                influx = influx[bidx==iseg]
                intime = intime[bidx==iseg]

                #fun = lambda x: np.sum(np.square(np.divide(influx,np.median(influx))-x*scipy.interpolate.splev(intime,pp)))
                fun = lambda x: np.sum(np.square(np.divide(influx,np.median(influx))-x*pp(intime)))
                tscale = np.append(tscale,sciopt.fminbound(fun,0.9,1.5)) #this is a last fix to scaling, not currently used
                tbidx = deepcopy(bidx)

            bin_size = bin_size/2

        # TODO: Remove in final version. Used to test execution time
        print(f"Spline Fit, Time: {time.time()-start_time}")

        #Correct the lightcurve
        lc_corr = deepcopy(lc)
        scale = 1.0
        lc_corr /= scale*pp(lc.time)

        # TODO: Remove in final version. Used to test execution time
        print(f"\nFull correction function, Time: {time.time()-fstart_time}")

        # ax = lc.plot(marker='o', label="Original LC")
        # lc_corr.plot(ax=ax, color='orange', marker='o', markersize=3, ls='--', label="Correction")
        # plt.show()

        return lc_corr, STATUS.OK
