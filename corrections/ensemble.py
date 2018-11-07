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
        fstart_time = time.time()

        # Determine distance of all stars to target. Array of star indexes by distance to target and array of the distance
        idx = np.arange(len(self.star_names))!=ifile
        dist = np.sqrt((self.eclat[idx] - self.eclat[ifile])**2 + (self.eclon[idx] - self.eclon[ifile])**2)
        distance_index = dist.argsort()
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
        ensemble_list = []
        # Initial number of closest stars to consider and variable to increase number
        initial_num_stars = 20
        star_count = initial_num_stars
        # (Alternate param) Initial distance at which to consider stars around the target
        # initial_search_radius = -1
        
        # Follows index of dist array to restart search after the last addded star
        i = 0
        # Start loop to build ensemble
        while True:

            # First get a list of indexes of a specified number of stars to build the ensemble
            while len(ensemble_list) < star_count:
                # Stars evaluated by order of distance to target
                star_index = distance_index[i]
                # Stars are added to ensemble if they fulfill the requirements
                if (np.log10(self.star_array[star_index].meta['drange']) < min_range and self.star_array[star_index].meta['drange'] < 10*lc.meta['drange']):
                    ensemble_list.append(star_index)
                i += 1


            # Now populate the arrays of data with the stars in the ensemble
            full_time = np.concatenate([self.star_array[i].time for i in ensemble_list]).ravel()
            tflux = np.concatenate(np.array([self.star_array[i].flux / self.star_array[i].meta['fmean'] for i in ensemble_list])).ravel()
            full_weight = np.concatenate(np.array([np.full(self.star_array[i].flux.size, star_array[i].meta['fmean'] / self.star_array[i].meta['fstd']) for i in ensemble_list])).ravel()
            full_flux = np.multiply(tflux, full_weight)

            # TODO: As of now the code begins by ensuring 20 stars are added to the ensemble and then adds one by one. Might have to change to use a search radius
            # Fetch distance of last added star to ensemble to use as search radius
            search_radius = distance[i-1]

            # Set up time array with 0.5-day resolution which spans the time range of the time series then histogram the data based on that array
            gx = np.arange(time_start,time_end,0.5)
            n = np.histogram(full_time, gx)[0]
            n2 = np.histogram(lc.time, gx)[0]
            # If the least-populated bin has less than 2000 points, increase the size of the ensemble by first
            # increasing the level of acceptable variability until it exceeds the variability of the star. Once that happens,
            # increase the search radius and reset acceptable variability back to initial value. If the search radius exceeds
            # a limiting value (pi/4 at this point), accept that we can't do any better.
            # if np.min(n[0])<400:
            # print np.min(n[n2>0])
            if np.min(n[n2>0]) < 1000:
                # print(min_range)
                min_range = min_range+0.3
                if min_range > np.log10(np.max(lc.meta['drange'])):
                    #if (search_radius < 0.5):
                    if (search_radius < 100):
                        # search_radius += 10
                        star_count += 1
                    else:
                        # search_radius *= 1.1
                        star_count += 1
                        min_range = min_range0

                # if search_radius > np.pi/4:
                if search_radius > 400:
                    break
            else:
                    break

        # Ensemble is now built
        # Clean up ensemble points by removing NaNs
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
        idx = np.logical_and(full_time > time_start, full_time < time_end)
        full_time = full_time[idx]
        full_flux = full_flux[idx]
        full_weight = full_weight[idx]

        temp_time = full_time
        temp_flux = full_flux
        temp_weight = full_weight


        start_time = time.time()
        #initialize bin size in days. We will fit the ensemble with splines
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
        print(f"Spline Fit, Time: {time.time()-start_time}")

        #Correct the lightcurve
        lc_corr = deepcopy(lc)
        scale = 1.0
        lc_corr /= scale*pp(lc.time)

        print(f"Full correction function, Time: {time.time()-fstart_time}")

        ax = lc.plot()
        lc_corr.plot(ax=ax)
        plt.show()

        return lc_corr, STATUS.OK
