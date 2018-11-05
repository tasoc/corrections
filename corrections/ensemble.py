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
.. codeauthor:: Fillipe Pereira
"""
import numpy as np
import glob
import os
import scipy.interpolate
import scipy.optimize as sciopt
import sqlite3
import sys
from tqdm import tqdm
from operator import add
from copy import deepcopy
from time import sleep

from corrections import BaseCorrector, STATUS


# from star import Star
import lightkurve


################################<PLACEHOLDERS>

#TODO: These shouldn't be hard coded!
# data_folder = "../TESS_Collab_Data"
__sql_folder__ = "../../data/Rasmus"
__data_folder__ = "../../data/Rasmus/data"
__sector__ = "sector02"

def read_todolist(): # NOTE: deprecated! remove me?
    """
    Function to read in the sql to do list for the globally defined sector (__sector__).

    Returns:
        star_names (ndarray): Labels of all the names of the stars in a given sector.
        Tmag (ndarray): TESS photometry apparent magnitude for all stars.
        variability (ndarray): A parameter describing the level of intrinsic variability for all stars.
        eclat (ndarray): Ecliptic latitude for all stars.
        eclon (ndarray): Ecliptic longitude for all stars.
    """

    #open sql file and find list of all stars in segment 2, camera 1, ccd 1
    #TODO: These should not  be hard coded!
    conn = sqlite3.connect('../tests/input/todo.sqlite')
    conn.row_factory = sqlite3.Row
    # conn = sqlite3.connect('/media/derek/data/TESS/TDA-4 data/Rasmus/todo-sector02.sqlite')
    # conn = sqlite3.connect('{}/todo-sector02.sqlite'.format(data_folder))
    c = conn.cursor()
    c.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority = diagnostics.priority WHERE camera = 1 AND ccd = 1 AND mean_flux > 0 ;")
    data = c.fetchall()
    conn.close()

    star_names = np.array([row['starid'] for row in data])
    variability = np.array([row['variability'] for row in data])
    eclat = np.array([row['eclat'] for row in data])
    eclon = np.array([row['eclon'] for row in data])

    return star_names, variability, eclat, eclon

def read_stars(star_names): # NOTE: deprecated! remove me?
    """
    Function to read in the flux timeseries for all stars in a sector given in
    star_names. Time, flux, and some additional metadata are stored in a list of
    class instances. For this, we use the lightkurve open-source Python package.

    Parameters:
        star_names (ndarray): Labels of all the names of the stars in a given
            sector.

    Returns:
        star_array (ndarray): An array of lightkurve.TessLightCurve class
            instances holding metadata on each star. (i.e. flux, time, mean
            flux, std of flux).
    """
    # Read star data from each file and instanciate a Star object for each with all data
    star_array = np.empty(star_names.size, dtype=object)
    for name_index in tqdm(range(star_names.size)):
        filename =  '{}/noisy_by_sectors/Star{}-{}.noisy'.format(__data_folder__, star_names[name_index], __sector__)
        mafs = np.loadtxt(filename, usecols=range(0,2)).T

        #Build lightkurve object and associated metadata
        lc = lightkurve.TessLightCurve(mafs[0], mafs[1]).remove_nans()
        frange = np.percentile(lc.flux, 95) - np.percentile(lc.flux, 5) / np.mean(lc.flux)
        drange = np.std(np.diff(lc.flux)) / np.mean(lc.flux)
        lc.meta = { 'fmean' : np.max(lc.flux),
                    'fstd' : np.std(np.diff(lc.flux)),
                    'frange' : frange,
                    'drange' : drange}
        star_array[name_index] = lc
    return star_array

################################</PLACEHOLDERS>

class EnsembleCorrector(BaseCorrector):
    """
    DOCSTRING
    """
    #NOTE: the __init__() was removed b/c it was overriding (overloading?) the base __init__() and breaking everything

    def do_correction(self):
        """
        Function that takes all input stars for a sector and uses them to find a detrending
        function using ensemble photometry for a star 'star_names[ifile]', where ifile is the
        index for the star in the star_array and star_names list.

        Parameters:
            lc (lightkurve.TessLightCurve object): Raw lightcurve stored
                in a TessLightCurve object.
            ifile (int): TEMPORARY index of the relevant star in the star_names
                list.

        Returns:
            lc_corr (lightkurve.TessLightCurve object): Corrected light-
                curve stored in a TessLightCurve object.
        """

        #Add metadata to the lightcurve object
        #Build lightkurve object and associated metadata
        frange = np.percentile(self.lc.flux, 95) - np.percentile(self.lc.flux, 5)
        frange /= np.mean(self.lc.flux)
        drange = np.std(np.diff(self.lc.flux)) / np.mean(self.lc.flux)
        self.lc.meta.update({ 'fmean' : np.max(self.lc.flux),   # TODO: why is mean flux set to the maximum flux? is this an error?
                    'fstd' : np.std(np.diff(self.lc.flux)),
                    'frange' : frange,
                    'drange' : drange})

        
        #set up start/end times for stellar time series
        time_start = np.amin(self.lc.time)
        time_end = np.max(self.lc.time)

        #set minimum range parameter...this is log10 photometric range, and stars more variable than this will be
        #excluded from the ensemble
        min_range = -2.0
        min_range0 = min_range
        flag = 1

        #NOTE: no longer done in a loop
        #num_star is number of stars in ensemble
        num_star = 0
        #full_time,flux,weight are time,flux,weight points in the ensemble
        full_time = np.array([])
        full_flux = np.array([])
        full_flag = np.array([])
        full_weight = np.array([])
        tflux = np.array([])
        comp_list = np.array([])

        #loop through all other stars to build ensemble
        #exclude stars outside search radius, flagged, too active (either relatively or absolutely)
        #excluding stars with negative flux is only required because the synthetic data have some flawed
        #light curves that are <0. Probably can remove this with real data.
        star_names = self.search_targets()

        # #Put the selection conditions into a boolean array for all stars simultaneously
        # sel = (dist[:,1] < search_radius) & (np.log10(drange) < min_range) & (drange < 10*drange[ifile])
        for test_star in star_names:

            num_star+=1
            #calculate relative flux for star to be added to ensemble
            new_star = self.load_lightcurve(test_star).remove_nans() # just do it here and save so, so much headache 
            test0 = np.asarray(new_star.time)
            test1 = np.asarray(new_star.flux)
            test1 = test1/np.max(np.asarray(new_star.flux))
            #calculate weight for star to be added to the ensemble. weight is whitened stdev relative to mean flux
            weight = np.ones_like(test1)
            weight = weight*np.max(np.asarray(new_star.flux))/np.std(np.diff(np.asarray(new_star.flux)))

            #add time, flux, weight to ensemble light curve. flux is weighted flux
            full_time = np.append(full_time,test0)
            full_flux = np.append(full_flux,np.multiply(test1,weight))
            full_weight = np.append(full_weight,weight)
            #tflux is total unweighted flux
            tflux = np.append(tflux,test1)
            comp_list = np.append(comp_list,test_star)

        #set up time array with 0.5-day resolution which spans the time range of the time series
        #then histogram the data based on that array
        gx = np.arange(time_start,time_end,0.5)
        n = np.histogram(full_time,gx)
        n = np.asarray(n[0])
        n2 = np.histogram(self.lc.time,gx)
        n2 = np.asarray(n2[0])
        #if the least-populated bin has less than 2000 points, increase the size of the ensemble by first
        #increasing the level of acceptable variability until it exceeds the variability of the star. Once that happens,
        #increase the search radius and reset acceptable variability back to initial value. If the search radius exceeds
        #a limiting value (pi/4 at this point), accept that we can't do any better.
        #if np.min(n[0])<400:
        #print np.min(n[n2>0])
        # if np.min(n[n2>0])<1000:
        #     #print min_range
        #     min_range = min_range+0.3
        #     if min_range > np.log10(np.max(lc.meta['drange'])):
        #         #if (search_radius < 0.5):
        #         if (search_radius < 100):
        #             #search_radius = search_radius+0.1
        #             search_radius = search_radius+10
        #         else:
        #             search_radius = search_radius*1.1
        #             min_range = min_range0

        #     #if search_radius > np.pi/4:
        #     if search_radius > 400:
        #         break
        # else:
        #         break
        self.lc.remove_nans()
        #clean up ensemble points by removing NaNs
        # TODO: this should be checked, but is likely unnecessary b/c new_star was called w/ removenans()
        full_time = full_time[~np.isnan(full_flux)]
        full_weight = full_weight[~np.isnan(full_flux)]
        full_flux = full_flux[~np.isnan(full_flux)]
        tflux = tflux[~np.isnan(full_flux)]

        #sort ensemble into time order
        idx = np.argsort(full_time)
        full_time = full_time[idx]
        full_flux = full_flux[idx]
        full_weight = full_weight[idx]

        #temporary copies of ensemble components
        full_time0 = full_time
        full_flux0 = full_flux
        full_weight0 = full_weight

        #set up temporary files

        temp_time = full_time
        temp_flux = full_flux
        temp_weight = full_weight

        #simplify by discarding ensemble points outside the temporal range of the stellar time series
        temp_time = full_time[(full_time>time_start) & (full_time<time_end)]
        temp_flux = full_flux[(full_time>time_start) & (full_time<time_end)]
        temp_weight = full_weight[(full_time>time_start) & (full_time<time_end)]

        full_time = temp_time
        full_flux = temp_flux
        full_weight = temp_weight

        #identify locations where there is a break in the time series. If there is at least one break, identify
        #segments and label ensemble points by segment; bidx2 is the label. If there are no breaks, then identify
        #only one segment and label accordingly
        break_locs = np.where(np.diff(full_time)>0.1)
        if np.size(break_locs)>0:
            if (break_locs[0][-1] < np.size(full_time)):
                break_locs = np.append(break_locs, np.size(full_time)-1)
                break_locs = np.insert(break_locs,0,0)
                cts, bin_edges = np.histogram(full_time,full_time[break_locs])
                bidx2 = np.digitize(full_time,full_time[break_locs])
                num_segs = np.size(break_locs)-1
        else: #TODO: check the indentation level here! some weird copy/pasting might have happened
            cts, bin_edges = np.histogram(full_time,np.squeeze(np.append(full_time[0],full_time[-1])))
            bidx2 = np.digitize(full_time,np.squeeze(np.append(full_time[0],full_time[-1]+1)))
            num_segs = 1;
            break_locs = np.append(0,np.size(full_time)-1)

        #pp will be components of spline fit to ensemble for each segment
        pp_ensemble = []
        #set up influx, inweight,intime as flux/weight/time of ensemble segment-by-segment
        for iseg in range(num_segs):
            influx = full_flux[bidx2-1==iseg]
            inweight = full_weight[bidx2-1==iseg]
            intime = full_time[bidx2-1==iseg]

            intime0 = intime;
            influx0 = influx;

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

            break_locs = np.where(np.diff(self.lc.time)>0.1) #find places where there is a break in time
            break_locs = np.array(break_locs)
            if break_locs.size>0: #set up boundaries to correspond with breaks
                break_locs = np.array(break_locs)+1
                break_locs.astype(int)
                if (np.max(break_locs) < len(self.lc.time)):
                    break_locs = np.append(break_locs, len(self.lc.time)-1)
                digit_bounds = self.lc.time
                digit_bounds = np.array(digit_bounds)
                digit_bounds = digit_bounds[break_locs]
                if digit_bounds[0] > np.min(full_time):
                    digit_bounds = np.append(np.min(full_time)-1e-5, digit_bounds)
                if digit_bounds[-1] < np.max(full_time):
                    digit_bounds = np.append(digit_bounds,np.max(full_time)+1e-5)
                if digit_bounds[0] > np.min(self.lc.time):
                    digit_bounds = np.append(np.min(self.lc.time)-1e-5, digit_bounds)
                if digit_bounds[-1] < np.max(self.lc.time):
                    digit_bounds = np.append(digit_bounds,np.max(self.lc.time)+1e-5)

                bincts, edges = np.histogram(self.lc.time,digit_bounds)
                bidx = np.digitize(self.lc.time, digit_bounds) #binning for star
                bidx = bidx-1
                bincts2, edges = np.histogram(full_time,full_time[break_locs])
                bidx2 = np.digitize(full_time, full_time[break_locs]) #binning for ensemble
                bidx2 = bidx2-1
                num_segs = len(break_locs)
            else:
                bincts, edges = np.histogram(self.lc.time,[self.lc.time[0],self.lc.time[-1]])
                bidx = np.digitize(self.lc.time, [self.lc.time[0],self.lc.time[-1]]) #binning for star
                bidx = bidx-1
                bincts2, edges = np.histogram(full_time,[full_time[0],full_time[-1]])
                bidx2 = np.digitize(full_time, [full_time[0],full_time[-1]]) #binning for ensemble
                bidx2 = bidx2-1
                num_segs = 1

            tscale = []
            for iseg in range(num_segs):
                influx = np.array(self.lc.flux)
                intime = np.array(self.lc.time)
                influx = influx[bidx==iseg]
                intime = intime[bidx==iseg]

                #fun = lambda x: np.sum(np.square(np.divide(influx,np.median(influx))-x*scipy.interpolate.splev(intime,pp)))
                fun = lambda x: np.sum(np.square(np.divide(influx,np.median(influx))-x*pp(intime)))
                tscale = np.append(tscale,sciopt.fminbound(fun,0.9,1.5)) #this is a last fix to scaling, not currently used
                tbidx = deepcopy(bidx)

            bin_size = bin_size/2

        #Correct the lightcurve
        lc_corr = deepcopy(self.lc)
        scale = 1.0
        lc_corr.flux = np.divide(self.lc.flux,(scale*pp(self.lc.time)))
        #self._status = status(1)

        return lc_corr, STATUS.OK

def correct(*args): # NOTE: deprecated! remove me?
    star_names, Tmag, variability, eclat, eclon = read_todolist()

    C = EnsembleCorrector(BaseCorrector)
    star_array = C.star_array

    '''Get the correction, apply the correction, output the data.'''
    for ifile in tqdm(range(len(star_names[:15]))):

        ax = star_array[ifile].plot()
        lc_corr = C.do_correction(star_array[ifile], ifile)
        lc_corr.plot(ax=ax)
        plt.show()

        outfile = '../../data/Rasmus/toutput2/'+str(star_names[ifile])+'.noisy_detrend'
        file = open(outfile,'w')
        #np.savetxt(file,np.column_stack((star_array[ifile].time,star_array[ifile].flux, fcorr2[ifile])), fmt = '%f')
        np.savetxt(file,np.column_stack((star_array[ifile].time,star_array[ifile].flux, lc_corr.flux)), fmt = '%f')
        file.close()
