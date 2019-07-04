#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
The ensemble photometry detrending class.

Created on Thu Mar 29 09:58:55 2018
.. codeauthor:: Derek Buzasi
.. codeauthor:: Oliver J. Hall
.. codeauthor:: Lindsey Carboneau
.. codeauthor:: Filipe Pereira
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize as sciopt
from copy import deepcopy
import time
import lightkurve
import logging
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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

    def fast_median(self, lc_ensemble):
        """
        A small utility function for calculating the ensemble median for use in
        correcting the target light curve
        Parameters:
            lc_ensemble: an array-like collection of each light curve in the ensemble
        Returns:
            lc_medians: an array-like (list) that represents the median value of
                    each light curve in the ensemble at each cadence 
        """
        lc_medians = []
        col, row = np.asarray(lc_ensemble).shape
        for i in range(row):
            # find the median of the ensemble
            temp =[]
            for j in range(col):
                temp.append(lc_ensemble[j][i])
            lc_medians.append(np.median(temp))
        
        return lc_medians

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
        logger = logging.getLogger(__name__)
        logger.info("Data Source: {}".format(lc.meta['task']['datasource']))

        # TODO: Remove in final version. Used to test execution time
        full_start = time.time()

        # Clean up the lightcurve by removing nans and ignoring data points with bad quality flags
        og_time = lc.time.copy()
        lc = lc.remove_nans()
        lc_quality_mask = (lc.quality == 0)
        # lc.time = lc.time[lc_quality_mask]
        # lc.flux = lc.flux[lc_quality_mask]
        # lc.flux_err = lc.flux_err[lc_quality_mask]

        # Set up basic statistical parameters for the light curves. 
        # frange is the light curve range from the 5th to the 95th percentile,
        # drange is the relative standard deviation of the differenced light curve (to whiten the noise)
        frange = (np.percentile(lc.flux, 95) - np.percentile(lc.flux, 5) )/ np.mean(lc.flux)

        drange = np.std(np.diff(lc.flux)) / np.mean(lc.flux)
        lc.meta.update({ 'fmean' : np.median(lc.flux),
                        'fstd' : np.std(np.diff(lc.flux)),
                        'frange' : frange,
                        'drange' : drange})

        logger.info(lc.meta.get("drange"))
        logger.info(" ")

        # StarID, pixel positions and lightcurve filenames are retrieved from the database
        select_params = ["todolist.starid", "pos_row", "pos_column"]
        search_params = ["camera={:d}".format(lc.camera), "ccd={:d}".format(lc.ccd), "mean_flux>0", "datasource='{:s}'".format(lc.meta["task"]["datasource"])]
        db_raw = self.search_database(select=select_params, search=search_params)
        starid = np.array([row['starid'] for row in db_raw])
        pixel_coords = np.array([[row['pos_row'], row['pos_column']] for row in db_raw])

        # TODO: We can leave the target star for the distance comparison and get its exact index for free. Just need to be careful with the entry 0 of dist 
        # Determine distance of all stars to target. Array of star indexes by distance to target and array of the distance. Pixel distance used
        idx = (starid == lc.targetid)
        dist = np.sqrt((pixel_coords[:,0] - pixel_coords[idx,0])**2 + (pixel_coords[:,1] - pixel_coords[idx,1])**2)
        distance_index = dist.argsort()
        target_index = distance_index[0]
        distance = dist[distance_index]

        # Set up start/end times for stellar time series
        time_start = np.amin(lc.time)
        time_end = np.max(lc.time)

        # Set minimum range parameter...this is log10 photometric range, and stars more variable than this will be excluded from the ensemble
        min_range = 0.0
        # min_range can be changed later on, so we establish a min_range0 for when we want to reset min_range back to its initial value
        min_range0 = min_range
        #flag = 1

        # Define variables to use in the loop to build the ensemble of stars
        # List of star indexes to be included in the ensemble
        temp_list = []
        # Initial number of closest stars to consider and variable to increase number
        initial_num_stars = 10
        star_count = initial_num_stars
        # (Alternate param) Initial distance at which to consider stars around the target
        # initial_search_radius = -1

        # Follows index of dist array to restart search after the last addded star. Starts at 1 as the first star ordered by distance is the target
        i = 1
        # Setup search and select params to use in loop
        select_loop = ["todolist.starid", "camera", "ccd", "lightcurve"]
        search_loop = ["camera={:d}".format(lc.camera), "ccd={:d}".format(lc.ccd), "mean_flux>0", "datasource='{:s}'".format(lc.meta["task"]["datasource"])]
        # Start loop to build ensemble
        ensemble_start = time.time()
        lc_ensemble = []
        target_flux = deepcopy(lc.flux)
        sum_ensemble = np.zeros(len(target_flux)) # to check for a large enough ensemble for dimmer stars
        mtarget_flux = target_flux - np.median(target_flux)

        logger.info(str(np.median(target_flux)))

        # First get a list of indexes of a specified number of stars to build the ensemble
        while len(temp_list) < star_count:
            
            # Get lightkurve for next star closest to target
            # NOTE: This seems needlessly complicated. Probably can just change load_lightcurve
            try:
                next_star_index = distance_index[i]
            except IndexError:
                return None, STATUS.SKIPPED
            search_loop.append("todolist.starid={:}".format(starid[next_star_index]))
            next_star_task = self.search_database(search=search_loop, select=select_loop)[0]
            next_star_lc = self.load_lightcurve(next_star_task).remove_nans()
            search_loop.pop(-1)
            
            next_star_lc_quality_mask = (next_star_lc.quality == 0)
            # next_star_lc.time = next_star_lc.time[next_star_lc_quality_mask]
            # next_star_lc.flux = next_star_lc.flux[next_star_lc_quality_mask]
            # next_star_lc.flux_err = next_star_lc.flux_err[next_star_lc_quality_mask]

            # Compute the rest of the statistical parameters for the next star to be added to the ensemble.
            frange = (np.percentile(next_star_lc.flux, 95) - np.percentile(next_star_lc.flux, 5) )/ np.mean(next_star_lc.flux)
            drange = np.std(np.diff(next_star_lc.flux)) / np.mean(next_star_lc.flux)
            
            next_star_lc.meta.update({ 'fmean' : np.median(next_star_lc.flux),
                                        'fstd' : np.std(np.diff(next_star_lc.flux)),
                                        'frange' : frange,
                                        'drange' : drange})

            logger.info(next_star_lc.meta.get("drange"))
            # Stars are added to ensemble if they fulfill the requirements. These are (1) drange less than min_range, (2) drange less than 10 times the 
            # drange of the target (to ensure exclusion of relatively noisy stars), and frange less than 0.03 (to exclude highly variable stars)
            if (np.log10(next_star_lc.meta['drange']) < min_range and next_star_lc.meta['drange'] < 10*lc.meta['drange'] and next_star_lc.meta['frange'] < 0.4):
                
                ###################################################################
                # median subtracted flux of target and ensemble candidate
                temp_lc = deepcopy(next_star_lc)
                time_ens = temp_lc.time
                ens_flux = temp_lc.flux
                if self.debug:
                    plt.plot(time_ens, ens_flux)
                    plt.show(block=True)
                
                #ens_flux = ens_flux[np.isin(next_star_lc.time, lc.time, assume_unique=True)]
                mens_flux = ens_flux - np.median(ens_flux)
                
                #logger.info(str(len(mtarget_flux)) +" "+ str(len(mens_flux))))

                # 2 sigma
                ens2sig = 2 * np.std(mens_flux)
                targ2sig = 2 * np.std(mtarget_flux)
                
                # absolute balue
                abstarg = np.absolute(mtarget_flux)
                absens = np.absolute(mens_flux)

                logger.info("2 sigma")
                logger.info(str(ens2sig) + " , " + str(targ2sig))

                # sigma clip the flux used to fit, but don't use that flux again
                clip_target_flux = np.where(
                    np.where(abstarg < targ2sig, True, False)
                    & 
                    np.where(absens < ens2sig, True, False),
                    mtarget_flux, 1)
                clip_ens_flux = np.where(
                    np.where(abstarg < targ2sig, True, False)
                    & 
                    np.where(absens < ens2sig, True, False),
                    mens_flux, 1)

                logger.info(str(np.median(target_flux)))
                logger.info(str(np.median(ens_flux)))

                #this is where I'll try adding the background correction portion
                #first get scaled target flux
                scale_target_flux = clip_target_flux/np.median(target_flux)

                args = tuple((clip_ens_flux+np.median(ens_flux),clip_target_flux+np.median(target_flux)))


                def func1(scaleK,*args):
                    temp = (((args[0]+scaleK)/np.median(args[0]+scaleK))-1)-((args[1]/np.median(args[1]))-1)
                    temp = (args[1]/np.median(args[1])) - ((args[0]+scaleK)/np.median(args[0]+scaleK))
                    temp = temp - np.median(temp)
                    return np.sum(np.square(temp))
                
                scale0 = 100    
                res = minimize(func1,scale0,args,method='Powell')

                logger.info("Fit param1: {}".format(res.x))
                logger.info(str(np.median(ens_flux)))
                logger.info(str(res.x))
                
                ens_flux = ens_flux+res.x
                mens_flux = mens_flux+res.x
                clip_ens_flux = clip_ens_flux+res.x

                if self.debug: 
                    plt.plot(ens_flux/np.median(ens_flux))
                    plt.show(block=True)
                next_star_lc.flux = (ens_flux/np.median(ens_flux))
                temp_list.append([next_star_index, next_star_lc.copy()])
                lc_ensemble.append(ens_flux/np.median(ens_flux))
                sum_ensemble = sum_ensemble + np.array(ens_flux)
                logger.info(str(next_star_lc.targetid)+'\n')
                ###################################################################
            i += 1
        
        logger.info("Build ensemble, Time: {}".format(time.time()-ensemble_start))
        
        
        lc_medians = self.fast_median(lc_ensemble)
        
        
        if self.debug:
            plt.plot(lc.time, lc_medians)
            plt.plot(lc.time, (lc.flux/np.median(lc.flux)))
            plt.show(block=True)
            plt.plot(lc.time, lc_medians)
            plt.show(block=True)
        lc_medians = np.asarray(lc_medians)


        args = tuple((lc.flux, lc_medians))
        def func2(scalef,*args):
            num1 = np.sum(np.abs(np.diff(np.divide(args[0],args[1]+scalef))))
            denom1 = np.median(np.divide(args[0],args[1]+scalef))
            return num1/denom1

        scale0 = 1.0
        res = minimize(func2,scale0,args)

        logger.info("Fit param: {}".format(res.x))
    
        #fitf2 = 1+((lc_medians-1)*res.x)

        # Correct the lightcurve
        lc_corr = lc.copy()

        k_corr = res.x

        median_only_flux = np.divide(lc_corr.flux, lc_medians)
        lc_corr.flux = np.divide(lc_corr.flux, (k_corr+lc_medians))
        lc_corr.flux = np.divide(lc_corr.flux, np.median(lc_corr.flux))
        lc_corr.flux = lc_corr.flux*np.median(lc.flux)

        
        if self.debug:
            plt.scatter(lc_corr.time, median_only_flux, marker='.', label="Median Only")
            plt.scatter(lc_corr.time, lc_corr.flux, marker='.', label="Corrected LC")
            plt.show(block=True)
        #######################################################################################################

        # TODO: Remove in final version. Used to test execution time
        logger.info("Full do_correction, Time: {}".format(time.time()-full_start))

        # We probably want to return additional information, including the list of stars in the ensemble, and potentially other things as well. 
        
        logger.info(temp_list)
        
        #sys.exit()
        # Replace removed points with NaN's so the info can be saved to the FITS
        lc_corr.time = lc_corr.time[lc_quality_mask]
        lc_corr.flux = lc_corr.flux[lc_quality_mask]
        lc_corr.flux_err = lc_corr.flux_err[lc_quality_mask]
        if len(lc_corr.flux) != len(og_time):
            fix_flux = np.asarray(lc_corr.flux.copy())
            indices = np.array(np.where(np.isin(og_time, lc_corr.time, assume_unique=True, invert=True)))[0]
            indices.tolist()

            for ind in indices:
                fix_flux = np.insert(fix_flux, ind, np.nan)

            lc_corr.flux = fix_flux.tolist()
            lc_corr.time = og_time
            
        

        if self.plot:
            ax = lc.plot(marker='o', label="Original LC")
            lc_corr.plot(ax=ax, color='orange', marker='o', ls='--', label="Corrected LC")
            #plt.show() #block=True)
            plt.savefig("./temp/" + str(lc.targetid) + "_testrun.png")
            logger.info(np.nanstd(lc.flux))
            logger.info(np.nanstd(lc_corr.flux))
            logger.info(np.nanmedian(lc.flux))
            logger.info(np.nanmedian(lc_corr.flux))
        return lc_corr, STATUS.OK
