#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 02:01:29 2011

@author: lunnde
"""

from __future__ import division
import numpy as np
cimport numpy as np
from numpy import array, diff, isnan, inf, NaN
from bottleneck import nanmedian, median
from collections import deque
from bisect import bisect_left, insort
from blist import blist

cimport cython
np.seterr(all='ignore')
DTYPE = float
ctypedef np.float_t DTYPE_t

#==============================================================================
#
#==============================================================================
@cython.boundscheck(False)
cpdef gap_fill(np.ndarray[DTYPE_t, ndim=1] t, np.ndarray[DTYPE_t, ndim=1] y, double maxgap=inf):
    # Declare variables used:
    cdef Py_ssize_t i, j
    cdef int times, times_max=0
    cdef double d, step, stepcut
    cdef np.ndarray[DTYPE_t, ndim=1] D = diff(t)
    time_tot = blist([])
    data_tot = blist([])
    ori_or_not = blist([])

    # Calculate the desired regular step size:
    step = median(D)
    stepcut = 1.5*step

    # test:
    if maxgap != inf:
        times_max = int((maxgap/2)/step)+1

    for i in xrange(len(t)-1):
        # Add the original point:
        time_tot.append(t[i])
        data_tot.append(y[i])
        ori_or_not.append(1)

        d = D[i]

        if d > maxgap:
            # Insert half the maximum number of points in the beginning and end of gap:
            for j in xrange(1, times_max):
                time_tot.append(t[i]+j*step)
                data_tot.append(NaN)
                ori_or_not.append(0)
            # Insert half the maximum number of points in the beginning and end of gap:
            for j in xrange(times_max,0,-1):
                time_tot.append(t[i+1]-j*step)
                data_tot.append(NaN)
                ori_or_not.append(0)
        elif d > stepcut:
             # Calculate the number of points to be inserted and insert them:
            times = int(d/step)-1
            for j in xrange(times):
               time_tot.append(t[i]+(j+1)*step)
               data_tot.append(NaN)
               ori_or_not.append(0)

    # Special treatment of last point:
    time_tot.append(t[-1])
    data_tot.append(y[-1])
    ori_or_not.append(1)

    return array(time_tot), array(data_tot), array(ori_or_not, dtype=bool)


#==============================================================================
# This implements the Theil-Sen linear regression estimator for 2d data points.
# The jist of it is:
# It returns the median all computed slope value between pairs (x_i, y_i), (x_j, y_j), (x_i > x_j)
# where slope = (y_i - y_j)/(x_i - x_j)
#
# Very robust to outliers.
#==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef theil_sen(np.ndarray[DTYPE_t, ndim=1, negative_indices=False] x, np.ndarray[DTYPE_t, ndim=1, negative_indices=False] y, int n_samples):
    """Computes the Theil-Sen estimator for 2d data

    parameters:
        x: 1-d np array, the control variate
        y: 1-d np.array, the ind variate.
        n_samples: how many points to sample.

    This complexity is O(n**2), which can be poor for large n. We will perform a sampling
    of data points to get an unbiased, but larger variance estimator.
    The sampling will be done by picking two points at random, and computing the slope,
    up to n_samples times.
    """

    cdef np.ndarray[long, ndim=1, negative_indices=False] i1, i2
    cdef np.ndarray[DTYPE_t, ndim=1, negative_indices=False] intercepts, slopes
    cdef int i, n
    cdef double intercept_, slope_

    assert x.shape[0] == y.shape[0], "x and y must be the same shape."
    n = x.shape[0]

    i1 = np.random.randint(0, n, n_samples)
    i2 = np.random.randint(0, n, n_samples)
    slopes = slope( x[i1], x[i2], y[i1], y[i2] )

    slope_ = nanmedian( slopes )
    #find the optimal b as the median of y_i - slope*x_i
    intercepts = np.empty( n , dtype=float)
    for i in xrange(n):
        intercepts[i] = y[i] - slope_*x[i]
    intercept_ = median( intercepts )

    return np.array( [slope_, intercept_] )


cdef slope(np.ndarray[DTYPE_t, ndim=1, negative_indices=False]  x_1, np.ndarray[DTYPE_t, ndim=1, negative_indices=False] x_2, np.ndarray[DTYPE_t, ndim=1, negative_indices=False] y_1, np.ndarray[DTYPE_t, ndim=1, negative_indices=False] y_2):
    return (1 - 2*(x_1>x_2) )*( (y_2 - y_1)/np.abs((x_2-x_1)) )
