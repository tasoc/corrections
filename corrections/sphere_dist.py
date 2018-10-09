# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:05:22 2018

.. code author :: Derek Buzasi
"""

def sphere_dist(lat1, lon1, lat2, lon2):
    import numpy as np
    import math
    deltaLat = lat2 - lat1
    #deltaLat = np.pi*deltaLat/180
    deltaLon = lon2 - lon1
    #deltaLon = np.pi*deltaLon/180
    a = np.sin(np.deg2rad(deltaLat/2))**2 + np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.sin(np.deg2rad(deltaLon/2))**2
    
    #a = np.sin(deltaLat/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(deltaLon/2)**2
    #print a
    d1 = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    return d1
    
    
    
    