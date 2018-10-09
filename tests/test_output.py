#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: important info goes here

.. code author:: Rasmus Handberg <rasmush@phys.au.dk>

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob

globlist = glob.glob('../data/Rasmus/toutput2/*.noisy_detrend')

for sfile in globlist:
    f = np.loadtxt(sfile).T
    plt.plot(f[0], f[1])
    plt.plot(f[0], f[2])
    plt.show()
