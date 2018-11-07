#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightcurve correction using the KASOC Filter (Handberg et al. 2015).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import logging
import psycopg2 as psql
import getpass
from . import kasoc_filter as kf
from . import BaseCorrector, STATUS

class KASOCFilterCorrector(BaseCorrector):

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		logger = logging.getLogger(__name__)

		# Store a list of known periods of TESS Objects of Interest (TOIs):
		self.tois_periods = {}

		# Ask for username and password for the TASOC database:
		default_username = getpass.getuser()
		username = input('Username [%s]: ' % default_username)
		if username == '':
			username = default_username
		passwd = getpass.getpass('Password: ')

		# Contact TASOC database for list of TOIs:
		with psql.connect(host='trinity.phys.au.dk', user=username, password=passwd, database='db_aadc') as conn:
			with conn.cursor() as cursor:
				cursor.execute("SELECT starid,period FROM tasoc.toi WHERE NOT period IS NULL;")
				for row in cursor.fetchall():
					if row[0] in self.tois_periods:
						self.tois_periods[row[0]].append(row[1])
					else:
						self.tois_periods[row[0]] = [row[1]]

		logger.debug(self.tois_periods)


	def do_correction(self, lc):

		logger = logging.getLogger(__name__)

		#position = np.column_stack((lc.centroid_col, lc.centroid_row))
		position = None

		periods = np.array(self.tois_periods.get(lc.targetid, []))
		if len(periods) == 0:
			periods = None
		else:
			indx = (np.max(lc.time) - np.min(lc.time)) / np.array(periods) > 3.0
			periods = periods[indx]
			if len(periods) == 0: periods = None

		logger.debug(periods)

		jumps = None
		filter_timescale_long = 3.0
		filter_timescale_short = 1.0/24.0
		filter_phase_smooth_factor = 200
		filter_sigma_clip = 4.5
		filter_turnover_clip = 5.0
		filter_turnover_width = 1.0
		filter_position_mode = 'None'

		# Configure plotting in KASOC filter:
		if self.plot:
			kf.set_output(self.plot_folder(lc), '%011d_kf_' % lc.targetid, fmt='png')
		else:
			kf.set_output(None)

		# Run the KASOC Filter:
		time2, lc.flux, lc.flux_err, kasoc_quality, filt, turnover, xlong, xpos, xtransit, xshort = kf.filter(
			lc.time,
			lc.flux,
			quality=lc.quality,
			P=periods,
			jumps=jumps,
			position=position,
			timescale_long=filter_timescale_long,
			timescale_short=filter_timescale_short,
			phase_smooth_factor=filter_phase_smooth_factor,
			sigma_clip=filter_sigma_clip,
			scale_clip=filter_turnover_clip,
			scale_width=filter_turnover_width
		)

		# Set headers that will be saved to the FITS file:
		#lc.meta['additional_headers']['KF_MODE'] = (filter_operation_mode, 'KASOC filter: operation mode')
		lc.meta['additional_headers']['KF_POSS'] = (filter_position_mode, 'KASOC filter: star positions used')
		lc.meta['additional_headers']['KF_LONG'] = (filter_timescale_long, '[d] KASOC filter: long timescale')
		lc.meta['additional_headers']['KF_SHORT'] = (filter_timescale_short, '[d] KASOC filter: short timescale')
		lc.meta['additional_headers']['KF_SCLIP'] = (filter_sigma_clip, 'KASOC filter: sigma clipping')
		lc.meta['additional_headers']['KF_TCLIP'] = (filter_turnover_clip, 'KASOC filter: turnover clip')
		lc.meta['additional_headers']['KF_TWDTH'] = (filter_turnover_width, 'KASOC filter: turnover width')
		lc.meta['additional_headers']['KF_PSMTH'] = (filter_phase_smooth_factor, 'KASOC filter: phase smooth factor')
		# Add information about removed periods to the header:
		if not periods is None:
			lc.meta['additional_headers']['NUM_PER'] = (len(periods), 'number of periods removed')
			for k,p in enumerate(periods):
				lc.meta['additional_headers']['PER_%d'%(k+1)] = (p, 'period removed (days)')
		else:
			lc.meta['additional_headers']['NUM_PER'] = (0, 'number of periods removed')

		return lc, STATUS.OK