#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightcurve correction using the KASOC Filter (Handberg et al. 2015).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import logging
import requests
from collections import defaultdict
from . import kasoc_filter as kf
from . import BaseCorrector, STATUS
from .quality import CorrectorQualityFlags

class KASOCFilterCorrector(BaseCorrector):

	#----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super().__init__(*args, **kwargs)

		logger = logging.getLogger(__name__)

		# Store a list of known periods of TESS Objects of Interest (TOIs):
		self.tois_periods = defaultdict(list)

		# Contact TASOC database for list of TOIs:
		r = requests.get('https://tasoc.dk/pipeline/toilist.php')
		r.raise_for_status()
		for row in r.json():
			self.tois_periods[row['tic']].append(row['period'])

		logger.debug(self.tois_periods)

	#----------------------------------------------------------------------------------------------
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
			if len(periods) == 0:
				periods = None

		logger.debug(periods)

		jumps = None

		cadence = 86400*np.median(np.diff(lc.time))
		if cadence > 1000:
			filter_timescale_long = 6.0
			filter_timescale_short = 0.5
		else:
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
		lc_corr = lc.copy()
		time2, lc_corr.flux, lc_corr.flux_err, kasoc_quality, filt, turnover, xlong, xpos, xtransit, xshort = kf.filter(
			lc.time,
			lc.flux,
			quality=lc.pixel_quality,
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

		# Translate the quality flags from the KASOC filter to the real ones:
		lc_corr.quality[kasoc_quality & 2 != 0] |= CorrectorQualityFlags.JumpAdditiveConstant
		lc_corr.quality[kasoc_quality & 4 != 0] |= CorrectorQualityFlags.JumpAdditiveLinear
		lc_corr.quality[kasoc_quality & 256 != 0] |= CorrectorQualityFlags.JumpMultiplicativeConstant
		lc_corr.quality[kasoc_quality & 512 != 0] |= CorrectorQualityFlags.JumpMultiplicativeLinear
		lc_corr.quality[kasoc_quality & 8+128 != 0] |= CorrectorQualityFlags.SigmaClip

		# Set headers that will be saved to the FITS file:
		#lc_corr.meta['additional_headers']['KF_MODE'] = (filter_operation_mode, 'KASOC filter: operation mode')
		lc_corr.meta['additional_headers']['KF_POSS'] = (filter_position_mode, 'KASOC filter: star positions used')
		lc_corr.meta['additional_headers']['KF_LONG'] = (filter_timescale_long, '[d] KASOC filter: long timescale')
		lc_corr.meta['additional_headers']['KF_SHORT'] = (filter_timescale_short, '[d] KASOC filter: short timescale')
		lc_corr.meta['additional_headers']['KF_SCLIP'] = (filter_sigma_clip, 'KASOC filter: sigma clipping')
		lc_corr.meta['additional_headers']['KF_TCLIP'] = (filter_turnover_clip, 'KASOC filter: turnover clip')
		lc_corr.meta['additional_headers']['KF_TWDTH'] = (filter_turnover_width, 'KASOC filter: turnover width')
		lc_corr.meta['additional_headers']['KF_PSMTH'] = (filter_phase_smooth_factor, 'KASOC filter: phase smooth factor')

		# Add information about removed periods to the header:
		if periods is not None:
			lc_corr.meta['additional_headers']['NUM_PER'] = (len(periods), 'KASOC filter: number of periods removed')
			for k, p in enumerate(periods):
				lc_corr.meta['additional_headers']['PER_%d' % (k+1)] = (p, '[d] KASOC filter: period removed')
		else:
			lc_corr.meta['additional_headers']['NUM_PER'] = (0, 'KASOC filter: number of periods removed')

		return lc_corr, STATUS.OK
