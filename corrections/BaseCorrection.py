#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic photometry class for the TASOC Correction pipeline.
All other specific correction algorithms will inherit from BaseCorrection.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import six
from six.moves import range
import numpy as np
import logging

__docformat__ = 'restructuredtext'

class BaseCorrection(object):
	"""
	The basic correction class for the TASOC Correction pipeline.
	All other specific correction algorithms will inherit from this.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self):
		"""
		Initialize the correction object.
		"""

		logger = logging.getLogger(__name__)


	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		"""Close correction object."""
		pass
