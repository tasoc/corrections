#!/usr/bin/env python
"""
The basic correction class for the TASOC Photomety pipeline
All other specific correction classes will inherit from BaseCorrection.
Structure from `BasePhotometry by Rasmus Handberg <https://github.com/tasoc/photometry/blob/devel/photometry/BasePhotometry.py>`_

- :py:class:`STATUS`: Status flags for pipeline performance logging
- :py:class:`STATUS`: Status flags for pipeline performance logging

.. codeauthor:: Lindsey Carboneau 
.. code author:: Rasmus Handberg
"""

# TODO: imports; must be included in requirements.txt
# from package import function as key
import enum
import logging
import traceback
import sqlite3
import numpy as np
import matplotlib.pyplot as plt 
from lightkurve import TessLightCurve

__docformat__ = 'restructuredtext'

class STATUS(enum.Enum):
    """
    Status indicator of the status of the correction.

    """

    UNKNOWN = 0
    OK = 1
    ERROR = 2
    WARNING = 3
    # TODO: various statuses as required

class BaseCorrector(object):
    """
    The basic correction class for the TASOC Photometry pipeline.
    All other specific correction classes will inherit from BaseCorrector

    Attributes:
        # TODO
    """

    def __init__(self, starid, camera, ccd, cbv_area, eclon, eclat, input_folder, output_folder, priority, plot=False):
        """
        Initialize the correction object

        Parameters:
            # TODO

        Returns:
            # TODO

        Raises:
            IOError: If (target ID) could not be found (TODO: other values as well?)
            ValueError: (TODO: on a lot of places)
            NotImplementedError: Everywhere a function has a TODO/FIXME tag preventing execution
        """

        logger = logging.getLogger(__name__)

        self._status = STATUS.UNKNOWN
        self.starid = starid
        self.camera = camera
        self.ccd = ccd
        self.cbv_area
        self.eclon = eclon
        self.eclat = eclat
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.lc = self.load_lightcurve(self.starid)
    
    def __enter__(self):
	    return self
    
    def __exit__(self, *args):
	    self.close()

    def close(self):
        """Close correction object"""
        pass

    def status(self):
        """ The status of the corrections. From :py:class:`STATUS`."""
        return self._status

    def load_lightcurve(self, starid):
        """
        Load target lightcurve for given TIC/starid
        NOTE: ASSUMPTIONS MADE
            can't use self.starid b/c we might be loading in a neighbor
            if we read/load in the data for a star that is _not_ the target
            being corrected, than the assumption is that star is on the same
            camera, CCD, and focal area (CBV area) as the intended target
            I'm not really sure if it's possible that isn't true; it's not something
            that can be checked for using the simulated data, but when we start 
            working with the output of the photometry side of the pipeline we need:
            TODO: update to set camera, ccd (and cbv_area?) from FITS headers;
                  also add logic to ensure that those values match the correction target

        Returns:
            Lightkurve object
        """
        logger = logging.getLogger(__name__)

        try:
            data = np.loadtxt(self.input_folder + '/Star' + str(starid) +'.sysnoise')
            lightcurve = TessLightCurve(
                time=data[:,0],
                flux=data[:,1],
                #flux_err=data[:,2],
                quality=np.asarray(data[:,3], dtype='int32'),
                # NOTE: this is hardcoded and gross; but the original layout did not match the T'DA 3_2 noisy sim data
                time_format='jd',
                time_scale='tdb',
                targetid=starid,
                camera=self.camera, # for next three lines see docstring above
                ccd=self.ccd,
                meta = {'eclon':self.eclon, 'eclat':self.eclat, 'cbv_area':self.cbv_area}
            )
        except ValueError:
            logger.exception("Check input file: Star"+self.starid+'.sysnoise')
            trace = traceback.format_exc().strip()
            try:
                corr._status = STATUS.ERROR # TODO: should this be self._status = STATUS.ERROR ?
                corr.report_details(error=trace)
            except:
                pass
        return lightcurve
  
    def do_correction(self):
        """
        Apply corrections to target lightcurve.

        Returns:
            The status of the corrections.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("A helpful error message goes here") # TODO

    def correct(self, *args, **kwargs):
        """
        Run correction.

        """
        self.load_lightcurve(self.starid)

        self._status = self.do_correction(*args, **kwargs)

        # Check that the status has been changed:
        if self._status == STATUS.UNKNOWN:
            raise Exception("STATUS was not tset by do_correction")

        if self._status in (STATUS.OK, STATUS.WARNING):
            # TODO: set outputs; self._details = self.lightcurve, etc.

            pass

    def save_lightcurve(self, output_folder=None):
        """
		Save generated lightcurve to file.

		Parameters:
		    output_folder (string, optional): Path to directory where to save lightcurve. If ``None`` the directory specified in the attribute ``output_folder`` is used.

		Returns:
		    string: Path to the generated file.
		"""
        
