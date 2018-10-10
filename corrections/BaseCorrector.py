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
import numpy as np
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

    def __init__(self, starid, camera, ccd, eclon, eclat, input_folder, output_folder, plot=False):
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
        self.eclon = eclon
        self.eclat = eclat
        self.input_folder = input_folder

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

        Returns:
            Lightkurve object
        """
        logger = logging.getLogger(__name__)

        try:
            #TODO: Make this so that it reads in from wherever the files are stored
            data = np.loadtxt(starid+'.noisy')
            lightcurve = TessLightCurve(
                time=data[:,0],
                flux=data[:,1],
                flux_err=data[:,2],
                quality=np.asarray(data[:,3], dtype='int32'),
                time_format='jd',
                time_scale='tdb',
                targetid=task['starid'],
                camera=task['camera'],
                ccd=task['ccd'],
                meta = {'eclon':self.eclon, 'eclat':self.eclat}
            )
        except:
            logger.exception("Something happened")
            trace = traceback.format_exc().strip()
            try:
                corr._status = STATUS.ERROR
                corr.report_details(error=trace)
            except:
                pass
        return lightcurve

    def search_targets(eclon, eclat, radius, camera, ccd):
        """Return a list of targets within a search window
        Parameters:
            eclon (float): the ecliptic longitude of the target
            eclat (float): the ecliptic latitude of the target
            radius (float): the search radius around the target

        Returns:
            targetlist (ndarray): A list of targets within the search window

        Note: We don't actually do a radius search, we use the radius as half
        the width of the sides of a box search instead, for now.
        """
        #TODO Docstring
        #Upper and lower bounds on the current stamp
        eclon_min = np.round(eclon - radius,2)
        eclon_max = np.round(eclon + radius,2)
        eclat_min = np.round(eclat - radius,2)
        eclat_max = np.round(eclat + radius,2)

        conn = sqlite3.connect('{}/todo.sqlite'.format(__inputfolder__))
        cursor = conn.cursor()
        #TODO: Camera and CCD should not be hardcoded
        query = "SELECT todolist.starid FROM todolist INNER JOIN diagnostics ON todolist.priority = diagnostics.priority\
                WHERE camera = :camera AND ccd = :ccd AND mean_flux > 0\
                AND eclon BETWEEN :eclon_min AND :eclon_max\
                AND eclat BETWEEN :eclat_min AND :eclat_max;"

        if eclat_min < -90:
			# We are very close to the southern pole
			# Ignore everything about RA
			cursor.execute(query, {
                'camera'  : camera,
                'ccd'      : ccd,
				'eclon_min': 0,
				'eclon_max': 360,
				'eclat_min': -90,
				'eclat_max': eclat_max})
        elif eclat_max > 90:
			# We are very close to the northern pole
			# Ignore everything about RA
            cursor.execute(query, {
                'camera'  : camera,
                'ccd'      : ccd,
                'eclon_min': 0,
                'eclon_max': 360,
                'eclat_min': eclat_min,
                'eclat_max': 90})
        elif eclon_min < 0:
			cursor.execute("""SELECT todolist.starid FROM todolist INNER JOIN diagnostics ON todolist.priority = diagnostics.priority\
                    WHERE camera = :camera AND ccd = :camera AND mean_flux > 0\
                    WHERE eclon <= :eclon_max AND eclat BETWEEN :eclat_min AND :eclat_max\
			UNION
			SELECT todolist.starid FROM todolist INNER JOIN diagnostics ON todolist.priority = diagnostics.priority\
                    WHERE eclon BETWEEN :eclon_min AND 360\
                    AND eclat BETWEEN :eclat_min AND :eclat_max;""", {
                'camera'  : camera,
                'ccd'      : ccd,
				'eclon_min': 360 - abs(eclon_min),
				'eclon_max': eclon_max,
				'eclat_min': eclat_min,
				'eclat_max': eclat_max
			})
        elif eclon_max > 360:
			cursor.execute("""SELECT todolist.starid FROM todolist INNER JOIN diagnostics ON todolist.priority = diagnostics.priority\
                    WHERE eclon >= :eclon_min AND eclat BETWEEN :eclat_min AND :eclat_max
			UNION
			SELECT todolist.starid FROM todolist INNER JOIN diagnostics ON todolist.priority = diagnostics.priority\
                    WHERE eclon BETWEEN 0 AND :eclon_max AND eclat BETWEEN :eclat_min AND :eclat_max;""", {
                'camera'  : camera,
                'ccd'      : ccd,
				'eclon_min': eclon_min,
				'eclon_max': eclon_max - 360,
				'eclat_min': eclat_min,
				'eclat_max': eclat_max
			})
        else:
			cursor.execute(query, {
                'camera'  : camera,
                'ccd'      : ccd,
				'eclon_min': eclon_min,
				'eclon_max': eclon_max,
				'eclat_min': eclat_min,
				'eclat_max': eclat_max
			})

        #Output the list of target names
        targetlist = np.array(cursor.fetchall()).T[0]
        cursor.close()
        return targetlist

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
