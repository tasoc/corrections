#!/usr/bin/env python
"""
The basic correction class for the TASOC Photomety pipeline
All other specific correction classes will inherit from BaseCorrection.
Structure from `BasePhotometry by Rasmus Handberg <https://github.com/tasoc/photometry/blob/devel/photometry/BasePhotometry.py>`_

- :py:class:`STATUS`: Status flags for pipeline performance logging
- :py:class:`STATUS`: Status flags for pipeline performance logging

.. codeauthor:: Lindsey Carboneau 
.. code author:: 
"""

# TODO: imports; must be included in requirements.txt
# from package import function as key
import enum
import logging

class STATUS(enum.enum):
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
    All other specific correction classes will inherit from BaseCorrection

    Attributes: 
        # TODO
    """

    def __init__():
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

    def status(self):
        """ The status of the corrections. From :py:class:`STATUS`."""
        return self._status
    
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

        self._status = self.do_correction(*args, **kwargs)

        # Check that the status has been changed:
        if self._status == STATUS.UNKNOWN:
            raise Exception("STATUS was not tset by do_correction")

        if self._status in (STATUS.OK, STATUS.WARNING):
            # TODO: set outputs; self._details = self.lightcurve, etc.

	def save_lightcurve(self, output_folder=None):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (string, optional): Path to directory where to save lightcurve. If ``None`` the directory specified in the attribute ``output_folder`` is used.

		Returns:
			string: Path to the generated file.
		"""