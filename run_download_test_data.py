#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script that will download data for extensive tests.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import requests
import tqdm
import zipfile

#--------------------------------------------------------------------------------------------------
def download_file(url, destination):
	"""
	Download file from URL and place into specified destination.

	Parameters:
		url (string): URL to file to be downloaded.
		destination (string): Path where to save file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	print("Downloading %s" % url)
	try:
		response = requests.get(url, stream=True, allow_redirects=True)

		# Throw an error for bad status codes
		response.raise_for_status()

		total_size = int(response.headers.get('content-length', 0))
		block_size = 1024
		with open(destination, 'wb') as handle:
			with tqdm.tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
				for block in response.iter_content(block_size):
					handle.write(block)
					pbar.update(len(block))

		if os.path.getsize(destination) != total_size:
			raise RuntimeError("File not downloaded correctly")

	except: # noqa: E722
		if os.path.exists(destination):
			os.remove(destination)
		raise

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	input_dir = os.path.join(os.path.dirname(__file__), 'tests', 'input')
	zip_path = os.path.join(input_dir, 'corrections_tests_input.zip')

	# Download the ZIP file if it doesn't already exists:
	if not os.path.exists(zip_path):
		download_file('https://tasoc.dk/pipeline/corrections_tests_input_v2.zip', zip_path)

	# Extract files into the input directory:
	print("Extracting files...")
	with zipfile.ZipFile(zip_path) as myzip:
		for member in tqdm.tqdm(myzip.infolist()):
			myzip.extract(member, path=input_dir)

	# Create dummy file that will indicate that data is available
	with open(os.path.join(input_dir, 'test_data_available_v2.txt'), 'w') as fid:
		fid.write("Test data has been downloaded")

	# Delete the ZIP file again:
	print("Cleaning up after ourselves...")
	os.remove(zip_path)

	print("Done.")
