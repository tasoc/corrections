# -*- coding: utf-8 -*-
import os
from numpy import get_include
import pyximport

if os.name == 'nt':
	if os.environ.get('CPATH'):
		os.environ['CPATH'] = os.environ['CPATH'] + get_include()
	else:
		os.environ['CPATH'] = get_include()

	# XXX: we're assuming that MinGW is installed in C:\MinGW (default)
	mingw_paths = (r'C:\MinGW\bin', r'C:\MinGW32-xy\bin')
	for mingw_path in mingw_paths:
		if os.path.exists(mingw_path) and not mingw_path in os.environ['PATH'].split(';'):
			if os.environ.get('PATH'):
				os.environ['PATH'] = os.environ['PATH'] + ';' + mingw_path
			else:
				os.environ['PATH'] = mingw_path
			break

	mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }
	#pyximport.install(setup_args=mingw_setup_args)
	pyximport.install()

elif os.name == 'posix':
	if os.environ.get('CFLAGS'):
		os.environ['CFLAGS'] = os.environ['CFLAGS'] + ' -I' + get_include()
	else:
		os.environ['CFLAGS'] = ' -I' + get_include()

	pyximport.install()
