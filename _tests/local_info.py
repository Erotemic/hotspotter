from __future__ import division, print_function
import sys
import os
from os.path import dirname, join, expanduser, exists, split


def ensure_hotspotter():
    hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
    if not exists(hotspotter_dir):
        print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
    hotspotter_location = split(hotspotter_dir)[0]
    sys.path.append(hotspotter_location)
ensure_hotspotter()

from hotspotter.dbgimport import *  # NOQA

helpers.printvar2('cv2.__version__')
helpers.printvar2('multiprocessing.cpu_count()')
helpers.printvar2('sys.platform')
helpers.printvar2('os.getcwd()')
