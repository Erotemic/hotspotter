'''
This module tries to ensure that the system paths are correctly setup for
hotspotter to run.
'''
from __future__ import print_function, division
import sys
from os.path import join

# Macports python directories
ports_pyframework = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/'
ports_site_packages = join(ports_pyframework, 'lib/python2.7/site-packages/')


def ensure_pythonpath():
    if sys.platform == 'darwin':
        if not ports_pyframework in sys.path:
            sys.path.append(ports_site_packages)
