from __future__ import print_function, division
import sys

macports_site_packages = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/'
if sys.platform == 'darwin':
    if not macports_site_packages in sys.path:
        sys.path.append(macports_site_packages)
