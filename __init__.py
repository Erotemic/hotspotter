# the hotspotter python module
from __future__ import division
print('Calling hotspotter.__init__')
#import scipy.ndimage.filters as filters

import hotspotter.experiments as experiments
import hotspotter.Pref as pref
import hotspotter.algos as algos
import hotspotter.chip_compute2 as cc2
import hotspotter.drawing_functions2 as df2
import hotspotter.helpers as helpers
import hotspotter.load_data2 as ld2
import hotspotter.load_data2 as load_data2
import hotspotter.match_chips2 as mc2
import hotspotter.params as params
import hotspotter.report_results2 as report_results2
import hotspotter.report_results2 as rr2
import hotspotter.spatial_verification2 as sv2
import hotspotter.tpl.extern_feat as extern_feat

import matplotlib.pyplot as plt

import skimage
import skimage.morphology
import skimage.filter.rank
import skimage.exposure
import skimage.util

import numpy as np
import os
import scipy.signal
import sys
import textwrap 
import warnings
import functools
import itertools
import cStringIO
import inspect
import imp
#1-866-2420

from os.path import join, relpath, realpath, normpath, dirname
from PIL import Image
from hotspotter.Parallelize import parallel_compute
from hotspotter.Printable import DynStruct
from hotspotter.helpers import *

#import scipy.ndimage.filters
__version__ = '1.9.9+'+repr(np.complex(0,.001))
__author__  = 'Jon Crall'
__email__   = 'hotspotter.ir@gmail.com'

DEVMODE = True
if DEVMODE:
    __file__ = realpath('../hotspotter/__init__.py')
HSDIR = dirname(__file__)

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

def hotspotter_modulenames():
    modpath_list = helpers.glob(HSDIR, '*.py')
    def just_name(path):
        return os.path.splitext(os.path.split(path)[1])[0]
    modname_list = [just_name(path) for path in modpath_list]
    return modname_list

def get_loaded_hotspotter_modules():
    hots_modnames = set(hotspotter_modulenames())
    hots_modlist = []
    for name, module in sys.modules.iteritems():
        print name
        if name in hots_modnames:
            hots_modlist.append(module)
        elif 'hotspotter.'+name in hots_modnames:
            hots_modlist.append(module)
    return hots_modlist

def reload_all_hotspotter_modules():
    hots_modlist = get_loaded_hotspotter_modules()
    for module in hots_modlist:
        print('reloading %r ' % (module,))
        imp.reload(module)
            
def reload():
    'Reloads all modules'
    pref.reload_module()
    algos.reload_module()
    cc2.reload_module()
    df2.reload_module()
    helpers.reload_module()
    ld2.reload_module()
    load_data2.reload_module()
    mc2.reload_module()
    params.reload_module()
    report_results2.reload_module()
    rr2.reload_module()
    sv2.reload_module()
    extern_feat.reload_module()
    #exec(open('helpers.py','r').read())

'''
__file__ = realpath('../hotspotter/__init__.py')
HSDIR = dirname(__file__)
'''


def rrr():
    'alias for reload'
    reload()

if __name__ == '__main__':
    experiments.demo()
