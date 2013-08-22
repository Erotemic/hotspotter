# the hotspotter python module
from __future__ import division
print('Calling hotspotter2.__init__')
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
import hotspotter.tpl.external_features as extern_feats

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

from os.path import join, relpath, realpath, normpath
from PIL import Image
from hotspotter.Parallelize import parallel_compute
from hotspotter.Printable import DynStruct
from hotspotter.helpers import ensure_path, mystats, myprint

#import scipy.ndimage.filters

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

__version__ = '1.9.9+'+repr(np.complex(0,.001))
__author__  = 'Jon Crall'
__email__   = 'hotspotter.ir@gmail.com'

def rrr():
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
    extern_feats.reload_module()

def rrr():
    'alias for reload'
    reload()

if __name__ == '__main__':
    experiments.demo()
