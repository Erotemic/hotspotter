# the hotspotter python module
print('Calling hotspotter2.__init__')
import hotspotter.drawing_functions2 as df2
import hotspotter.algos as algos
import hotspotter.helpers as helpers
import hotspotter.load_data2 as load_data2
import hotspotter.report_results2 as report_results2
import hotspotter.match_chips2 as mc2
import hotspotter.params as params

from hotspotter.Parallelize import parallel_compute
from hotspotter.Printable import DynStruct

import matplotlib.pyplot as plt

import numpy as np
import os
import sys
from os.path import join, relpath, realpath, normpath

import textwrap 

from PIL import Image

__version__ = '1.9.9+'+repr(np.complex(0,.001))
__author__  = 'Jon Crall'
__email__   = 'hotspotter.ir@gmail.com'

def rrr():
    algos.reload_module()
    params.reload_module()
    load_data2.reload_module()
