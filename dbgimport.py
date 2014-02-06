from __future__ import division, print_function
# Python
from collections import OrderedDict
from os.path import (dirname, realpath, join, exists, normpath, splitext,
                     expanduser)
import imp
import itertools
from itertools import izip, chain
from itertools import product as iprod
import multiprocessing
import os
import re
import sys
import site
import shutil
# Matplotlib
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
# Scientific
import numpy as np  # NOQA
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
# Qt
import PyQt4
from PyQt4 import QtCore, QtGui
from PyQt4.Qt import (QAbstractItemModel, QModelIndex, QVariant, QWidget,
                      Qt, QObject, pyqtSlot, QKeyEvent)
# HotSpotter
from hotspotter import Config
from hotspotter import DataStructures as ds
from hotspotter import HotSpotterAPI
from hotspotter import HotSpotterAPI as api
from hotspotter import algos
from hotspotter import chip_compute2 as cc2
from hotspotter import feature_compute2 as fc2
from hotspotter import load_data2 as ld2
from hotspotter import match_chips3 as mc3
from hotspotter import matching_functions as mf
from hotspotter import segmentation
from hotspotter import voting_rules2 as vr2
from hotspotter import nn_filters
from hscom import tools
from hscom import Parallelize as parallel
from hscom import cross_platform as cplat
from hscom import fileio as io
from hscom import helpers as helpers
from hsviz import draw_func2 as df2
from hsviz import extract_patch
from hsviz import viz
import dev

if __name__ == 'main':
    multiprocessing.freeze_support()
    #exec(open('dbgimport.py').read())
    if '--img' in sys.argv:
        img_fpath_ = helpers.dbg_get_imgfpath()
        np_img_ = io.imread(img_fpath_)
        img_ = Image.open(img_fpath_)
