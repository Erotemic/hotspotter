from __future__ import division, print_function
# Python
import itertools  # NOQA
import textwrap  # NOQA
import itertools
import re
import sys
import textwrap
from os.path import join
from collections import OrderedDict
# Scientific
import numpy as np  # NOQA
# Qt
import PyQt4
from PyQt4 import QtCore, QtGui
from PyQt4.Qt import (QAbstractItemModel, QModelIndex, QVariant, QWidget,
                      Qt, QObject, pyqtSlot, QKeyEvent)
# HotSpotter
import DataStructures as ds
import HotSpotter
import Parallelize as parallel
import algos
import chip_compute2 as cc2
import dev
import draw_func2 as df2  # NOQA
import feature_compute2 as fc2
import fileio as io
import helpers as helpers
import load_data2 as ld2
import match_chips3 as mc3
import matching_functions as mf
import voting_rules2 as vr2
import vizualizations as viz

if __name__ == 'main':
    exec(open('dbgimport.py').read())

#img_fpath = helpers.get_img_fpath()
