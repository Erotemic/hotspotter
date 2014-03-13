# Developer Functions
from PyQt4.Qt import QApplication, QMainWindow,\
        QMessageBox, QAbstractItemView, QObject, QInputDialog
from hsapi.QueryManager import RawResults, QueryResult
from PIL import Image
import types
import os
from os.path import expanduser, join, relpath, normpath, exists, dirname
from hsapi.util import *
from hsapi.other.AbstractPrintable import *
from hsapi.other.ConcretePrintable import *
from hsapi.other.logger import *
import hotspotter.ChipFunctions
import cv2
import matplotlib.pyplot as plt
from hsapi.ChipFunctions import normalize, read_oriented_chip

# Get commonly used variables for command line usage
hs    = fac.hs
qhs   = hs
uim   = hs.uim
hsgui = uim.hsgui
if hsgui != None:
    epw = hsgui.epw

cm = hs.cm
nm = hs.nm
gm = hs.gm 
am = hs.am
vm = hs.vm
dm = hs.dm
iom = hs.iom
em = hs.em

cx  = 1
qcx = 1

def indentLine(line, spaces=8):
    return (' ' * spaces) + string.lstrip(line)

def reindentBlock(s, numSpaces):
    s = string.split(s, '\n')
    s = map(lambda a, ns=numSpaces: indentLine(a, ns), s)
    s = string.join(s, '\n')
    return s

# Run some common snippets
import inspect
function_body_list = [hotspotter.ChipFunctions.compute_chiprep_args,
                      hotspotter.ChipFunctions.compute_chiprep]
function_body_list = []
for function_body in function_body_list:
    src_lines_all = inspect.getsource(function_body).split('\n')
    src_lines_ind = []
    for line in src_lines_all[1:]:
        if line.strip(' ').find('return') == 0:
            src_lines_ind.append(line.replace('return', 'pass # return'))
        elif line.strip(' ').find('#') == 0 or line.strip(' ') == '':
            pass
        else:
            src_lines_ind.append(line)

    line1 = src_lines_ind[0]
    indent1 = line1[:len(line1)-len(line1.lstrip())]

    src_lines = [line[len(indent1):] for line in src_lines_ind]
    src_code = '\n'.join(src_lines)
    print src_code
    exec(src_code)


def MSER_PREFS():
    hs.am.algo_prefs.chiprep.kpts_detector = 'MSER'
    import os, sys
    PYTHONPATH = os.environ['PYTHONPATH'].split(os.pathsep)
    PATH = os.environ['PATH'].split(os.pathsep)
    PATH = sys.path

'''
    def vd_HOTSPOTTER():
        vd(HOTSPOTTER_INSTALL_PATH)
    def vd_SITEPACKAGES():
        vd(SITEPACKAGES_PATH)
    def vd_OPENCV():
        vd(OPENCV_PATH)
'''

# TODO: Add to setup TPL
'''
OpenCV
Numpy 
Scipy
Matplotlib
Qt
msvcp
msvcr
LibFLANN
LibINRIA
LIB_GCC
LIB_SDC++
'''


