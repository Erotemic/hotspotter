# Developer Functions
from PyQt4.Qt import QApplication, QMainWindow,\
        QMessageBox, QAbstractItemView, QObject, QInputDialog
from hotspotter.QueryManager import RawResults, QueryResult
from PIL import Image
import types
import os
from os.path import expanduser, join, relpath, normpath, exists, dirname
from hotspotter.helpers import *
from hotspotter.other.AbstractPrintable import *
from hotspotter.other.ConcretePrintable import *
from hotspotter.other.logger import *

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
