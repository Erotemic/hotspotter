from other.logger import *
def get_namespace(fac):
    # DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS
    # DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS
    # DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS DEVELOPERS
    return \
'''from core.QueryManager import RawResults, QueryResult
from other.helpers import *
from PyQt4.Qt   import QString, QApplication, QMainWindow, QTableWidgetItem, QMessageBox, QAbstractItemView, QObject
from PyQt4.QtCore import SIGNAL, Qt, pyqtSlot, pyqtSignal
from PIL import Image
import types
# Get commonly used variables for command line usage
hs = fac.hs
uim = hs.uim
hsgui = uim.hsgui
if hsgui != None:
    epw = hsgui.epw
cm,  nm,  gm,  am,  dm,  vm,  qm,  iom,  em = hs.get_managers(
'cm','nm','gm','am','dm','vm','qm','iom','em')

qcx = 1'''
     


def force_grayscale_uint8_0_255(chip):
    from numpy import uint8, array, floor
    from skimage.color import rgb2gray
    if len(chip.shape) > 2: chip = rgb2gray(chip) # remove color
    if floor(chip.max()) <= 1: chip *= 255. # force to range 0,255
    if chip.dtype != np.uint8:
        chip = array(chip, dtype=uint8)
    return chip
    
def test_bilateral_filter(fac):
    from tpl.other.shiftableBF import shiftableBF
    cm = fac.hs.cm
    cx = cm.get_valid_cxs()[0]
    chip  = force_grayscale_uint8_0_255(cm.cx2_chip(cx))
    chip2 = shiftableBF(chip, sigmaS=1.6, sigmaR=40, tol=.01)

def preform_memory_dump(fpath):
    from meliae import scanner
    scanner.dump_all_objects( fpath ) 


def parallel_speed_test(hs):
    arg_list = [(x,) for x in xrange(1,20)]
    #method_fn, remove_pat = hs.precompute_chips, '*.png'
    method_fn, remove_pat = hs.precompute_chipreps, '*.npz'
    outlist = []
    for args in iter(arg_list):
        hs.iom.remove_computed_files_with_pattern(remove_pat)
        with Timer(outlist) as t:
            method_fn(*args)
        print('Times So Far: ', outlist)
    print('Output Times:')
    for outtup in zip(arg_list,outlist):
        print outtup
    
        
