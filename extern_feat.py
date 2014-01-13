from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[extern_feat]')
# Standard
import os
import sys
from os.path import dirname, realpath, join, expanduser, exists, split
# Scientific
import numpy as np

OLD_HESAFF = False or '--oldhesaff' in sys.argv
if '--newhesaff' in sys.argv:
    OLD_HESAFF = False


def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

EXE_EXT = {'win32': '.exe', 'darwin': '.mac', 'linux2': '.ln'}[sys.platform]

if not '__file__' in vars():
    __file__ = os.path.realpath('extern_feat.py')
EXE_PATH = realpath(dirname(__file__))
if not os.path.exists(EXE_PATH):
    EXE_PATH = realpath('tpl/extern_feat')
if not os.path.exists(EXE_PATH):
    EXE_PATH = realpath('hotspotter/tpl/extern_feat')

HESAFF_EXE = join(EXE_PATH, 'hesaff' + EXE_EXT)
INRIA_EXE  = join(EXE_PATH, 'compute_descriptors' + EXE_EXT)

KPTS_DTYPE = np.float64
DESC_DTYPE = np.uint8


#---------------------------------------
# Define precompute functions
def precompute(rchip_fpath, feat_fpath, dict_args, compute_fn):
    kpts, desc = compute_fn(rchip_fpath, dict_args)
    np.savez(feat_fpath, kpts, desc)
    return kpts, desc


def precompute_hesaff(rchip_fpath, feat_fpath, dict_args):
    return precompute(rchip_fpath, feat_fpath, dict_args, compute_hesaff)


#---------------------------------------
# Work functions which call the external feature detectors
# Helper function to call commands
try:
    import _tpl.extern_feat.pyhesaff as pyhesaff

    def detect_kpts_new(rchip_fpath, dict_args):
        kpts, desc = pyhesaff.detect_kpts(rchip_fpath, **dict_args)
        return kpts, desc
    print('[extern_feat] new hessaff is available')
except ImportError as ex:
    print('[extern_feat] new hessaff is not available: %r' % ex)
    if '--strict' in sys.argv:
        raise

try:
    import _tpl.extern_feat.pyhesaffexe as pyhesaffexe

    def detect_kpts_old(rchip_fpath, dict_args):
        kpts, desc = pyhesaffexe.detect_kpts(rchip_fpath, **dict_args)
        return kpts, desc
    print('[extern_feat] old hessaff is available')
except ImportError as ex:
    print('[extern_feat] old hessaff is not available: %r' % ex)
    if '--strict' in sys.argv:
        raise


if OLD_HESAFF:
    detect_kpts = detect_kpts_old
    print('[extern_feat] using: old hessian affine')
else:
    detect_kpts = detect_kpts_new
    print('[extern_feat] using: new pyhesaff')


#----
def compute_hesaff(rchip_fpath, dict_args):
    return detect_kpts(rchip_fpath, dict_args)


if __name__ == '__main__':
    print('[TPL] Test Extern Features')
    import multiprocessing
    multiprocessing.freeze_support()

    def ensure_hotspotter():
        import matplotlib
        matplotlib.use('Qt4Agg', warn=True, force=True)
        # Look for hotspotter in ~/code
        hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
        if not exists(hotspotter_dir):
            print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
        # Append hotspotter location (not dir) to PYTHON_PATH (i.e. sys.path)
        hotspotter_location = split(hotspotter_dir)[0]
        sys.path.append(hotspotter_location)

    # Import hotspotter io and drawing
    ensure_hotspotter()
    from hotspotter import draw_func2 as df2
    from hotspotter import vizualizations as viz
    from hotspotter import fileio as io

    # Read Image
    img_fpath = realpath('lena.png')
    image = io.imread(img_fpath)

    def spaced_elements(list_, n):
        indexes = np.arange(len(list_))
        stride = len(indexes) // n
        return list_[indexes[0:-1:stride]]

    def test_detect(n=None, fnum=1, old=True):
        try:
            # Select kpts
            detect_kpts_func = detect_kpts_old if old else detect_kpts_new
            kpts, desc = detect_kpts_func(img_fpath, {})
            kpts_ = kpts if n is None else spaced_elements(kpts, n)
            desc_ = desc if n is None else spaced_elements(desc, n)
            # Print info
            np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
            print('----')
            print('detected %d keypoints' % len(kpts))
            print('drawing %d/%d kpts' % (len(kpts_), len(kpts)))
            print(kpts_)
            print('----')
            # Draw kpts
            viz.interact_keypoints(image, kpts_, desc_, fnum)
            #df2.imshow(image, fnum=fnum)
            #df2.draw_kpts2(kpts_, ell_alpha=.9, ell_linewidth=4,
                           #ell_color='distinct', arrow=True, rect=True)
            df2.set_figtitle('old' if old else 'new')
        except Exception as ex:
            import traceback
            traceback.format_exc()
            print(ex)
        return locals()

    test_detect(n=10, fnum=1, old=True)
    test_detect(n=10, fnum=2, old=False)
    exec(df2.present())
