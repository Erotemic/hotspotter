from __future__ import division, print_function
# Standard
import subprocess
#import warnings
import os
import sys
from os.path import dirname, realpath, join, expanduser, exists, split
# Scientific
import numpy as np
from numpy import diag, sqrt, abs
#from numpy.linalg import det
import cv2

# Set to use old hesaff for the time being
OLD_HESAFF = True or '--oldhesaff' in sys.argv   # override for chuck
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


def svd(M):
    #U, S, V = np.linalg.svd(M)
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U, S, V


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
def execute_extern(cmd):
    'Executes a system call'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if proc.returncode != 0:
        raise Exception('\n'.join(['* External detector returned 0',
                                   '* Failed calling: ' + cmd, '* Process output: ',
                                   '------------------', out, '------------------']))

try:
    import pyhesaff

    def detect_kpts_new(rchip_fpath, dict_args):
        kpts, desc = pyhesaff.detect_hesaff_kpts(rchip_fpath, dict_args)
        # TODO: Move this into C++
        kpts, desc = filter_kpts_scale(kpts, desc, **dict_args)
        return kpts, desc
    print('[extern_feat] new hessaff is available')
except ImportError:
    print('[extern_feat] new hessaff is not available')
    if '--strict' in sys.argv:
        raise


def detect_kpts_old(rchip_fpath, dict_args):
    'Runs external perdoch detector'
    outname = rchip_fpath + '.hesaff.sift'
    hesaff_exe = join(EXE_PATH, 'hesaff' + EXE_EXT)
    args = '"' + rchip_fpath + '"'
    cmd  = hesaff_exe + ' ' + args
    execute_extern(cmd)
    kpts, desc = read_text_feat_file(outname)
    if len(kpts) == 0:
        return np.empty((0, 5), dtype=KPTS_DTYPE), np.empty((0, 5), dtype=DESC_DTYPE)
    kpts = fix_kpts_hack(kpts)
    kpts, desc = filter_kpts_scale(kpts, desc, **dict_args)
    return kpts, desc

if OLD_HESAFF:
    detect_kpts = detect_kpts_old
    print('[extern_feat] using: old hessian affine')
else:
    detect_kpts = detect_kpts_new
    print('[extern_feat] using: new pyhesaff')


#----
def compute_hesaff(rchip_fpath, dict_args):
    return detect_kpts(rchip_fpath, dict_args)


#---------------------------------------
# Helper function to read external file formats
def read_text_feat_file(outname, be_clean=True):
    'Reads output from external keypoint detectors like hesaff'
    file = open(outname, 'r')
    # Read header
    ndims = int(file.readline())  # assert ndims == 128
    nkpts = int(file.readline())  #
    lines = file.readlines()
    file.close()
    if be_clean:
        os.remove(outname)
    # Preallocate output
    kpts = np.zeros((nkpts, 5), dtype=KPTS_DTYPE)
    desc = np.zeros((nkpts, ndims), dtype=DESC_DTYPE)
    for kx, line in enumerate(lines):
        data = line.split(' ')
        kpts[kx, :] = np.array([KPTS_DTYPE(_) for _ in data[0:5]], dtype=KPTS_DTYPE)
        desc[kx, :] = np.array([DESC_DTYPE(_) for _ in data[5:]],  dtype=DESC_DTYPE)
    return (kpts, desc)


def filter_kpts_scale(kpts, desc, scale_max=None, scale_min=None, **kwargs):
    #max_scale=1E-3, min_scale=1E-7
    #from hotspotter import helpers
    if len(kpts) == 0 or \
       scale_max is None or scale_min is None or\
       scale_max < 0 or scale_min < 0 or\
       scale_max < scale_min:
        return kpts, desc
    acd = kpts.T[2:5]
    det_ = acd[0] * acd[2]
    scale = sqrt(det_)
    #print('scale.stats()=%r' % helpers.printable_mystats(scale))
    #is_valid = np.bitwise_and(scale_min < scale, scale < scale_max).flatten()
    is_valid = np.logical_and(scale_min < scale, scale < scale_max).flatten()
    #scale = scale[is_valid]
    kpts = kpts[is_valid]
    desc = desc[is_valid]
    #print('scale.stats() = %s' % str(helpers.printable_mystats(scale)))
    return kpts, desc


def fix_kpts_hack(kpts, method=1):
    ''' Transforms:
        [E_a, E_b]        [A_a,   0]
        [E_b, E_d]  --->  [A_c, A_d]
    '''
    'Hack to put things into acd foramat'
    xyT   = kpts.T[0:2]
    invET = kpts.T[2:5]
    # Expand into full matrix
    invE_list = expand_invET(invET)
    # Decompose using singular value decomposition
    invXWYt_list = [svd(invE) for invE in invE_list]
    # Rebuild the ellipse -> circle matrix
    A_list = [invX.dot(diag(1 / sqrt(invW))) for (invX, invW, invYt) in invXWYt_list]
    # Flatten the shapes for fast rectification
    abcd  = np.vstack([A.flatten() for A in A_list])
    # Rectify up
    acd = rectify_up_abcd(abcd)
    kpts = np.vstack((xyT, acd.T)).T
    return kpts


def rectify_up_abcd(abcd):
    '''
    Based on:
    void rectifyAffineTransformationUpIsUp(float &a11, float &a12, float &a21, float &a22)
    {
    double a = a11, b = a12, c = a21, d = a22;
    double det = sqrt(abs(a*d-b*c));
    double b2a2 = sqrt(b*b + a*a);
    a11 = b2a2/det;             a12 = 0;
    a21 = (d*b+c*a)/(b2a2*det); a22 = det/b2a2;
    }
    '''
    #(a, b, c, d) = abcd.T
    #absdet_ = abs(a * d - b * c)
    #sqtdet_ = sqrt(absdet_)
    #b2a2 = sqrt(b * b + a * a)
    #a11 = sqtdet_ * b2a2
    #a21 = (d * b + c * a) / (b2a2)
    #a22 = absdet_ / b2a2
    #acd = np.vstack([a11, a21, a22]).T
    (a, b, c, d) = abcd.T
    absdet_ = abs(a * d - b * c)
    sqtdet_ = sqrt(absdet_)
    b2a2 = sqrt(b * b + a * a)
    # Build rectified ellipse matrix
    a11 = b2a2 / sqtdet_
    a21 = (d * b + c * a) / (sqtdet_ * b2a2)
    a22 = sqtdet_ / b2a2
    acd = np.vstack([sqtdet_ * a11, sqtdet_ * a21, sqtdet_ * a22]).T
    return acd


#---------
def expand_invET(invET):
    # Put the inverse elleq in a list of matrix structure
    e11 = invET[0]
    e12 = invET[1]
    e21 = invET[1]
    e22 = invET[2]
    invE_list = np.array(((e11, e12), (e21, e22))).T
    return invE_list


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
