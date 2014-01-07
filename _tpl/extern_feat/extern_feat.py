from __future__ import division, print_function
# Standard
import subprocess
#import warnings
import os
import sys
from os.path import dirname, realpath, join
# Scientific
import numpy as np
from numpy import diag, sqrt, abs
#from numpy.linalg import det
import cv2

OLD_HESAFF = False or '--oldhesaff' in sys.argv


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


def dict_has(dict_, flag_list):
    return any([dict_.has(flag) for flag in iter(flag_list)])
    # ['scale_min', 'scale_max']


#---------------------------------------
# Define precompute functions
def precompute(rchip_fpath, feat_fpath, dict_args, compute_fn):
    #if dict_has(dict_args, ['scale_min', 'scale_max']):
        #kpts, desc = compute_fn(rchip_fpath)
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
    if OLD_HESAFF:
        raise ImportError
    import pyhesaff
    print('using: new pyhesaff')

    def detect_kpts(rchip_fpath, dict_args):
        kpts, desc = pyhesaff.detect_hesaff_kpts(rchip_fpath, dict_args)
        # TODO: Move this into C++
        kpts, desc = filter_kpts_scale(kpts, desc, **dict_args)
        return kpts, desc
except ImportError:
    if '--strict' in sys.argv and not OLD_HESAFF:
        raise
    print('using: old hessian affine')

    def detect_kpts(rchip_fpath, dict_args):
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
    kpts = np.zeros((nkpts, 5), dtype=float)
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
    A_list = [invX.dot(diag(1 / sqrt(invW))) for (invX, invW, _invYt) in invXWYt_list]
    # Flatten the shapes for fast rectification
    abcd  = np.vstack([A.flatten() for A in A_list])
    # Rectify up
    acd = rectify_up_abcd(abcd)
    kpts = np.vstack((xyT, acd.T)).T
    return kpts


def rectify_up_abcd(abcd):
    (a, b, c, d) = abcd.T
    det2_  = sqrt(abs(a * d - b * c))
    b2a2 = sqrt(b * b + a * a)
    a11 = b2a2 / det2_
    a21 = (d * b + c * a) / (b2a2)
    a22 = det2_ * det2_ / b2a2
    acd = np.vstack([a11, a21, a22]).T
    return acd


def rotate_downwards(invA):
    (a_, b_,
     c_, d_)  = invA.flatten()  # abcd_.T
    det_      = np.abs(a_ * d_ - b_ * c_)  # idk why abs either (dets cant be negative?)
    mag_ab_   = np.sqrt(b_ ** 2 + a_ ** 2)
    idk_      = (d_ * b_ + c_ * a_)
    a = mag_ab_
    b = 0
    c = idk_ / (mag_ab_)
    d = det_ / mag_ab_
    Aup = np.array(((a, b), (c, d)))
    return Aup


def rectify_up_A(A):
    (a, b, c, d) = A.flatten()
    det_ = sqrt(abs(a * d - b * c))
    b2a2 = sqrt(b * b + a * a)
    a11 = b2a2 / det_
    a21 = (d * b + c * a) / (b2a2 * det_)
    a22 = det_ / b2a2
    Aup = np.array(((a11, 0), (a21, a22)))
    return Aup, det_


def expand_invET(invET):
    # Put the inverse elleq in a list of matrix structure
    e11 = invET[0]
    e12 = invET[1]
    e21 = invET[1]
    e22 = invET[2]
    invE_list = np.array(((e11, e12), (e21, e22))).T
    return invE_list


def expand_acd(acd):
    A_list = [np.array(((a, 0), (c, d))) for (a, c, d) in acd]
    return A_list


def A_to_E(A):
    #U,S,Vt = svd(A)
    #E3 = Vt.dot(diag(S**2)).dot(Vt.T)
    E = A.dot(A.T)
    return E


def A_to_E2(A):
    U, S, Vt = svd(A)
    E = U.dot(diag(S ** 2)).dot(U.T)
    return E


def invE_to_E(invE):
    # This is just the pseudo inverse...
    # if m = n and A is full rank then, pinv(A) = inv(A)
    # if A is full rank. The pseudo-inverse for the case where A is not full
    # rank will be considered below
    #E = invX.dot(diag(1/invW[::-1])).dot(invYt)
    invX, invW, invYt = svd(invE)
    E = invX.dot(diag(1 / invW)).dot(invYt)
    return E


def E_to_invE(E):
    X, W, Yt = svd(E)
    invE = X.dot(diag(1 / W)).dot(Yt)
    return invE


'''
Matrix Properties
http://en.wikipedia.org/wiki/Transpose

   (A + B).T = A.T + B.T
** (A * B).T = B.T * A.T
   (cA).T    = cA.T
   det(A.T)  = det(A)

Orthogonal Matrix Q
http://en.wikipedia.org/wiki/Orthogonal_matrix
Q.T = inv(Q)

'''


def invE_to_A(invE, integrate_det=True):
    '''
    Known: A is a transformation mapping points on the ellipse to points on a unit circle
    Known: E is symmetric
    Known: A.T * A = E
    '''
    '''
    eg.
    # From oxford.hesaff.sift
    invE = np.array([[0.00812996, 0.00553573], [0.00553573, 0.0159823]])
    Y, iW_, Xt = svd(invE)
    iW = diag(iW_)
    iS = sqrt(iW)
    invA = Y.dot(iS)
    A = inv(invA)
    E = inv(invE)
    # CAN CONFIRM THAT:
    np.all(np.abs(invA.dot(invA.T) - invE) < 1E-9)
    np.all(np.abs(A.T.dot(A) - E) < 1E-9)

    U, S_, V = svd(A)
    S = diag(S_)

    U.dot(inv(S)).dot(inv(S)).dot(U.T)

    # Ok, so they are not the same, but the eigvals are.
    # It looks like its just up to an arbitrary rotation.
    eig(U.dot(inv(S)).dot(inv(S)).dot(U.T))
    eig(invE)



    Because E is symetric X == Yt
    X, W, Yt = svd(E)

    # We really do get E as input and convert to inv(A)
    X *     W  * X.T = svd(E)
    X * inv(W) * X.T = svd(inv(E))

    U *      S * V.T  = svd(A)
    V * inv(S) * U.T = svd(inv(A))


    Known:
    A.T             *             A = E
    // Sub in for SVD(A)
    (U * S * V.T).T * (U * S * V.T) = E
    (V * S * U.T)   * (U * S * V.T) = E
    // Transpose Identity
    (V * S * U.T * U * S * V.T) = E
    // U is orthorogonal
    (V * S * S * V.T) = E
    // S is diagonal
    (V * S^2 * V.T) = E

    //Because
    (X * W * X.T) = E
    //Therefore
    X = V
    W = S^2

    //Therefore
    (X.T * 1/W * X) = inv(E)
    # Now thinking about A
    U * S * Vt = svd(A)
    A.T * A = E
    (U * S * Vt).T * (U * S * Vt) = E
    (V * S * U.T)  * (U * S * Vt) = E
    (V * S * S * Vt)              = E
    // Therefore
    V    = X
    S**2 = W
    S    = sqrt(W)
    // Therefore
    A = X * sqrt(W)

    # Now thinking about inv(A)
    Vt.T * inv(S) * U.T = svd(inv(A))
    Vt.T * inv(sqrt(W)) * U.T = svd(inv(A))
    X * inv(sqrt(W)) * U.T = svd(inv(A))
    '''
    # Let E be an ellipse equation
    # Let A go from ellipse to circle
    # Let inv(A) go from circle to ellipse
    #
    #X * W * Yt = svd(E)  // Definition of E
    #// E is symetric so X == Y
    #X * W * X.T = svd(E)
    #(X * sqrt(W))     * (sqrt(W) * X.T) = E
    #(X * sqrt(W)).T.T * (sqrt(W) * X.T) = E
    #(X * sqrt(W)).T   * (X * sqrt(W))   = E
    #
    # //Therefore
    # A = (X * sqrt(W)) = (Y.T * sqrt(W))
    #
    #X * W * Yt = svd(E)
    #Yt.T * inv(W) * X.T = svd(inv(E))
    #(Yt.T * inv(sqrt(W)))        * (inv(sqrt(W)) * X.T) = svd(inv(E))
    #(inv(sqrt(W)).T * Yt.T.T)).T * (inv(sqrt(W)) * X.T) = svd(inv(E))
    #(inv(sqrt(W)).T * Yt)).T     * (inv(sqrt(W)) * X.T) = svd(inv(E))
    X, W, Yt = svd(invE)
    A = X.dot(diag(1 / sqrt(W)))
    Aup, det_ = rectify_up_A(A)
    if integrate_det:
        A = Aup * det_
        return A
    else:
        return Aup, det_


if __name__ == '__main__':
    print('[TPL] Test Extern Features')
    import multiprocessing
    from hotspotter import draw_func2 as df2
    multiprocessing.freeze_support()
    df2.show()
    exec(df2.present())
