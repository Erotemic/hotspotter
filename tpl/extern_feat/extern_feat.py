from __future__ import division, print_function
import subprocess
import numpy as np
import os, sys
from os.path import dirname, realpath, join
from PIL import Image
from numpy import uint8, float32, diag, sqrt, abs
import numpy.linalg as npla
DESC_FACTOR = 3.0*np.sqrt(3.0)

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

EXE_EXT = {'win32':'.exe', 'darwin':'.mac', 'linux2':'.ln'}[sys.platform]

#__file__ = os.path.realpath('external_feature_interface.py')
EXE_PATH = realpath(dirname(__file__))
try: # for debugging
    __IPYTHON__
    EXE_PATH = realpath('tpl/extern_feat')
except Exception as ex:
    pass

HESAFF_EXE = join(EXE_PATH, 'hesaff'+EXE_EXT)
INRIA_EXE  = join(EXE_PATH, 'compute_descriptors'+EXE_EXT)

# Create directory for temporary files (if needed)
TMP_DIR = os.path.join(EXE_PATH, '.tmp_external_features') 
if not os.path.exists(TMP_DIR):
    print('Making directory: '+TMP_DIR)
    os.mkdir(TMP_DIR)
 
#---------------------------------------
# Define precompute functions
def __precompute(rchip_fpath, feat_fpath, compute_fn):
    kpts, desc = compute_fn(rchip_fpath)
    np.savez(feat_fpath, kpts, desc)
    return kpts, desc

# TODO Dynamiclly add descriptor types
valid_extractors = ['sift', 'gloh']
valid_detectors = ['mser', 'hessaff']

def precompute_harris(rchip_fpath, feat_fpath):
    return __precompute(rchip_fpath, feat_fpath, __compute_harris)

def precompute_mser(rchip_fpath, feat_fpath):
    return __precompute(rchip_fpath, feat_fpath, __compute_mser)

def precompute_hesaff(rchip_fpath, feat_fpath):
    return __precompute(rchip_fpath, feat_fpath, __compute_hesaff)

#---------------------------------------
# Defined temp compute functions
def __temp_compute(rchip, compute_fn):
    tmp_fpath = TMP_DIR + '/tmp.ppm'
    rchip_pil = Image.fromarray(rchip)
    rchip_pil.save(tmp_fpath, 'PPM')
    (kpts, desc) = compute_fn(tmp_fpath)
    return (kpts, desc)

def compute_hesaff(rchip):
    return __temp_compute(rchip, __compute_hesaff)

def compute_descriptors(rchip, detect_type, extract_type):
    return __temp_compute(rchip, __compute_hesaff)

#---------------------------------------
# Work functions which call the external feature detectors

# Helper function to call commands
def __execute_extern(cmd):
    #print('tpl.execute_extern> '+cmd)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if proc.returncode != 0:
        raise Exception('\n'.join(['* External detector returned 0',
                                   '* Failed calling: '+cmd,
                                   '* Process output: ',
                                   '------------------',
                                   out,
                                   '------------------']))

def inria_cmd(rchip_fpath, detect_type, extract_type):
    ''' -noangle causes a crash on windows '''
    detect_arg  = '-'+detect_type
    extract_arg = '-'+extract_type
    input_arg   = '-i "' + rchip_fpath + '"'
    other_args  = ''# '-noangle'
    cmd = ' '.join([INRIA_EXE, input_arg, detect_arg, extract_arg, other_args])
    return cmd

def hesaff_cmd(rchip_fpath):
    args = '"' + rchip_fpath + '"'
    cmd  = HESAFF_EXE + ' ' + args
    return cmd

#----
    
def __compute_inria_text_feats(rchip_fpath, detect_type, extract_type):
    'Runs external keypoint detetectors like hesaff'
    outname = rchip_fpath + '.'+detect_type+'.'+extract_type
    cmd = inria_cmd(rchip_fpath, detect_type, extract_type)
    __execute_extern(cmd)
    return outname

def __compute_perdoch_text_feats(rchip_fpath):
    outname = rchip_fpath + '.hesaff.sift'
    cmd  = hesaff_cmd(rchip_fpath)
    __execute_extern(cmd)
    return outname

#----

def __compute_inria_feats(rchip_fpath, detect_type, extract_type):
    '''Runs external inria detector
    detect_type = 'harris'
    extract_type = 'sift'
    '''
    outname = __compute_inria_text_feats(rchip_fpath, detect_type, extract_type)
    kpts, desc = __read_text_feat_file(outname)
    kpts = fix_kpts_hack(kpts)
    #kpts, desc = filter_kpts_scale(kpts, desc)
    return kpts, desc

def __compute_perdoch_feats(rchip_fpath):
    'Runs external perdoch detector'
    outname = __compute_perdoch_text_feats(rchip_fpath)
    kpts, desc = __read_text_feat_file(outname)
    kpts = fix_kpts_hack(kpts)
    #kpts, desc = filter_kpts_scale(kpts, desc)
    return kpts, desc

#----

def __compute_mser(rchip_fpath):
    return __compute_inria_feats(rchip_fpath, 'mser', 'sift')
def __compute_harris(rchip_fpath):
    return __compute_inria_feats(rchip_fpath, 'harris', 'sift')
def __compute_hesaff(rchip_fpath):
    return __compute_perdoch_feats(rchip_fpath)

#---------------------------------------
# Helper function to read external file formats
def __read_text_feat_file(outname):
    'Reads output from external keypoint detectors like hesaff'
    file = open(outname, 'r')
    # Read header
    ndims = int(file.readline()) # assert ndims == 128
    nkpts = int(file.readline()) #
    lines = file.readlines()
    file.close()
    # Preallocate output
    kpts = np.zeros((nkpts, 5), dtype=float)
    desc = np.zeros((nkpts, ndims), dtype=uint8)
    for kx, line in enumerate(lines):
        data = line.split(' ')
        kpts[kx,:] = np.array([float32(_) for _ in data[0:5]], dtype=float32)
        desc[kx,:] = np.array([uint8(_) for _ in data[5: ]], dtype=uint8)
    return (kpts, desc)

def filter_kpts_scale(kpts, desc, max_scale=1E-3, min_scale=1E-7):
    #max_scale=1E-3, min_scale=1E-7
    acd = kpts.T[2:5]
    det_ = acd[0] * acd[2]
    is_valid = np.bitwise_and(det_ > min_scale, det_ < max_scale).flatten()
    kpts = kpts[is_valid]
    desc = desc[is_valid]
    return kpts, desc

def fix_kpts_hack(kpts, method=1):
    'Hack to put things into acd foramat'
    xyT   = kpts.T[0:2]
    invET = kpts.T[2:5]
    acd = convert_invET_to_acd(invET, method=method)
    kpts = np.vstack((xyT, acd.T)).T
    return kpts

def convert_invET_to_acd(invET, method=1):
    ''' Transforms: 
        [E_a, E_b]        [A_a,   0]
        [E_b, E_d]  --->  [A_c, A_d]
    '''
    # Expand into full matrix
    invE_list = expand_invET(invET)
    # Decompose using singular value decomposition
    invXWYt_list = [svd(invE) for invE in invE_list]
    # Rebuild the ellipse -> circle matrix
    if method == 1:
        A_list = [invX.dot(diag(sqrt(invW))) for (invX, invW, invYt) in invXWYt_list]
    if method == 2:
        A_list = [invX.dot(diag(1/sqrt(invW))) for (invX, invW, invYt) in invXWYt_list]
    # Flatten the shapes for fast rectification
    abcd  = np.vstack([A.flatten() for A in A_list])
    # Rectify up
    acd = rectify_up_abcd(abcd)
    return acd

def rectify_up_abcd(abcd):
    (a, b, c, d) = abcd.T
    det_ = sqrt(abs(a*d - b*c))
    b2a2 = sqrt(b*b + a*a)
    a11 = b2a2 / det_
    a21 = (d*b + c*a) / (b2a2*det_)
    a22 = det_ / b2a2
    acd = np.vstack([det_*a11, det_*a21, det_*a22]).T
    return acd
#---------

def test_inria_feats():
    detect_type_list = [_.strip() for _ in '''
    harris, hessian, harmulti, hesmulti,
    harhesmulti, harlap, heslap, dog, 
    mser, haraff, hesaff, dense 6 6
    '''.strip(' \n').split(',')]

    extract_type_list = ['sift','gloh']
    extract_type = 'sift'

    rchip_fpath = os.path.realpath('lena.png')

    for detect_type in detect_type_list:
        for extract_type in extract_type_list:
            cmd = inria_cmd(rchip_fpath, detect_type, extract_type)
            print('Execute: '+cmd)
            __execute_extern(cmd+' -DP')

def expand_invET(invET):
    # Put the inverse elleq in a list of matrix structure
    e11 = invET[0]; e12 = invET[1]
    e21 = invET[1]; e22 = invET[2]
    invE_list = np.array(((e11, e12), (e21, e22))).T
    return invE_list

def expand_acd(acd):
    A_list = [np.array(((a,0),(c,d))) for (a,c,d) in acd]
    return A_list

def test_keypoint_extraction():
    '''
    __file__ = 'tpl/extern_feat/extern_feat.py'
    exec(open('tpl/extern_feat/extern_feat.py'))
    sc = 12.0158; s = 2.31244
    A = np.array([( 1.79978, 0), (0.133274, 0.555623)])
    array([[ 21.62579652,   0.        ], [  1.60139373,   6.67625484]])
    invE = np.array([[ 0.00226126, -0.00166135], [-0.00166135,  0.0224354 ]])
    '''
    import draw_func2 as df2
    import cv2
    import helpers
    rchip = cv2.cvtColor(cv2.imread(rchip_fpath, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    outname = __compute_perdoch_text_feats(rchip_fpath)
    #outname = __compute_inria_text_feats(rchip_fpath, 'harris', 'sift')
    # Keep the wrong way to compare
    kpts0, desc = __read_text_feat_file(outname)
    invE = expand_invET(kpts0[:,2:5].T)[0]
    kpts1 = fix_kpts_hack(kpts0[:], method=1)
    A1 = expand_acd(kpts1[:,2:5])[0]
    kpts2 = fix_kpts_hack(kpts0[:], method=2)
    A2 = expand_acd(kpts2[:,2:5])[0]
    #kpts, desc = filter_kpts_scale(kpts, desc)

    # Dare to compare
    df2.figure(1, doclf=True)
    df2.imshow(rchip, plotnum=(1,3,1), title='0')
    df2.draw_kpts2(kpts0, wrong_way=True)
    df2.imshow(rchip, plotnum=(1,3,2), title='1')
    df2.draw_kpts2(kpts1, wrong_way=False)
    df2.imshow(rchip, plotnum=(1,3,3), title='2')
    df2.draw_kpts2(kpts2, wrong_way=False)

    # TEST
    hprint = helpers.horiz_print
    invA1 = inv(A1)
    invA2 = inv(A2)
    hprint('invE = ', invE)
    hprint('A1 = ', A1)
    hprint('A2 = ', A2)
    hprint('invA1 = ', invA1)
    hprint('invA2 = ', invA2)

def A_to_E(A):
    #U,S,Vt = np.linalg.svd(A)
    #E3 = Vt.dot(diag(S**2)).dot(Vt.T)
    E = A.dot(A.T)
    return E
def A_to_E2(A):
    U, S, Vt = np.linalg.svd(A)
    E = U.dot(diag(S**2)).dot(U.T)
    return E
def invE_to_E(invE):
    # This is just the pseudo inverse...
    # if m = n and A is full rank then, pinv(A) = inv(A)
    # if A is full rank. The pseudo-inverse for the case where A is not full
    # rank will be considered below
    #E = invX.dot(diag(1/invW[::-1])).dot(invYt)
    invX, invW, invYt = np.linalg.svd(invE)
    E = invX.dot(diag(1/invW)).dot(invYt)
    return E
def E_to_invE(E):
    X, W, Yt = np.linalg.svd(E)
    invE = X.dot(diag(1/W)).dot(Yt)
    return invE

def invE_to_A(invE):
    #_X * _W * _Yt = _E
    #(_X * sqrt(_W)) * (sqrt(_W) * _Yt) = _E
    #(_X * sqrt(_W)) * (sqrt(_W) * _Yt) = _E
    invX, invW, invYt = np.linalg.svd(invE)
    A = invX.dot(diag(1/sqrt(invW)))
    Aup, det_ = rectify_up(A)
    A = Aup * det_
    return A

if __name__ == '__main__':
    #import cv2
    #test_inria_feats()
    rchip_fpath = 'tpl/extern_feat/lena.png'
    rchip_fpath = os.path.realpath(rchip_fpath)
