from __future__ import division, print_function
import subprocess
import warnings
import numpy as np
import os, sys
from os.path import dirname, realpath, join
from PIL import Image
from numpy import uint8, float32, diag, sqrt, abs
from numpy.linalg import det
import cv2
DESC_FACTOR = 3.0*np.sqrt(3.0)

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

EXE_EXT = {'win32':'.exe', 'darwin':'.mac', 'linux2':'.ln'}[sys.platform]

if not '__file__' in vars():
    __file__ = os.path.realpath('extern_feat.py')
EXE_PATH = realpath(dirname(__file__))
if not os.path.exists(EXE_PATH):
    EXE_PATH = realpath('tpl/extern_feat')
if not os.path.exists(EXE_PATH):
    EXE_PATH = realpath('hotspotter/tpl/extern_feat')

HESAFF_EXE = join(EXE_PATH, 'hesaff'+EXE_EXT)
INRIA_EXE  = join(EXE_PATH, 'compute_descriptors'+EXE_EXT)

# Create directory for temporary files (if needed)
TMP_DIR = os.path.join(EXE_PATH, '.tmp_external_features') 
if not os.path.exists(TMP_DIR):
    print('Making directory: '+TMP_DIR)
    os.mkdir(TMP_DIR)
 
def svd(M):
    #U, S, V = np.linalg.svd(M)
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U,S,V

#---------------------------------------
# Define precompute functions
def precompute(rchip_fpath, feat_fpath, compute_fn):
    kpts, desc = compute_fn(rchip_fpath)
    np.savez(feat_fpath, kpts, desc)
    return kpts, desc

# TODO Dynamiclly add descriptor types
valid_extractors = ['sift', 'gloh']
valid_detectors = ['mser', 'hessaff']

def precompute_harris(rchip_fpath, feat_fpath):
    return precompute(rchip_fpath, feat_fpath, compute_harris)

def precompute_mser(rchip_fpath, feat_fpath):
    return precompute(rchip_fpath, feat_fpath, compute_mser)

def precompute_hesaff(rchip_fpath, feat_fpath):
    return precompute(rchip_fpath, feat_fpath, compute_hesaff)

#---------------------------------------
# Defined temp compute functions
def temp_compute(rchip, compute_fn):
    tmp_fpath = TMP_DIR + '/tmp.ppm'
    rchip_pil = Image.fromarray(rchip)
    rchip_pil.save(tmp_fpath, 'PPM')
    (kpts, desc) = compute_fn(tmp_fpath)
    return (kpts, desc)

def compute_hesaff(rchip):
    return temp_compute(rchip,compute_hesaff)

def compute_descriptors(rchip, detect_type, extract_type):
    return temp_compute(rchip, compute_hesaff)

#---------------------------------------
# Work functions which call the external feature detectors

# Helper function to call commands
def execute_extern(cmd):
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
    
def compute_inria_text_feats(rchip_fpath, detect_type, extract_type):
    'Runs external keypoint detetectors like hesaff'
    outname = rchip_fpath + '.'+detect_type+'.'+extract_type
    cmd = inria_cmd(rchip_fpath, detect_type, extract_type)
    execute_extern(cmd)
    return outname

def compute_perdoch_text_feats(rchip_fpath):
    outname = rchip_fpath + '.hesaff.sift'
    cmd  = hesaff_cmd(rchip_fpath)
    execute_extern(cmd)
    return outname

#----

def compute_inria_feats(rchip_fpath, detect_type, extract_type):
    '''Runs external inria detector
    detect_type = 'harris'
    extract_type = 'sift'
    '''
    outname = compute_inria_text_feats(rchip_fpath, detect_type, extract_type)
    kpts, desc = read_text_feat_file(outname)
    kpts = fix_kpts_hack(kpts)
    kpts, desc = filter_kpts_scale(kpts, desc)
    return kpts, desc

def compute_perdoch_feats(rchip_fpath):
    'Runs external perdoch detector'
    outname = compute_perdoch_text_feats(rchip_fpath)
    kpts, desc = read_text_feat_file(outname)
    kpts = fix_kpts_hack(kpts)
    kpts, desc = filter_kpts_scale(kpts, desc)
    return kpts, desc

#----

def compute_mser(rchip_fpath):
    return compute_inria_feats(rchip_fpath, 'mser', 'sift')
def compute_harris(rchip_fpath):
    return compute_inria_feats(rchip_fpath, 'harris', 'sift')
def compute_hesaff(rchip_fpath):
    return compute_perdoch_feats(rchip_fpath)

#---------------------------------------
# Helper function to read external file formats
def read_text_feat_file(outname, be_clean=True):
    'Reads output from external keypoint detectors like hesaff'
    file = open(outname, 'r')
    # Read header
    ndims = int(file.readline()) # assert ndims == 128
    nkpts = int(file.readline()) #
    lines = file.readlines()
    file.close()
    if be_clean:
        os.remove(outname)
    # Preallocate output
    kpts = np.zeros((nkpts, 5), dtype=float)
    desc = np.zeros((nkpts, ndims), dtype=uint8)
    for kx, line in enumerate(lines):
        data = line.split(' ')
        kpts[kx,:] = np.array([float32(_) for _ in data[0:5]], dtype=float32)
        desc[kx,:] = np.array([uint8(_) for _ in data[5: ]], dtype=uint8)
    return (kpts, desc)

from hotspotter import helpers
def filter_kpts_scale(kpts, desc, max_scale=250, min_scale=100):
    #max_scale=1E-3, min_scale=1E-7
    acd = kpts.T[2:5]
    det_ = acd[0] * acd[2]
    scale = sqrt(det_)
    print('scale.stats()=%r' % helpers.printable_mystats(scale))
    is_valid = np.bitwise_and(min_scale < scale, scale < max_scale).flatten()
    scale = scale[is_valid]
    kpts = kpts[is_valid]
    desc = desc[is_valid]
    print('scale.stats() = %s' % str(helpers.printable_mystats(scale)))
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
    A_list = [invX.dot(diag(1/sqrt(invW))) for (invX, invW, invYt) in invXWYt_list]
    # Flatten the shapes for fast rectification
    abcd  = np.vstack([A.flatten() for A in A_list])
    # Rectify up
    acd = rectify_up_abcd(abcd)
    kpts = np.vstack((xyT, acd.T)).T
    return kpts

def rectify_up_A(A):
    (a, b, c, d) = A.flatten()
    det_ = sqrt(abs(a*d - b*c))
    b2a2 = sqrt(b*b + a*a)
    a11 = b2a2 / det_
    a21 = (d*b + c*a) / (b2a2*det_)
    a22 = det_ / b2a2
    Aup = np.array(((a11,0),(a21,a22)))
    return Aup, det_

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

    rchip_fpath = os.path.realpath('zebra.png')

    for detect_type in detect_type_list:
        for extract_type in extract_type_list:
            cmd = inria_cmd(rchip_fpath, detect_type, extract_type)
            print('Execute: '+cmd)
            execute_extern(cmd+' -DP')

def expand_invET(invET):
    # Put the inverse elleq in a list of matrix structure
    e11 = invET[0]; e12 = invET[1]
    e21 = invET[1]; e22 = invET[2]
    invE_list = np.array(((e11, e12), (e21, e22))).T
    return invE_list

def expand_acd(acd):
    A_list = [np.array(((a,0),(c,d))) for (a,c,d) in acd]
    return A_list

def A_to_E(A):
    #U,S,Vt = svd(A)
    #E3 = Vt.dot(diag(S**2)).dot(Vt.T)
    E = A.dot(A.T)
    return E
def A_to_E2(A):
    U, S, Vt = svd(A)
    E = U.dot(diag(S**2)).dot(U.T)
    return E
def invE_to_E(invE):
    # This is just the pseudo inverse...
    # if m = n and A is full rank then, pinv(A) = inv(A)
    # if A is full rank. The pseudo-inverse for the case where A is not full
    # rank will be considered below
    #E = invX.dot(diag(1/invW[::-1])).dot(invYt)
    invX, invW, invYt = svd(invE)
    E = invX.dot(diag(1/invW)).dot(invYt)
    return E
def E_to_invE(E):
    X, W, Yt = svd(E)
    invE = X.dot(diag(1/W)).dot(Yt)
    return invE

def invE_to_A(invE, integrate_det=True):
    #_X * _W * _Yt = _E
    #(_X * sqrt(_W)) * (sqrt(_W) * _Yt) = _E
    #(_X * sqrt(_W)) * (sqrt(_W) * _Yt) = _E
    invX, invW, invYt = svd(invE)
    A = invX.dot(diag(1/sqrt(invW)))
    Aup, det_ = rectify_up_A(A)
    if integrate_det:
        A = Aup * det_
        return A
    else:
        return Aup, det_

def test_extract_hesaff():
  with warnings.catch_warnings():
    from hotspotter import helpers
    warnings.simplefilter("ignore")
    from hotspotter import params
    from os.path import join, exists
    img_dir = join(params.GZ, 'images')
    
    rchip_fpath = join(img_dir, 'NewHack_zimg-0000236.jpg')
    # ('NewHack_zimg-0000254.jpg')
    print(rchip_fpath)
    #if not exists(rchip_fpath):
        #rchip_fpath = os.path.realpath('lena.png')
    #rchip_fpath = os.path.realpath('zebra.jpg')
    rchip = cv2.cvtColor(cv2.imread(rchip_fpath, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    outname = compute_perdoch_text_feats(rchip_fpath)
    #outname = compute_inria_text_feats(rchip_fpath, 'harris', 'sift')
    # Keep the wrong way to compare
    kpts0, desc = read_text_feat_file(outname)
    kpts = fix_kpts_hack(kpts0[:], method=1)
    kpts, desc = filter_kpts_scale(kpts, desc)
    
    kpts0 = None
    keypoint_interaction(rchip, kpts, desc, kpts0)

def keypoint_interaction(rchip, kpts, desc=None, kpts0=None, fnum=1, **kwargs):
    from hotspotter import helpers
    import linalg_helpers as lh
    index = 20+37
    index = 0
    hprint = helpers.horiz_print

    kpts_CLEAN = kpts.copy()
    desc_CLEAN = desc.copy()
    if not kpts0 is None:
        kpts0_CLEAN = kpts0.copy()

    def keypoint_info(index):
        if not kpts0 is None:
            kp0 = kpts0[index]
        kp = kpts[index]
        print(kp)
        x,y,a,c,d = kp
        if not kpts0 is None:
            x0,y0,a0,b0,d0 = kp0
        A = np.array(([a,0],[c,d]))
        if not kpts0 is None:
            invE = np.array(([a0,b0],[b0,d0]))
        print('--kp info--')
        def kpts57_target():
            a = 1.03575
            b = 0
            c = -.883026
            d = .965485
            s = 2.38525
            return np.array(([a,b],[c,d]))

        if not kpts0 is None:
            At = kpts57_target()
            A_fE, s_fE= invE_to_A(invE, False)

        if not kpts0 is None:
            E = np.linalg.inv(invE)
            hprint('invE = ', invE)
            hprint('E = ', E)
        hprint('A = ', A)
        hprint('A.T.dot(A) = ', A.T.dot(A))
        hprint('A.dot(A.T) = ', A.dot(A.T))
        invA = np.linalg.inv(A)
        hprint('invA.dot(invA.T) = ', invA.dot(invA.T))
        hprint('invA.T.dot(invA) = ', invA.T.dot(invA))
        hprint('Qdet(A) = ', sqrt(det(A)))
        hprint('A*= ', At)
        hprint('A?= ', A_fE)
        def print_eig(A, name=''):
            print('[eig]==================')
            print('[eig] '+name+' EigenDecomp')
            evals, evecs = np.linalg.eig(A)
            hprint('[eig] '+name+' Evals=', evals)
            hprint('[eig] '+name+' Evecs=', evecs)
            print('[eig] '+name+' Unit-EigenDecomp')
            U,S,Vt = svd(A)
            A_ = U.dot(Vt)
            evals, evecs = np.linalg.eig(A_)
            hprint('[eig] '+name+' Unit-Evals=', evals)
            hprint('[eig] '+name+' Unit-Evecs=', evecs)
            print('[eig]==================')
        print_eig(A, '*A   *')
        print_eig(invA, '*invA*')
        if not kpts0 is None:
            print_eig(invE, '*invE*')
            print_eig(E, '*E*')
        print('--------------------------------------')
        # I have discovered A* is used as a transform from 
        # a unit_circle(x,y) ==> ellipse(x,y)

    def show_ith_patch(index):
        print('-------------------------------------------')
        print('[extern] show ith=%r patch' % index)
        is_valid = kpts_CLEAN.T[2]*kpts_CLEAN.T[4] > 30
        kpts = kpts_CLEAN[is_valid]
        desc = desc_CLEAN[is_valid]
        if not 'kpts0' in vars(): kpts0 = None

        if not kpts0 is None:
            kpts0 = kpts0_CLEAN[is_valid]

        kp = kpts[index]
        if not kpts0 is None:
            kp0 = kpts0[index]
        sift = desc[index]
        np.set_printoptions(precision=5)
        #keypoint_info(index)
        #
        df2.plt.cla()
        fig1 = df2.figure(fnum, **kwargs)
        df2.imshow(rchip, plotnum=(2,1,1))
        #df2.imshow(rchip, plotnum=(1,2,1), title='inv(sqrtm(invE*)')
        #df2.imshow(rchip, plotnum=(1,2,2), title='inv(A)')
        ell_args = {'ell_alpha':.4, 'ell_linewidth':1.8, 'rect':False}
        if not kpts0 is None:
            df2.draw_kpts2(kpts0, ell_color=df2.RED, wrong_way=True, **ell_args)
            df2.draw_kpts2(kpts0[index:index+1], ell_color=df2.YELLOW, wrong_way=True, **ell_args)
        df2.draw_kpts2(kpts, ell_color=df2.ORANGE, **ell_args)
        df2.draw_kpts2(kpts[index:index+1], ell_color=df2.BLUE, **ell_args)
        ax = df2.plt.gca()
        ax.set_title(str(index)+' old=b(inv(sqrtm(invE*)) and new=o(A=invA)')
        #
        extract_patch.draw_keypoint_patch(rchip, kp, sift, plotnum=(2,2,3))
        extract_patch.draw_keypoint_patch(rchip, kp, sift, warped=True, plotnum=(2,2,4))
        golden_wh = lambda x:map(int,map(round,(x*.618 , x*.312)))
        Ooo_50_50 = {'num_rc':(1,1), 'wh':golden_wh(1400*2)}
        #df2.present(**Ooo_50_50)
        #df2.update()
        fig1.show()
        fig1.canvas.draw()
        #df2.show()

    fig = df2.plt.figure(1)
    xy = kpts.T[0:2].T
    import pyflann
    # Flann doesn't help here at all
    use_flann = False
    flann_ptr = [None]
    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata))
        x,y = event.xdata, event.ydata
        if not use_flann:
            dist = (kpts.T[0] - x)**2 + (kpts.T[1] - y)**2
            index = dist.argsort()[0]
            show_ith_patch(index)
        else:
            flann, = flann_ptr
            if flann is None:
                flann = pyflann.FLANN()
                flann.build_index(xy, algorithm='kdtree', trees=1)
            query = np.array(((x,y)))
            knnx, kdist = flann.nn_index(query, 1, checks=8)
            index=knnx[0]
            show_ith_patch(index)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    show_ith_patch(index)

    #while True:
        #show_ith_patch(index)
        #index += 1
        #ans = raw_input('continue? (any key to quit)')
        #use_flann = True
        #if ans != '': break

if __name__ == '__main__':
    print('[TPL] Test Extern Features')
    from hotspotter import draw_func2 as df2
    from hotspotter import extract_patch as extract_patch
    df2.DARKEN = .5
    test_extract_hesaff()
    df2.show()
    #exec(df2.present())
