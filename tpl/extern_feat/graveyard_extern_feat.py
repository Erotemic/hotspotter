from __future__ import division, print_function
import subprocess
import numpy as np
import os, sys
from os.path import dirname, realpath, join
from PIL import Image
from numpy import uint8, float32
#import threading
#__hesaff_lock = threading.Lock()

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
# Defined temp compute functions
# Create directory for temporary files (if needed)
#TMP_DIR = os.path.join(EXE_PATH, '.tmp_external_features') 
#if not os.path.exists(TMP_DIR):
    #print('Making directory: '+TMP_DIR)
    #os.mkdir(TMP_DIR)
#def temp_compute(rchip, compute_fn):
    #tmp_fpath = TMP_DIR + '/tmp.ppm'
    #rchip_pil = Image.fromarray(rchip)
    #rchip_pil.save(tmp_fpath, 'PPM')
    #(kpts, desc) = compute_fn(tmp_fpath)
    #return (kpts, desc)
#def compute_perdoch(rchip, dict_args):
    #return temp_compute(rchip,compute_hesaff)
#def compute_inria(rchip, detect_type, extract_type):
    #return temp_compute(rchip, compute_hesaff)

 
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

def inria_cmd(rchip_fpath, detect_type, extract_type):
    detect_arg  = '-'+detect_type
    extract_arg = '-'+extract_type
    input_arg   = '-i "' + rchip_fpath + '"'
    other_args  = '-noangle'
    args = INRIA_EXE + ' ' + ' '.join([input_arg, detect_arg,
                                       extract_arg, other_args])
    return args

def __compute_descriptors(rchip_fpath, detect_type, extract_type):
    'Runs external keypoint detetectors like hesaff'
    outname = rchip_fpath + '.'+detect_type+'.'+extract_type
    cmd = inria_cmd(rchip_fpath, detect_type, extract_type)
    __execute_extern(cmd)
    kpts, desc = __read_text_feat_file(outname)
    return kpts, desc

def __compute_mser(rchip_fpath):
    __compute_descriptors(rchip_fpath, 'mser', 'sift')

def __compute_harris(rchip_fpath):
    __compute_descriptors(rchip_fpath, 'harris', 'sift')

def __compute_hesaff(rchip_fpath):
    'Runs external keypoint detetectors like hesaff'
    outname = rchip_fpath + '.hesaff.sift'
    args = '"' + rchip_fpath + '"'
    cmd  = HESAFF_EXE + ' ' + args
    print(cmd)
    __execute_extern(cmd)
    kpts, desc = __read_text_feat_file(outname)
    return kpts, desc

#---------------------------------------

def rectify_up_is_up(abcd):
    abcdT = abcd.T
    (a, b, c, d) = abcdT
    # Logic taken from Perdoch's code
    sqrt_det = np.sqrt(np.abs(a*d - b*c))
    sqrt_b2a2 = np.sqrt(b*b + a*a)
    a11 = sqrt_b2a2 / sqrt_det
    a12 = 0
    a21 = (d*b + c*a)/(sqrt_b2a2*sqrt_det)
    a22 = sqrt_det/sqrt_b2a2
    acd = np.vstack([a11, a21, a22]).T
    return acd, sqrt_det

DESC_FACTOR = 3.0*np.sqrt(3.0)
from numpy.linalg import svd, det, inv
from numpy import diag, sqrt, abs
import numpy.linalg as npla

def expand_invET(invET):
    # Put the inverse elleq in a list of matrix structure
    e11 = invET[0]; e12 = invET[1]
    e21 = invET[1]; e22 = invET[2]
    invE_list = np.array(((e11, e12), (e21, e22))).T
    return invE_list

def expand_acd(acd):
    A_list = [np.array(((a,0),(c,d))) for (a,c,d) in acd]
    return A_list

def convert_invE_to_abcd(invET):
    ''' Transforms: 
        [E_a, E_b]        [A_a,   0]
        [E_b, E_d]  --->  [A_c, A_d]
    '''
    invE_list = expand_invET(invET)

    # Decompose using singular value decomposition
    USV_list = [svd(invE) for invE in invE_list]
    U_list, S_list, V_list = zip(*USV_list)
    # Deintegrate the scale
    sc_list = [1.0 / (sqrt(sqrt(S[0] * S[1]))) for S in S_list]
    sigma_list = [sc / DESC_FACTOR for sc in sc_list]

    # Rebuild the ellipse -> circle matrix
    abcd_list = [(U.dot(diag(sqrt(S[::-1])*sc)).dot(V)).flatten() for (sc, (U,S,V)) in zip(sc_list, USV_list)]
    abcd = np.vstack(abcd_list)

    # Enforce a lower triangular matrix
    acd = rectify_up_is_up(abcd)
    


# Helper function to read external file formats
def __read_text_feat_file(outname):
    'Reads output from external keypoint detectors like hesaff'
    file = open(outname, 'r')
    # Read header
    ndims = int(file.readline())
    nkpts = int(file.readline())
    lines = file.readlines()
    file.close()
    # Preallocate output
    kpts = np.zeros((nkpts, 5), dtype=float)
    desc = np.zeros((nkpts, ndims), dtype=uint8)
    for kx, line in enumerate(lines):
        data = line.split(' ')
        kpts[kx,:] = np.array([float32(_) for _ in data[0:5]], dtype=float32)
        desc[kx,:] = np.array([uint8(_) for _ in data[5: ]], dtype=uint8)
    # Hack to put things into acd foramat
    invET = kpts.T[2:5]

    #
    acd = kpts.T[2:5]
    det = acd[0] * acd[2]
    is_valid = np.bitwise_and(det.T < 1E-3, det.T > 1E-7)
    kpts = kpts[is_valid.flatten()]
    desc = desc[is_valid.flatten()]
    return (kpts, desc)

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

def test_deintegrate_scale(invET):
    '''
    %run feature_compute2.py
    __file__ = 'tpl/extern_feat/extern_feat.py'
    exec(open('tpl/extern_feat/extern_feat.py'))

    outname = rchip_fpath + '.hesaff.sift'
    args = '"' + rchip_fpath + '"'
    cmd  = HESAFF_EXE + ' ' + args
    print(cmd)
    __execute_extern(cmd)

    file = open(outname, 'r')
    # Read header
    ndims = int(file.readline())
    nkpts = int(file.readline())
    lines = file.readlines()
    file.close()
    # Preallocate output
    kpts = np.zeros((nkpts, 5), dtype=float)
    desc = np.zeros((nkpts, ndims), dtype=uint8)
    for kx, line in enumerate(lines):
        data = line.split(' ')
        kpts[kx,:] = np.array([float32(_) for _ in data[0:5]], dtype=float32)
        desc[kx,:] = np.array([uint8(_) for _ in data[5: ]], dtype=uint8)
    # Hack to put things into acd foramat
    invET = kpts.T[2:5]

    invE = array([[ 0.00226126, -0.00166135],
                  [-0.00166135,  0.0224354 ]])
    '''

    def A_from_E(E=None, invE=None):
        if E is None: 
            E = inv(invE)
        # convert E to A
        det_invE = np.linalg.det(invE)
        det_E = np.linalg.det(E)
        #
        invX,invW,invYt = np.linalg.svd(invE)
        E = invX.dot(diag(1/invW[::-1])).dot(invYt)

        X,W,Yt = np.linalg.svd(E)
        invE2 = X.dot(diag(1/W[::-1])).dot(Yt)

        invX,invW,invYt = np.linalg.svd(invE2)
        E2 = invX.dot(diag(1/invW[::-1])).dot(invYt)

        X,W,Yt = np.linalg.svd(E2)
        invE3 = X.dot(diag(1/W[::-1])).dot(Yt)

        A = invX.dot(diag(1/sqrt(invW[::-1])))

        U,S,Vt = np.linalg.svd(A)
        E3 = Vt.dot(diag(S**2)).dot(Vt.T)
        E3 = A.dot(A.T)
        print(E3)
        print(E2)
        #invE4 = A.T.dot(A)
        #invE4 = U.dot(diag(S)).dot(U.T)
        #invE4 = U.dot(diag(S[::-1])).dot(Vt.T)
        #invE4 = V.dot(diag(S[::-1])).dot(Vt.T)
        #invE4 = U.T.dot(diag(S[::-1])).dot(Vt.T)
        #print(invE)
        #print(invE4)

        (a, b, c, d) = A.flatten()
        det_2 = sqrt(abs(a*d - b*c))
        b2a2 = sqrt(b*b + a*a)
        a11 = b2a2 / det_2
        a12 = 0
        a21 = (d*b + c*a)/(b2a2*det_2)
        a22 = det_2/b2a2
        Aup = np.array(((a11, a12), (a21, a22)))
        print(Aup.dot(Aup.T))
        U,S,Vt = np.linalg.svd(Aup)
        E4 = U.dot(diag(det_2**2 * S**2)).dot(U.T)
        print(E4)
        print(E3)
        print(E)



        #invA = U.dot(diag(1/S)).dot(Vt)
        #invS1 = 1/S[::-1]

        #invU,invS,invVt = np.linalg.svd(invA)
        #A2 = invV.T.dot(diag()).dot(invU.T)
        print(E)
        print(E2)
        print(E3)
        print('--')
        print(invE)
        print(invE2)
        print(invE3)
        print(invE4)
        print('---')
        print(A)
        print(A2)
        
        

        sigma = np.linalg.det(S)
        print('E')
        print(E)
        print(invE)
        print(A)
        print(invA)
        print(det_invE)
        print(det_E)


        U = X
        S = np.diag(np.sqrt(W))
        A = U.dot(S)
        det_A = np.linalg.det(A)
        print(det_A)
        U,S,Vt = np.linalg.svd(invA)
        
        #Breakpoint 1
        #isE1 = A.dot(A.T)
        #print('=====')
        #print('isE1? svd(A.dot(A.T))=')
        #print(isE1)
        #print(E)
        #print('----')
        #print('\n'.join(map(str, list(svd(isE)))))
        #print('----')
        A = A.dot(np.eye(2)/(det*det))
        det_ = np.linalg.det(A)
        def recify_up(Au):
            (a, b, c, d) = Au.flatten()
            det_2 = sqrt(abs(a*d - b*c))
            b2a2 = sqrt(b*b + a*a)
            a11 = b2a2 / det_2
            a12 = 0
            a21 = (d*b + c*a)/(b2a2*det_2)
            a22 = det_2/b2a2
            Aup = np.array(((a11, a12), (a21, a22)))
            A   = Aup * sqrt(det_2)

        print(isE)
        print(E)
        assert all(abs(isE - E) < 1E18)
        return A, Au, det_2
    A, Au, det = A_from_E(invE=invE)


    invET 
    import helpers
    import textwrap
    helpers.rrr()
    invE_list = expand_invET(invET)
    # Decompose using singular value decomposition
    invE = invE_list[0]

    hstr = helpers.horiz_string
    np.set_printoptions(precision=8)
    import cv2
    def print_2x2_svd(M, name=''):
        #S, U, V = cv2.SVDecomp(M, flags=cv2.SVD_FULL_UV)
        #S = S.flatten()
        #print(hstr([U,S,V]))
        U, S, V = np.linalg.svd(M)
        #print(hstr([U,S,V]))
        # Try and conform to opencv
        Sm = diag(S)
        print('---- SVD of '+name+' ----')
        print(name+' =\n%s' % M)
        print('= U * S * V =')
        print(hstr([U, ' * ', Sm, ' * ', V]))
        print('=')
        print(U.dot(Sm).dot(V))
        print('--')
        # SVD is not rotation, scale rotation...
        # That is only for a shear matrix
        asin = np.arcsin; acos = np.arccos
        thetaU11 = acos(U[0,0])
        thetaU12 = -asin(U[0,1])
        thetaU21 = asin(U[1,0])
        thetaU22 = acos(U[1,1])
        print([thetaU11,thetaU12,thetaU21,thetaU22])
        thetaV11 = acos(V.T[0,0])
        thetaV12 = -asin(V.T[0,1])
        thetaV21 = asin(V.T[1,0])
        thetaV22 = acos(V.T[1,1])
        print([thetaV11,thetaV12,thetaV21,thetaV22])
        thetaU = thetaU11*360/(2*np.pi)
        thetaV = thetaV11*360/(2*np.pi)
        print('theta_U = %r' % thetaU)
        print('theta_V = %r' % thetaV)
        print('---------------------\n')
        return U, S, V, Sm
    U, S, V, Sm = print_2x2_svd(invE, 'invE')

    def print_extract_scale(S):
        sc = (1.0 / (sqrt(sqrt(S[0] * S[1]))))
        sigma = sc / DESC_FACTOR
        print('---- Scale Extraction ---')
        print('sc = 1.0 / (sqrt(sqrt(S[0] * S[1])))')
        print('sc = 1.0 / (sqrt(sqrt(%f * %f])))' % (S[0], S[1]))
        print('sc = %.3f' % sc)
        print('sigma = %.3f / DESC_FACTOR' % sc)
        print('sigma = %.3f / %.3f' % (sc, DESC_FACTOR))
        print('sigma = %.3f' % (sigma))
        print('---------------------\n')
        return sc, sigma
    sc, sigma = print_extract_scale(S)

    def print_reconstruct_unit(U, S, V, sc, name=''):
        flip_Sm = diag(S[::-1])
        Sm_unit = diag(sqrt(S[::-1])*sc)
        M_unit = (U.dot(Sm_unit).dot(V)).flatten().reshape(2,2)
        print('---- Reconstruct Unit A ---')
        print('sc = %.3f' % sc)
        print(name+'_unit = U * sqrt(S[::-1])*sc * V')
        print('=')
        print(hstr([U, ' * ', flip_Sm, '*' , ('%.3f' % sc), ' * ', V]))
        print('=')
        print(hstr([U, ' * ', Sm_unit, ' * ', V]))
        print('=')
        print(M_unit)
        print('--')
        print('sqrt(det('+name+'_unit)) = %.3f' % sqrt(det(M_unit)))
        print('---------------------\n')
        return M_unit
    A_unit = print_reconstruct_unit(U, S, V, sc, name='A')

    def print_rectify_up(M, name=''):
        print('---- Recify '+name+' up is up ---')
        (a, b, c, d) = M.flatten()
        print(name+' =\n%s' % M)
        det_ = sqrt(abs(a*d - b*c))
        b2a2 = sqrt(b*b + a*a)
        a11 = b2a2 / det_
        a12 = 0
        a21 = (d*b + c*a)/(b2a2*det_)
        a22 = det_/b2a2
        M_up = np.array(((a11, a12), (a21, a22)))
        print('det = sqrt(abs(a*d - b*c))')
        print('det = sqrt(abs(%.3f*%.3f - %.3f*%.3f))' % (a, b, c, d))
        print('det = %.3f' % det_)
        print('--')
        print('b2a2 = sqrt(b*b + a*a)')
        print('b2a2 = sqrt(%.3f*%.3f + %.3f*%.3f)' % (b, b, a, a))
        print('b2a2 = %.3f' % b2a2)
        print('--')
        print('A ='+textwrap.dedent('''
        [[             b2a2/det, 0       ],
        [(d*b + c*a)/(b2a2*det), det/b2a2]]'''))
        print('= '+name+'_up = ')
        print(M_up)
        print('--')
        print('det('+name+'_up) = %.3f' % det(M_up))
        print('---------------------\n')
        return M_up
    A_up = print_rectify_up(A_unit, 'A')
    A = A_up

    def print_integrate_scale(M, sc, name):
        U, S, V = svd(M)
        #S, U, V = cv2.SVDecomp(M)
        #S = S.flatten()
        scaled_Sm = diag(S) / sc
        print('---- Integrate scale into '+name+' ----')
        print('sc = %.3f' % sc)
        print(name+' =\n%s' % M)
        print(name+'\' = U * (S/sc) * V =')
        print(hstr([U, ' * ', scaled_Sm, ' * ', V]))
        print('=')
        scaled_M = U.dot(scaled_Sm).dot(V)
        scaled_M = helpers.correct_zeros(scaled_M)
        print(scaled_M)
        print('---------------------\n')
        return scaled_M

    scaled_M = print_integrate_scale(A, sc, 'A')
    print(hstr(('A.T.dot(A) =', str(A.T.dot(A)))))
    print(hstr(('A.dot(A.T) =', str(A.dot(A.T)))))
    print_2x2_svd(A, 'A')

    print('A =\n%s' % (A,))
    print('Sm =\n%s' % (Sm,))
    print(Sm.dot(A))
    print(A.dot(Sm))

    print('---- Check things ---')
    #AScaleUnit = A_up.dot(
if __name__ == '__main__':
    #import cv2
    #test_inria_feats()
    rchip_fpath = 'tpl/extern_feat/lena.png'
    rchip_fpath = os.path.realpath(rchip_fpath)

'''
Interest points:
     -harris - harris detector
     -hessian - hessian detector
     -harmulti - multi-scale harris detector
     -hesmulti - multi-scale hessian detector
     -harhesmulti - multi-scale harris-hessian detector
     -harlap - harris-laplace detector
     -heslap - hessian-laplace detector
     -dog    - DoG detector
     -mser   - mser detector
     -haraff - harris-affine detector
     -hesaff - hessian-affine detector
     -harhes - harris-hessian-laplace detector
     -dense dx dy - dense sampling
Interest points parameters:
     -density 100 - feature density per pixels (1 descriptor per 100pix)
     -harThres - harris threshold [100]
     -hesThres  - hessian threshold [200]
     -edgeLThres  - lower canny threshold [5]
     -edgeHThres  - higher canny threshold [10]
 Descriptors:
     -sift - sift [D. Lowe]
     -gloh - gloh [KM]
 Descriptor paramenters:
     -color - color sift [KM]
     -dradius - patch radius for computing descriptors at scale 1
     -fface ..../facemodel.dat - frontal face detector
Input/Output:
     -i image.png  - input image pgm, ppm, png, jpg, tif
     -p1 image.pgm.points - input regions format 1
     -p2 image.pgm.points - input regions format 2
     -o1 out.desc - saves descriptors in out.desc output format 1
     -o2 out.desc - saves descriptors in out.desc output format 2
     -noangle - computes rotation variant descriptors (no rotation esimation)
     -DP - draws features as points in out.desc.png
     -DC - draws regions as circles in out.desc.png
     -DE - draws regions as ellipses in out.desc.png
     -c 255 - draws points in grayvalue [0,...,255]
     -lparams params.par - load parameter settings from file
     -sparams params.par - save parameter settings to file
     -pca input.basis - projects the descriptors with pca basis
example:       compute_descriptors.exe -sift -i image.png -p1 image.png.points -DR
               compute_descriptors.exe -harlap -sift -i image.png  -DC -pca harhessift.basis

               compute_descriptors.exe -harhes -sift -color -i image.png  -DC

               compute_descriptors.exe -params har.params -i image.png


--------------------
 file format 2:
#comments: x y cornerness scale=patch_size angle object_index point_type laplacian_value extremum_type mi11 mi12 mi21 mi
22 ...sift descriptor
m_nb_of_descriptors_in file
k_number_of_parameters
n_descriptor_dimension
p1_1 ... p1_k d1_1 d1_2 d1_3 ... d1_n
:

pm_1 ... pm_k dm_1 dm_2 dm_3 ... dm_n


--------------------
 file format 1:
n_descriptor_dimension
m_nb_of_descriptors_in file
 y1 a1 b1 c1 desc1_1 desc1_2 ......desc1_descriptor_dimension
:
 ym am bm cm descm_1 descm_2 ......descm_descriptor_dimension
--------------------

where a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1

 file format 2:
vector_dimension
nb_of_descriptors
x y cornerness scale/patch_size angle object_index  point_type laplacian_value extremum_type mi11 mi12 mi21 mi22 desc_1
...... desc_vector_dimension
--------------------

distance=(descA_1-descB_1)^2+...+(descA_vector_dimension-descB_vector_dimension)^2

 input.basis format:
nb_of_dimensions
mean_v1
mean_v2
.
.
mean_vnb_of_dimensions
nb_of_dimensions*nb_of_pca_vectors
pca_vector_v1
pca_vector_v2
.
.
--------------------

'''
