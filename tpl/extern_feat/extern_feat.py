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
    det = np.sqrt(np.abs(a*d - b*c))
    b2a2 = np.sqrt(b*b + a*a)
    a11 = b2a2 / det
    a12 = 0
    a21 = (d*b + c*a)/(b2a2*det)
    a22 = det/b2a2
    acd = np.vstack([a11, a21, a22]).T
    return acd

DESC_FACTOR = 3.0*np.sqrt(3.0)
from numpy.linalg import svd, det
from numpy import diag, sqrt, abs

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
    abcd_list = [(U.dot(diag(S[::-1]*sc)).dot(V)).flatten() for (sc, (U,S,V)) in zip(sc_list, USV_list)]
    abcd = np.vstack(abcd_list)

    # Enforce a lower triangular matrix
    acd = rectify_up_is_up(abcd)


def test_deintegrate_scale(invET):
    invE_list = expand_invET(invET)
    # Decompose using singular value decomposition
    USV_list = [svd(invE) for invE in invE_list]
    U_list, S_list, V_list = zip(*USV_list)
    A_list = expand_acd(acd)
    invE = invE_list[0]
    S = S_list[0]
    Sm = diag(S)
    sc = (1.0 / (sqrt(sqrt(S[0] * S[1]))))
    sigma = sc / DESC_FACTOR
    A_unit = (U.dot(diag(S[::-1]*sc)).dot(V)).flatten().reshape(2,2)
    U = U_list[0]
    V = V_list[0]
    A = A_list[0]
    import helpers
    helpers.rrr()
    hstr = helpers.horiz_string
    np.set_printoptions(precision=3)
    print('---- SVD of invE ----')
    print('invE =\n%s' % invE)
    print('= U * S * V =')
    print(hstr([U, ' * ', Sm, ' * ', V]))
    print('=')
    print(U.dot(Sm).dot(V))
    #print('1/det(invE) = %f' % (1/sqrt(sqrt(det(invE)))))
    print('---------------------\n')

    print('---- Scale Extraction ---')
    print('sc = 1.0 / (sqrt(sqrt(S[0] * S[1])))')
    print('sc = 1.0 / (sqrt(sqrt(%f * %f])))' % (S[0], S[1]))
    print('sc = %.3f' % sc)
    print('sigma = %.3f / DESC_FACTOR' % sc)
    print('sigma = %.3f / %.3f' % (sc, DESC_FACTOR))
    print('sigma = %.3f' % (sigma))
    print('---------------------\n')

    print('---- Rebuild Unit A ---')
    print('A = U * S[::-1]*sc * V')
    print('=')
    print(hstr([U, ' * ', diag(S[::-1]), '*' , ('%.3f' % sc), ' * ', V]))
    print('=')
    print(hstr([U, ' * ', diag(S[::-1]*sc), ' * ', V]))
    print('=')
    print(A_unit)
    print('--')
    print('sqrt(det(A)) = %.3f' % sqrt(det(A_unit)))
    import textwrap

    print('---- Recify A up is up ---')
    (a, b, c, d) = A_unit.flatten()
    det_ = sqrt(abs(a*d - b*c))
    b2a2 = sqrt(b*b + a*a)
    a11 = b2a2 / det_
    a12 = 0
    a21 = (d*b + c*a)/(b2a2*det_)
    a22 = det_/b2a2
    A_up = np.array(((a11, a12), (a21, a22)))
    print('det = sqrt(abs(a*d - b*c))')
    print('det = sqrt(abs(%.3f*%.3f - %.3f*%.3f))' % (a, b, c, d))
    print('det = %.3f' % det_)
    print('--')
    print('b2a2 = sqrt(b*b + a*a)')
    print('b2a2 = sqrt(%.3f*%.3f + %.3f*%.3f)' % (b, b, a, a))
    print('b2a2 = %.3f' % b2a2)
    print('--')
    print('A =')
    print(textwrap.dedent('''
    [[            b2a2 / det, 0       ],
     [(d*b + c*a)/(b2a2*det), det/b2a2]]'''))
    print('=')
    print(A_up)
    print('--')
    print('det(A) = %.3f' % det(A_up))

    print('A =\n%s' % (A,))
    print('Sm =\n%s' % (Sm,))
    print(Sm.dot(A))
    print(A.dot(Sm))

    print('---- Check things ---')
    #AScaleUnit = A_up.dot(
    


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

if __name__ == '__main__':
    import cv2
    test_inria_feats()
    #rchip_fpath = os.path.realpath('lena.png')

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
