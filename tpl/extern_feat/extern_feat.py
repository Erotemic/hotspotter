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
def __precompute(rchip_fpath, chiprep_fpath, compute_fn):
    kpts, desc = compute_fn(rchip_fpath)
    np.savez(chiprep_fpath, kpts, desc)
    return kpts, desc

# TODO Dynamiclly add descriptor types
valid_extractors = ['sift', 'gloh']
valid_detectors = ['mser', 'hessaff']

def precompute_harris(rchip_fpath, chiprep_fpath):
    return __precompute(rchip_fpath, chiprep_fpath, __compute_harris)

def precompute_mser(rchip_fpath, chiprep_fpath):
    return __precompute(rchip_fpath, chiprep_fpath, __compute_mser)

def precompute_hesaff(rchip_fpath, chiprep_fpath):
    return __precompute(rchip_fpath, chiprep_fpath, __compute_hesaff)

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
    cmd = inria_args(rchip_fpath, detect_type, extract_type)
    __execute_extern(cmd)
    kpts, desc = __read_text_chiprep_file(outname)
    return kpts, desc

def __compute_mser(rchip_fpath):
    __compute_descriptors(rchip_fath, 'mser', 'sift')

def __compute_harris(rchip_fpath):
    __compute_descriptors(rchip_fath, 'harris', 'sift')

def __compute_hesaff(rchip_fpath):
    'Runs external keypoint detetectors like hesaff'
    outname = rchip_fpath + '.hesaff.sift'
    args = '"' + rchip_fpath + '"'
    cmd  = HESAFF_EXE + ' ' + args
    __execute_extern(cmd)
    kpts, desc = __read_text_chiprep_file(outname)
    return kpts, desc

#---------------------------------------

# Helper function to read external file formats
def __read_text_chiprep_file(outname):
    'Reads output from external keypoint detectors like hesaff'
    file = open(outname, 'r')
    # Read header
    ndims = int(file.readline())
    nkpts = int(file.readline())
    lines = file.readlines()
    file.close()
    # Preallocate output
    kpts = np.zeros((nkpts, 5), dtype=float32)
    desc = np.zeros((nkpts, ndims), dtype=uint8)
    for kx, line in enumerate(lines):
        data = line.split(' ')
        kpts[kx,:] = np.array([float32(_) for _ in data[0:5]], dtype=float32)
        desc[kx,:] = np.array([uint8(_) for _ in data[5: ]], dtype=uint8)
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
    harris
    hessian
    harmulti
    hesmulti
    harhesmulti
    harlap
    heslap
    dog
    mser
    haraff
    hesaff
    dense 6 6
    '''.strip().split('\n')]

    extract_type_list = ['sift','gloh']
    extract_type = 'sift'

    rchip_fpath = os.path.realpath('lena.png')

    for detect_type in detect_type_list:
        cmd = inria_cmd(rchip_fpath, detect_type, extract_type)
        __execute_extern(cmd+' -DP')


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
