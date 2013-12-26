import ctypes
#from ctypes.util import find_library
import numpy as np
from os.path import join, exists, abspath, dirname, normpath
import sys


def get_lib_fname_list(libname):
    'returns possible library names given the platform'
    if sys.platform == 'win32':
        libnames = ['lib' + libname + '.dll', libname + '.dll']
    elif sys.platform == 'darwin':
        libnames = ['lib' + libname + '.dylib']
    elif sys.platform == 'linux2':
        libnames = ['lib' + libname + '.so']
    else:
        raise Exception('Unknown operating system: %s' % sys.platform)
    return libnames


def get_lib_dpath_list(root_dir):
    'returns possible lib locations'
    get_lib_dpath_list = [root_dir,
                          join(root_dir, 'lib'),
                          join(root_dir, 'build'),
                          join(root_dir, 'build', 'lib')]
    return get_lib_dpath_list


def find_lib_fpath(libname, root_dir, recurse_down=True):
    lib_fname_list = get_lib_fname_list(libname)
    while root_dir is not None:
        for lib_fname in lib_fname_list:
            for lib_dpath in get_lib_dpath_list(root_dir):
                lib_fpath = normpath(join(lib_dpath, lib_fname))
                #print('testing: %r' % lib_fpath)
                if exists(lib_fpath):
                    print('using: %r' % lib_fpath)
                    return lib_fpath
            _new_root = dirname(root_dir)
            if _new_root == root_dir:
                root_dir = None
                break
            else:
                root_dir = _new_root
        if not recurse_down:
            break
    raise ImportError('Cannot find dynamic library.')


def load_library2(libname, rootdir):
    lib_fpath = find_lib_fpath(libname, root_dir)
    try:
        clib = ctypes.cdll[lib_fpath]
    except Exception as ex:
        print('Caught exception: %r' % ex)
        raise ImportError('Cannot load dynamic library. Did you compile FLANN?')
    return clib

#def load_hesaff_lib():
# LOAD LIBRARY
root_dir = abspath(dirname(__file__))
libname = 'hesaff'
hesaff_lib = load_library2(libname, root_dir)

# Define types
ctype_flags = 'aligned, c_contiguous'
# numpy dtypes
kpts_dtype = np.float64
desc_dtype = np.uint8
# ctypes
voidp_t = ctypes.c_void_p
int_t = ctypes.c_int
kpts_t = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=ctype_flags)
desc_t = np.ctypeslib.ndpointer(dtype=desc_dtype, ndim=2, flags=ctype_flags)
intp_t = ctypes.POINTER(ctypes.c_long)
cstring_t = ctypes.c_char_p

# Test
hesaff_lib.make_hesaff.restype   = voidp_t
hesaff_lib.make_hesaff.argtypes  = [cstring_t]
hesaff_lib.detect.argtypes       = [voidp_t, intp_t]
hesaff_lib.exportArrays.argtypes = [voidp_t, kpts_t, desc_t]
hesaff_lib.extractDesc.argtypes  = [voidp_t, int_t, kpts_t, desc_t]


def hesaff_detect(img_fpath):
    # Make detector and read image
    hesaff_ptr = hesaff_lib.make_hesaff(abspath(img_fpath))
    # Return the number of keypoints detected
    nKpts_ptr = ctypes.pointer(ctypes.c_long(0))
    hesaff_lib.detect(hesaff_ptr, nKpts_ptr)
    nKpts = nKpts_ptr.contents.value
    print('hesafflib detected: %r keypoints' % nKpts)
    # Allocate arrays
    kpts = np.empty((nKpts, 5), kpts_dtype)
    desc = np.empty((nKpts, 128), desc_dtype)
    # Populate arrays
    hesaff_lib.exportArrays(hesaff_ptr, kpts, desc)
    return kpts, desc

#print(kpts)
#print(desc)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from hotspotter import fileio as io
    from hotspotter import draw_func2 as df2
    image = io.imread('lena.png')
    kpts, desc = hesaff_detect('lena.png')
    df2.imshow(image)
