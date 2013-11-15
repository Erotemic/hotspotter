from __future__ import division, print_function
from os.path import normpath, exists, realpath, join
import os
import os.path
import fnmatch
import pickle
import cPickle
import shelve

import datetime
import timeit

import helpers
import params
import numpy as np
import sys

def print(*args, **kwargs): pass
def noprint(*args, **kwargs): pass
def realprint(*args, **kwargs):
    sys.stdout.write(args[0]+'\n')
def print_on():
    global print
    print = realprint
def print_off():
    global print
    print = noprint
print_on()

def reload_module():
    import imp, sys
    print('[io] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

# --- Saving ---
def save_npy(fpath, data):
    with open(fpath, 'wb') as file:
        np.save(file, data)
def save_npz(fpath, data):
    with open(fpath, 'wb') as file:
        np.savez(file, data)
def save_cPkl(fpath, data):
    with open(fpath, 'wb') as file:
        cPickle.dump(data, file, cPickle.HIGHEST_PROTOCOL)
def save_pkl(fpath, data):
    with open(fpath, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

# --- Loading ---
def load_npz_memmap(fpath):
    with open(fpath, 'rb') as file:
        npz = np.load(file, mmap_mode='r')
        data = npz['arr_0']
        npz.close()
    return data
def load_npz(fpath):
    with open(fpath, 'rb') as file:
        npz = np.load(file, mmap_mode=None)
        data = npz['arr_0']
        npz.close()
    return data
def load_npy(fpath):
    with open(fpath, 'rb') as file:
        data = np.load(file)
    return data
def load_cPkl(fpath):
    with open(fpath, 'rb') as file:
        data = cPickle.load(file)
    return data
def load_pkl(fpath):
    with open(fpath, 'rb') as file:
        data = pickle.load(file)
    return data

ext2_load_func = {
    '.npy'  : load_npy,
    '.npz'  : load_npz,
    '.cPkl' : load_cPkl,
    '.pkl'  : load_pkl }

ext2_save_func = {
    '.npy'  : save_npy,
    '.npz'  : save_npz,
    '.cPkl' : save_cPkl,
    '.pkl'  : save_pkl }


def debug_smart_load(dpath='', fname='*', uid='*', ext='*'):
    pattern = fname+uid+ext
    print('[io] debug_smart_load(): dpath=%r' % (dpath))
    for fname_ in os.listdir(dpath):
        if fnmatch.fnmatch(fname_, pattern):
            fpath = join(dpath, fname_)
            print(fname_)

# --- Smart Load/Save ---
#----
def __args2_fpath(dpath, fname, uid, ext):
    if len(ext) > 0 and ext[0] != '.':
        raise Exception('Fatal Error: Please be explicit and use a dot in ext')
    fpath = normpath(join(dpath, fname+uid+ext))
    return fpath
    
def smart_save(data, dpath='', fname='', uid='', ext='', verbose=params.VERBOSE_IO):
    ''' Saves data to the direcotry speficied '''
    fpath = __args2_fpath(dpath, fname, uid, ext)
    if verbose:
        if verbose > 1: print('[io]')
        print(('[io] smart_save(dpath=%r,\n'+' '*11+'fname=%r, uid=%r, ext=%r)')\
              % (dpath, fname, uid, ext))
    ret = __smart_save(data, fpath, verbose)
    if verbose > 1: print('[io]')
    return ret
#----
def smart_load(dpath='', fname='', uid='', ext='', verbose=params.VERBOSE_IO, **kwargs):
    ''' Loads data to the direcotry speficied '''
    fpath = __args2_fpath(dpath, fname, uid, ext)
    if verbose:
        if verbose > 1: print('[io]')
        print(('[io] smart_load(dpath=%r,\n'+' '*11+'fname=%r, uid=%r, ext=%r)')\
              % (dpath, fname, uid, ext))
    data = __smart_load(fpath, verbose, **kwargs)
    if verbose > 1: print('[io]')
    return data
#----
#----
def __smart_save(data, fpath, verbose):
    ' helper '
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    save_func = ext2_save_func[ext_]
    if verbose > 1:
        print('[io] saving: %r' % (type(data),))
    try: 
        save_func(fpath, data)
        if verbose > 1:
            print('[io] saved %s ' % (filesize_str(fpath),))
    except Exception as ex: 
        print('[io] ! Exception will saving %r' % fpath)
        print(helpers.indent(repr(ex),'[io]    '))
        raise
#----
def __smart_load(fpath, verbose, allow_alternative=False, can_fail=True, **kwargs):
    ' helper '
    # Get components of the filesname
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    # If exact path doesnt exist
    if not exists(fpath):
        print('[io] fname=%r does not exist' % fname)
        if allow_alternative:
            # allows alternative extension
            convert_alternative(fpath, verbose, can_fail=can_fail, **kwargs)
    # Ensure a valid extension
    if ext_ == '':
        raise NotImplemented('')
    else:
        load_func = ext2_load_func[ext_]
        # Do actual data loading
        try: 
            if verbose > 1:
                print('[io] loading '+filesize_str(fpath))
            data = load_func(fpath)
        except Exception as ex: 
            print('[io] ! Exception will loading %r' % fpath)
            print('[io] caught ex=%r' % (ex,))
            data = None
            if not can_fail:
                raise
        print('[io]... loaded data')
    if data is None:
        print('[io]... did not load %r' % fpath)
    return data
#----

# --- Util ---
def convert_alternative(fpath, verbose, can_fail):
    # check for an alternative (maybe old style or ext) file
    alternatives = find_alternatives(fpath, verbose)
    dpath, fname = os.path.split(fpath)
    if len(alternatives) == 0:
        fail_msg = '[io] ...no alternatives to %r' % fname
        if verbose: print(fail_msg)
        if can_fail: return None
        else: raise IOError(fail_msg)
    else:
        #load and convert alternative
        alt_fpath = alternatives[0]
        if verbose > 1:
            print('[io] ...converting %r' % alt_fpath)
        data = __smart_load(alt_fpath, verbose, allow_alternative=False)
        __smart_save(data, fpath, verbose)
        return data

def find_alternatives(fpath, verbose):
    # Check if file is in another format
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    fpath_noext = join(dpath, fname_noext)
    alternatives = []
    # Find files with a different
    for alt_ext in list(['.npy', '.npz', '.cPkl', '.pkl']):
        alt_fpath = fpath_noext + alt_ext
        if exists(alt_fpath):
            alternatives.append(alt_fpath)
    if verbose > 1:
        # Print num alternatives / filesizes
        print('[io] Found %d alternate(s)' % len(alternatives))
        for alt_fpath in iter(alternatives):
            print('[io] '+filesize_str(alt_fpath))
    return alternatives

def sanatize_fpath(fpath, ext=None): #UNUSED!
    'Ensures a filepath has correct the extension'
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    if not ext is None and ext_ != ext:
        fname = fname_noext + ext
    fpath = normpath(join(dpath, fname))
    return fpath

def print_filesize(fpath):
    print(filesize_str(fpath))

def filesize_str(fpath):
    _, fname = os.path.split(fpath)
    mb_str = helpers.file_megabytes_str(fpath)
    return 'filesize(%r)=%s' % (fname, mb_str)

# --- Global Cache ---
GLOBAL_CACHE_DIR = realpath('.hotspotter/global_cache')
helpers.ensuredir(GLOBAL_CACHE_DIR)

def global_cache_read(cache_id):
    cache_fname = join(GLOBAL_CACHE_DIR, 'cached_dir_'+str(cache_id)+'.txt')
    return helpers.read_from(cache_fname) if exists(cache_fname) else '.'

def global_cache_write(cache_id, newdir):
    cache_fname = join(GLOBAL_CACHE_DIR, 'cached_dir_'+str(cache_id)+'.txt')
    helpers.write_to(cache_fname, newdir)

# --- Shelve Caching ---
#def read_cache(fpath):
    #pass
#def write_cache(fpath): 
    #with open(fpath, 'wa') as file_
        #shelf = shelve.open(file_)
#def cached_keys(fpath):
    #pass

# --- Main Test --- 

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    data1 = (255 * np.random.rand(10000,10000)).astype(np.uint8)
    data2 = np.random.rand(10000,10000).astype(np.float64)
    data3 = (255 * np.random.rand(10000,10000)).astype(np.int32)

    print('[io] Created arrays')
    save_npy.ext = '.npy'
    save_npz.ext = '.npz'
    save_cPkl.ext = '.cPkl'
    save_pkl.ext = '.pkl'

    load_npy.ext = '.npy'
    load_npz.ext = '.npz'
    load_cPkl.ext = '.cPkl'
    load_pkl.ext = '.pkl'

    fpath_list = ['data1','data2','data3']
    data_list  =  [data1, data2, data3]

    save_func_list = [save_npy, save_npz, save_cPkl, save_pkl]
    load_func_list = [load_npy, load_npz, load_cPkl, load_pkl]

    fpath = '/media/Store/data/work/Oxford_Buildings/.hs_internals/'

    # Test Save
    for save_func in save_func_list:
        print('[io] Testing: '+save_func.__name__)
        print('[io]  withext: '+save_func.ext)
        tt_total = helpers.tic(save_func.__name__)

        for fpath, data, in zip(fpath_list, data_list):
            fpath += save_func.ext
            tt_single = helpers.tic(fpath)
            save_func(fpath, data)
            helpers.toc(tt_single)
        helpers.toc(tt_total)
        print('------------------')

    # Test memory:
    for save_func in save_func_list:
        for fpath in fpath_list:
            fpath += save_func.ext
            print(helpers.file_megabytes_str(fpath))

    # Test Load
    for load_func in load_func_list:
        print('Testing: '+load_func.__name__)
        print(' withext: '+load_func.ext)
        tt_total = helpers.tic(load_func.__name__)

        for fpath, data, in zip(fpath_list, data_list):
            fpath += load_func.ext
            tt = helpers.tic(fpath)
            data2 = load_func(fpath)
            helpers.toc(tt)
        helpers.toc(tt_total)
        print('------------------')


    fpath = ld2.OXFORD+'/.hs_internals/computed/cache/cx2_desc_HESAFF_szorig.npz'
    fpath_py = ld2.OXFORD+'/.hs_internals/computed/cache/cx2_desc_HESAFF_szorig.npy'
    fpath_pyz = ld2.OXFORD+'/.hs_internals/computed/cache/cx2_desc_TEMP.npz'


    print(helpers.file_megabytes_str(fpath))

    tic = helpers.tic
    toc = helpers.toc

    #tt = tic(fpath_py)
    #with open(fpath, 'wb') as file_:
        #npz = np.load(file_, fpath)
        #data = npz['arr_0']
        #npz.close()
    #toc(tt)

    tt = tic(fpath_py)
    with open(fpath_py, 'wb') as file_:
        np.save(file_, data)
    toc(tt)

    tt = tic(fpath_pyz)
    with open(fpath_pyz, 'wb') as file_:
        np.savez(file_, data)
    toc(tt)

    tt = tic(fpath_py)
    with open(fpath_py, 'rb') as file_:
        npy_data = np.load(file_)
    toc(tt)
    print(helpers.file_megabytes_str(fpath_py))


    tt = tic(fpath_pyz)
    with open(fpath_pyz, 'rb') as file_:
        npz = np.load(file_)
        npz_data = npz['arr_0']
        npz.close()
    toc(tt)
    print(helpers.file_megabytes_str(fpath_pyz))

    tt = tic(fpath_pyz)
    with open(fpath_pyz, 'rb') as file_:
        npz = np.load(file_, mmap_mode='r+')
        npz_data = npz['arr_0']
        npz.close()
    toc(tt)

    tt = helpers.tic(fpath)
    data2 = load_func(fpath)

    with Timer():
        with open(fpath, 'rb') as file_:
            npz = np.load(file_, mmap_mode='r')
            data = npz['arr_0']
            npz.close()

    with Timer():
        with open(fpath, 'rb') as file_:
            npz2 = np.load(file_, mmap_mode=None)
            data2 = npz['arr_0']
            npz2.close()
