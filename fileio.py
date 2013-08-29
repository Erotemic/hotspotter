from os.path import normpath, exists, realpath

import os
import os.path
import fnmatch
import pickle
import cPickle

import datetime
import timeit

import hotspotter.helpers as helpers
import numpy as np



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
def load_cPkl(fpath):
    with open(fpath, 'rb') as file:
        return cPickle.load(file)
def load_pkl(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)

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

# --- Smart Load/Save ---

def debug_smart_load(dpath='', fname='*', uid='*', ext='*'):
    pattern = fname+uid+ext
    print('debug_smart_load> In directory: '+dpath)
    for fname_ in os.listdir(dpath):
        if fnmatch.fnmatch(fname_, pattern):
            fpath = join(dpath, fname_)
            print fname_

def smart_load(dpath='', fname='', uid='', ext='', verbose=True, **kwargs):
    fpath = join(dpath, fname+uid+ext)
    if verbose:
        print(('smart_load(dpath=%r,\n'+' '*11+'fname=%r, uid=%r, ext=%r)')\
              % (dpath, fname, uid, ext))
    return __smart_load(fpath, **kwargs)

def __smart_load(fpath, convert=True, **kwargs):
    # Get components of the filesname
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    # If exact path doesnt exist
    if not exists(fpath):
        print('smart_load>fname=%r does not exist' % fname)
        data = load_alternative(fpath, **kwargs)
        if convert:
            print('smart_load> ...converting')
            smart_save(data, fpath=fpath)
        return data
    elif ext_ == '':
        raise NotImplemented('')
    else:
        print('smart_load>'+filesize_str(fpath))
        load_func = ext2_load_func[ext_]
        data = load_func(fpath)
        return data

def smart_save(data, fpath=None):
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    save_func = ext2_save_func[ext_]
    print('smart_save(fname=%r)' % fname)
    save_func(fpath, data)
    print('smart_save>'+filesize_str(fpath))

# --- Util ---
def sanatize_fpath(fpath, ext=None):
    'Ensures a filepath has correct the extension'
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    if not ext is None and ext_ != ext:
        fname = fname_noext + ext
    fpath = normpath(join(dpath, fname))
    return fpath

def find_alternative(fpath):
    # Check if file is in another format
    dpath, fname = os.path.split(fpath)
    fname_noext, ext_ = os.path.splitext(fname)
    fpath_noext = join(dpath, fname_noext)
    alternates = []
    for alt_ext in list(['.npy', '.npz', '.cPkl', '.pkl']):
        alt_fpath = fpath_noext + alt_ext
        if exists(alt_fpath):
            alternates.append(alt_fpath)
    if len(alternates) == 0:
        print(' * no alternates')
        alternative = None
    else:
        print(' * Found %d alternate(s) (picking first)' % len(alternates))
        for alt_fpath in iter(alternates):
            print('   * '+filesize_str(alt_fpath))
        alternative = alternates[0]
    return alt_fpath

def print_filesize(fpath):
    print(filesize_str(fpath))

def filesize_str(fpath):
    _, fname = os.path.split(fpath)
    mb_str = helpers.file_megabytes_str(fpath)
    return 'filesize(%r)=%s' % (fname, mb_str)

def load_alternative(fpath, can_fail=False, convert=True):
    # check for an alternative
    alt_fpath = find_alternative(fpath)
    if alt_fpath is None:
        # failed to find file path
        fail_msg = '...no alternatives to %r' % fpath
        print(fail_msg)
        if canfail:
            return None
        else:
            raise IOError(fail_msg)
    else:
        #load and convert alternative
        data = __smart_load(alt_fpath)
        return data

if __name__ == '__main__':
    data1 = (255 * np.random.rand(10000,10000)).astype(np.uint8)
    data2 = np.random.rand(10000,10000).astype(np.float64)
    data3 = (255 * np.random.rand(10000,10000)).astype(np.int32)

    print('Created arrays')
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
        print('Testing: '+save_func.__name__)
        print(' withext: '+save_func.ext)
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
