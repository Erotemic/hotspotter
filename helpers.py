'''
This is less of a helper function file and more of a pile of things 
where I wasn't sure of where to put. 
A lot of things could probably be consolidated or removed. There are many
non-independent functions which are part of HotSpotter. They should be removed
and put into their own module. The standalone functions should be compiled 
into a global set of helper functions.

Wow, pylint is nice for cleaning.
'''
from __future__ import division, print_function
from Printable import printableVal
from os.path import join, relpath, normpath, join, split, isdir, isfile, exists, islink, ismount
import cPickle
import cStringIO
import code
import datetime
import fnmatch
import hashlib
import inspect
import numpy as np
import os, os.path, sys
import shutil
import sys
import textwrap
import time
import types
import warnings
#print('LOAD_MODULE: helpers.py')

# reloads this module when I mess with it
def reload_module():
    import imp
    import sys
    print('[helpers] reloading '+__name__)
    imp.reload(sys.modules[__name__])

def rrr():
    reload_module()

# --- Globals ---

IMG_EXTENSIONS = set(['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ppm'])

PRINT_CHECKS = False # True
__PRINT_WRITES__ = False
__CHECKPATH_VERBOSE__ = False

VERY_VERBOSE = False

def __DEPRICATED__(func):
    'deprication decorator'
    warn_msg = 'Depricated call to: %s' % func.__name__
    def __DEP_WRAPPER(*args, **kwargs):
        printWARN(warn_msg, category=DeprecationWarning)
        return func(*args, **kwargs)
    __DEP_WRAPPER.__name__ = func.__name__
    __DEP_WRAPPER.__doc__ = func.__doc__
    __DEP_WRAPPER.__dict__.update(func.__dict__)
    return __DEP_WRAPPER

# --- Images ----

def num_images_in_dir(path):
    'returns the number of images in a directory'
    num_imgs = 0
    for root, dirs, files in os.walk(path):
        for fname in files:
            if matches_image(fname):
                num_imgs += 1
    return num_imgs

def matches_image(fname):
    fname_ = fname.lower()
    return any([fnmatch.fnmatch(fname_, pat) for pat in ['*.jpg', '*.png']])

def list_images(img_dpath, ignore_list=[], recursive=True, fullpath=False):
    ignore_set = set(ignore_list)
    gname_list_ = []
    assert_path(img_dpath)
    # Get all the files in a directory recursively
    for root, dlist, flist in os.walk(img_dpath):
        for fname in iter(flist):
            gname = join(relpath(root, img_dpath), fname).replace('\\','/').replace('./','')
            if fullpath:
                gname_list_.append(join(root, gname))
            else:
                gname_list_.append(gname)
        if not recursive:
            break
    # Filter out non images or ignorables
    gname_list = [gname for gname in iter(gname_list_) 
                  if not gname in ignore_set and matches_image(gname)]
    return gname_list


# --- Strings ----

def remove_chars(instr, illegals_chars):
    outstr = instr
    for ill_char in iter(illegals_chars):
        outstr = outstr.replace(ill_char, '')
    return outstr

def indent(string, indent = '    '):
    return indent+string.replace('\n','\n'+indent)

def truncate_str(str, maxlen=110):
    if len(str) < maxlen:
        return str
    else:
        truncmsg = ' /* TRUNCATED */ '
        maxlen_= maxlen - len(truncmsg)
        lowerb = int(maxlen_ * .8) 
        upperb = maxlen_ - lowerb
        return str[:lowerb]+truncmsg+str[-upperb:]

def pack_into(instr, textwidth=160, breakchars=' ', break_words=True):
    newlines = ['']
    word_list = instr.split(breakchars)
    for word in word_list:
        if len(newlines[-1]) + len(word) > textwidth:
            newlines.append('')
        while break_words and len(word) > textwidth:
            newlines[-1] += word[:textwidth]
            newlines.append('')
            word = word[textwidth:]
        newlines[-1] += word+' '
    return '\n'.join(newlines)

    
# --- Lists ---

def list_replace(instr, search_list=[], repl_list=None):
    repl_list = [''] * len(search_list) if repl_list is None else repl_list
    for ser, repl in zip(search_list, repl_list):
        instr = instr.replace(ser, repl)
    return instr

def intersect_ordered(list1, list2):
    'returns list1 elements that are also in list2 preserves order of list1'
    set2 = set(list2)
    new_list = [item for item in iter(list1) if item in set2]
    #new_list =[]
    #for item in iter(list1):
        #if item in set2:
            #new_list.append(item)
    return new_list

# --- Info Strings ---

def printable_mystats(_list):
    stat_dict = mystats(_list)
    ret = '{'
    ret += ', '.join(['%r:%.1f' % (key, val) for key, val in stat_dict.items()])
    ret += '}'
    return ret

def mystats(_list):
    from collections import OrderedDict
    if len(_list) == 0:
        return {'empty_list':True}
    nparr = np.array(_list)
    return OrderedDict([
        ('max',nparr.max()),
        ('min',nparr.min()),
        ('mean',nparr.mean()),
        ('std',nparr.std())])

def myprint(input=None, prefix='', indent='', lbl=''):
    if len(lbl) > len(prefix): 
        prefix = lbl
    if len(prefix) > 0:
        prefix += ' '
    _print(indent+prefix+str(type(input))+' ')
    if type(input) == types.ListType:
        _println(indent+'[')
        for item in iter(input):
            myprint(item, indent=indent+'  ')
        _println(indent+']')
    elif type(input) == types.StringType:
        _println(input)
    elif type(input) == types.DictType:
        _println(printableVal(input))
    else: #
        _println(indent+'{')
        attribute_list = dir(input)
        for attr in attribute_list:
            if attr.find('__') == 0: continue
            val = str(input.__getattribute__(attr))
            #val = input[attr]
            # Format methods nicer
            #if val.find('built-in method'):
                #val = '<built-in method>'
            _println(indent+'  '+attr+' : '+val)
        _println(indent+'}')

def info(var, lbl):
    if types(var) == np.ndarray:
        return npinfo(var, lbl)
    if types(var) == types.ListType:
        return listinfo(var, lbl)

def npinfo(ndarr, lbl='ndarr'):
    info = ''
    info+=(lbl+': shape=%r ; dtype=%r' % (ndarr.shape, ndarr.dtype))
    return info

def listinfo(list, lbl='ndarr'):
    if type(list) != types.ListType:
        raise Exception('!!')
    info = ''
    type_set = set([])
    for _ in iter(list): type_set.add(str(type(_)))
    info+=(lbl+': len=%r ; types=%r' % (len(list), type_set))
    return info

#expected_type = np.float32
#expected_dims = 5
def numpy_list_num_bits(nparr_list, expected_type, expected_dims): 
    num_bits = 0
    num_items = 0
    num_elemt = 0
    bit_per_item = {np.float32:32, np.uint8:8}[expected_type]
    for nparr in iter(nparr_list):
        arr_len, arr_dims = nparr.shape
        if not nparr.dtype.type is expected_type: 
            msg = 'Expected Type: '+repr(expected_type)
            msg += 'Got Type: '+repr(nparr.dtype)
            raise Exception(msg)
        if arr_dims != expected_dims: 
            msg = 'Expected Dims: '+repr(expected_dims)
            msg += 'Got Dims: '+repr(arr_dims)
            raise Exception(msg)
        num_bits += len(nparr) * expected_dims * bit_per_item
        num_elemt += len(nparr) * expected_dims
        num_items += len(nparr) 
    return num_bits,  num_items, num_elemt

def public_attributes(input):
    public_attr_list = []
    all_attr_list = dir(input)
    for attr in all_attr_list:
        if attr.find('__') == 0: continue
        public_attr_list.append(attr)
    return public_attr_list

def explore_stack():
    stack = inspect.stack()
    tup = stack[0]
    for ix, tup in reversed(list(enumerate(stack))):
        frame = tup[0]
        print('--- Frame %2d: ---' % (ix))
        print_frame(frame)
        print('\n')
        #next_frame = curr_frame.f_back

def explore_module(module_, seen=None, maxdepth=2, nonmodules=False):
    def __childiter(module):
        for aname in iter(dir(module)):
            if aname.find('_') == 0: 
                continue
            try:
                yield module.__dict__[aname], aname
            except KeyError as ex:
                print(repr(ex))
                pass
    def __explore_module(module, indent, seen, depth, maxdepth, nonmodules):
        valid_children = []
        ret = u''
        modname = str(module.__name__)
        #modname = repr(module)
        for child, aname in __childiter(module):
            try: 
                childtype = type(child)
                if not childtype == types.ModuleType:
                    if nonmodules:
                        #print_(depth)
                        fullstr = indent+'    '+str(aname)+' = '+repr(child)
                        truncstr = truncate_str(fullstr)+'\n'
                        ret +=  truncstr
                    continue
                childname = str(child.__name__)
                if not seen is None:
                    if childname in seen: 
                        continue
                    elif maxdepth is None:
                      seen.add(childname)
                if childname.find('_') == 0: 
                    continue
                valid_children.append(child)
            except Exception as ex:
                print(repr(ex))
                pass
        # Print 
        #print_(depth)
        ret += indent+modname+'\n'
        # Recurse
        if not maxdepth is None and depth >= maxdepth: 
            return ret
        ret += ''.join([__explore_module(child,
                                         indent+'    ', 
                                         seen, depth+1,
                                         maxdepth,
                                         nonmodules) \
                       for child in iter(valid_children)])
        return ret
    #ret += 
    #println('#module = '+str(module_))
    ret = __explore_module(module_, '     ', seen, 0, maxdepth, nonmodules)
    #print(ret)
    flush()
    return ret

# --- Util --- 

def configure_matplotlib():
    import matplotlib
    if matplotlib.get_backend() != 'Qt4Agg':
        print('[helpers] Configuring matplotlib for Qt4')
        matplotlib.use('Qt4Agg', warn=True, force=True)
        matplotlib.rcParams['toolbar'] = 'None'
    
def alloc_lists(num_alloc):
    'allocates space for a numpy array of lists'
    return [[] for _ in xrange(num_alloc)]

def get_timestamp(format='filename'):
    now = datetime.datetime.now()
    time_tup = (now.year, now.month, now.day, now.hour, now.minute)
    time_formats = {
        'filename': 'ymd-%04d-%02d-%02d_hm-%02d-%02d',
        'comment' : '# (yyyy-mm-dd hh:mm) %04d-%02d-%02d %02d:%02d' }
    stamp = time_formats[format] % time_tup
    return stamp

def make_progress_fmt_str(max_val, lbl='Progress: '):
    r'makes format string that prints progress: %Xd/MAX_VAL with backspaces'
    max_str = str(max_val)
    dnumstr = str(len(max_str))
    fmt_str = lbl+'%'+dnumstr+'d/'+max_str
    fmt_str = '\b'*(len(fmt_str)-len(dnumstr)+len(max_str)) + fmt_str
    return fmt_str

def normalize(array, dim=0):
    return norm_zero_one(array, dim)

def norm_zero_one(array, dim=0):
    'normalizes a numpy array from 0 to 1'
    array_max  = array.max(dim)
    array_min  = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    return np.divide(np.subtract(array, array_min), array_exnt)

def find_std_inliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

def win_shortcut(source, link_name):
    import ctypes
    csl = ctypes.windll.kernel32.CreateSymbolicLinkW
    csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    csl.restype = ctypes.c_ubyte
    flags = 1 if isdir(source) else 0
    retval = csl(link_name, source, flags)
    if retval == 0:
        warn_msg = '[helpers] Unable to create symbolic link on windows.'
        print(warn_msg)
        warnings.warn(warn_msg, category=UserWarning)
        if checkpath(link_name):
            return True
        raise ctypes.WinError()

def symlink(source, link_name, noraise=False):
    if os.path.islink(link_name):
        print('[helpers] symlink %r exists' % (link_name))
        return
    print('[helpers] Creating symlink: source=%r link_name=%r' % (source, link_name))
    try: 
        os_symlink = getattr(os, "symlink", None)
        if callable(os_symlink):
            os_symlink(source, link_name)
        else:
            win_shortcut(source, link_name)
    except Exception as ex:
        checkpath(link_name, True)
        checkpath(source, True)
        if not noraise:
            raise

# --- Context ---

def inIPython():
    try:
        __IPYTHON__
        return True
    except NameError as nex:
        return False

def haveIPython():
    try:
        import IPython
        return True
    except NameError as nex:
        return False

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    print('*** keyboard> INTERACTING WITH IPYTHON ***')
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print("# Ctrl-D  Use quit() to exit :) Happy debugging!")
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        import IPython
        IPython.embed_kernel(module=None,local_ns=namespace)
    except SystemExit:
        pass
    except Exception as ex:
        print(repr(ex))
        print('*** keyboard> FAILED TO INTERACT WITH IPYTHON ***')
        print('probably want to up up')
        import pdb
        pdb.set_trace()


def print_frame(frame):
    frame = frame if 'frame' in vars() else inspect.currentframe()
    attr_list = ['f_code.co_name', 'f_back', 'f_lineno',
                   'f_code.co_names', 'f_code.co_filename']
    obj_name = 'frame'
    execstr_print_list = ['print("%r=%%r" %% (%s,))' % (_execstr, _execstr)
                         for _execstr in execstr_attr_list(obj_name, attr_list)]
    execstr = '\n'.join(execstr_print_list)
    exec(execstr)
    local_varnames = pack_into('; '.join(frame.f_locals.keys()))
    print(local_varnames)
    #if len(local_varnames) > 360: 
        #print(local_varnames[0:360]+'...')#hack
    #else:
    print('--- End Frame ---')

def search_stack_for_localvar(varname):
    curr_frame = inspect.currentframe()
    print(' * Searching parent frames for: '+str(varname))
    frame_no = 0
    while not curr_frame.f_back is None:
        if varname in curr_frame.f_locals.keys():
            print(' * Found in frame: '+str(frame_no))
            return curr_frame.f_locals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    print('... Found nothing in all '+str(frame_no)+' frames.')
    return None

def get_parent_locals():
    this_frame = inspect.currentframe()
    call_frame = this_frame.f_back
    parent_frame = call_frame.f_back
    if parent_frame is None:
        return None
    return parent_frame.f_locals

# --- Convinience ----

def vd(dname=None):
    'view directory'
    print('[helpers] view_dir(%r) ' % dname)
    dname = os.getcwd() if dname is None else dname
    open_prog = {'win32' :'explorer.exe',
                 'linux2':'nautilus',
                 'darwin':'open'}[sys.platform]
    os.system(open_prog+' '+normpath(dname))
        
def str2(obj):
    if type(obj) == types.DictType:
        return str(obj).replace(', ','\n')[1:-1]
    if type(obj) == types.TypeType:
        return str(obj).replace('<type \'','').replace('\'>','')
    else:
        return str(obj)

def random_indexes(max_index, subset_size):
    subst_ = np.arange(0, max_index)
    np.random.shuffle(subst_)
    subst = subst_[0:min(subset_size, max_index)]
    return subst

def gvim(fname):
    'its the only editor that matters'
    import subprocess
    proc = subprocess.Popen(['gvim',fname])

def cmd(command):
    os.system(command)

# --- Path ---

#---------------

@__DEPRICATED__
def filecheck(fpath):
    return exists(fpath)

@__DEPRICATED__
def dircheck(dpath,makedir=True):
    if not exists(dpath):
        if not makedir:
            #print('Nonexistant directory: %r ' % dpath)
            return False
        print('Making directory: %r' % dpath)
        os.makedirs(dpath)
    return True

def remove_file(fpath, verbose=True):
    try:
        if verbose:
            print('[helpers] Removing '+fpath)
        os.remove(fpath)
    except OSError as e:
        printWARN('OSError: %s,\n Could not delete %s' % (str(e), fpath))
        return False
    return True

def remove_files_in_dir(dpath, fname_pattern='*', recursive=False):
    print('[helpers] Removing files:')
    print('  * in dpath = %r ' % dpath) 
    print('  * matching pattern = %r' % fname_pattern) 
    print('  * recursive = %r' % recursive) 
    num_removed, num_matched = (0,0)
    if not exists(dpath):
        printWARN('!!! dir = %r does not exist!' % dpath)
    for root, dname_list, fname_list in os.walk(dpath):
        for fname in fnmatch.filter(fname_list, fname_pattern):
            num_matched += 1
            num_removed += remove_file(join(root, fname))
        if not recursive:
            break
    print('[helpers] ... Removed %d/%d files' % (num_removed, num_matched))
    return True

def longest_existing_path(_path):
    while True: 
        _path_new = os.path.dirname(_path)
        if exists(_path_new):
            _path = _path_new
            break
        if _path_new == _path: 
            print('!!! This is a very illformated path indeed.')
            _path = ''
            break
        _path = _path_new
    return _path

def path_ndir_split(path, n):
    path, ndirs = split(path)
    for i in xrange(n-1):
        path, name = split(path)
        ndirs = name + os.path.sep + ndirs
    return ndirs

def get_caller_name():
    frame = inspect.currentframe()
    frame = frame.f_back
    caller_name = None
    while caller_name in [None, 'ensurepath']:
        frame = frame.f_back
        if frame is None: 
            break 
        caller_name = frame.f_code.co_name
    return caller_name

def checkpath(path_, verbose=PRINT_CHECKS):
    'returns true if path_ exists on the filesystem'
    path_ = normpath(path_)
    if verbose:
        print = print_
        pretty_path = path_ndir_split(path_, 2)
        caller_name = get_caller_name()
        print('[%s] checkpath(%r)' % (caller_name, pretty_path))
        if exists(path_):
            path_type = ''
            if isfile(path_):  path_type += 'file'
            if isdir(path_):   path_type += 'directory'
            if islink(path_):  path_type += 'link'
            if ismount(path_): path_type += 'mount'
            path_type = 'file' if isfile(path_) else 'directory'
            println('...(%s) exists' % (path_type,))
        else:
            print('... does not exist\n')
            if __CHECKPATH_VERBOSE__:
                print('[helpers] \n  ! Does not exist\n')
                _longest_path = longest_existing_path(path_)
                print('[helpers] ... The longest existing path is: %r\n' % _longest_path)
            return False
        return True
    else:
        return exists(path_)
def check_path(path_):
    return checkpath(path_)

def ensurepath(path_):
    if not checkpath(path_):
        print('[helpers] mkdir(%r)' % path_)
        os.makedirs(path_)
    return True
def ensuredir(path_):
    return ensurepath(path_)
def ensure_path(path_):
    return ensurepath(path_)
def assertpath(path_):
    if not checkpath(path_):
        raise AssertionError('Asserted path does not exist: '+path_)
def assert_path(path_):
    return assertpath(path_)

def join_mkdir(*args):
    'join and creates if not exists'
    output_dir = join(*args)
    if not exists(output_dir):
        print('Making dir: '+output_dir)
        os.mkdir(output_dir)
    return output_dir

# ---File Copy---

def copy_task(cp_list, test=False, nooverwrite=False, print_tasks=True):
    '''
    Input list of tuples: 
        format = [(src_1, dst_1), ..., (src_N, dst_N)] 
    Copies all files src_i to dst_i
    '''
    num_overwrite = 0
    _cp_tasks = [] # Build this list with the actual tasks
    if nooverwrite:
        print('[helpers] Removed: copy task ')
    else:
        print('[helpers] Begining copy+overwrite task.')
    for (src, dst) in iter(cp_list):
        if exists(dst):
            num_overwrite += 1
            if print_tasks:
                print('[helpers] !!! Overwriting ')
            if not nooverwrite:
                _cp_tasks.append((src, dst))
        else:
            if print_tasks:
                print('[helpers] ... Copying ')
                _cp_tasks.append((src, dst))
        if print_tasks:
            print('[helpers]    '+src+' -> \n    '+dst)
    print('[helpers] About to copy %d files' % len(cp_list))
    if nooverwrite:
        print('[helpers] Skipping %d tasks which would have overwriten files' % num_overwrite)
    else:
        print('[helpers] There will be %d overwrites' % num_overwrite)
    if not test: 
        print('[helpers]... Copying')
        for (src, dst) in iter(_cp_tasks):
            shutil.copy(src, dst)
        print('[helpers]... Finished copying')
    else:
        print('[helpers]... In test mode. Nothing was copied.')

def copy(src, dst):
    if exists(dst):
        print('[helpers] !!! Overwriting ')
    else:
        print('[helpers] ... Copying ')
    print('[helpers]    '+src+' -> \n    '+dst)
    shutil.copy(src, dst)

def copy_all(src_dir, dest_dir, glob_str_list):
    if type(glob_str_list) != types.ListType:
        glob_str_list = [glob_str_list]
    for _fname in os.listdir(src_dir):
        for glob_str in glob_str_list:
            if fnmatch.fnmatch(_fname, glob_str):
                src = normpath(join(src_dir, _fname))
                dst = normpath(join(dest_dir, _fname))
                copy(src, dst)
                break
# ---File / String Search----

def grep(string, pattern):
    if not type(string) is types.StringType:#-> convert input to a string
        string = repr(string)
    matching_lines = [] # Find all matching lines
    for line in string.split('\n'):
        if not fnmatch.fnmatch(string, pattern):
            continue
        matching_lines.append(line)
    return matching_lines

def glob(dirname, pattern, recursive=False):
    matching_fnames = []
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            if not fnmatch.fnmatch(fname, pattern): 
                continue
            matching_fnames.append(join(root, fname))
        if not recursive: 
            break
    return matching_fnames

def print_grep(*args, **kwargs):
    matching_lines = grep(*args, **kwargs)
    print('Matching Lines:') # Print matching lines
    print('\n    '.join(matching_lines))

def print_glob(*args, **kwargs):
    matching_fnames = glob(*args, **kwargs)
    print('Matching Fnames:') # Print matching fnames
    print('\n    '.join(matching_fnames))

#---------------
# save / load / cache functions

def sanatize_fname(fname):
    ext = '.pkl'
    if fname.rfind(ext) != max(len(fname) - len(ext), 0):
        fname += ext
    return fname

def eval_from(fpath, err_onread=True):
    'evaluate a line from a test file'
    print('[helpers] Evaling: fpath=%r' % fpath)
    text = read_from(fpath)
    if text is None:
        if err_onread:
            raise Exception('Error reading: fpath=%r' % fpath)
        print('[helpers] * could not eval: %r ' % fpath)
        return None
    return eval(text)

def read_from(fpath):
    if not checkpath(fpath):
        println('[helpers] * FILE DOES NOT EXIST!')
        return None
    print('[helpers] * Reading text file: %r ' % split(fpath)[1])
    try: 
        text = open(fpath,'r').read()
    except Exception as ex:
        print('[helpers] * Error reading fpath=%r' % fpath)
        raise
    if VERY_VERBOSE:
        print('[helpers] * Read %d characters' % len(text))
    return text

def write_to(fpath, to_write):
    if __PRINT_WRITES__:
        println('[helpers] * Writing to text file: %r ' % fpath)
    with open(fpath, 'w') as file:
        file.write(to_write)

def save_pkl(fpath, data):
    with open(fpath, 'wb') as file:
        cPickle.dump(data, file)

def load_pkl(fpath):
    with open(fpath, 'wb') as file:
        return cPickle.load(file)

def save_npz(fpath, *args, **kwargs):
    print_(' * save_npz: %r ' % fpath)
    flush()
    np.savez(fpath, *args, **kwargs)
    print('... success')

def load_npz(fpath):
    print('[helpers] load_npz: %r ' % split(fpath)[1])
    print('[helpers] filesize is: '+ file_megabytes_str(fpath))
    npz = np.load(fpath, mmap_mode='r+')
    data = tuple(npz[key] for key in sorted(npz.keys()))
    #print(' * npz.keys() = %r '+str(npz.keys()))
    npz.close()
    return data

def hashstr(data, trunc_pos=8):
    # Get a 128 character hex string
    hashstr = hashlib.sha512(data).hexdigest()
    # Convert to base 57
    hashstr2 = hex2_base57(hashstr)
    # Truncate
    hashstr = hashstr2[:trunc_pos]
    return hashstr

#def valid_filename_ascii_chars():
    ## Find invalid chars
    #ntfs_inval = '< > : " / \ | ? *'.split(' ')
    #other_inval = [' ', '\'', '.']
    ##case_inval = map(chr, xrange(97, 123))
    #case_inval = map(chr, xrange(65, 91))
    #invalid_chars = set(ntfs_inval+other_inval+case_inval)
    ## Find valid chars
    #valid_chars = []
    #for index in xrange(32, 127):
        #char = chr(index)
        #if not char in invalid_chars:
            #print index, chr(index)
            #valid_chars.append(chr(index))
    #return valid_chars
#valid_filename_ascii_chars()
ALPHABET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  'a', 'b', 'c',
            'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ';', '=', '@',
            '[', ']', '^', '_', '`', '{', '}', '~', '!', '#', '$', '%', '&',
            '(', ')', '+', ',', '-']
BIGBASE = len(ALPHABET)

def hex2_base57(hexstr):
    x = int(hexstr, 16)
    if x==0: return '0'
    sign = 1 if x > 0 else -1
    x *= sign
    digits = []
    while x:
        digits.append(ALPHABET[x % BIGBASE])
        x //= BIGBASE
    if sign < 0:
        digits.append('-')
        digits.reverse()
    newbase_str = ''.join(digits)
    return newbase_str

def hashstr_md5(data):
    hashstr = hashlib.md5(data).hexdigest()
    bin(int(my_hexdata, scale))
    return hashstr

def load_cache_npz(input_data, uid='', cache_dir='.', is_sparse=False):
    data_fpath = __cache_data_fpath(input_data, uid, cache_dir)
    cachefile_exists = checkpath(data_fpath)
    if cachefile_exists:
        try:
            print('helpers.load_cache> Trying to load cached data: %r' % split(data_fpath)[1])
            print('helpers.load_cache> Cache filesize: ' + file_megabytes_str(data_fpath))
            flush()
            if is_sparse:
                with open(data_fpath, 'rb') as file_:
                    data = cPickle.load(file_)
            else:
                npz = np.load(data_fpath)
                data = npz['arr_0']
                npz.close()
            print('...success')
            return data
        except Exception as ex:
            print('...failure')
            print('helpers.load_cache> %r ' % ex)
            print('helpers.load_cache>...cannot load data_fpath=%r ' % data_fpath)
            raise CacheException(repr(ex))
    else:
        raise CacheException('nonexistant file: %r' % data_fpath)
    raise CacheException('other failure')

def save_cache_npz(input_data, data, uid='', cache_dir='.', is_sparse=False):
    data_fpath = __cache_data_fpath(input_data, uid, cache_dir)
    print('[helpers] caching data: %r' % split(data_fpath)[1])
    flush()
    if is_sparse:
        with open(data_fpath, 'wb') as outfile:
            cPickle.dump(data, outfile, cPickle.HIGHEST_PROTOCOL)
    else:
        np.savez(data_fpath, data)
    print('...success')

def cache_npz_decorator(npz_func):
    def __func_wrapper(input_data, *args, **kwargs):
        ret = npz_func(*args, **kwargs)

class CacheException(Exception):
    pass

def __cache_data_fpath(input_data, uid, cache_dir):
    hashstr    = hashstr(input_data)
    shape_lbl  = str(input_data.shape).replace(' ','')
    data_fname = uid+'_'+shape_lbl+'_'+hashstr+'.npz'
    data_fpath = join(cache_dir, data_fname)
    return data_fpath

def file_bytes(fpath):
    return os.stat(fpath).st_size

def file_megabytes(fpath):
    return os.stat(fpath).st_size / (2.0 ** 20)

def file_megabytes_str(fpath):
    return ('%.2f MB' % file_megabytes(fpath))

    
# --- Timing ---

def tic(msg=None):
    return (msg, time.time())

def toc(tt):
    (msg, start_time) = tt
    ellapsed = (time.time() - start_time)
    if not msg is None: 
        sys.stdout.write('...toc(%.4fs, ' % ellapsed + '"' + str(msg) + '"' + ')\n')
    return ellapsed

# from http://stackoverflow.com/questions/6796492/python-temporarily-redirect-stdout-stderr
class RedirectStdout(object):
    def __init__(self, lbl=None):
        self._stdout_old = sys.stdout
        self.stream = cStringIO.StringIO()
        self.record = '<no record>'
        self.lbl = lbl
    def start(self):
        sys.stdout.flush()
        sys.stdout = self.stream
    def stop(self):
        self.stream.flush()
        sys.stdout = self._stdout_old
        self.stream.seek(0)
        self.record = self.stream.read()
        return self.record
    def dump(self):
        print(indent(self.record, self.lbl))
    def __enter__(self):
        self.start()
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        if not self.lbl is None:
            self.dump()

def choose(n,k):
    import scipy.misc
    return scipy.misc.comb(n,k,True)

class Timer(object):
    ''' Used to time statments with a with statment
    e.g with Timer() as t: some_function()'''
    def __init__(self, msg='', verbose=True, newline=True):
        self.msg = msg
        self.verbose = verbose
        self.newline = newline
        self.tstart = -1
        self.tic()

    def tic(self):
        if self.verbose:
            sys.stdout.flush()
            sys.stdout.write('\ntic(%r)' % self.msg)
            if self.newline:
                sys.stdout.write('\n')
            sys.stdout.flush()
        self.tstart = time.time()

    def toc(self):
        ellapsed = (time.time() - self.tstart)
        if self.verbose:
            sys.stdout.write('...toc(%r)=%.4fs\n' % (self.msg, ellapsed))
            sys.stdout.flush()
        return ellapsed 

    def __enter__(self):
        #if not self.msg is None:
            #sys.stdout.write('---tic---'+self.msg+'  \n')
        #self.tic()
        pass

    def __exit__(self, type, value, trace):
        self.toc()

# --- Exec Strings ---

IPYTHON_EMBED_STR = r'''
try: 
    import IPython
    print('Presenting in new ipython shell.')
    embedded = True
    IPython.embed()
except Exception as ex:
    printWARN(repr(ex)+'\n!!!!!!!!')
    embedded = False
'''

def execstr_parent_locals():
    parent_locals = get_parent_locals()
    return execstr_dict(parent_locals, 'parent_locals')

def execstr_attr_list(obj_name, attr_list=None):
    #if attr_list is None:
        #exec(execstr_parent_locals())
        #exec('attr_list = dir('+obj_name+')')
    execstr_list = [obj_name+'.'+attr for attr in attr_list]
    return execstr_list

def execstr_dict(dict_, local_name, exclude_list=None):
    #if local_name is None:
        #local_name = dict_
        #exec(execstr_parent_locals())
        #exec('dict_ = local_name')
    if exclude_list is None: 
        execstr = '\n'.join((key+' = '+local_name+'['+repr(key)+']'
                            for (key, val) in dict_.iteritems()))
    else:
        if not type(exclude_list) == types.ListType:
            exclude_list = [exclude_list]
        exec_list = []
        for (key, val) in dict_.iteritems():
            if not any((fnmatch.fnmatch(key, pat) for pat in iter(exclude_list))):
                exec_list.append(key+' = '+local_name+'['+repr(key)+']')
        execstr = '\n'.join(exec_list)
    return execstr

def execstr_timeitsetup(dict_, exclude_list=[]):
    '''
    Example: 
    import timeit
    local_dict = locals().copy()
    exclude_list=['_*', 'In', 'Out', 'rchip1', 'rchip2']
    local_dict = locals().copy()
    setup = helpers.execstr_timeitsetup(local_dict, exclude_list)
    timeit.timeit('somefunc', setup)
    '''
    old_thresh =  np.get_printoptions()['threshold']
    np.set_printoptions(threshold=1000000000)
    matches = fnmatch.fnmatch
    excl_valid_keys = [key for key in dict_.iterkeys() if not any((matches(key, pat) for pat in iter(exclude_list)))]
    valid_types = set([np.ndarray, np.float32, np.float64, np.int64, int, float])
    type_valid_keys = [key for key in iter(excl_valid_keys) if type(dict_[key]) in valid_types]
    exec_list = []
    for key in type_valid_keys:
        val = dict_[key]
        try:
            val_str = np.array_repr(val)
        except Exception as ex:
            val_str = repr(val)
        exec_list.append(key+' = '+repr(dict_[key]))
    exec_str  = '\n'.join(exec_list)
    import_str = textwrap.dedent('''
    import numpy as np
    from numpy import array, float32, float64, int32, int64
    import helpers
    from spatial_verification2 import *
                                 ''')
    setup = import_str + exec_str
    np.set_printoptions(threshold=old_thresh)
    return setup


@__DEPRICATED__
def dict_execstr(dict_, local_name=None):
    return execstr_dict(dict_, local_name)


def execstr_func(func):
    print(' ! Getting executable source for: '+func.func_name)
    _src = inspect.getsource(func)
    execstr = textwrap.dedent(_src[_src.find(':')+1:])
    # Remove return statments
    while True:
        stmtx = execstr.find('return')  # Find first 'return'
        if stmtx == -1: break # Fail condition
        # The characters which might make a return not have its own line
        stmt_endx = len(execstr)-1
        for stmt_break in '\n;':
            print(execstr)
            print('')
            print(stmtx)
            stmt_endx_new = execstr[stmtx:].find(stmt_break)
            if -1 < stmt_endx_new < stmt_endx:
                stmt_endx = stmt_endx_new
        # now have variables stmt_x, stmt_endx
        before = execstr[:stmtx]
        after  = execstr[stmt_endx:]
        execstr = before+after
    return execstr
def execstr_src(func):
    return execstr_func(func)
@__DEPRICATED__
def get_exec_src(func):
    return execstr_func(func)

# --- Profiling ---

def unit_test(test_func):
    test_name = test_func.func_name
    def __unit_test_wraper():
        print('Testing: '+test_name)
        try:
            ret = test_func()
        except Exception as ex:
            print(repr(ex))
            print('Tested: '+test_name+' ...FAILURE')
            raise
        print('Tested: '+test_name+' ...SUCCESS')
        return ret
    return __unit_test_wraper


def profile(cmd, globals_=globals(), locals_=locals()):
    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
    import cProfile, sys, os
    print('[helpers] Profiling Command: '+cmd)
    cProfOut_fpath = 'OpenGLContext.profile'
    cProfile.runctx( cmd, globals_, locals_, filename=cProfOut_fpath )
    # RUN SNAKE
    print('[helpers] Profiled Output: '+cProfOut_fpath)
    if sys.platform == 'win32':
        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
    else:
        rsr_fpath = 'runsnake'
    view_cmd = rsr_fpath+' "'+cProfOut_fpath+'"'
    os.system(view_cmd)
    return True

def profile_lines(fname):
    import __init__
    import shutil
    hs_path = split(__init__.__file__)
    lineprofile_path = join(hs_path, '.lineprofile')
    ensurepath(lineprofile_path)
    shutil.copy('*', lineprofile_path+'/*')

def memory_profile():
    #http://stackoverflow.com/questions/2629680/deciding-between-subprocess-multiprocessing-and-thread-in-python
    import guppy
    import gc
    print('Collecting garbage')
    gc.collect()
    hp = guppy.hpy()
    print('Waiting for heap output...')
    heap_output = hp.heap()
    print(heap_output)
    # Graphical Browser
    #hp.pb()

def garbage_collect():
    import gc
    gc.collect()
#http://www.huyng.com/posts/python-performance-analysis/
#Once youve gotten your code setup with the @profile decorator, use kernprof.py to run your script.
#kernprof.py -l -v fib.py

#---------------
# printing and logging
#---------------

__STDOUT__ = sys.stdout
__STDERR__ = sys.stdout

def reset_streams():
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = __STDOUT__
    sys.stderr = __STDERR__
    sys.stdout.flush()
    sys.stderr.flush()
    print('helprs> Reset stdout and stderr')

def print_list(list):
    if list is None: return 'None'
    msg = '\n'.join([repr(item) for item in list])
    print(msg)
    return msg

def _print(msg):
    sys.stdout.write(msg)

def _println(msg):
    sys.stdout.write(msg+'\n')

def println(msg, *args):
    args = args+tuple('\n',)
    return print_(msg, *args)
def print_(msg, *args):
    msg_ = str(msg)+''.join(map(str,args))
    sys.stdout.write(msg_)
    return msg_
def flush():
    sys.stdout.flush()
    return ''
def endl():
    print_('\n')
    sys.stdout.flush()
    return '\n'
def printINFO(msg, *args):
    msg = 'INFO: '+str(msg)+''.join(map(str,args))
    return println(msg, *args)

def printDBG(msg, *args):
    msg = 'DEBUG: '+str(msg)+''.join(map(str,args))
    return println(msg, *args)

def printERR(msg, *args):
    msg = 'ERROR: '+str(msg)+''.join(map(str,args))
    raise Exception(msg)
    return println(msg, *args)

def printWARN(warn_msg, category=UserWarning):
    warn_msg = 'Warning: '+warn_msg
    sys.stdout.write(warn_msg+'\n')
    sys.stdout.flush()
    warnings.warn(warn_msg, category=category)
    sys.stdout.flush()
    return warn_msg

#---------------
def try_cast(var, type_):
    if type_ is None:
        return var
    try:
        return type_(var)
    except Exception as ex:
        return None

def get_arg_after(arg, type_=None):
    arg_after = None
    try:
        arg_index = sys.argv.index(arg)
        if arg_index < len(sys.argv):
            arg_after = try_cast(sys.argv[arg_index+1], type_)
    except Exception as ex:
        pass
    return arg_after

def listfind(list_, tofind):
    try: 
        return list_.index(tofind)
    except ValueError:
        return None

# Tests for data types
def is_int(num):
    valid_int_types = (np.int64,  np.int32,  np.int16,  np.int8,
                            np.uint64, np.uint32, np.uint16, np.uint8)
    valid_int_types = (np.typeDict['int64'],
                            np.typeDict['int32'],
                            np.typeDict['uint8'],
                            types.LongType,
                            types.IntType)
    flag = type(num) in valid_int_types
    return flag

def is_float(num):
    valid_float_types = (float, np.float64, np.float32, np.float16)
    valid_float_types = (types.FloatType,
                    np.typeDict['float64'],
                    np.typeDict['float32'],
                    np.typeDict['float16'])
    flag = type(num) in valid_float_types
    return flag

def num_fmt(num, max_digits=2):
    if is_float(num):
        return '%.'+str(max_digits)+'f'
    elif is_int(num):
        return '%d'
    else:
        return '%r'

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('[helpers] You ran helpers as main!')
    import algos
    import sklearn
    module = sys.modules[__name__]
    seen = set(['numpy','matplotlib', 'scipy', 'pyflann', 'sklearn', 'skimage', 'cv2'])

    hs2_basic = set(['draw_func2', 'params', 'mc2'])
    python_basic = set(['os','sys', 'warnings', 'inspect','copy', 'imp','types'])
    tpl_basic = set(['pyflann', 'cv2'])
    science_basic = set(['numpy',
                         'matplotlib',
                         'matplotlib.pyplot',
                         'scipy',
                         'scipy.sparse'])
    seen = set(list(python_basic) + list(science_basic) + list(tpl_basic))
    seen = set([])
    print('[helpers] seen=%r' % seen)
    explore_module(module, maxdepth=0, seen=seen, nonmodules=False)
