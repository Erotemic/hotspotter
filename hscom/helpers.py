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
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[helpers]')
# Scientific
import numpy as np
# Standard
from collections import OrderedDict
from itertools import product as iprod
from os.path import (join, relpath, normpath, split, isdir, isfile, exists,
                     islink, ismount, expanduser)
import cPickle
import cStringIO
import datetime
import decimal
import fnmatch
import hashlib
import inspect
import os
import platform
import shutil
import sys
import textwrap
import time
import types
import warnings
# HotSpotter
import tools
from Printable import printableVal
#print('LOAD_MODULE: helpers.py')

# --- Globals ---

__IMG_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ppm']
__LOWER_EXTS = [ext.lower() for ext in __IMG_EXTS]
__UPPER_EXTS = [ext.upper() for ext in __IMG_EXTS]
IMG_EXTENSIONS =  set(__LOWER_EXTS + __UPPER_EXTS)

PRINT_CHECKS = False  # True
__PRINT_WRITES__ = False
__CHECKPATH_VERBOSE__ = False

VERY_VERBOSE = False


def DEPRICATED(func):
    'deprication decorator'
    warn_msg = 'Depricated call to: %s' % func.__name__

    def __DEP_WRAPPER(*args, **kwargs):
        raise Exception('dep')
        warnings.warn(warn_msg, category=DeprecationWarning)
        #printWARN(warn_msg, category=DeprecationWarning)
        return func(*args, **kwargs)
    __DEP_WRAPPER.__name__ = func.__name__
    __DEP_WRAPPER.__doc__ = func.__doc__
    __DEP_WRAPPER.__dict__.update(func.__dict__)
    return __DEP_WRAPPER


def try_get_path(path_list):
    tried_list = []
    for path in path_list:
        if path.find('~') != -1:
            path = expanduser(path)
        tried_list.append(path)
        if exists(path):
            return path
    return (False, tried_list)


def get_lena_fpath():
    possible_lena_locations = [
        'lena.png',
        '~/code/hotspotter/_tpl/extern_feat/lena.png',
        '_tpl/extern_feat/lena.png',
        '../_tpl/extern_feat/lena.png',
        '~/local/lena.png',
        '../lena.png',
        '/lena.png',
        'C:\\lena.png']
    lena_fpath = try_get_path(possible_lena_locations)
    if not isinstance(lena_fpath, str):
        raise Exception('cannot find lena: tried: %r' % (lena_fpath,))
    return lena_fpath


def horiz_print(*args):
    toprint = horiz_string(args)
    print(toprint)


def horiz_string(str_list):
    '''
    str_list = ['A = ', str(np.array(((1,2),(3,4)))), ' * ', str(np.array(((1,2),(3,4))))]
    '''
    all_lines = []
    hpos = 0
    for sx in xrange(len(str_list)):
        str_ = str(str_list[sx])
        lines = str_.split('\n')
        line_diff = len(lines) - len(all_lines)
        # Vertical padding
        if line_diff > 0:
            all_lines += [' ' * hpos] * line_diff
        # Add strings
        for lx, line in enumerate(lines):
            all_lines[lx] += line
            hpos = max(hpos, len(all_lines[lx]))
        # Horizontal padding
        for lx in xrange(len(all_lines)):
            hpos_diff = hpos - len(all_lines[lx])
            if hpos_diff > 0:
                all_lines[lx] += ' ' * hpos_diff
    ret = '\n'.join(all_lines)
    return ret


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
    img_pats = ['*' + ext for ext in IMG_EXTENSIONS]
    return any([fnmatch.fnmatch(fname_, pat) for pat in img_pats])


def list_images(img_dpath, ignore_list=[], recursive=True, fullpath=False):
    ignore_set = set(ignore_list)
    gname_list_ = []
    assert_path(img_dpath)
    # Get all the files in a directory recursively
    for root, dlist, flist in os.walk(img_dpath):
        for fname in iter(flist):
            gname = join(relpath(root, img_dpath), fname).replace('\\', '/').replace('./', '')
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


def indent(string, indent='    '):
    return indent + string.replace('\n', '\n' + indent)


def truncate_str(str, maxlen=110):
    if len(str) < maxlen:
        return str
    else:
        truncmsg = ' ~~~TRUNCATED~~~ '
        maxlen_ = maxlen - len(truncmsg)
        lowerb  = int(maxlen_ * .8)
        upperb  = maxlen_ - lowerb
        return str[:lowerb] + truncmsg + str[-upperb:]


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
        newlines[-1] += word + ' '
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


@DEPRICATED
def array_index(array, item):
    return np.where(array == item)[0][0]


@DEPRICATED
def index_of(item, array):
    'index of [item] in [array]'
    return np.where(array == item)[0][0]


def intersect2d_numpy(A, B):
    #http://stackoverflow.com/questions/8317022/
    #get-intersecting-rows-across-two-2d-numpy-arrays/8317155#8317155
    nrows, ncols = A.shape
    # HACK to get consistent dtypes
    assert A.dtype is B.dtype, 'A and B must have the same dtypes'
    dtype = np.dtype([('f%d' % i, A.dtype) for i in range(ncols)])
    try:
        C = np.intersect1d(A.view(dtype), B.view(dtype))
    except ValueError:
        C = np.intersect1d(A.copy().view(dtype), B.copy().view(dtype))
    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C


def intersect2d(A, B):
    Cset  =  set(tuple(x) for x in A).intersection(set(tuple(x) for x in B))
    Ax = np.array([x for x, item in enumerate(A) if tuple(item) in Cset], dtype=np.int)
    Bx = np.array([x for x, item in enumerate(B) if tuple(item) in Cset], dtype=np.int)
    C = np.array(tuple(Cset))
    return C, Ax, Bx


def unique_keep_order(arr):
    'pandas.unique preseves order and seems to be faster due to index overhead'
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


# --- Info Strings ---
def printable_mystats(_list):
    stat_dict = mystats(_list)
    stat_strs = ['%r:%s' % (key, val) for key, val in stat_dict.iteritems()]
    ret = '{' + ', '.join(stat_strs) + '}'
    return ret
#def mystats2_latex(mystats):
    #statdict_ = eval(mystats)


def mystats(_list):
    if len(_list) == 0:
        return {'empty_list': True}
    nparr = np.array(_list)
    min_val = nparr.min()
    max_val = nparr.max()
    nMin = np.sum(nparr == min_val)  # number of entries with min val
    nMax = np.sum(nparr == max_val)  # number of entries with min val
    return OrderedDict([('max',   np.float32(max_val)),
                        ('min',   np.float32(min_val)),
                        ('mean',  np.float32(nparr.mean())),
                        ('std',   np.float32(nparr.std())),
                        ('nMin',  np.int32(nMin)),
                        ('nMax',  np.int32(nMax)),
                        ('shape', repr(nparr.shape))])


def myprint(input=None, prefix='', indent='', lbl=''):
    if len(lbl) > len(prefix):
        prefix = lbl
    if len(prefix) > 0:
        prefix += ' '
    _print(indent + prefix + str(type(input)) + ' ')
    if isinstance(input, list):
        _println(indent + '[')
        for item in iter(input):
            myprint(item, indent=indent + '  ')
        _println(indent + ']')
    elif isinstance(input, str):
        _println(input)
    elif isinstance(input, dict):
        _println(printableVal(input))
    else:
        _println(indent + '{')
        attribute_list = dir(input)
        for attr in attribute_list:
            if attr.find('__') == 0:
                continue
            val = str(input.__getattribute__(attr))
            #val = input[attr]
            # Format methods nicer
            #if val.find('built-in method'):
                #val = '<built-in method>'
            _println(indent + '  ' + attr + ' : ' + val)
        _println(indent + '}')


def info(var, lbl):
    if isinstance(var, np.ndarray):
        return npinfo(var, lbl)
    if isinstance(var, list):
        return listinfo(var, lbl)


def npinfo(ndarr, lbl='ndarr'):
    info = ''
    info += (lbl + ': shape=%r ; dtype=%r' % (ndarr.shape, ndarr.dtype))
    return info


def listinfo(list_, lbl='ndarr'):
    if not isinstance(list_, list):
        raise Exception('!!')
    info = ''
    type_set = set([])
    for _ in iter(list_):
        type_set.add(str(type(_)))
    info += (lbl + ': len=%r ; types=%r' % (len(list_), type_set))
    return info


#expected_type = np.float32
#expected_dims = 5
def numpy_list_num_bits(nparr_list, expected_type, expected_dims):
    num_bits = 0
    num_items = 0
    num_elemt = 0
    bit_per_item = {
        np.float32: 32,
        np.uint8: 8
    }[expected_type]
    for nparr in iter(nparr_list):
        arr_len, arr_dims = nparr.shape
        if not nparr.dtype.type is expected_type:
            msg = 'Expected Type: ' + repr(expected_type)
            msg += 'Got Type: ' + repr(nparr.dtype)
            raise Exception(msg)
        if arr_dims != expected_dims:
            msg = 'Expected Dims: ' + repr(expected_dims)
            msg += 'Got Dims: ' + repr(arr_dims)
            raise Exception(msg)
        num_bits += len(nparr) * expected_dims * bit_per_item
        num_elemt += len(nparr) * expected_dims
        num_items += len(nparr)
    return num_bits,  num_items, num_elemt


def public_attributes(input):
    public_attr_list = []
    all_attr_list = dir(input)
    for attr in all_attr_list:
        if attr.find('__') == 0:
            continue
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
                if not isinstance(childtype, types.ModuleType):
                    if nonmodules:
                        #print_(depth)
                        fullstr = indent + '    ' + str(aname) + ' = ' + repr(child)
                        truncstr = truncate_str(fullstr) + '\n'
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
        # print_(depth)
        ret += indent + modname + '\n'
        # Recurse
        if not maxdepth is None and depth >= maxdepth:
            return ret
        ret += ''.join([__explore_module(child,
                                         indent + '    ',
                                         seen, depth + 1,
                                         maxdepth,
                                         nonmodules)
                       for child in iter(valid_children)])
        return ret
    #ret +=
    #println('#module = ' + str(module_))
    ret = __explore_module(module_, '     ', seen, 0, maxdepth, nonmodules)
    #print(ret)
    flush()
    return ret


# --- Util ---
def alloc_lists(num_alloc):
    'allocates space for a numpy array of lists'
    return [[] for _ in xrange(num_alloc)]


def ensure_list_size(list_, size_):
    'extend list to max_cx'
    lendiff = (size_) - len(list_)
    if lendiff > 0:
        extension = [None for _ in xrange(lendiff)]
        list_.extend(extension)


def get_timestamp(format_='filename', use_second=False):
    now = datetime.datetime.now()
    if use_second:
        time_tup = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        time_formats = {
            'filename': 'ymd_hms-%04d-%02d-%02d_%02d-%02d-%02d',
            'comment': '# (yyyy-mm-dd hh:mm:ss) %04d-%02d-%02d %02d:%02d:%02d'}
    else:
        time_tup = (now.year, now.month, now.day, now.hour, now.minute)
        time_formats = {
            'filename': 'ymd_hm-%04d-%02d-%02d_%02d-%02d',
            'comment': '# (yyyy-mm-dd hh:mm) %04d-%02d-%02d %02d:%02d'}
    stamp = time_formats[format_] % time_tup
    return stamp

VALID_PROGRESS_TYPES = ['none', 'dots', 'fmtstr', 'simple']


# TODO: Return start_prog, make_prog, end_prog
def progress_func(max_val=0, lbl='Progress: ', mark_after=-1,
                  flush_after=4, spacing=0, line_len=80,
                  progress_type='fmtstr'):
    '''Returns a function that marks progress taking the iteration count as a
    parameter. Prints if max_val > mark_at. Prints dots if max_val not
    specified or simple=True'''
    # Tell the user we are about to make progress
    if progress_type in ['simple', 'fmtstr'] and max_val < mark_after:
        return lambda count: None, lambda: None
    print(lbl)
    # none: nothing
    if progress_type == 'none':
        mark_progress =  lambda count: None
    # simple: one dot per progress. no flush.
    if progress_type == 'simple':
        mark_progress = lambda count: sys.stdout.write('.')
    # dots: spaced dots
    if progress_type == 'dots':
        indent_ = '    '
        sys.stdout.write(indent_)

        if spacing > 0:
            # With spacing
            newline_len = spacing * line_len // spacing

            def mark_progress_sdot(count):
                sys.stdout.write('.')
                count_ = count + 1
                if (count_) % newline_len == 0:
                    sys.stdout.write('\n' + indent_)
                    sys.stdout.flush()
                elif (count_) % spacing == 0:
                    sys.stdout.write(' ')
                    sys.stdout.flush()
                elif (count_) % flush_after == 0:
                    sys.stdout.flush()
            mark_progress = mark_progress_sdot
        else:
            # No spacing
            newline_len = line_len

            def mark_progress_dot(count):
                sys.stdout.write('.')
                count_ = count + 1
                if (count_) % newline_len == 0:
                    sys.stdout.write('\n' + indent_)
                    sys.stdout.flush()
                elif (count_) % flush_after == 0:
                    sys.stdout.flush()
            mark_progress = mark_progress_dot
    # fmtstr: formated string progress
    if progress_type == 'fmtstr':
        fmt_str = progress_str(max_val, lbl=lbl)

        def mark_progress_fmtstr(count):
            count_ = count + 1
            sys.stdout.write(fmt_str % (count_))
            if (count_) % flush_after == 0:
                sys.stdout.flush()
        mark_progress = mark_progress_fmtstr
    # FIXME idk why argparse2.ARGS_ is none here.
    if '--aggroflush' in sys.argv:
        def mark_progress_agressive(count):
            mark_progress(count)
            sys.stdout.flush()
        return mark_progress_agressive

    def end_progress():
        print('')
    return mark_progress, end_progress
    raise Exception('unkown progress type = %r' % progress_type)


def progress_str(max_val, lbl='Progress: '):
    r'makes format string that prints progress: %Xd/MAX_VAL with backspaces'
    max_str = str(max_val)
    dnumstr = str(len(max_str))
    fmt_str = lbl + '%' + dnumstr + 'd/' + max_str
    fmt_str = '\b' * (len(fmt_str) - len(dnumstr) + len(max_str)) + fmt_str
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


def my_computer_names():
    return ['Ooo', 'Hyrule', 'BakerStreet']


def get_computer_name():
    return platform.node()


def win_shortcut(source, link_name):
    import ctypes
    csl = ctypes.windll.kernel32.CreateSymbolicLinkW
    csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    csl.restype = ctypes.c_ubyte
    flags = 1 if isdir(source) else 0
    retval = csl(link_name, source, flags)
    if retval == 0:
        #warn_msg = '[helpers] Unable to create symbolic link on windows.'
        #print(warn_msg)
        #warnings.warn(warn_msg, category=UserWarning)
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
    except Exception:
        checkpath(link_name, True)
        checkpath(source, True)
        if not noraise:
            raise


# --- Context ---
def inIPython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def haveIPython():
    try:
        import IPython  # NOQA
        return True
    except NameError:
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
        IPython.embed_kernel(module=None, local_ns=namespace)
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
        #print(local_varnames[0:360] + '...')#hack
    #else:
    print('--- End Frame ---')


def search_stack_for_localvar(varname):
    curr_frame = inspect.currentframe()
    print(' * Searching parent frames for: ' + str(varname))
    frame_no = 0
    while not curr_frame.f_back is None:
        if varname in curr_frame.f_locals.keys():
            print(' * Found in frame: ' + str(frame_no))
            return curr_frame.f_locals[varname]
        frame_no += 1
        curr_frame = curr_frame.f_back
    print('... Found nothing in all ' + str(frame_no) + ' frames.')
    return None


def get_parent_locals():
    this_frame = inspect.currentframe()
    call_frame = this_frame.f_back
    parent_frame = call_frame.f_back
    if parent_frame is None:
        return None
    return parent_frame.f_locals


def get_caller_locals():
    this_frame = inspect.currentframe()
    call_frame = this_frame.f_back
    if call_frame is None:
        return None
    return call_frame.f_locals


# --- Convinience ----
def vd(dname=None):
    'view directory'
    print('[helpers] view_dir(%r) ' % dname)
    dname = os.getcwd() if dname is None else dname
    open_prog = {'win32': 'explorer.exe',
                 'linux2': 'nautilus',
                 'darwin': 'open'}[sys.platform]
    os.system(open_prog + ' ' + normpath(dname))


def str2(obj):
    if isinstance(obj, dict):
        return str(obj).replace(', ', '\n')[1:-1]
    if isinstance(obj, type):
        return str(obj).replace('<type \'', '').replace('\'>', '')
    else:
        return str(obj)


def tiled_range(range, cols):
    return np.tile(np.arange(range), (cols, 1)).T
    #np.tile(np.arange(num_qf).reshape(num_qf, 1), (1, k_vsmany))


def random_indexes(max_index, subset_size):
    subst_ = np.arange(0, max_index)
    np.random.shuffle(subst_)
    subst = subst_[0:min(subset_size, max_index)]
    return subst


#def gvim(fname):
    #'its the only editor that matters'
    #import subprocess
    #proc = subprocess.Popen(['gvim',fname])


def cmd(command):
    os.system(command)


# --- Path ---
#@DEPRICATED
def filecheck(fpath):
    return exists(fpath)


@DEPRICATED
def dircheck(dpath, makedir=True):
    if not exists(dpath):
        if not makedir:
            #print('Nonexistant directory: %r ' % dpath)
            return False
        print('Making directory: %r' % dpath)
        os.makedirs(dpath)
    return True


def remove_file(fpath, verbose=True, dryrun=False, **kwargs):
    try:
        if dryrun:
            if verbose:
                print('[helpers] Dryrem %r' % fpath)
        else:
            if verbose:
                print('[helpers] Removing %r' % fpath)
            os.remove(fpath)
    except OSError as e:
        printWARN('OSError: %s,\n Could not delete %s' % (str(e), fpath))
        return False
    return True


def remove_dirs(dpath, dryrun=False, **kwargs):
    print('[helpers] Removing directory: %r' % dpath)
    try:
        shutil.rmtree(dpath)
    except OSError as e:
        printWARN('OSError: %s,\n Could not delete %s' % (str(e), dpath))
        return False
    return True


def remove_files_in_dir(dpath, fname_pattern='*', recursive=False, verbose=True,
                        dryrun=False, **kwargs):
    print('[helpers] Removing files:')
    print('  * in dpath = %r ' % dpath)
    print('  * matching pattern = %r' % fname_pattern)
    print('  * recursive = %r' % recursive)
    num_removed, num_matched = (0, 0)
    if not exists(dpath):
        msg = ('!!! dir = %r does not exist!' % dpath)
        print(msg)
        warnings.warn(msg, category=UserWarning)
    for root, dname_list, fname_list in os.walk(dpath):
        for fname in fnmatch.filter(fname_list, fname_pattern):
            num_matched += 1
            num_removed += remove_file(join(root, fname), verbose=verbose,
                                       dryrun=dryrun, **kwargs)
        if not recursive:
            break
    print('[helpers] ... Removed %d/%d files' % (num_removed, num_matched))
    return True


def delete(path, dryrun=False, recursive=True, verbose=True, **kwargs):
    print('[helpers] Deleting path=%r' % path)
    rmargs = dict(dryrun=dryrun, recursive=recursive, verbose=verbose, **kwargs)
    if not exists(path):
        msg = ('..does not exist!')
        print(msg)
        return False
    if isdir(path):
        flag = remove_files_in_dir(path, **rmargs)
        flag = flag and remove_dirs(path, **rmargs)
    elif isfile(path):
        flag = remove_file(path, **rmargs)
    return flag


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
    for i in xrange(n - 1):
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
        print = sys.stdout.write
        pretty_path = path_ndir_split(path_, 2)
        caller_name = get_caller_name()
        print_('[%s] checkpath(%r)' % (caller_name, pretty_path))
        if exists(path_):
            path_type = ''
            if isfile(path_):
                path_type += 'file'
            if isdir(path_):
                path_type += 'directory'
            if islink(path_):
                path_type += 'link'
            if ismount(path_):
                path_type += 'mount'
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
        raise AssertionError('Asserted path does not exist: ' + path_)


def assert_path(path_):
    return assertpath(path_)


def join_mkdir(*args):
    'join and creates if not exists'
    output_dir = join(*args)
    if not exists(output_dir):
        print('Making dir: ' + output_dir)
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
    _cp_tasks = []  # Build this list with the actual tasks
    if nooverwrite:
        print('[helpers] Removed: copy task ')
    else:
        print('[helpers] Begining copy + overwrite task.')
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
            print('[helpers]    ' + src + ' -> \n    ' + dst)
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
    if exists(src):
        if exists(dst):
            prefix = 'C+O'
            print('[helpers] [Copying + Overwrite]:')
        else:
            prefix = 'C'
            print('[helpers] [Copying]: ')
        print('[%s] | %s' % (prefix, src))
        print('[%s] ->%s' % (prefix, dst))
        shutil.copy(src, dst)
    else:
        prefix = 'Miss'
        print('[helpers] [Cannot Copy]: ')
        print('[%s] src=%s does not exist!' % (prefix, src))
        print('[%s] dst=%s' % (prefix, dst))


def copy_all(src_dir, dest_dir, glob_str_list):
    if not isinstance(glob_str_list, list):
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
    if not isinstance(string, str):  # -> convert input to a string
        string = repr(string)
    matching_lines = []  # Find all matching lines
    for line in string.split('\n'):
        if not fnmatch.fnmatch(string, pattern):
            continue
        matching_lines.append(line)
    return matching_lines


def correct_zeros(M):
    index_gen = iprod(*[xrange(_) for _ in M.shape])
    for index in index_gen:
        if M[index] < 1E-18:
            M[index] = 0
    return M


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
    print('Matching Lines:')  # Print matching lines
    print('\n    '.join(matching_lines))


def print_glob(*args, **kwargs):
    matching_fnames = glob(*args, **kwargs)
    print('Matching Fnames:')  # Print matching fnames
    print('\n    '.join(matching_fnames))


#---------------
# save / load / cache functions
def sanatize_fname2(fname):
    fname = fname.replace(' ', '_')
    return fname


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
        text = open(fpath, 'r').read()
    except Exception:
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
    print('[helpers] filesize is: ' + file_megabytes_str(fpath))
    npz = np.load(fpath, mmap_mode='r + ')
    data = tuple(npz[key] for key in sorted(npz.keys()))
    #print(' * npz.keys() = %r ' + str(npz.keys()))
    npz.close()
    return data


def dict_union2(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def dict_union(*args):
    return dict([item for dict_ in iter(args) for item in dict_.iteritems()])


def hashstr_arr(arr, lbl='arr', **kwargs):
    if isinstance(arr, list):
        arr = tuple(arr)
    if isinstance(arr, tuple):
        arr_shape = '(' + str(len(arr)) + ')'
    else:
        arr_shape = str(arr.shape).replace(' ', '')
    arr_hash = hashstr(arr, **kwargs)
    arr_uid = ''.join((lbl, '(', arr_shape, arr_hash, ')'))
    return arr_uid


def hashstr(data, trunc_pos=8):
    if isinstance(data, tuple):
        data = repr(data)
    # Get a 128 character hex string
    hashstr = hashlib.sha512(data).hexdigest()
    # Convert to base 57
    hashstr2 = hex2_base57(hashstr)
    # Truncate
    hashstr = hashstr2[:trunc_pos]
    return hashstr


class ModulePrintLock():
    '''Temporarily turns off printing while still in scope
    chosen modules must have a print_off function
    '''
    def __init__(self, *args):
        self.module_list = args
        for module in self.module_list:
            module.print_off()

    def __del__(self):
        for module in self.module_list:
            module.print_on()

#def valid_filename_ascii_chars():
    ## Find invalid chars
    #ntfs_inval = '< > : " / \ | ? *'.split(' ')
    #other_inval = [' ', '\'', '.']
    ##case_inval = map(chr, xrange(97, 123))
    #case_inval = map(chr, xrange(65, 91))
    #invalid_chars = set(ntfs_inval + other_inval + case_inval)
    ## Find valid chars
    #valid_chars = []
    #for index in xrange(32, 127):
        #char = chr(index)
        #if not char in invalid_chars:
            #print index, chr(index)
            #valid_chars.append(chr(index))
    #return valid_chars
#valid_filename_ascii_chars()
# I Removed two characters that made awkward filenames
#ALPHABET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  'a', 'b', 'c',
            #'d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            #'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ';', '=', '@',
            #'[', ']', '^', '_', '`', '{', '}', '~', '!', '#', '$', '%', '&',
            #'(', ')', '+', ',', '-']
ALPHABET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  'a', 'b', 'c',
            'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ';', '=', '@',
            '[', ']', '^', '_', '`', '{', '}', '~', '!', '#', '$', '%', '&',
            '+', ',']

BIGBASE = len(ALPHABET)


def hex2_base57(hexstr):
    x = int(hexstr, 16)
    if x == 0:
        return '0'
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
    #bin(int(my_hexdata, scale))
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


#def cache_npz_decorator(npz_func):
    #def __func_wrapper(input_data, *args, **kwargs):
        #ret = npz_func(*args, **kwargs)


class CacheException(Exception):
    pass


def __cache_data_fpath(input_data, uid, cache_dir):
    hashstr_   = hashstr(input_data)
    shape_lbl  = str(input_data.shape).replace(' ', '')
    data_fname = uid + '_' + shape_lbl + '_' + hashstr_ + '.npz'
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
    def __init__(self, lbl=None, autostart=False, show_on_exit=True):
        self._stdout_old = sys.stdout
        self.stream = cStringIO.StringIO()
        self.record = '<no record>'
        self.lbl = lbl
        self.show_on_exit = show_on_exit
        if autostart:
            self.start()

    def start(self):
        sys.stdout.flush()
        sys.stdout = self.stream

    def stop(self):
        self.stream.flush()
        sys.stdout = self._stdout_old
        self.stream.seek(0)
        self.record = self.stream.read()
        return self.record

    def update(self):
        self.stop()
        self.dump()
        self.start()

    def dump(self):
        print(indent(self.record, self.lbl))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        if not self.lbl is None:
            if self.show_on_exit:
                self.dump()


class Indenter(RedirectStdout):
    def __init__(self, lbl='    '):
        super(Indenter, self).__init__(lbl=lbl, autostart=True)


def choose(n, k):
    import scipy.misc
    return scipy.misc.comb(n, k, True)


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


def ipython_execstr():
    return textwrap.dedent(r'''
    import matplotlib.pyplot as plt
    import sys
    embedded = False
    try:
        __IPYTHON__
        in_ipython = True
    except NameError:
        in_ipython = False
    try:
        import IPython
        have_ipython = True
    except NameError:
        have_ipython = False
    if in_ipython:
        print('Presenting in current ipython shell.')
    elif '--cmd' in sys.argv or 'devmode' in vars():
        print('[helpers] Requested IPython shell with --cmd argument.')
        if have_ipython:
            print('[helpers] Found IPython')
            try:
                import IPython
                print('[helpers] Presenting in new ipython shell.')
                embedded = True
                IPython.embed()
            except Exception as ex:
                print(repr(ex)+'\n!!!!!!!!')
                embedded = False
        else:
            print('[helpers] IPython is not installed')
    ''')


def execstr_parent_locals():
    parent_locals = get_parent_locals()
    return execstr_dict(parent_locals, 'parent_locals')


def execstr_attr_list(obj_name, attr_list=None):
    #if attr_list is None:
        #exec(execstr_parent_locals())
        #exec('attr_list = dir('+obj_name+')')
    execstr_list = [obj_name + '.' + attr for attr in attr_list]
    return execstr_list


def execstr_dict(dict_, local_name, exclude_list=None):
    #if local_name is None:
        #local_name = dict_
        #exec(execstr_parent_locals())
        #exec('dict_ = local_name')
    if exclude_list is None:
        execstr = '\n'.join((key + ' = ' + local_name + '[' + repr(key) + ']'
                            for (key, val) in dict_.iteritems()))
    else:
        if not isinstance(exclude_list, list):
            exclude_list = [exclude_list]
        exec_list = []
        for (key, val) in dict_.iteritems():
            if not any((fnmatch.fnmatch(key, pat) for pat in iter(exclude_list))):
                exec_list.append(key + ' = ' + local_name + '[' + repr(key) + ']')
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
        except Exception:
            val_str = repr(val)  # NOQA
        exec_list.append(key + ' = ' + repr(dict_[key]))
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


@DEPRICATED
def dict_execstr(dict_, local_name=None):
    return execstr_dict(dict_, local_name)


def execstr_func(func):
    print(' ! Getting executable source for: ' + func.func_name)
    _src = inspect.getsource(func)
    execstr = textwrap.dedent(_src[_src.find(':') + 1:])
    # Remove return statments
    while True:
        stmtx = execstr.find('return')  # Find first 'return'
        if stmtx == -1:
            break  # Fail condition
        # The characters which might make a return not have its own line
        stmt_endx = len(execstr) - 1
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
        execstr = before + after
    return execstr


def execstr_src(func):
    return execstr_func(func)


@DEPRICATED
def get_exec_src(func):
    return execstr_func(func)


# --- Profiling ---
def unit_test(test_func):
    test_name = test_func.func_name

    def __unit_test_wraper():
        print('Testing: ' + test_name)
        try:
            ret = test_func()
        except Exception as ex:
            print(repr(ex))
            print('Tested: ' + test_name + ' ...FAILURE')
            raise
        print('Tested: ' + test_name + ' ...SUCCESS')
        return ret
    return __unit_test_wraper


def runprofile(cmd, globals_=globals(), locals_=locals()):
    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
    import cProfile
    import sys
    import os
    print('[helpers] Profiling Command: ' + cmd)
    cProfOut_fpath = 'OpenGLContext.profile'
    cProfile.runctx( cmd, globals_, locals_, filename=cProfOut_fpath)
    # RUN SNAKE
    print('[helpers] Profiled Output: ' + cProfOut_fpath)
    if sys.platform == 'win32':
        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
    else:
        rsr_fpath = 'runsnake'
    view_cmd = rsr_fpath + ' "' + cProfOut_fpath + '"'
    os.system(view_cmd)
    return True

'''
def profile_lines(fname):
    import __init__
    script = 'dev.py'
    args = '--db MOTHERS --nocache-feat'
    runcmd = 'kernprof.py %s %s' % (script, args)
    viewcmd = 'python -m line_profiler %s.lprof' % script
    hs_path = split(__init__.__file__)
    lineprofile_path = join(hs_path, '.lineprofile')
    ensurepath(lineprofile_path)
    shutil.copy('*', lineprofile_path + '/*')
    '''


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
    if list is None:
        return 'None'
    msg = '\n'.join([repr(item) for item in list])
    print(msg)
    return msg


def _print(msg):
    sys.stdout.write(msg)


def _println(msg):
    sys.stdout.write(msg + '\n')


def println(msg, *args):
    args = args + tuple('\n',)
    return print_(msg + ''.join(args))


def flush():
    sys.stdout.flush()
    return ''


def endl():
    print_('\n')
    sys.stdout.flush()
    return '\n'


def printINFO(msg, *args):
    msg = 'INFO: ' + str(msg) + ''.join(map(str, args))
    return println(msg, *args)


def printDBG(msg, *args):
    msg = 'DEBUG: ' + str(msg) + ''.join(map(str, args))
    return println(msg, *args)


def printERR(msg, *args):
    msg = 'ERROR: ' + str(msg) + ''.join(map(str, args))
    raise Exception(msg)
    return println(msg, *args)


def printWARN(warn_msg, category=UserWarning):
    warn_msg = 'Probably not a big issue, but you should know...: ' + warn_msg
    sys.stdout.write(warn_msg + '\n')
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
    except Exception:
        return None


def get_arg(arg, type_=None, default=None):
    arg_after = default
    try:
        arg_index = sys.argv.index(arg)
        if arg_index < len(sys.argv):
            arg_after = try_cast(sys.argv[arg_index + 1], type_)
    except Exception:
        pass
    return arg_after


def get_flag(arg, default=False):
    'Checks if the commandline has a flag or a corresponding noflag'
    if arg.find('--') != 0:
        raise Exception(arg)
    if arg.find('--no') == 0:
        arg = arg.replace('--no', '--')
    noarg = arg.replace('--', '--no')
    if arg in sys.argv:
        return True
    elif noarg in sys.argv:
        return False
    else:
        return default
    return default


def listfind(list_, tofind):
    try:
        return list_.index(tofind)
    except ValueError:
        return None


def num_fmt(num, max_digits=1):
    if tools.is_float(num):
        return ('%.' + str(max_digits) + 'f') % num
    elif tools.is_int(num):
        return int_comma_str(num)
    else:
        return '%r'


def int_comma_str(num):
    int_str = ''
    reversed_digits = decimal.Decimal(num).as_tuple()[1][::-1]
    for i, digit in enumerate(reversed_digits):
        if (i) % 3 == 0 and i != 0:
            int_str += ','
        int_str += str(digit)
    return int_str[::-1]


def fewest_digits_float_str(num, n=8):
    int_part = int(num)
    dec_part = num - int_part
    x = decimal.Decimal(dec_part, decimal.Context(prec=8))
    decimal_list = x.as_tuple()[1]
    nonzero_pos = 0
    for i in range(0, min(len(decimal_list), n)):
        if decimal_list[i] != 0:
            nonzero_pos = i
    sig_dec = int(dec_part * 10 ** (nonzero_pos + 1))
    float_str = int_comma_str(int_part) + '.' + str(sig_dec)
    return float_str
    #x.as_tuple()[n]


def commas(num, n=8):
    if tools.is_float(num):
        #ret = sigfig_str(num, n=2)
        ret = '%.3f' % num
        return ret
        #return fewest_digits_float_str(num, n)
    return '%d' % num
    #return int_comma_str(num)


def printshape(arr_name, locals_):
    arr = locals_[arr_name]
    if type(arr) is np.ndarray:
        print(arr_name + '.shape = ' + str(arr.shape))
    else:
        print('len(%s) = %r' % (arr_name, len(arr)))


def printvar2(varstr, attr=''):
    locals_ = get_parent_locals()
    printvar(locals_, varstr, attr)


def printvar(locals_, varname, attr='.shape'):
    import tools
    npprintopts = np.get_printoptions()
    np.set_printoptions(threshold=5)
    dotpos = varname.find('.')
    # Locate var
    if dotpos == -1:
        var = locals_[varname]
    else:
        varname_ = varname[:dotpos]
        dotname_ = varname[dotpos:]
        var_ = locals_[varname_]  # NOQA
        var = eval('var_' + dotname_)
    # Print in format
    typestr = tools.get_type(var)
    if isinstance(var, np.ndarray):
        varstr = eval('str(var' + attr + ')')
        print('[var] %s %s = %s' % (typestr, varname + attr, varstr))
    elif isinstance(var, list):
        if attr == '.shape':
            func = 'len'
        else:
            func = ''
        varstr = eval('str(' + func + '(var))')
        print('[var] %s len(%s) = %s' % (typestr, varname, varstr))
    else:
        print('[var] %s %s = %r' % (typestr, varname, var))
    np.set_printoptions(**npprintopts)


def format(num, n=8):
    '''makes numbers pretty e.g.
    nums = [9001, 9.053]
    print([format(num) for num in nums])
    '''
    if num is None:
        return 'None'
    if tools.is_float(num):
        ret = ('%.' + str(n) + 'E') % num
        exp_pos  = ret.find('E')
        exp_part = ret[(exp_pos + 1):]
        exp_part = exp_part.replace('+', '')
        if exp_part.find('-') == 0:
            exp_part = '-' + exp_part[1:].strip('0')
        exp_part = exp_part.strip('0')
        if len(exp_part) > 0:
            exp_part = 'E' + exp_part
        flt_part = ret[:exp_pos].strip('0').strip('.')
        ret = flt_part + exp_part
        return ret
    return '%d' % num


def cartesian(arrays, out=None):
    '''
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6], [1, 4, 7], [1, 5, 6], [1, 5, 7],
           [2, 4, 6], [2, 4, 7], [2, 5, 6], [2, 5, 7],
           [3, 4, 6], [3, 4, 7], [3, 5, 6], [3, 5, 7]])
    '''
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def float_to_decimal(f):
    # http://docs.python.org/library/decimal.html#decimal-faq
    "Convert a floating point number to a Decimal with no loss of information"
    n, d = f.as_integer_ratio()
    numerator, denominator = decimal.Decimal(n), decimal.Decimal(d)
    ctx = decimal.Context(prec=60)
    result = ctx.divide(numerator, denominator)
    while ctx.flags[decimal.Inexact]:
        ctx.flags[decimal.Inexact] = False
        ctx.prec *= 2
        result = ctx.divide(numerator, denominator)
    return result


#http://stackoverflow.com/questions/2663612/nicely-representing-a-floating-point-number-in-python
def sigfig_str(number, sigfig):
    # http://stackoverflow.com/questions/2663612/nicely-representing-a-floating-point-number-in-python/2663623#2663623
    assert(sigfig > 0)
    try:
        d = decimal.Decimal(number)
    except TypeError:
        d = float_to_decimal(float(number))
    sign, digits, exponent = d.as_tuple()
    if len(digits) < sigfig:
        digits = list(digits)
        digits.extend([0] * (sigfig - len(digits)))
    shift = d.adjusted()
    result = int(''.join(map(str, digits[:sigfig])))
    # Round the result
    if len(digits) > sigfig and digits[sigfig] >= 5:
        result += 1
    result = list(str(result))
    # Rounding can change the length of result
    # If so, adjust shift
    shift += len(result) - sigfig
    # reset len of result to sigfig
    result = result[:sigfig]
    if shift >= sigfig - 1:
        # Tack more zeros on the end
        result += ['0'] * (shift - sigfig + 1)
    elif 0 <= shift:
        # Place the decimal point in between digits
        result.insert(shift + 1, '.')
    else:
        # Tack zeros on the front
        assert(shift < 0)
        result = ['0.'] + ['0'] * (-shift - 1) + result
    if sign:
        result.insert(0, '-')
    return ''.join(result)


def ensure_iterable(obj):
    if np.iterable(obj):
        return obj
    else:
        return [obj]


def npfind(arr):
    found = np.where(arr)[0]
    pos = -1 if len(found) == 0 else found[0]
    return pos


def all_dict_combinations(varied_dict):
    viter = varied_dict.iteritems()
    tups_list = [[(key, val) for val in val_list] for (key, val_list) in viter]
    dict_list = [{key: val for (key, val) in tups} for tups in iprod(*tups_list)]
    return dict_list


def stash_testdata(*args):
    import shelve
    shelf = shelve.open('test_data.shelf')
    locals_ = get_parent_locals()
    for key in args:
        print('Stashing key=%r' % key)
        shelf[key] = locals_[key]
    shelf.close()


def load_testdata(*args):
    import shelve
    shelf = shelve.open('test_data.shelf')
    ret = [shelf[key] for key in args]
    shelf.close()
    if len(ret) == 1:
        ret = ret[0]
    return ret


def import_testdata():
    from hscom import helpers
    import shelve
    shelf = shelve.open('test_data.shelf')
    print('importing\n * ' + '\n * '.join(shelf.keys()))
    shelf_exec = helpers.execstr_dict(shelf, 'shelf')
    exec(shelf_exec)
    shelf.close()
    return import_testdata.func_code.co_code
