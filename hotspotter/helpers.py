from __future__ import print_function
'''
This is less of a helper function file and more of a pile of things 
where I wasn't sure of where to put. 
A lot of things could probably be consolidated or removed. There are many
non-independent functions which are part of HotSpotter. They should be removed
and put into their own module. The standalone functions should be compiled 
into a global set of helper functions.

Wow, pylint is nice for cleaning.
'''

from hotspotter.other.AbstractPrintable import printableVal
import cPickle
import code
import numpy as np
import os, os.path, sys
import sys
import time
import types
import shutil
import warnings
import fnmatch
from sys import stdout as sout

def _print(msg):
    sout.write(msg)
def _println(msg):
    sout.write(msg+'\n')

img_ext_set = set(['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ppm'])

def public_attributes(input):
    public_attr_list = []
    all_attr_list = dir(input)
    for attr in all_attr_list:
        if attr.find('__') == 0: continue
        public_attr_list.append(attr)
    return public_attr_list

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

def longest_existing_path(_path):
    while True: 
        _path_new = os.path.dirname(_path)
        if os.path.exists(_path_new):
            _path = _path_new
            break
        if _path_new == _path: 
            print('!!! This is a very illformated path indeed.')
            _path = ''
            break
        _path = _path_new
    return _path

def normalize(array, dim=0):
    'normalizes a numpy array from 0 to 1'
    array_max  = array.max(dim)
    array_min  = array.min(dim)
    array_exnt = np.subtract(array_max, array_min)
    return np.divide(np.subtract(array, array_min), array_exnt)

def checkpath(_path):
    '''Checks to see if the argument _path exists.'''
    # Do the work
    _path = os.path.normpath(_path)
    sys.stdout.write('Checking ' + repr(_path))
    if os.path.exists(_path):
        if os.path.isfile(_path):
            path_type = 'file'
        if os.path.isdir(_path): 
            path_type = 'directory'
        sys.stdout.write('... exists ('+path_type+')\n')
    else:
        print('\n  ! Does not exist')
        _longest_path = longest_existing_path(_path)
        print('... The longest existing path is: ' + repr(_longest_path))
        return False
    return True
def check_path(_path):
    return checkpath(_path)

def copy_task(cp_list, test=False, nooverwrite=False, print_tasks=True):
    '''
    Input list of tuples: 
        format = [(src_1, dst_1), ..., (src_N, dst_N)] 
    Copies all files src_i to dst_i
    '''
    num_overwrite = 0
    _cp_tasks = [] # Build this list with the actual tasks
    if nooverwrite:
        print('Removed: copy task ')
    else:
        print('Begining copy+overwrite task.')
    for (src, dst) in iter(cp_list):
        if os.path.exists(dst):
            num_overwrite += 1
            if print_tasks:
                print('!!! Overwriting ')
            if not nooverwrite:
                _cp_tasks.append((src, dst))
        else:
            if print_tasks:
                print('... Copying ')
                _cp_tasks.append((src, dst))
        if print_tasks:
            print('    '+src+' -> \n    '+dst)
    print('About to copy %d files' % len(cp_list))
    if nooverwrite:
        print('Skipping %d tasks which would have overwriten files' % num_overwrite)
    else:
        print('There will be %d overwrites' % num_overwrite)
    if not test: 
        print('... Copying')
        for (src, dst) in iter(_cp_tasks):
            shutil.copy(src, dst)
        print('... Finished copying')
    else:
        print('... In test mode. Nothing was copied.')

def copy(src, dst):
    if os.path.exists(dst):
        print('!!! Overwriting ')
    else:
        print('... Copying ')
    print('    '+src+' -> \n    '+dst)
    shutil.copy(src, dst)

def copy_all(src_dir, dest_dir, glob_str_list):
    if type(glob_str_list) != types.ListType:
        glob_str_list = [glob_str_list]
    for _fname in os.listdir(src_dir):
        for glob_str in glob_str_list:
            if fnmatch.fnmatch(_fname, glob_str):
                src = os.path.normpath(os.path.join(src_dir, _fname))
                dst = os.path.normpath(os.path.join(dest_dir, _fname))
                copy(src, dst)
                break

def ensurepath(_path):
    if not checkpath(_path):
        print('... Making directory: ' + _path)
        os.makedirs(_path)
    return True

def ensure_path(_path):
    return ensurepath(_path)

def assertpath(_path):
    if not checkpath(_path):
        raise AssertionError('Asserted path does not exist: '+_path)
def assert_path(_path):
    return assertpath(_path)

def join_mkdir(*args):
    'os.path.join and creates if not exists'
    output_dir = os.path.join(*args)
    if not os.path.exists(output_dir):
        print('Making dir: '+output_dir)
        os.mkdir(output_dir)
    return output_dir

def vd(dname=None):
    'View directory'
    if dname is None: 
        dname = os.getcwd()
    os_type       = sys.platform
    open_prog_map = {'win32':'explorer.exe', 'linux2':'nautilus', 'darwin':'open'}
    open_prog     = open_prog_map[os_type]
    os.system(open_prog+' '+dname)
        
def str2(obj):
    if type(obj) == types.DictType:
        return str(obj).replace(', ','\n')[1:-1]
    if type(obj) == types.TypeType:
        return str(obj).replace('<type \'','').replace('\'>','')
    else:
        return str(obj)
#---------------
'''
def __getstate__(self):
    out_dict = self.__dict__.copy()
    return odict
def __setstate__(self, in_dict):
    self.__dict__.update(in_dict)
'''
#---------------
import cPickle
def sanatize_fname(fname):
    ext = '.pkl'
    if fname.rfind(ext) != max(len(fname) - len(ext), 0):
        fname += ext
    return fname

def save_pkl(fname, obj):
    with open(fname, 'wb') as file:
        cPickle.dump(obj, file)

def load_pkl(fname):
    with open(fname, 'wb') as file:
        return cPickle.load(file)

def save_npz(fname, *args):
    with open(fname, 'wb') as file:
        np.savez(file, *args)

def load_npz(fname):
    npz = np.load(fname)
    return tuple(npz[key] for key in sorted(npz.keys()))


def hashstr_md5(data):
    import hashlib
    return hashlib.md5(data).hexdigest()
    


#---------------
def printWARN(warn_msg, category=UserWarning):
    print(warn_msg)
    warnings.warn(warn_msg, category=category)
#---------------
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
#---------------
@__DEPRICATED__
def filecheck(fpath):
    return os.path.exists(fpath)
#----------------
@__DEPRICATED__
def dircheck(dpath,makedir=True):
    if not os.path.exists(dpath):
        if not makedir:
            #print('Nonexistant directory: %r ' % dpath)
            return False
        print('Making directory: %r' % dpath)
        os.makedirs(dpath)
    return True

def in_IPython():
    try:
        __IPYTHON__
        return True
    except NameError as nex:
        return False

def have_IPython():
    try:
        import IPython
        return True
    except NameError as nex:
        return False

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
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
        return

def alloc_lists(num_alloc):
    'allocates space for a numpy array of lists'
    return [[] for _ in xrange(num_alloc)]

def mystats(_list):
    nparr = np.array(_list)
    return {'min'   : nparr.min(),
            'mean'  : nparr.mean(),
            'stddev': np.sqrt(nparr.var()),
            'max'   : nparr.max()}

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


def tic(msg=None):
    return (msg, time.time())

def toc(tt):
    ellapsed = (time.time() - tt[1])
    sys.stdout.write('...toc(%.4fs, ' % ellapsed + '"' + str(tt[0]) + '"' + ')\n')
    return ellapsed


import sys
class Timer(object):
    ''' Used to time statments with a with statment
    e.g with Timer() as t: some_function()'''
    def __init__(self, outlist=[], msg=''):
        # outlist is a list to append output to
        self.outlist = outlist
        self.msg = msg
        self.tstart = -1

    def __enter__(self):
        #if not self.msg is None:
            #sys.stdout.write('---tic---'+self.msg+'  \n')
        sys.stdout.flush()
        self.tstart = time.time()

    def __exit__(self, type, value, trace):
        ellapsed = (time.time() - self.tstart)
        if not self.msg is None and len(self.msg) <= 0:
            self.outlist.append(ellapsed)
        #sys.stdout.write('___toc___'+self.msg+' = %.4fs \n\n' % ellapsed)
        #sys.stdout.write('___toc___'+self.msg+' = %.4fs \n\n' % ellapsed)
        sys.stdout.write('...toc(%.4fs, ' % ellapsed + '"' + self.msg + '"' + ')\n')
        sys.stdout.flush()

import matplotlib.pyplot as plt
def figure(fignum, doclf=False, title=None, **kwargs):
    fig = plt.figure(fignum, **kwargs)
    axes_list = fig.get_axes()
    if not 'user_stat_list' in fig.__dict__.keys() or doclf:
        fig.user_stat_list = []
        fig.user_notes = []
    if doclf or len(axes_list) == 0:
        fig.clf()
        ax = plt.subplot(111)
    else: 
        ax  = axes_list[0]
    if not title is None:
        ax.set_title(title)
        fig.canvas.set_window_title('fig '+str(fignum)+' '+title)
    return fig


def reload_modules():
    import imp
    import drawing_functions2
    import hotspotter.helpers
    imp.reload(drawing_functions2)
    imp.reload(hotspotter.helpers)


def symlink(source, link_name, noraise=False):
    try: 
        import os
        os_symlink = getattr(os, "symlink", None)
        if callable(os_symlink):
            os_symlink(source, link_name)
        else:
            import ctypes
            csl = ctypes.windll.kernel32.CreateSymbolicLinkW
            csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
            csl.restype = ctypes.c_ubyte
            flags = 1 if os.path.isdir(source) else 0
            if csl(link_name, source, flags) == 0:
                warnings.warn(warn_msg, category=UserWarning)
                print(' Unable to create symbolic liwk on windows.')
                raise ctypes.WinError()
    except Exception as ex:
        if not noraise:
            raise

def get_exec_src(func):
    import inspect
    import textwrap
    _src = inspect.getsource(func)
    src = textwrap.dedent(_src[_src.find(':')+1:])
    # Remove return statments
    while True:
        ret_start = src.find('return')
        if ret_start == -1:
            break
        middle   = src[ret_start:]
        ret_end1 = middle.find(';')
        ret_end2 = middle.find('\n')
        if ret_end1 == -1:
            ret_end1 = ret_end2
        ret_end = min(ret_end1, ret_end2)
        if ret_end == -1 or ret_end == len(src):
            ret_end = len(src)-1
        ret_end = ret_start + ret_end + 1
        before = src[:ret_start]
        after  = src[ret_end:]
        src = before+after
    return src

def remove_file(fpath, verbose=True):
    try:
        if verbose:
            print('Removing '+fpath)
        os.remove(fpath)
    except OSError as e:
        printWARN('OSError: %s,\n Could not delete %s' % (str(e), fpath))
        return False
    return True

def remove_files_in_dir(dpath, fname_pattern='*', recursive=False):
    print('Removing files:')
    print('  * in dpath = %r ' % dpath) 
    print('  * matching pattern = %r' % fname_pattern) 
    print('  * recursive = %r' % recursive) 
    num_removed, num_matched = (0,0)
    if not os.path.exists(dpath):
        printWARN('!!! dir = %r does not exist!' % dpath)
    for root, dname_list, fname_list in os.walk(dpath):
        for fname in fnmatch.filter(fname_list, fname_pattern):
            num_matched += 1
            num_removed += remove_file(os.path.join(root, fname))
        if not recursive:
            break
    print('... Removed %d/%d files' % (num_removed, num_matched))
    return True

def profile(cmd):
    # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
    import cProfile, sys, os
    print('Profiling Command: '+cmd)
    cProfOut_fpath = 'OpenGLContext.profile'
    cProfile.runctx( cmd, globals(), locals(), filename=cProfOut_fpath )
    # RUN SNAKE
    print('Profiled Output: '+cProfOut_fpath)
    if sys.platform == 'win32':
        rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
    else:
        rsr_fpath = 'runsnake'
    view_cmd = rsr_fpath+' "'+cProfOut_fpath+'"'
    os.system(view_cmd)
    return True

#http://www.huyng.com/posts/python-performance-analysis/
#Once youve gotten your code setup with the @profile decorator, use kernprof.py to run your script.
#kernprof.py -l -v fib.py
