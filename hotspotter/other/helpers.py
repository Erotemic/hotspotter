'''
This is less of a helper function file and more of a pile of things 
where I wasn't sure of where to put. 
A lot of things could probably be consolidated or removed. There are many
non-independent functions which are part of HotSpotter. They should be removed
and put into their own module. The standalone functions should be compiled 
into a global set of helper functions.

Wow, pylint is nice for cleaning.
'''

import code
import os.path
import sys
import time
import types
from numpy import empty

def str2(obj):
    if type(obj) == types.DictType:
        return str(obj).replace(', ','\n')[1:-1]
    if type(obj) == types.TypeType:
        return str(obj).replace('<type \'','').replace('\'>','')
    else:
        return str(obj)


#---------------
def filecheck(fpath):
    return os.path.exists(fpath)
#----------------
def dircheck(dpath,makedir=True):
    if not os.path.exists(dpath):
        if not makedir:
            #print('Nonexistant directory: %r ' % dpath)
            return False
        print('Making directory: %r' % dpath)
        os.makedirs(dpath)
    return True


def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Ctrl-D  Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
        #import IPython
        #IPython.embed_kernel(module=None,local_ns=namespace)
    except SystemExit:
        return

def alloc_lists(num_alloc):
    'allocates space for a numpy array of lists'
    alloc_data = empty(num_alloc, dtype=list)
    for i in xrange(num_alloc): alloc_data[i] = [] 
    return alloc_data

class Timer(object):
    ''' Used to time statments with a with statment
    e.g with Timer() as t: some_function()'''
    def __init__(self, outlist=[]):
        # outlist is a list to append output to
        self.outlist   = outlist
        self.tstart = -1

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, trace):
        tend = time.time()
        ellapsed = (tend - self.tstart)
        self.outlist.append(ellapsed)
        print 'Elapsed: %s seconds' % ellapsed
