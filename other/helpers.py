'''
This is less of a helper function file and more of a pile of things 
where I wasn't sure of where to put. 
A lot of things could probably be consolidated or removed. There are many
non-independent functions which are part of HotSpotter. They should be removed
and put into their own module. The standalone functions should be compiled 
into a global set of helper functions.

Wow, pylint is nice for cleaning.
'''

from numpy import empty, log10
from other.logger import logmsg, logerr, logdbg
import code
import logging
import numpy
import os.path
import sys
import time
import types
import re

#---------------
# Constants !!!
eps = numpy.spacing(1)

def str2(obj):
    if type(obj) == types.DictType:
        return str(obj).replace(', ','\n')[1:-1]
    if type(obj) == types.TypeType:
        return str(obj).replace('<type \'','').replace('\'>','')
    else:
        return str(obj)

#----------------
def dircheck(dpath,makedir=True):
    if not os.path.exists(dpath):
        if not makedir:
            logdbg('Nonexistant directory: %r ' % dpath)
            return False
        logio('Making directory: %r' % dpath)
        os.makedirs(dpath)
    #logdbg('SUCCESS')
    return True

#---------------
def filecheck(fpath):
    return os.path.exists(fpath)

#---------------
def lbls2_headers(lbls):
    _lbl2_header = {
        'cid'  : 'ChipID'      ,\
        'nid'  : 'NameID'      ,\
        'gid'  : 'ImgID'       ,\
        'roi'  : 'roi[tl_x  tl_y  w  h]',\
        'cx'   : 'ChipIndex'   ,\
        'nx'   : 'NameIndex'   ,\
        'gx'   : 'ImageIndex'  ,\
        'cxs'  : 'ChipIndexes' ,\
        'cids' : 'ChipIDs',\
        'name' : 'Name',\
        'gname': 'ImageName',\
        'num_c': 'Num Chips',\
        'aif'  : 'AllIndexesFound',\
    }
    return [_lbl2_header[l] for l in lbls]

def lbls2_maxvals(lbls, hs):
    '''
    Finds the maximum value seen so far in the managers
    Uses this to figure out how big to make column spacing
    '''
    cm = hs.cm
    nm = hs.nm
    gm = hs.gm
    _lbl2_maxval = {
        'cid'  : int(cm.max_cid),\
        'aif'  : 2,\
        'nid'  : int(nm.max_nid),\
        'gid'  : int(gm.max_gid),\
        'roi'  :      cm.max_roi,\
        'cx'   : int(cm.max_cx) ,\
        'nx'   : int(nm.max_nx) ,\
        'gx'   : int(gm.max_gx) ,\
        'cxs'  : int(cm.max_cx) ,\
        'cids' :         '',\
        'name' : nm.max_name,\
        'gname': gm.max_gname,\
        'num_c': 10
    }
    return [_lbl2_maxval[l] for l in lbls]

def lbls2_format(lbls, hs):
    headers = lbls2_headers(lbls)
    maxvals = lbls2_maxvals(lbls, hs)
    #A list of (space,format) tuples
    _spcfmt = [__table_fmt(m, h) for m, h in zip(maxvals, headers)]
    header_space_list = [ t[0] for t in _spcfmt ]
    data_format_list  = [ t[1] for t in _spcfmt ]
    head_format_list  = ', '.join(['{:>%d}']*len(lbls)) % tuple(header_space_list)
    header = head_format_list.format(*headers)
    data_format = ', '.join(data_format_list)
    return (header, data_format)

def __table_fmt(max_val, lbl=""):
    '''
    Table Formater: gives you the python string to format your data
    Input:  longest value
    Output: (nSpaces, formatStr)
    '''
    if max_val == 0:
        max_val = 1
    if type(max_val) is types.IntType or type(max_val) == numpy.uint32:
        spaces = max(int(log10(max_val)), len(lbl))+1
        fmtstr = '{:>%dd}' % spaces
    elif type(max_val) is types.FloatType:
        _nDEC = 3
        if _nDEC == 0:
            spaces = max(int(log10(max_val)), len(lbl))+1
            fmtstr = '{:>%d.0f}' % (spaces)
        else:
            spaces = max(int(log10(max_val))+1+_nDEC, len(lbl))+1
            fmtstr = '{:>%d.%df}' % (spaces, _nDEC)
    elif type(max_val) is types.ListType:
        _SEP    = '  '
        _rBrace = ' ]'
        _lBrace = '[ '
        # Recursively format elements in the list
        _items  = [__table_fmt(x) for x in max_val]
        _spc    = [ t[0] for t in _items]
        _fmt    = [ t[1] for t in _items]
        spaces  = sum(_spc)+((len(_items)-1)*len(_SEP))+len(_rBrace)+len(_lBrace)
        if spaces < len(lbl):
            _lBrace = ' '*(len(lbl)-spaces) + _lBrace
            #raise Exception('The label is expected to be shorter than the list')
        fmtstr  = _lBrace+_SEP.join(_fmt)+_rBrace
    elif type(max_val) is types.StringType:
        spaces = len(max_val)+1
        fmtstr = '{:>%d}' % (spaces) 
    else:
        raise Exception('Unknown Type for '+str(type(max_val))+'\n label:\"'+str(lbl)+'\" max_val:'+str(max_val) )
    return (spaces, fmtstr)

#---------------

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
    def __init__(self, name=''):
        self.name = name
        self.tstart = -1

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, trace):
        tend = time.time()
        if self.name != '':
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (tend - self.tstart)
