from __future__ import division, print_function
# Premature optimization is the root of all evil
import itertools
import sys
import os
import warnings
import textwrap
# Hotspotter Frontend Imports
import draw_func2 as df2
# Hotspotter Imports
import fileio as io
import helpers
from helpers import Timer, tic, toc, printWARN
from Printable import DynStruct
import algos
import helpers
import spatial_verification2 as sv2
import load_data2
import params
# Math and Science Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyflann
import scipy as sp
import scipy.sparse as spsparse
import sklearn.preprocessing 
from itertools import izip, chain
import investigate_chip as invest
import DataStructures as ds
import matching_functions as mf

#<PRINT FUNCTIONS>
import sys
import __builtin__
def print_(*args, **kwargs): pass
def print(*args, **kwargs): pass
def noprint(*args, **kwargs): pass
def realprint(*args, **kwargs):
    __builtin__.print(*args, **kwargs)
def realprint_(*args, **kwargs):
    sys.stdout.write(*args)
def print_on():
    global print
    global print_
    print = realprint
    print_ = realprint_
def print_off():
    global print
    global print_
    print = noprint
    print_ = noprint
print_on()
#</PRINT FUNCTIONS>

def reload_module():
    import imp, sys
    print('[mc3] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def make_nn_index(hs, sx2_cx=None):
    if sx2_cx is None:
        sx2_cx = hs.indexed_sample_cx
    data_index = ds.NNIndex(hs, sx2_cx)
    return data_index

def prequery(hs, q_cfg=None, **kwargs):
    if q_cfg is None:
        q_cfg = ds.QueryConfig(**kwargs)
    if q_cfg.a_cfg.query_type == 'vsmany':
        dcxs = hs.indexed_sample_cx
        dcxs_ = tuple(dcxs)
        q_cfg.dcxs2_index[dcxs_] = make_nn_index(hs, dcxs)
    return q_cfg

def vsone_groundtruth(hs, qcx, q_cfg=None, **kwargs):
    print('[mc3] vsone groundtruth')
    gt_cxs = hs.get_other_cxs(qcx)
    return execute_query_safe(hs, q_cfg, [qcx], gt_cxs, **kwargs)

def vsmany_database(hs, qcx, q_cfg=None, **kwargs):
    kwargs['invert_query'] = True
    return execute_query_safe(hs, q_cfg, [qcx], **kwargs)

qcxs = [0]
def execute_query_safe(hs, q_cfg=None, qcxs=None, dcxs=None, use_cache=True, **kwargs):
    print('[query] Execute query safe: q%s' % hs.cxstr(qcxs))
    if dcxs is None: dcxs = hs.indexed_sample_cx
    q_cfg.qcxs = qcxs
    q_cfg.dcxs = dcxs
    #---------------
    # Flip if needebe
    query_type = q_cfg.a_cfg.query_type
    if query_type == 'vsone': # On the fly computation
        dcxs = q_cfg.qcxs
        qcxs = q_cfg.dcxs 
    elif query_type == 'vsmany':
        qcxs = q_cfg.qcxs
        dcxs = q_cfg.dcxs 
    # caching
    if use_cache:
        print_('[query] query result cache')
        mf.print_off()
        cache_='cache'
        matchesCACHE = {qcx:cache_ for qcx in qcxs}
        result_list = [
            mf.chipmatch_to_resdict(hs, matchesCACHE, q_cfg, '+NN'),
            mf.chipmatch_to_resdict(hs, matchesCACHE, q_cfg, '+FILT'),
            mf.chipmatch_to_resdict(hs, matchesCACHE, q_cfg, '+SVER'),
        ]
        mf.print_on()
        if not any([res is None for res in result_list]):
            print_('... hit\n')
            return result_list
        else:
            print_('... miss\n')
    dcxs_ = tuple(dcxs)
    if not q_cfg.dcxs2_index.has_key(dcxs_):
        q_cfg.dcxs2_index[dcxs_] = make_nn_index(hs, dcxs)
    q_cfg.data_index = q_cfg.dcxs2_index[dcxs_]
    print('[query] qcxs=%r' % q_cfg.qcxs)
    print('[query] len(dcxs)=%r' % len(q_cfg.dcxs))
    # Real function names
    nearest_neighbs = mf.nearest_neighbors
    weight_neighbs  = mf.weight_neighbors
    filter_neighbs  = mf.filter_neighbors
    build_matches   = mf.build_chipmatches
    verify_matches  = mf.spatial_verification
    build_result    = mf.chipmatch_to_resdict
    # Nearest neighbors
    neighbs = nearest_neighbs(hs, qcxs, q_cfg)
    # Nearest neighbors weighting and scoring
    weights  = weight_neighbs(hs, neighbs, q_cfg)
    # Thresholding and weighting
    nnfiltORIG = filter_neighbs(hs, neighbs, {}, q_cfg)
    nnfiltFILT = filter_neighbs(hs, neighbs, weights, q_cfg)
    # Nearest neighbors to chip matches
    matchesORIG = build_matches(hs, neighbs, nnfiltORIG, q_cfg)
    matchesFILT = build_matches(hs, neighbs, nnfiltFILT, q_cfg)
    # Spatial verification
    matchesSVER = verify_matches(hs, matchesFILT, q_cfg)
    # Query results format
    result_list = [
        build_result(hs, matchesORIG, q_cfg, '+NN'),
        build_result(hs, matchesFILT, q_cfg, '+FILT'),
        build_result(hs, matchesSVER, q_cfg, '+SVER'),
    ]
    return result_list

def matcher_test(hs, qcx, fnum=1, **kwargs):
    print('=================================')
    print('[mc2] MATCHER TEST qcx=%r' % qcx)
    print('=================================')
    # Exececute Queries Args
    qcx    = vars().get('qcx', 0)
    fnum   = vars().get('fnum', 1)
    kwargs = vars().get('kwargs', {})
    q_cfg  = prequery(hs, **kwargs)
    match_type = 'vsmany'
    compare_to = 'SVER'
    kwshow = dict(show_query=0, vert=1)
    N = 4
    # Exececute Queries Helpers
    def build_res_(taug=''):
        qcx2_resORIG, qcx2_resFILT, qcx2_resSVER = execute_query_safe(hs, q_cfg, [qcx])
        resORIG = qcx2_resORIG[qcx]
        resFILT = qcx2_resFILT[qcx]
        resSVER = qcx2_resSVER[qcx]
        return resORIG, resFILT, resSVER, taug
    # Exececute Queries Driver
    res_list = [
        (build_res_()),
    ]
    # Show Helpers
    def smanal_(res, fnum, aug='', resCOMP=None):
        SV = res.get_SV()
        docomp   = (resCOMP is None) or (res is resCOMP)
        comp_cxs = None if docomp else resCOMP.topN_cxs(N)
        df2.show_match_analysis(hs, res, N, fnum, aug, SV=SV,
                                compare_cxs=comp_cxs, **kwshow) 
    def show_(resORIG, resFILT, resSVER, fnum, aug=''):
        resCOMP = None if compare_to is None else eval('res'+compare_to)
        smanal_(resORIG, fnum+0, resORIG.title, resCOMP) 
        smanal_(resFILT, fnum+1, resFILT.title, resCOMP) 
        smanal_(resSVER, fnum+2, resSVER.title, resCOMP) 
        return fnum + 3
    # Show Driver
    for (res1, res2, res3, taug) in res_list:
        fnum = show_(res1, res2, res3, fnum, taug)
    df2.update()
    return fnum

#def execute_query_fast(hs, qcx, q_cfg):
# fast should be the current sota execute_query that doesn't perform checks and
# need to have precomputation done beforehand. 
# safe should perform all checks and be easilly callable on the fly. 
if __name__ == '__main__':
    #exec(open('match_chips3.py').read())
    df2.rrr()
    df2.reset()
    mf.rrr()
    ds.rrr()
    main_locals = invest.main()
    execstr = helpers.execstr_dict(main_locals, 'main_locals')
    exec(execstr)
    qcx = qcxs[0]
    #df2.DARKEN = .5
    df2.DISTINCT_COLORS = True
    #matcher_test(hs, qcx, qnum=1, K=2, Krecip=4, roidist_thresh=None,
                 #scale_thresh=(2, 10), xy_thresh=1, use_chip_extent=True,
                 #query_type='vsmany', ratio_thresh=None, lnbnn_weight=0, ratio_weight=1)
    matcher_test(hs, qcx, qnum=1, Krecip=0, Knorm=1, K=5, ratio_thresh=None,
                 lnbnn_weight=0, roidist_thresh=.5, query_type='vsmany')
    exec(df2.present())
