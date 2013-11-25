from __future__ import division, print_function
import __builtin__
import sys
# Premature optimization is the root of all evil
import itertools
import sys
import os
import warnings
import textwrap
import re
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

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass
# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[mc3] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

#----------------------
# Convinience Functions 
#----------------------
def __dict_default_func(dict_):
    # Sets keys only if they dont exist
    def set_key(key, val):
        if not dict_.has_key(key):
            dict_[key] = val
    return set_key

def get_vsmany_cfg(**kwargs):
    kwargs['query_type'] = 'vsmany'
    kwargs_set = __dict_default_func(kwargs)
    kwargs_set('lnbnn_weight', .001)
    kwargs_set('K', 2)
    kwargs_set('Knorm', 1)
    q_cfg = ds.QueryConfig(**kwargs)
    return q_cfg

def get_vsone_cfg(**kwargs):
    kwargs['query_type'] = 'vsone'
    kwargs_set = __dict_default_func(kwargs)
    kwargs_set('lnbnn_weight', 0)
    kwargs_set('checks', 256)
    kwargs_set('K', 1)
    kwargs_set('Knorm', 1)
    kwargs_set('ratio_weight', 1.0)
    kwargs_set('ratio_thresh', 1.5)
    q_cfg = ds.QueryConfig(**kwargs)
    return q_cfg

def query_dcxs(hs, qcx, dcxs, q_cfg=None, **kwargs):
    result_list = execute_query_safe(hs, q_cfg, [qcx], dcxs, **kwargs)
    res = result_list[0].values()[0]
    return res

def query_groundtruth(hs, qcx, q_cfg=None, **kwargs):
    print('[mc3] query groundtruth')
    gt_cxs = hs.get_other_cxs(qcx)
    print('[mc3] len(gt_cxs) = %r' % (gt_cxs,))
    return query_dcxs(hs, qcx, gt_cxs, q_cfg, **kwargs)

def query_database(hs, qcx, q_cfg=None, **kwargs):
    print('[mc3] query database')
    dcxs = hs.indexed_sample_cx
    return query_dcxs(hs, qcx, dcxs, q_cfg, **kwargs)

def make_nn_index(hs, sx2_cx=None):
    if sx2_cx is None:
        sx2_cx = hs.indexed_sample_cx
    data_index = ds.NNIndex(hs, sx2_cx)
    return data_index

def unify_cfgs(cfg_list):
    # Super HACK so all query configs share the same nearest neighbor indexes
    GLOBAL_dcxs2_index = cfg_list[0].dcxs2_index
    for q_cfg in cfg_list:
        q_cfg.dcxs2_index = GLOBAL_dcxs2_index

def simplify_test_uid(test_uid):
    # Remove extranious characters from test_uid
    test_uid = re.sub(r'_trainID\([0-9]*,........\)','', test_uid)
    test_uid = re.sub(r'_indxID\([0-9]*,........\)','', test_uid)
    test_uid = re.sub(r'_dcxs\(........\)','', test_uid)
    test_uid = re.sub(r'HSDB_zebra_with_mothers','', test_uid)
    test_uid = re.sub(r'GZ_ALL','', test_uid)
    test_uid = re.sub(r'HESAFF_sz750','', test_uid)
    test_uid = test_uid.strip(' _')
    return test_uid

#----------------------
# Helper Functions
#----------------------
def ensure_nn_index(hs, q_cfg, dcxs):
    dcxs_ = tuple(dcxs)
    if not q_cfg.dcxs2_index.has_key(dcxs_):
        data_index = ds.NNIndex(hs, dcxs)
        q_cfg.dcxs2_index[dcxs_] = data_index
    q_cfg.data_index = q_cfg.dcxs2_index[dcxs_]

def prequery(hs, q_cfg=None, **kwargs):
    if q_cfg is None:
        q_cfg = ds.QueryConfig(**kwargs)
    if q_cfg.a_cfg.query_type == 'vsmany':
        dcxs = hs.indexed_sample_cx
        ensure_nn_index(hs, q_cfg, dcxs)
    return q_cfg

def load_cached_query(hs, q_cfg, aug_list=['']):
    print('[query] query result cache')
    qcxs = q_cfg.qcxs
    result_list = []
    for aug in aug_list:
        qcx2_res = mf.load_resdict(hs, qcxs, q_cfg, aug)
        if qcx2_res is None: 
            return None
        result_list.append(qcx2_res)
    print('[query] cache hit\n')
    return result_list

#----------------------
# Main Query Logic
#----------------------
def execute_query_safe(hs, q_cfg=None, qcxs=None, dcxs=None, use_cache=True, **kwargs):
    '''Executes a query, performs all checks, callable on-the-fly'''
    print('[query]-------')
    print('[query] Execute query safe: q%s' % hs.cxstr(qcxs))
    if q_cfg is None: q_cfg = ds.QueryConfig(**kwargs)
    if dcxs is None: dcxs = hs.indexed_sample_cx
    q_cfg.qcxs = qcxs
    q_cfg.dcxs = dcxs
    #---------------
    # Flip if needebe
    query_type = q_cfg.a_cfg.query_type
    if query_type == 'vsone': 
        (dcxs, qcxs) = (q_cfg.qcxs, q_cfg.dcxs)
    elif query_type == 'vsmany':
        (dcxs, qcxs) = (q_cfg.dcxs, q_cfg.qcxs)
    # caching
    if not hs.args.nocache_query:
        result_list = load_cached_query(hs, q_cfg)
        if not result_list is None: 
            return result_list
    print('[query] qcxs=%r' % q_cfg.qcxs)
    print('[query] len(dcxs)=%r' % len(q_cfg.dcxs))
    ensure_nn_index(hs, q_cfg, dcxs)
    result_list = execute_query_fast(hs, q_cfg, qcxs, dcxs)
    for qcx2_res in result_list:
        for qcx, res in qcx2_res.iteritems():
            res.save(hs)
    return result_list

from helpers import tic, toc
def execute_query_fast(hs, q_cfg, qcxs, dcxs):
    '''Executes a query and assumes q_cfg has all precomputed information'''
    # Nearest neighbors
    nn_tt = tic()
    neighbs = mf.nearest_neighbors(hs, qcxs, q_cfg)
    nn_time = toc(nn_tt)
    # Nearest neighbors weighting and scoring
    weight_tt = tic()
    weights  = mf.weight_neighbors(hs, neighbs, q_cfg)
    weight_time = toc(weight_tt)
    # Thresholding and weighting
    filt_tt = tic()
    nnfiltFILT = mf.filter_neighbors(hs, neighbs, weights, q_cfg)
    filt_time = toc(filt_tt)
    # Nearest neighbors to chip matches
    build_tt = tic()
    matchesFILT = mf.build_chipmatches(hs, neighbs, nnfiltFILT, q_cfg)
    build_time = toc(build_tt)
    # Spatial verification
    verify_tt = tic()
    matchesSVER = mf.spatial_verification(hs, matchesFILT, q_cfg)
    verify_time = toc(verify_tt)
    # Query results format
    result_list = [
        mf.chipmatch_to_resdict(hs, matchesSVER, q_cfg),
    ]
    # Add timings to the results
    for res in result_list[0].itervalues():
        res.nn_time     = nn_time
        res.filt_time   = filt_time
        res.build_time  = build_time
        res.verify_time = verify_time
    return result_list

#----------------------
# Tests
#----------------------

def matcher_test(hs, qcx, fnum=1, **kwargs):
    print('=================================')
    print('[mc2] MATCHER TEST qcx=%r' % qcx)
    print('=================================')
    # Exececute Queries Args
    qcx    = vars().get('qcx', 0)
    fnum   = vars().get('fnum', 1)
    kwargs = vars().get('kwargs', {})
    q_cfg = ds.QueryConfig(**kwargs)
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
