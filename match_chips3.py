from __future__ import division, print_function
# Premature optimization is the root of all evil
from __future__ import division
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
def print_(*args, **kwargs): pass
def print(*args, **kwargs): pass
def noprint(*args, **kwargs): pass
def realprint(*args, **kwargs):
    sys.stdout.write(args[0]+'\n')
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

def prequery(hs, q_params=None, **kwargs):
    if q_params is None:
        q_params = ds.QueryParams(**kwargs)
    if q_params.query_type == 'vsmany':
        dcxs = hs.indexed_sample_cx
        dcxs_ = tuple(dcxs)
        q_params.dcxs2_index[dcxs_] = make_nn_index(hs, dcxs)
    return q_params

qcxs = [0]
def execute_query_safe(hs, q_params=None, qcxs=None, dcxs=None, **kwargs):
    print('------------------')
    print('Execute query safe')
    print('------------------')
    kwargs = vars().get('kwargs', {})
    #---------------
    # Ensure q_params
    if not 'q_params' in vars() or q_params is None:
        kwargs = {}
        q_params = prequery(hs, **kwargs)
    if dcxs is None: dcxs = hs.indexed_sample_cx
    q_params.qcxs = qcxs
    q_params.dcxs = dcxs
    #---------------
    # Flip if needebe
    if q_params.query_type == 'vsone': # On the fly computation
        dcxs = q_params.qcxs
        qcxs = q_params.dcxs 
    elif  q_params.query_type == 'vsmany':
        qcxs = q_params.qcxs
        dcxs = q_params.dcxs 
    dcxs_ = tuple(dcxs)
    if not q_params.dcxs2_index.has_key(dcxs_):
        q_params.dcxs2_index[dcxs_] = make_nn_index(hs, dcxs)
    q_params.data_index = q_params.dcxs2_index[dcxs_]
    print('[query] qcxs=%r' % qcxs)
    print('[query] len(dcxs)=%r' % len(dcxs))
    # Real function names
    nearest_neighbs = mf.nearest_neighbors
    weight_neighbs  = mf.weight_neighbors
    score_neighbs   = mf.score_neighbors
    to_chipmatches  = mf.neighbors_to_chipmatch
    sv_matches      = mf.spatially_verify_matches
    to_res          = mf.chipmatch_to_res
    # Nearest neighbors
    neighbs = nearest_neighbs(hs, qcxs, q_params)
    # Nearest neighbors weighting and scoring
    weights  = weight_neighbs(hs, neighbs, q_params)
    # Thresholding and weighting
    scoresORIG = score_neighbs(hs, neighbs, {}, q_params)
    scoresFILT = score_neighbs(hs, neighbs, weights, q_params)
    # Chip matches
    matchesORIG = to_chipmatches(hs, neighbs, scoresORIG, q_params)
    matchesFILT = to_chipmatches(hs, neighbs, scoresFILT, q_params)
    # Spatial verification
    matchesSVER = sv_matches(hs, matchesFILT, q_params)
    # Query results
    result_list = [
        to_res(hs, matchesORIG, q_params, '+ORIG'),
        to_res(hs, matchesFILT, q_params, '+FILT'),
        to_res(hs, matchesSVER, q_params, '+SVER'),
        to_res(hs, matchesSVER, q_params, '+SVPL')
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
    q_params = prequery(hs, **kwargs)
    match_type = 'vsmany'
    compare_to = 'SVER'
    kwshow = dict(show_query=0, vert=1)
    N = 4
    # Exececute Queries Helpers
    def build_res_(taug=''):
        qcx2_resORIG, qcx2_resFILT, qcx2_resSVER = execute_query_safe(hs, q_params, [qcx])
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

#def execute_query_fast(hs, qcx, q_params):
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
