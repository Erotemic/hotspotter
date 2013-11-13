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

def prequery(hs, query_params=None, **kwargs):
    if query_params is None:
        query_params = ds.QueryParams(**kwargs)
    if query_params.query_type == 'vsmany':
        dcxs = hs.indexed_sample_cx
        dcxs_ = tuple(dcxs)
        query_params.dcxs2_index[dcxs_] = make_nn_index(hs, dcxs)
    return query_params

qcxs = [0]
def execute_query_safe(hs, query_params=None, qcxs=None, dcxs=None, **kwargs):
    print('------------------')
    print('Execute query safe')
    print('------------------')
    kwargs = vars().get('kwargs', {})
    #---------------
    # Ensure query_params
    if not 'query_params' in vars() or query_params is None:
        kwargs = {}
        query_params = prequery(hs, **kwargs)
    if dcxs is None: dcxs = hs.indexed_sample_cx
    query_params.qcxs = qcxs
    query_params.dcxs = dcxs
    #---------------
    # Flip if needebe
    if query_params.query_type == 'vsone': # On the fly computation
        dcxs = query_params.qcxs
        qcxs = query_params.dcxs 
    elif  query_params.query_type == 'vsmany':
        qcxs = query_params.qcxs
        dcxs = query_params.dcxs 
    dcxs_ = tuple(dcxs)
    if not query_params.dcxs2_index.has_key(dcxs_):
        query_params.dcxs2_index[dcxs_] = make_nn_index(hs, dcxs)
    query_params.data_index = query_params.dcxs2_index[dcxs_]
    print('[query] qcxs=%r' % qcxs)
    print('[query] len(dcxs)=%r' % len(dcxs))
    # Nearest Neighbors
    qcx2_nns = mf.nearest_neighbors(hs, qcxs, query_params)
    # Nearest Neighbors Weighting and Scoring
    filt2_weights  = mf.weight_neighbors(hs, qcx2_nns, query_params)
    qcx2_nnscoresORIG = mf.score_neighbors(hs, qcx2_nns, {}, query_params)
    qcx2_nnscoresFILT = mf.score_neighbors(hs, qcx2_nns, filt2_weights, query_params)
    # Chip Matches
    qcx2_chipmatchORIG = mf.neighbors_to_chipmatch(hs, qcx2_nns, qcx2_nnscoresORIG, query_params)
    qcx2_chipmatchFILT = mf.neighbors_to_chipmatch(hs, qcx2_nns, qcx2_nnscoresFILT, query_params)
    qcx2_chipmatchSVER = mf.spatially_verify_matches(hs, qcx2_chipmatchFILT, query_params)
    qcx2_chipmatch = qcx2_chipmatchSVER
    # Query Results
    qcx2_resORIG = mf.chipmatch_to_res(hs, qcx2_chipmatchORIG, query_params)
    qcx2_resFILT = mf.chipmatch_to_res(hs, qcx2_chipmatchFILT, query_params, scored=True)
    qcx2_resSVER = mf.chipmatch_to_res(hs, qcx2_chipmatchSVER, query_params, SV=True)
    
    #print('[query] '+str(qcx2_resORIG.keys()))
    for qcx in query_params.qcxs:
        qcx2_resORIG[qcx].title=' +ORIG '+query_params.get_uid(False, False, False, True)
        qcx2_resFILT[qcx].title=' +FILT '+query_params.get_uid(False, True, False, False)
        qcx2_resSVER[qcx].title=' +SVER '+query_params.get_uid(True, False, False, False)
    #print('[query] qcx2_resSVER = ')
    #qcx2_resSVER[0].printme()
    # Score each database chip
    #qcx2_res = mf.score_matches(hs, qcx2_nns, data_index, filt2_weights, score_params, nn_params)
    #cache_results(qcx2_res)
    #cache_results(qcx2_resSVER)
    '''
    df2.reset()
    res1.show_topN(hs, SV=False, fignum=1)
    res2.show_topN(hs, SV=True, fignum=2)
    df2.update()
    '''
    return qcx2_resORIG, qcx2_resFILT, qcx2_resSVER



def matcher_test(hs, qcx, fnum=1, **kwargs):
    print('=================================')
    print('[mc2] MATCHER TEST qcx=%r' % qcx)
    print('=================================')
    # Exececute Queries Args
    qcx    = vars().get('qcx', 0)
    fnum   = vars().get('fnum', 1)
    kwargs = vars().get('kwargs', {})
    query_params = prequery(hs, **kwargs)
    match_type = 'vsmany'
    compare_to = 'SVER'
    kwshow = dict(show_query=0, vert=1)
    N = 4
    # Exececute Queries Helpers
    def build_res_(taug=''):
        qcx2_resORIG, qcx2_resFILT, qcx2_resSVER = execute_query_safe(hs, query_params, [qcx])
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

#def execute_query_fast(hs, qcx, query_params):
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
