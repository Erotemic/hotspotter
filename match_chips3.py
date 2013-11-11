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

def make_nn_index(hs, sx2_cx=None, **kwargs):
    if sx2_cx is None:
        sx2_cx = hs.indexed_sample_cx
    data_index = ds.NNIndex(hs, sx2_cx, **kwargs)
    return data_index

def prequery(hs, query_params=None, **kwargs):
    if query_params is None:
        query_params = ds.QueryParams(**kwargs)
    data_index_dict = {
        'vsmany' : make_nn_index(hs, **kwargs),
        'vsone'  : None, }
    query_params.data_index = data_index_dict[query_params.query_type]
    return query_params

qcxs = [0]

def execute_query_safe(hs, qcxs, query_params=None, dcxs=None, **kwargs):
    kwargs = vars().get('kwargs', {})
    query_params = vars().get('query_params', None)
    if query_params is None:
        kwargs = {}
        ds.rrr()
        query_params = prequery(hs, **kwargs)
        print(query_params)
        print(query_params.get_uid())
    if query_params.query_type == 'vsone': # On the fly computation
        qcxs_ = tuple(qcxs)
        if qcxs_ in query_params.qcxs2_index.keys():
            query_params.qcxs2_index[qcxs_] = make_nn_index(hs, qcxs, **kwargs)
        data_index = query_params.qcxs2_index[qcxs_key]
    elif  query_params.query_type == 'vsmany':
        data_index = query_params.data_index
    sv_params    = query_params.sv_params
    score_params = query_params.score_params
    nn_params    = query_params.nn_params
    # Assign Nearest Neighors
    qcx2_neighbors = mf.nearest_neighbors(hs, qcxs, data_index, nn_params)
    # Apply cheap filters
    filter_weights = {}
    for nnfilter in nn_params.nnfilter_list:
        nnfilter_fn = eval('mf.nn_'+nnfilter+'_weight')
        filter_weights[nnfilter] = nnfilter_fn(hs, qcx2_neighbors, data_index, nn_params)
    # Convert to the plotable res objects
    qcx2_res = mf.neighbors_to_res(hs, qcx2_neighbors, filter_weights, query_params)
    # Score each database chip
    qcx2_res = mf.score_matches(hs, qcx2_neighbors, data_index, filter_weights, score_params, nn_params)
    #cache_results(qcx2_res)
    # Spatial Verify
    qcx2_neighborsSV = mf.spatially_verify_matches(hs, qcxs, qcx2_res, qcx2_neighbors, filter_weights, sv_params)
    #cache_results(qcx2_resSV)
    return qcx2_res, qcx2_resSV

def matcher_test(hs, qcx, fnum=1, **kwargs):
    print('=================================')
    print('[mc2] MATCHER TEST qcx=%r' % qcx)
    print('=================================')
    query_params = prequery(hs, **kwargs)
    use_cache = kwargs.get('use_cache', False)
    match_type = 'vsmany'
    kwbuild = dict(use_cache=use_cache, remove_init=False,
                   save_changes=True, match_type=match_type)
    kwshow = dict(SV=1, show_query=1, compare_SV=1, vert=1)
    N = 4
    def build_res_(recip, spatial):
        return execute_query_safe(hs, [qcx], recip=recip, spatial=spatial, **kwbuild)
    def show_(res, fnum, figtitle=''):
        df2.show_match_analysis(hs, res, N, fnum, figtitle, **kwshow) 
        return fnum + 1
    res_list = [
        (build_res_(False, False), 'knn'),
        (build_res_(True,  False), 'kRnn'),
        (build_res_(False, True),  'kSnn'),
        (build_res_(True,  True),  'kRSnn'),
    ]
    for (res, taug) in res_list:
        fnum = show_(res, fnum, taug)
    return fnum

#def execute_query_fast(hs, qcx, query_params):
# fast should be the current sota execute_query that doesn't perform checks and
# need to have precomputation done beforehand. 
# safe should perform all checks and be easilly callable on the fly. 
if __name__ == '__main__':
    main_locals = invest.main()
    execstr = helpers.execstr_dict(main_locals, 'main_locals')
    exec(execstr)
