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

FM_DTYPE  = np.uint32
FS_DTYPE  = np.float32

def reload_module():
    import imp, sys
    print('[mc3] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def precompute_data_index(hs, sx2_cx=None, **kwargs):
    if sx2_cx is None:
        sx2_cx = hs.indexed_sample_cx
    data_index = ds.NNIndex(hs, sx2_cx, **kwargs)
    return data_index

def prequery(hs, query_params=None, **kwargs):
    if query_params is None:
        query_params = ds.QueryParams(**kwargs)
    data_index_dict = {
        'vsmany' : precompute_data_index(hs, **kwargs),
        'vsone'  : None, }
    query_params.data_index = data_index_dict[query_params.query_type]
    return query_params

qcxs = [0]

def execute_query_safe(hs, qcxs, query_params=None, **kwargs):
    if not 'query_params' in vars() or query_params is None:
        kwargs = {}
        query_params = prequery(hs, **kwargs)
    if query_params.query_type == 'vsone': # On the fly computation
        query_params.query_index = precompute_data_index(hs, qcxs, **kwargs)
        data_index = query_params.query_index
    elif  query_params.query_type == 'vsmany':
        data_index = query_params.data_index
    sv_params = query_params.sv_params
    nn_params = query_params.nn_params
    # Assign Nearest Neighors
    qcx2_neighbors = mf.nearest_neighbors(hs, qcxs, data_index, nn_params)
    # Apply cheap filters
    key2_cx_qfx2_weights = {}
    for nnfilter in nn_params.nnfilter_list:
        nnfilter_fn = eval('mf.nn_'+nnfilter+'_weight')
        key2_cx_qfx2_weights[nnfilter] = nnfilter_fn(hs, qcx2_neighbors, data_index, nn_params)
    # Score each database chip
    score_params = query_params.score_params
    qcx2_res = mf.score_matches(hs, qcx2_neighbors, data_index, key2_cx_qfx2_weights, score_params, nn_params)
    #cache_results(qcx2_res)
    # Spatial Verify
    qcx2_neighborsSV = mf.spatially_verify_matches(hs, qcxs, qcx2_res, qcx2_neighbors,
                                                key2_cx_qfx2_weights, sv_params)
    cache_results(qcx2_resSV)
    return qcx2_res, qcx2_resSV

#def execute_query_fast(hs, qcx, query_params):
# fast should be the current sota execute_query that doesn't perform checks and
# need to have precomputation done beforehand. 
# safe should perform all checks and be easilly callable on the fly. 
if __name__ == '__main__':
    main_locals = invest.main()
    execstr = helpers.execstr_dict(main_locals, 'main_locals')
    exec(execstr)
