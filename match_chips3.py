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

class NNParams(DynStruct):
    def __init__(nn_params, **kwargs):
        super(NNParams, nn_params).__init__
        # Core
        nn_params.K = 2
        nn_params.Knorm = 1
        # Filters
        nn_params.nnfilter_list = ['reciprocal', 'roidist']
        #['reciprocal', 'roidist', 'frexquency', 'ratiotest', 'bursty']
        nn_params.K_reciprocal   = 1 # 0 := off
        nn_params.roidist_thresh = 1 # 1 := off
        nn_params.ratio_thresh   = 1 # 1 := off
        nn_params.freq_thresh    = 1 # 1 := off
        nn_params.checks = 128
        nn_params.__dict__.update(**kwargs)

class SpatialVerifyParams(DynStruct):
    def __init__(sv_params, **kwargs):
        super(SpatialVerifyParams, sv_params).__init__
        sv_params.scale_thresh  = (.5, 2)
        sv_params.xy_thresh = .002
        sv_params.shortlist_len = 100
        sv_params.__dict__.update(kwargs)

class ScoringParams(DynStruct):
    def __init__(score_params, **kwargs):
        super(ScoringParams, score_params).__init__
        score_params.aggregation_method = 'ChipSum' # ['NameSum', 'NamePlacketLuce']
        score_params.meta_params = {
            'roidist'    : (.5),
            'reciprocal' : (0), 
            'ratio'      : (1.2), 
            'scale'      : (.5),
            'bursty'     : (1),
            'lnbnn'      : 0, 
        }
        score_params.num_shortlist  = 100
        score_params.__dict__.update(kwargs)

def scoring_func(hs, qcx2_neighbors, data_index, nnweights, scoring_params, nn_params):
    meta_params = score_params.meta_params
    agg_method = score_params.aggregation_method
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dcxs = np.arange(len(hs.feats.cx2_desc))
    K = nn_params.K
    qcx2_res = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, _) = qcx2_neighbors[qcx]
        nQuery = len(qfx2_dx)
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_score = np.ones(qfx2_nn.shape)
        qfx2_valid = np.ones(qfx2_nn.shape, dtype=np.bool)
        for key, cx2_weights in nnweights.iteritems():
            qfx2_weights = cx2_weights[qcx]
            thresh = meta_params[key]
            print('%r, %r' % (key, thresh))
            qfx2_valid = np.bitwise_and(qfx2_valid, qfx2_weights <= thresh)
            #qfx2_score
        if agg_method == 'ChipSum':
            qfx2_cx = dx2_cx[qfx2_nn]
            qfx2_fx = dx2_fx[qfx2_nn]
            # Build feature matches
            cx2_fm = [[] for _ in xrange(len(dcxs))]
            cx2_fs = [[] for _ in xrange(len(dcxs))]
            qfx2_qfx = helpers.tiled_range(nQuery, K)
            #iter_matches = izip(qfx2_qfx.flat, qfx2_cx.flat, qfx2_fx.flat, qfx2_score.flat)
            iter_matches = izip(qfx2_qfx[qfx2_valid],
                                qfx2_cx[qfx2_valid],
                                qfx2_fx[qfx2_valid],
                                qfx2_score[qfx2_valid])
            for qfx, cx, fx, score in iter_matches:
                if qcx == cx: 
                    continue # dont vote for yourself
                cx2_fm[cx].append((qfx, fx))
                cx2_fs[cx].append(score)
            # Convert to numpy
            for cx in xrange(len(dcxs)):
                fm = np.array(cx2_fm[cx], dtype=FM_DTYPE)
                fm = fm.reshape(len(fm), 2)
                cx2_fm[cx] = fm
            for cx in xrange(len(dcxs)): 
                fs = np.array(cx2_fs[cx], dtype=FS_DTYPE)
                #fs.shape = (len(fs), 1)
                cx2_fs[cx] = fs
            cx2_fm = np.array(cx2_fm)
            cx2_fs = np.array(cx2_fs)
            cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
            qcx2_res[qcx] = (cx2_fm, cx2_fs, cx2_score)
        return qcx2_res
    #s2coring_func  = [LNBNN, PlacketLuce, TopK, Borda]
    #load_precomputed(cx, query_params)

class QueryParams(DynStruct):
    def __init__(query_params, **kwargs):
        super(QueryParams, query_params).__init__
        query_params.nn_params = NNParams(**kwargs)
        query_params.score_params = ScoringParams(**kwargs)
        query_params.sv_params = SpatialVerifyParams( *kwargs)
        query_params.query_type = 'vsmany'
        query_params.__dict__.update(kwargs)

def precompute_data_index(hs, sx2_cx=None, **kwargs):
    if sx2_cx is None:
        sx2_cx = hs.indexed_sample_cx
    data_index = ds.NNIndex(hs, sx2_cx, **kwargs)
    return data_index

#def execute_query_fast(hs, qcx, query_params):
# fast should be the current sota execute_query that doesn't perform checks and
# need to have precomputation done beforehand. 
# safe should perform all checks and be easilly callable on the fly. 
def prequery(hs, query_params=None, **kwargs):
    if query_params is None:
        query_params = QueryParams(**kwargs)
    data_index_dict = {
        'vsmany' : precompute_data_index(hs, **kwargs),
        'vsone'  : None, }
    query_params.data_index = data_index_dict[query_params.query_type]
    return query_params

qcxs = [0]

def spatially_verify_matches(hs, qcxs, qcx2_res, qcx2_neighbors,
                             key2_cx_qfx2_weights, sv_params):
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts  = hs.feats.cx2_kpts
    cx2_resSV = {}
    for qcx in qcx2_res.iterkeys():
        kpts1     = cx2_kpts[qcx]
        (cx2_fm, cx2_fs, cx2_score) = qcx2_res[qcx]
        #cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
        top_cx     = cx2_score.argsort()[::-1]
        num_rerank = min(len(top_cx), sv_params.shortlist_len)
        # Precompute output container
        cx2_fm_V = [[] for _ in xrange(len(cx2_fm))]
        cx2_fs_V = [[] for _ in xrange(len(cx2_fs))]
        # spatially verify the top __NUM_RERANK__ results
        for topx in xrange(num_rerank):
            cx    = top_cx[topx]
            kpts2 = cx2_kpts[cx]
            fm    = cx2_fm[cx]
            fs    = cx2_fs[cx]
            rchip_size2 = cx2_rchip_size[cx]
            xy_thresh = sv_params.xy_thresh
            scale_thresh_low, scale_thresh_high = sv_params.scale_thresh
            (fm_V, fs_V) = spatially_verify(kpts1, kpts2, rchip_size2, fm, fs, xy_thresh,
                     scale_thresh_high, scale_thresh_low)
            cx2_fm_V[cx] = fm_V
            cx2_fs_V[cx] = fs_V
        # Rebuild the feature match / score arrays to be consistent
        for cx in xrange(len(cx2_fm_V)):
            fm = np.array(cx2_fm_V[cx], dtype=FM_DTYPE)
            fm = fm.reshape(len(fm), 2)
            cx2_fm_V[cx] = fm
        for cx in xrange(len(cx2_fs_V)): 
            cx2_fs_V[cx] = np.array(cx2_fs_V[cx], dtype=FS_DTYPE)
        cx2_fm_V = np.array(cx2_fm_V)
        cx2_fs_V = np.array(cx2_fs_V)
        # Rebuild the cx2_score arrays
        cx2_score_V = np.array([np.sum(fs) for fs in cx2_fs_V])
        resSV = (cx2_fm_V, cx2_fs_V, cx2_score_V)
        cx2_resSV[qcx] = resSV
    return cx2_resSV

def __default_sv_return():
    'default values returned by bad spatial verification'
    #H = np.eye(3)
    fm_V = np.empty((0, 2))
    fs_V = np.array((0, 1))
    return (fm_V, fs_V)
import params
import spatial_verification2 as sv2
def spatially_verify(kpts1, kpts2, rchip_size2, fm, fs, xy_thresh,
                     scale_thresh_high, scale_thresh_low):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers 
       returns fm_V, fs_V, H '''
    # Return if pathological
    min_num_inliers   = 4
    if len(fm) < min_num_inliers:
        return __default_sv_return()
    # Get homography parameters
    if params.__USE_CHIP_EXTENT__:
        diaglen_sqrd = rchip_size2[0]**2 + rchip_size2[1]**2
    else:
        x_m = kpts2[fm[:,1],0].T
        y_m = kpts2[fm[:,1],1].T
        diaglen_sqrd = sv2.calc_diaglen_sqrd(x_m, y_m)
    # Try and find a homography
    sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh, 
                                    scale_thresh_high, scale_thresh_low,
                                    diaglen_sqrd, min_num_inliers)
    if sv_tup is None:
        return __default_sv_return()
    # Return the inliers to the homography
    (H, inliers, Aff, aff_inliers) = sv_tup
    fm_V = fm[inliers, :]
    fs_V = fs[inliers]
    return fm_V, fs_V

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
    qcx2_res = scoring_func(hs, qcx2_neighbors, data_index, key2_cx_qfx2_weights, score_params, nn_params)
    #cache_results(qcx2_res)
    # Spatial Verify
    qcx2_neighborsSV = spatially_verify_matches(hs, qcxs, qcx2_res, qcx2_neighbors,
                                                key2_cx_qfx2_weights, sv_params)
    cache_results(qcx2_resSV)
    return qcx2_res, qcx2_resSV

'''
PRIORITY 1: 
* CREATE A SIMPLE TEST DATABASE
* Need simple testcases showing the validity of each step. Do this with a small
* database of three images: Query, TrueMatch, FalseMatch
* Manually remove a selection of keypoints. 

PRIORITY 2: 
* FIX QUERY CACHING
 QueryResult should save each step of the query. 
 * Initial Nearest Neighbors Result,
 * Filter Reciprocal Result
 * Filter Spatial Result
 * Filter Spatial Verification Result 
 You should have the ability to turn the caching of any part off. 

PRIORITY 3: 
 * Unifty vsone and vsmany
 * Just make a query params object
 they are the same process which accepts the parameters: 
     invert_query, qcxs, dcxs
'''
if __name__ == '__main__':
    main_locals = invest.main()
    execstr = helpers.execstr_dict(main_locals, 'main_locals')
    exec(execstr)
