from __future__ import division, print_function
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
import spatial_verification2 as sv2
import DataStructures as ds

def reload_module():
    import imp, sys
    print('[mc3] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()
    
#============================
# Nearest Neighbors
#============================
'''
Filter Example Data:
burst_thresh = 2
qfx2_fx   = np.array([(1, 3, 4), (8, 2, 3), (3, 6, 2), (3, 7, 1), (3, 2, 4), (1, 2, 2)])
qfx2_dist = np.array([(1, 2, 3), (4, 5, 6), (1, 6, 9), (0, 2, 4), (4, 3, 7), (2, 3, 4)])
qfx2_valid = np.ones(qfx2_fx.shape, dtype=np.bool)
'''
# All neighbors are valid
#qfx2_valid = np.ones(qfx2_dx.shape, dtype=np.bool)

def nearest_neighbors(hs, qcxs, data_index, query_params):
    'Plain Nearest Neighbors'
    nn_params = query_params.nn_params
    print('Step 1) Assign nearest neighbors: '+nn_params.get_uid())
    K = nn_params.K
    Knorm = nn_params.Knorm
    checks = nn_params.checks
    cx2_desc = hs.feats.cx2_desc
    nn_index = data_index.flann.nn_index
    nnfunc = lambda qcx: nn_index(cx2_desc[qcx], K+Knorm, checks=checks)
    #qcx2_neighbors = {qcx:func(qcx) for qcx in qcxs}
    qcx2_neighbors = {}
    for qcx in qcxs:
        (qfx2_dx, qfx2_dist) = nnfunc(qcx)
        qcx2_neighbors[qcx] = (qfx2_dx, qfx2_dist)
    return qcx2_neighbors


#-------
# Nearest Neighbor weights
#-------
def apply_neighbor_weights(hs, qcx2_neighbors, data_index, query_params):
    score_params = query_params.score_params
    print('Step 2) Weight neighbors: '+score_params.get_uid())
    nnfilter_list = score_params.nnfilter_list
    filter_weights = {}
    for nnfilter in nnfilter_list:
        nnfilter_fn = eval('nn_'+nnfilter+'_weight')
        filter_weights[nnfilter] = nnfilter_fn(hs, qcx2_neighbors, data_index, query_params)
    return filter_weights

eps = 1E-8
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist, vdist+eps)+1) 
def RATIO_fn(vdist, ndist): return np.divide(ndist, vdist+eps)
def LNBNN_fn(vdist, ndist): return (ndist - vdist) / 1000.0
# normweight_fn = LNBNN_fn
''''
ndist = np.array([[0, 1, 2], [3, 4, 5], [3, 4, 5], [3, 4, 5],  [9, 7, 6] ])
vdist = np.array([[3, 2, 1, 5], [3, 2, 5, 6], [3, 4, 5, 3], [3, 4, 5, 8],  [9, 7, 6, 3] ])
vdist1 = vdist[:,0:1]
vdist2 = vdist[:,0:2]
vdist3 = vdist[:,0:3]
vdist4 = vdist[:,0:4]
print(LNBNN_fn(vdist1, ndist)) * 1000
print(LNBNN_fn(vdist2, ndist)) * 1000
print(LNBNN_fn(vdist3, ndist)) * 1000
print(LNBNN_fn(vdist4, ndist)) * 1000
'''
def _nn_normalized_weight(normweight_fn, hs, qcx2_neighbors, data_index, query_params):
    # Only valid for vsone
    K = query_params.nn_params.K
    Knorm = query_params.nn_params.Knorm
    qcx2_norm_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (_, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nndist = qfx2_dist[:, 0:K]
        qfx2_normdist = qfx2_dist[:, -2:-1]
        qfx2_normweight = normweight_fn(qfx2_nndist, qfx2_normdist)
        qcx2_norm_weight[qcx] = qfx2_normweight
    return qcx2_norm_weight
def nn_ratio_weight(*args):
    return _nn_normalized_weight(RATIO_fn, *args)
def nn_lnbnn_weight(*args):
    return _nn_normalized_weight(LNBNN_fn, *args)
def nn_lnrat_weight(*args):
    return _nn_normalized_weight(LNRAT_fn, *args)

def nn_bursty_weight(hs, qcx2_neighbors, data_index, query_params):
    'Filters matches to a feature which is matched > burst_thresh #times'
    # Half-generalized to vsmany
    # Assume the first nRows-1 rows are the matches (last row is normalizer)
    K = query_params.nn_params.K
    Knorm = query_params.nn_params.Knorm
    qcx2_bursty_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nn = qfx2_dx[:, 0:K]
        dx2_frequency  = np.bincount(qfx2_nn.flatten())
        qfx2_bursty = dx2_frequency[qfx2_nn]
        qcx2_bursty_weight[qcx] = qfx2_bursty
    return qcx2_bursty_weight

def nn_recip_weight(hs, qcx2_neighbors, data_index, query_params):
    'Filters a nearest neighbor to only reciprocals'
    K = query_params.nn_params.K
    Krecip = query_params.score_params.Krecip
    checks = query_params.nn_params.checks
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dx2_data = data_index.ax2_data
    data_flann = data_index.flann
    qcx2_recip_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        nQuery = len(qfx2_dx)
        dim = dx2_data.shape[1]
        # Get the original K nearest features
        qx2_nndx = dx2_data[qfx2_dx[:, 0:K]]
        qx2_nndist = qfx2_dist[:, 0:K]
        qx2_nndx.shape = (nQuery*K, dim)
        # TODO: Have the option for this to be both indexes.
        (_nn2_rdx, _nn2_rdists) = data_flann.nn_index(qx2_nndx, Krecip, checks=checks)
        # Get the maximum distance of the Krecip reciprocal neighbors
        _nn2_rdists.shape = (nQuery, K, Krecip)
        qfx2_recipmaxdist = _nn2_rdists.max(2)
        # Test if nearest neighbor distance is less than reciprocal distance
        qfx2_reciprocalness = qfx2_recipmaxdist - qx2_nndist
        qcx2_recip_weight[qcx] = qfx2_reciprocalness
    return qcx2_recip_weight

def nn_roidist_weight(hs, qcx2_neighbors, data_index, query_params):
    'Filters a matches to those within roughly the same spatial arangement'
    K = query_params.nn_params.K
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts = hs.feats.cx2_kpts
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    cx2_roidist_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nn = qfx2_dx[:,0:K]
        # Get matched chip sizes #.0300s
        qfx2_kpts = cx2_kpts[qcx]
        nQuery = len(qfx2_dx)
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_chipsize2 = np.array([cx2_rchip_size[cx] for cx in qfx2_cx.flat])
        qfx2_chipsize2.shape = (nQuery, K, 2)
        qfx2_chipdiag2 = np.sqrt((qfx2_chipsize2**2).sum(2))
        # Get query relative xy keypoints #.0160s / #.0180s (+cast)
        qdiag = np.sqrt((np.array(cx2_rchip_size[qcx])**2).sum())
        qfx2_xy1 = np.array(qfx2_kpts[:, 0:2], np.float)
        qfx2_xy1[:,0] /= qdiag
        qfx2_xy1[:,1] /= qdiag
        # Get database relative xy keypoints
        qfx2_xy2 = np.array([cx2_kpts[cx][fx, 0:2] for (cx, fx) in
                            izip(qfx2_cx.flat, qfx2_fx.flat)], np.float)
        qfx2_xy2.shape = (nQuery, K, 2)
        qfx2_xy2[:,:,0] /= qfx2_chipdiag2
        qfx2_xy2[:,:,1] /= qfx2_chipdiag2
        # Get the relative distance # .0010s
        qfx2_K_xy1 = np.rollaxis(np.tile(qfx2_xy1, (K, 1, 1)), 1)
        qfx2_xydist = ((qfx2_K_xy1 - qfx2_xy2)**2).sum(2)
        cx2_roidist_weight[qcx] = qfx2_xydist
    return cx2_roidist_weight

def nn_scale_weight(hs, qcx2_neighbors, data_index, query_params):
    # Filter by scale for funzies
    K = query_params.nn_params.K
    cx2_scale_weight = {}
    for qcx in qcx2_neighbors.iterkeys():
        (qfx2_dx, qfx2_dist) = qcx2_neighbors[qcx]
        qfx2_nn = qfx2_dx[:,0:K]
        nQuery = len(qfx2_dx)
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_det1 = np.array(qfx2_kpts[:, [2,4]], np.float).prod(1)
        qfx2_det1 = np.sqrt(1.0/qfx2_det1)
        qfx2_K_det1 = np.rollaxis(np.tile(qfx2_det1, (K, 1)), 1)
        qfx2_det2 = np.array([cx2_kpts[cx][fx, [2,4]] for (cx, fx) in
                            izip(qfx2_cx.flat, qfx2_fx.flat)], np.float).prod(1)
        qfx2_det2.shape = (nQuery, K)
        qfx2_det2 = np.sqrt(1.0/qfx2_det2)
        qfx2_scaledist = qfx2_det2 / qfx2_K_det1
        cx2_scale_weight[qcx] = qfx2_scaledist
    return cx2_scale_weight


def nn_placketluce_score(hs, qcx2_neighbors, data_index, query_params):
    pass

def nn_positional_score(hs, qcx2_neighbors, data_index, query_params):
    pass


def fix_fmfs(cx2_fm, cx2_fs):
    # Convert to numpy
    for cx in xrange(len(cx2_fm)):
        fm = np.array(cx2_fm[cx], dtype=ds.FM_DTYPE)
        fm = fm.reshape(len(fm), 2)
        cx2_fm[cx] = fm
    for cx in xrange(len(cx2_fs)): 
        fs = np.array(cx2_fs[cx], dtype=ds.FS_DTYPE)
        #fs.shape = (len(fs), 1)
        cx2_fs[cx] = fs
    cx2_fm = np.array(cx2_fm)
    cx2_fs = np.array(cx2_fs)
    return cx2_fm, cx2_fs

def fmfs2_QueryResult(qcx, cx2_fm, cx2_fs, uid):
    cx2_fm, cx2_fs = fix_fmfs(cx2_fm, cx2_fs)
    cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
    res = ds.QueryResult(qcx, uid)
    res.cx2_fm = cx2_fm
    res.cx2_fs = cx2_fs
    res.cx2_score = cx2_score
    return res
    
def neighbors_to_res(hs, qcx2_neighbors, filter_weights, data_index, query_params):
    print('Step 3) Convert (qfx2_dx, qfx2_dist) to (cx2_fm, sx2_fs)')
    K = query_params.nn_params.K
    agg_method = query_params.score_params.aggregation_method
    filt2_tw = query_params.score_params.filt2_tw
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dcxs   = query_params.dcxs 
    #np.arange(len(hs.feats.cx2_desc))
    qcx2_res = {}
    scored = len(filter_weights.keys()) > 0
    invert_query = query_params.query_type == 'vsone'
    uid = query_params.get_uid(SV=False, scored=scored)
    if invert_query: #vsone
        assert len(query_params.qcxs) == 1
        cx2_fm = [[] for _ in xrange(len(query_params.dcxs))]
        cx2_fs = [[] for _ in xrange(len(query_params.dcxs))]
    for qcx in qcx2_neighbors.iterkeys():
        print(' * convering q'+hs.cxstr(qcx))
        (qfx2_dx, _) = qcx2_neighbors[qcx]
        nQuery = len(qfx2_dx)
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_score = np.ones(qfx2_nn.shape, dtype=ds.FS_DTYPE)
        qfx2_valid = np.ones(qfx2_nn.shape, dtype=np.bool)
        # Apply the filter weightings to determine feature validity and scores
        for key, cx2_weights in filter_weights.iteritems():
            qfx2_weights = cx2_weights[qcx]
            (sign, thresh), weight = filt2_tw[key]
            if not thresh is None:
                qfx2_passed = sign*qfx2_weights <= sign*thresh
                nValid  = qfx2_valid.sum()
                qfx2_valid  = np.bitwise_and(qfx2_valid, qfx2_passed)
                nPassed = (True - qfx2_passed).sum()
                nAdded = nValid - qfx2_valid.sum()
                print(sign*qfx2_weights)
                print(' * filt=%r thresh=%r, weight=%r, nFailed=%r, nFiltered=%r' % (key,
                                                                                  sign*thresh,
                                                                                  weight,
                                                                                  nPassed,
                                                                                  nAdded))
            if not weight == 0:
                qfx2_score  += weight * qfx2_weights
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        # Build feature matches
        if not invert_query: #vsmany
            cx2_fm = [[] for _ in xrange(len(query_params.dcxs))]
            cx2_fs = [[] for _ in xrange(len(query_params.dcxs))]
        qfx2_qfx = helpers.tiled_range(nQuery, K)
        v = qfx2_valid
        iter_matches = izip(qfx2_qfx[v], qfx2_cx[v], qfx2_fx[v], qfx2_score[v])
        if not invert_query: # vsmany
            for qfx, cx, fx, score in iter_matches:
                if qcx == cx: continue # dont vote for yourself
                cx2_fm[cx].append((qfx, fx))
                cx2_fs[cx].append(score)
            res = fmfs2_QueryResult(qcx, cx2_fm, cx2_fs, uid)
            qcx2_res[qcx] = res
        else:  # vsone
            for qfx, cx, fx, score in iter_matches:
                if qcx == cx: continue # dont vote for yourself
                cx2_fm[qcx].append((fx, qfx))
                cx2_fs[qcx].append(score)
    if agg_method == 'ChipSum':
        res = fmfs2_QueryResult(qcx, cx2_fm, cx2_fs, uid)
        qcx2_res[query_params.qcxs[0]] = res
    return qcx2_res
'''
def doit():
    df2.rrr()
    df2.reset()
    res.show_query(hs, SV=False)
    res.show_topN(hs, SV=False)
    df2.update()
    df2.bring_to_front(df2.plt.gcf())
'''
#-----
# Scoring Mechanism
#-----
def score_matches(hs, qcx2_neighbors, data_index, filter_weights, query_params):
    if agg_method == 'ChipSum':
        cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
        qcx2_res[qcx] = (cx2_fm, cx2_fs, cx2_score)
    return qcx2_res
    #s2coring_func  = [LNBNN, PlacketLuce, TopK, Borda]
    #load_precomputed(cx, query_params)

#-----
# Spatial Verification
#-----
def spatially_verify_matches(hs, qcx2_res, query_params):
    sv_params = query_params.sv_params
    print('Step 4) Spatial verification: '+sv_params.get_uid())
    xy_thresh = sv_params.xy_thresh
    slow_thresh, shigh_thresh = sv_params.scale_thresh
    use_chip_extent = sv_params.xy_thresh
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts  = hs.feats.cx2_kpts
    cx2_resSV = {}
    for qcx in qcx2_res.iterkeys():
        kpts1 = cx2_kpts[qcx]
        res = qcx2_res[qcx]
        (cx2_fm, cx2_fs, cx2_score) = (res.cx2_fm, res.cx2_fs, res.cx2_score) 
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
            np.set_printoptions(threshold=2)
            (fm_V, fs_V) = spatially_verify(kpts1, kpts2, rchip_size2, 
                                            fm, fs, xy_thresh,
                                            shigh_thresh, slow_thresh,
                                            use_chip_extent)
            cx2_fm_V[cx] = fm_V
            cx2_fs_V[cx] = fs_V
        # Rebuild the feature match / score arrays to be consistent
        for cx in xrange(len(cx2_fm_V)):
            fm = np.array(cx2_fm_V[cx], dtype=ds.FM_DTYPE)
            fm = fm.reshape(len(fm), 2)
            cx2_fm_V[cx] = fm
        for cx in xrange(len(cx2_fs_V)): 
            cx2_fs_V[cx] = np.array(cx2_fs_V[cx], dtype=ds.FS_DTYPE)
        cx2_fm_V = np.array(cx2_fm_V)
        cx2_fs_V = np.array(cx2_fs_V)
        # Rebuild the cx2_score arrays
        cx2_score_V = np.array([np.sum(fs) for fs in cx2_fs_V])
        resSV = ds.QueryResult(qcx, query_params.get_uid(SV=True))
        (resSV.cx2_fm_V, resSV.cx2_fs_V, resSV.cx2_score_V) = (cx2_fm_V, cx2_fs_V, cx2_score_V)
        cx2_resSV[qcx] = resSV
    return cx2_resSV

def spatially_verify(kpts1, kpts2, rchip_size2, fm, fs, xy_thresh,
                     shigh_thresh, slow_thresh, use_chip_extent):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers 
       returns fm_V, fs_V, H '''
    # Return if pathological
    min_num_inliers   = 4
    if len(fm) < min_num_inliers:
        return (np.empty((0, 2)), np.empty((0, 1)))
    # Get homography parameters
    if use_chip_extent:
        diaglen_sqrd = rchip_size2[0]**2 + rchip_size2[1]**2
    else:
        x_m = kpts2[fm[:,1],0].T
        y_m = kpts2[fm[:,1],1].T
        diaglen_sqrd = sv2.calc_diaglen_sqrd(x_m, y_m)
    # Try and find a homography
    sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh, 
                                    shigh_thresh, slow_thresh,
                                    diaglen_sqrd, min_num_inliers)
    if sv_tup is None:
        return (np.empty((0, 2)), np.empty((0, 1)))
    # Return the inliers to the homography
    (H, inliers, Aff, aff_inliers) = sv_tup
    fm_V = fm[inliers, :]
    fs_V = fs[inliers]
    return fm_V, fs_V
####

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
