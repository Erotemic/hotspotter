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
import nn_filters

def reload_module():
    import imp, sys
    print('[mc3] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()
    
#============================
# Nearest Neighbors
#============================
def nearest_neighbors(hs, qcxs, query_params):
    'Plain Nearest Neighbors'
    data_index = query_params.data_index
    nn_params  = query_params.nn_params
    print('[mf.1] Step 1) Assign nearest neighbors: '+nn_params.get_uid())
    K = nn_params.K
    Knorm = nn_params.Knorm
    checks = nn_params.checks
    cx2_desc = hs.feats.cx2_desc
    nn_index = data_index.flann.nn_index
    nnfunc = lambda qcx: nn_index(cx2_desc[qcx], K+Knorm, checks=checks)
    #qcx2_nns = {qcx:func(qcx) for qcx in qcxs}
    qcx2_nns = {}
    nFoundNN = 0
    print('')
    for qcx in qcxs:
        sys.stdout.write('.')
        (qfx2_dx, qfx2_dist) = nnfunc(qcx)
        qcx2_nns[qcx] = (qfx2_dx, qfx2_dist)
        nFoundNN += qfx2_dx.size
    print('\n[mf.1] * found %r nearest neighbors' % nFoundNN)
    return qcx2_nns

#============================
# Nearest Neighbor weights
#============================
def weight_neighbors(hs, qcx2_nns, query_params):
    score_params = query_params.score_params
    print('[mf.2] Step 2) Weight neighbors: '+score_params.get_uid())
    nnfilter_list = score_params.nnfilter_list
    filter_weights = {}
    for nnfilter in nnfilter_list:
        print('[mf.2] * computing %s weights' % nnfilter)
        nnfilter_fn = eval('nn_filters.nn_'+nnfilter+'_weight')
        filter_weights[nnfilter] = nnfilter_fn(hs, qcx2_nns, query_params)
    return filter_weights

def nn_placketluce_score(hs, qcx2_nns, data_index, query_params):
    pass

def nn_positional_score(hs, qcx2_nns, data_index, query_params):
    pass

#============================
# Conversion 
#============================

def _fix_fmfs(cx2_fm, cx2_fs):
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

def _fmfs2_QueryResult(qcx, cx2_fm, cx2_fs, uid):
    cx2_fm, cx2_fs = _fix_fmfs(cx2_fm, cx2_fs)
    cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
    res = ds.QueryResult(qcx, uid)
    res.cx2_fm = cx2_fm
    res.cx2_fs = cx2_fs
    res.cx2_score = cx2_score
    return res

def _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt2_tw):
    qfx2_score = np.ones(qfx2_nn.shape, dtype=ds.FS_DTYPE)
    qfx2_valid = np.ones(qfx2_nn.shape, dtype=np.bool)
    # Apply the filter weightings to determine feature validity and scores
    for filt, cx2_weights in filt2_weights.iteritems():
        qfx2_weights = cx2_weights[qcx]
        (sign, thresh), weight = filt2_tw[filt]
        print('[mf.3] * filt=%r ' % filt)
        if not thresh is None or not weight == 0:
            print('[mf.3] * \\ qfx2_weights = '+helpers.printable_mystats(qfx2_weights.flatten()))
        if not thresh is None:
            qfx2_passed = sign*qfx2_weights <= sign*thresh
            nValid  = qfx2_valid.sum()
            qfx2_valid  = np.bitwise_and(qfx2_valid, qfx2_passed)
            nPassed = (True - qfx2_passed).sum()
            nAdded = nValid - qfx2_valid.sum()
            #print(sign*qfx2_weights)
            print('[mf.3] * \\ *thresh=%r, nFailed=%r, nFiltered=%r' % \
                    (sign*thresh, nPassed, nAdded))
        if not weight == 0:
            print('[mf.3] * \\ weight=%r' % weight)
            qfx2_score  += weight * qfx2_weights
    return qfx2_score, qfx2_valid

def score_neighbors(hs, qcx2_nns, filt2_weights, query_params):
    print('[mf.3] Step 3) Scoring neighbors')
    qcx2_nnscores = {}
    data_index = query_params.data_index
    K = query_params.nn_params.K
    dx2_cx = data_index.ax2_cx
    filt2_tw = query_params.score_params.filt2_tw
    for qcx in qcx2_nns.iterkeys():
        print('[mf.3] * scoring q'+hs.cxstr(qcx))
        (qfx2_dx, _) = qcx2_nns[qcx]
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_score, qfx2_valid = _apply_filter_scores(qcx, qfx2_nn,
                                                      filt2_weights, filt2_tw)
        qfx2_cx = dx2_cx[qfx2_nn]
        # dont vote for yourself
        qfx2_notself_vote = qfx2_cx != qcx
        print('[mf.3] * Removed %d/%d self-votes' % (qfx2_notself_vote.sum(), qfx2_notself_vote.size))
        print('[mf.3] * %d/%d valid neighbors ' % (qfx2_valid.sum(), qfx2_valid.size))
        qfx2_valid = np.bitwise_and(qfx2_valid, qfx2_notself_vote) 
        qcx2_nnscores[qcx] = (qfx2_score, qfx2_valid)
    return qcx2_nnscores


def neighbors_to_res(hs, qcx2_nns, qcx2_nnscores, query_params, scored=True):
    print('[mf.4] Step 4) Convert (qfx2_dx/dist) to (cx2_fm/fs)')
    data_index = query_params.data_index
    K = query_params.nn_params.K
    agg_method = query_params.score_params.aggregation_method
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dcxs   = query_params.dcxs 
    invert_query = query_params.query_type == 'vsone'
    qcx2_res = {}
    uid = query_params.get_uid(SV=False, scored=scored)
    if invert_query: #vsone
        assert len(query_params.qcxs) == 1
        cx2_fm = [[] for _ in xrange(len(query_params.dcxs))]
        cx2_fs = [[] for _ in xrange(len(query_params.dcxs))]
    for qcx in qcx2_nns.iterkeys():
        print('[mf.3] * converting q'+hs.cxstr(qcx))
        (qfx2_dx, _) = qcx2_nns[qcx]
        # Build feature matches
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        (qfx2_score, qfx2_valid) = qcx2_nnscores[qcx]
        nQuery = len(qfx2_dx)
        qfx2_qfx = helpers.tiled_range(nQuery, K)
        v = qfx2_valid
        iter_matches = izip(qfx2_qfx[v], qfx2_cx[v], qfx2_fx[v], qfx2_score[v])
        if not invert_query: # vsmany
            cx2_fm = [[] for _ in xrange(len(query_params.dcxs))]
            cx2_fs = [[] for _ in xrange(len(query_params.dcxs))]
            for qfx, cx, fx, score in iter_matches:
                cx2_fm[cx].append((qfx, fx))
                cx2_fs[cx].append(score)
            if agg_method == 'ChipSum':
                res = _fmfs2_QueryResult(qcx, cx2_fm, cx2_fs, uid)
                qcx2_res[qcx] = res
        else:  # vsone
            for qfx, cx, fx, score in iter_matches:
                cx2_fm[qcx].append((fx, qfx))
                cx2_fs[qcx].append(score)
    if invert_query and agg_method == 'ChipSum':
        res = _fmfs2_QueryResult(qcx, cx2_fm, cx2_fs, uid)
        qcx2_res[query_params.qcxs[0]] = res
    return qcx2_res

#-----
# Scoring Mechanism
#-----
def score_matches(hs, qcx2_nns, filter_weights, query_params):
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
    print('[mf.5] Step 5) Spatial verification: '+sv_params.get_uid())
    xy_thresh = sv_params.xy_thresh
    slow_thresh, shigh_thresh = sv_params.scale_thresh
    use_chip_extent = sv_params.xy_thresh
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts  = hs.feats.cx2_kpts
    cx2_resSV = {}
    print('')
    for qcx in qcx2_res.iterkeys():
        sys.stdout.write('.')
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
            #np.set_printoptions(threshold=2)
            (fm_V, fs_V) = sv2.spatially_verify(kpts1, kpts2, rchip_size2, 
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
    print('\n[mf.5] Finished sv')
    return cx2_resSV
