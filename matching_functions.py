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
import scipy.optimize
import pandas as pd

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
    filt2_weights = {}
    for nnfilter in nnfilter_list:
        print('[mf.2] * computing %s weights' % nnfilter)
        nnfilter_fn = eval('nn_filters.nn_'+nnfilter+'_weight')
        filt2_weights[nnfilter] = nnfilter_fn(hs, qcx2_nns, query_params)
    return filt2_weights

#==========================
# Neighbor scoring (Voting Profiles)
#==========================
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
        qfx2_score, qfx2_valid = _apply_filter_scores(qcx, qfx2_nn, filt2_weights, filt2_tw)
        qfx2_cx = dx2_cx[qfx2_nn]
        # dont vote for yourself
        qfx2_notself_vote = qfx2_cx != qcx
        print('[mf.3] * Removed %d/%d self-votes' % (qfx2_notself_vote.sum(), qfx2_notself_vote.size))
        print('[mf.3] * %d/%d valid neighbors ' % (qfx2_valid.sum(), qfx2_valid.size))
        qfx2_valid = np.bitwise_and(qfx2_valid, qfx2_notself_vote) 
        qcx2_nnscores[qcx] = (qfx2_score, qfx2_valid)
    return qcx2_nnscores

#---
# RAW Spatial verification
#---

def spatial_verification_raw(hs, qcx2_nns, qcx2_nnscores, query_params):
    K = query_params.nn_params.K
    data_index = query_params.data_index
    dx2_cx = data_index.ax2_cx
    for qcx in qcx2_nns.iterkeys():
        (qfx2_dx, _) = qcx2_nns[qcx]
        (_, qfx2_valid) = qcx2_nnscores[qcx]
        dx2_cx

def score_chipmatch(hs, chipmatch, query_params):
    agg_method = query_params.score_params.aggregation_method
    if agg_method == 'ChipSum':
        (_, cx2_fs, _) = chipmatch
        cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
        return cx2_score

def nn_placketluce_score(hs, qcx2_nns, data_index, query_params):
    pass

def nn_positional_score(hs, qcx2_nns, data_index, query_params):
    pass

#============================
# Conversion qfx2 -> cx2
#============================
def neighbors_to_chipmatch(hs, qcx2_nns, qcx2_nnscores, query_params):
    print('[mf.4] Step 4) Convert (qfx2_dx/dist) to (cx2_fm/fs)')
    data_index = query_params.data_index
    K = query_params.nn_params.K
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dcxs   = query_params.dcxs 
    invert_query = query_params.query_type == 'vsone'
    qcx2_chipmatch = {}
    #Vsone
    if invert_query: 
        assert len(query_params.qcxs) == 1
        cx2_fm, cx2_fs, cx2_fk = new_fmfsfk(hs)
    # Iterate over chips with nearest neighbors
    for qcx in qcx2_nns.iterkeys():
        print('[mf.3] * building chipmatch q'+hs.cxstr(qcx))
        (qfx2_dx, _) = qcx2_nns[qcx]
        (qfx2_fs, qfx2_valid) = qcx2_nnscores[qcx]
        nQuery = len(qfx2_dx)
        # Build feature matches
        qfx2_nn = qfx2_dx[:, 0:K]
        qfx2_cx = dx2_cx[qfx2_nn]
        qfx2_fx = dx2_fx[qfx2_nn]
        qfx2_qfx = np.tile(np.arange(nQuery), (K, 1)).T
        qfx2_k   = np.tile(np.arange(K), (nQuery, 1))
        qfx2_tup = (qfx2_qfx, qfx2_cx, qfx2_fx, qfx2_fs, qfx2_k)
        match_iter = izip(*[qfx2[qfx2_valid] for qfx2 in qfx2_tup])
        # Vsmany
        if not invert_query: 
            cx2_fm, cx2_fs, cx2_fk = new_fmfsfk(hs)
            for qfx, cx, fx, fs, fk in match_iter:
                cx2_fm[cx].append((qfx, fx))
                cx2_fs[cx].append(fs)
                cx2_fk[cx].append(fk)
            chipmatch = _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk)
            qcx2_chipmatch[qcx] = chipmatch
            continue
        # Vsone
        for qfx, cx, fx, fs, fk in match_iter:
            cx2_fm[qcx].append((fx, qfx))
            cx2_fs[qcx].append(fs)
            cx2_fs[qcx].append(fk)
    #Vsone
    if invert_query:
        chipmatch = _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk)
        qcx2_chipmatch[query_params.qcxs[0]] = chipmatch
    return qcx2_chipmatch

#============================
# Conversion to cx2 -> qfx2
#============================
def chipmatch2_neighbors(hs, qcx2_chipmatch, query_params):
    qcx2_nns={}
    K = query_params.nn_params.K
    for qcx in qcx2_chipmatch.iterkeys():
        nQuery = len(hs.feats.cx2_kpts[qcx])
        # Stack the feature matches
        (cx2_fm, cx2_fs, cx2_fk) = qcx2_chipmatch[qcx]
        cxs = np.hstack([[cx]*len(cx2_fm[cx]) for cx in xrange(len(cx2_fm))])
        fms = np.vstack(cx2_fm)
        # Get the individual feature match lists
        qfxs = fms[:,0]
        fxs  = fms[:,0]
        fss  = np.hstack(cx2_fs)
        fks  = np.hstack(cx2_fk)
        # Rebuild the nearest neigbhor matrixes
        qfx2_cx = -np.ones((nQuery, K), np.int)
        qfx2_fx = -np.ones((nQuery, K), np.int)
        qfx2_fs = -np.ones((nQuery, K), ds.FS_DTYPE)
        qfx2_valid = np.zeros((nQuery, K), np.bool)
        # Populate nearest neigbhor matrixes
        for qfx, k in izip(qfxs, fks):
            assert qfx2_valid[qfx, k] == False
            qfx2_valid[qfx, k] = True
        for cx, qfx, k in izip(cxs, qfxs, fks): qfx2_cx[qfx, k] = cx
        for qfx, fx, k in izip(qfxs, fxs, fks): qfx2_fx[qfx, k] = fx
        for qfx, fs, k in izip(qfxs, fss, fks): qfx2_fs[qfx, k] = fs
        nns = (qfx2_cx, qfx2_fx, qfx2_fs, qfx2_valid)
        qcx2_nns[qcx] = nns
    return qcx2_nns


def score_chipmatch_PL(hs, qcx2_chipmatch, query_params):
    K = query_params.nn_params.K
    cx2_nx = hs.tables.cx2_nx
    nx2_cxs = hs.get_nx2_cxs()
    for qcx, chipmatch in qcx2_chipmatch.iteritems():
        # Run Placket Luce Model
        qfx2_utilities = _chipmatch2_utilities(hs, qcx, chipmatch, K)
        qfx2_utilities = _filter_utilities(qfx2_utilities)
        PL_matrix, altx2_tnx = _utilities2_pairwise_breaking(qfx2_utilities)
        M = PL_matrix
        gamma = _optimize(PLmatrix)
        altx2_prob = _PL_score(gamma)
        # Use probabilities as scores
        nx2_score = np.zeros(len(hs.tables.nx2_name))
        cx2_score = np.zeros(hs.num_cx)
        for altx, prob in enumerate(altx2_prob):
            tnx = altx2_tnx[altx]
            if tnx < 0: # account for temporary names
                cx2_score[-tnx] = prob
                nx2_score[1] += prob
            else:
                nx2_prob[tnx] = prob
                for cx in nx2_cxs[tnx]:
                    if cx == qcx: continue
                    cx2_score[cx] = prob

def _optimize(M):
    print('[vote] running optimization')
    m = M.shape[0]
    x0 = np.ones(m)/np.sqrt(m)
    f   = lambda x, M: linalg.norm(M.dot(x))
    con = lambda x: linalg.norm(x) - 1
    cons = {'type':'eq', 'fun': con}
    optres = scipy.optimize.minimize(f, x0, args=(M,), constraints=cons)
    x = optres['x']
    xnorm = linalg.norm(x)
    gamma = np.abs(x / xnorm)
    return gamma

def _PL_score(gamma):
    print('[vote] computing probabilities')
    nAlts = len(gamma)
    mask = np.ones(nAlts, dtype=np.bool)
    altx2_prob = np.zeros(nAlts)
    for ax in xrange(nAlts):
        mask[ax] = False
        altx2_prob[ax] = gamma[ax] / np.sum(gamma[mask])
        mask[ax] = True
    altx2_prob = altx2_prob / altx2_prob.sum()
    return altx2_prob

def _chipmatch2_utilities(hs, qcx, chipmatch, K):
    print('[vote] computing utilities')
    cx2_nx = hs.tables.cx2_nx
    qcx2_voters={}
    nQFeats = len(hs.feats.cx2_kpts[qcx])
    # Stack the feature matches
    (cx2_fm, cx2_fs, cx2_fk) = chipmatch
    cxs = np.hstack([[cx]*len(cx2_fm[cx]) for cx in xrange(len(cx2_fm))])
    cxs = np.array(cxs, np.int)
    fms = np.vstack(cx2_fm)
    # Get the individual feature match lists
    qfxs = fms[:,0]
    fss  = np.hstack(cx2_fs)
    fks  = np.hstack(cx2_fk)
    qfx2_utilities = [[] for _ in xrange(nQFeats)]
    for cx, qfx, fk, fs in izip(cxs, qfxs, fks, fss):
        nx = cx2_nx[cx]
        # Apply temporary uniquish name
        tnx = nx if nx >= 2 else -cx
        utility = (cx, tnx, fs, fk)
        qfx2_utilities[qfx].append(utility)
    for qfx in xrange(len(qfx2_utilities)):
        utilities = qfx2_utilities[qfx]
        utilities = sorted(utilities, key=lambda tup:tup[3])
        qfx2_utilities[qfx] = utilities
    return qfx2_utilities

def _filter_utilities(qfx2_utilities):
    print('[vote] filtering utilities')
    tnxs = [util[1] for utils in qfx2_utilities for util in utils]
    tnxs = np.array(tnxs)
    tnxs_min = tnxs.min()
    tnx2_freq = np.bincount(tnxs - tnxs_min)
    least_freq_tnxs = tnx2_freq.argsort() + tnxs_min
    return qfx2_utilities

def _utilities2_pairwise_breaking(qfx2_utilities):
    print('[vote] building pairwise matrix')
    arr_   = np.array
    hstack = np.hstack
    cartesian = helpers.cartesian
    tnxs = [util[1] for utils in qfx2_utilities for util in utils]
    altx2_tnx = pd.unique(tnxs)
    tnx2_altx = {nx:altx for altx, nx in enumerate(alts_tnxs)}
    nUtilities = len(qfx2_utilities)
    nAlts   = len(altx2_tnx)
    altxs   = np.arange(nAlts)
    pairwise_mat = np.zeros((nAlts, nAlts))
    qfx2_porder = [np.array([tnx2_altx[util[1]] for util in utils])
                   for utils in qfx2_utilities]
    def sum_win(ij):  # pairiwse wins on off-diagonal
        pairwise_mat[ij[0], ij[1]] += 1 
    def sum_loss(ij): # pairiwse wins on off-diagonal
        pairwise_mat[ij[1], ij[1]] -= 1
    nVoters = 0
    for qfx in xrange(nUtilities):
        sys.stdout.write('.')
        # partial and compliment order over alternatives
        porder = pd.unique(qfx2_porder[qfx])
        nReport = len(porder) 
        if nReport == 0: continue
        corder = np.setdiff1d(altxs, porder)
        # pairwise winners and losers
        pw_winners = [porder[r:r+1] for r in xrange(nReport)]
        pw_losers = [hstack((corder, porder[r+1:])) for r in xrange(nReport)]
        pw_iter = izip(pw_winners, pw_losers)
        pw_votes_ = [cartesian((winner, losers)) for winner, losers in pw_iter]
        pw_votes = np.vstack(pw_votes_)
        #pw_votes = [(w,l) for votes in pw_votes_ for w,l in votes if w != l]
        map(sum_win,  iter(pw_votes))
        map(sum_loss, iter(pw_votes))
        nVoters += 1
    print('')
    PLmatrix = pairwise_mat / nVoters     
    return pairwise_mat, altx2_tnx

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
def spatially_verify_matches(hs, qcx2_chipmatch, query_params):
    sv_params = query_params.sv_params
    print('[mf.5] Step 5) Spatial verification: %r' % sv_params.get_uid())
    nShortlist      = sv_params.shortlist_len
    xy_thresh       = sv_params.xy_thresh
    scale1, scale2  = sv_params.scale_thresh
    use_chip_extent = sv_params.use_chip_extent
    min_nInliers = sv_params.min_nInliers
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts  = hs.feats.cx2_kpts
    qcx2_chipmatchSV = {}
    for qcx in qcx2_chipmatch.iterkeys():
        (cx2_fm, cx2_fs, cx2_fk) = qcx2_chipmatch[qcx]
        cx2_score = score_chipmatch(hs, (cx2_fm, cx2_fs, cx2_fk), query_params)
        topx2_cx= cx2_score.argsort()[::-1]
        nRerank = min(len(topx2_cx), nShortlist)
        # Precompute output container
        cx2_fm_V, cx2_fs_V, cx2_fk_V = new_fmfsfk(hs)
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = np.zeros(nRerank)
        for topx in xrange(nRerank):
            cx = topx2_cx[topx]
            rchip_size2 = cx2_rchip_size[cx]
            if use_chip_extent:
                dlen_sqrd = rchip_size2[0]**2 + rchip_size2[1]**2
            else:
                fm = cx2_fm[cx]
                if len(fm) > 1:
                    kpts2 = cx2_kpts[cx]
                    x_m = kpts2[fm[:,1],0].T
                    y_m = kpts2[fm[:,1],1].T
                    dlen_sqrd = sv2.calc_diaglen_sqrd(x_m, y_m)
                else:
                    dlen_sqrd = 1
            topx2_dlen_sqrd[topx] = dlen_sqrd
        # Query Keypoints
        kpts1 = cx2_kpts[qcx]
        # spatially verify the top __NUM_RERANK__ results
        for topx in xrange(nRerank):
            cx = topx2_cx[topx]
            fm = cx2_fm[cx]
            if len(fm) < min_nInliers:
                sys.stdout.write('x')
                continue
            dlen_sqrd = topx2_dlen_sqrd[topx]
            kpts2 = cx2_kpts[cx]
            fs    = cx2_fs[cx]
            fk    = cx2_fk[cx]
            sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh, scale2,
                                            scale1, dlen_sqrd, min_nInliers)
            if sv_tup is None:
                sys.stdout.write('o')
                continue
            # Return the inliers to the homography
            (H, inliers, Aff, aff_inliers) = sv_tup
            cx2_fm_V[cx] = fm[inliers, :]
            cx2_fs_V[cx] = fs[inliers]
            cx2_fk_V[cx] = fk[inliers]
            sys.stdout.write('.')
            #np.set_printoptions(threshold=2)
        # Rebuild the feature match / score arrays to be consistent
        chipmatchSV = _fix_fmfsfk(cx2_fm_V, cx2_fs_V, cx2_fk_V)
        qcx2_chipmatchSV[qcx] = chipmatchSV
    print('\n[mf.5] Finished sv')
    return qcx2_chipmatchSV
#-----

def _fix_fmfsfk(cx2_fm, cx2_fs, cx2_fk):
    # Convert to numpy
    arr_ = np.array
    fm_dtype_ = ds.FM_DTYPE
    fs_dtype_ = ds.FS_DTYPE
    fk_dtype_ = ds.FK_DTYPE
    cx2_fm = np.array([arr_(fm, fm_dtype_) for fm in iter(cx2_fm)], list)
    for cx in xrange(len(cx2_fm)):
        cx2_fm[cx].shape = (len(cx2_fm[cx]), 2)
    cx2_fs = np.array([arr_(fs, fs_dtype_) for fs in iter(cx2_fs)], list)
    cx2_fk = np.array([arr_(fk, fk_dtype_) for fk in iter(cx2_fk)], list)
    chipmatch = (cx2_fm, cx2_fs, cx2_fk)
    return chipmatch

def new_fmfsfk(hs):
    cx2_fm = [[] for _ in xrange(hs.num_cx)]
    cx2_fs = [[] for _ in xrange(hs.num_cx)]
    cx2_fk = [[] for _ in xrange(hs.num_cx)]
    return cx2_fm, cx2_fs, cx2_fk

#----------
# QueryResult Format
#----------

def _fmfs2_QueryResult(hs, qcx, chipmatch, uid, query_params):
    res = ds.QueryResult(qcx, uid)
    cx2_score = score_chipmatch(hs, chipmatch, query_params)
    (res.cx2_fm, res.cx2_fs, res.cx2_fk) = chipmatch
    res.cx2_score = cx2_score
    return res

def chipmatch_to_res(hs, qcx2_chipmatch, query_params, SV=False, scored=False):
    uid = query_params.get_uid(SV=SV, scored=scored)
    qcx2_res = {}
    for qcx in qcx2_chipmatch.iterkeys():
        chipmatch = qcx2_chipmatch[qcx]
        res = _fmfs2_QueryResult(hs, qcx, chipmatch, uid, query_params)
        qcx2_res[qcx] = res
    return qcx2_res
