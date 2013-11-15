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
import voting_rules2 as vr2
import pandas as pd

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
    
#============================
# Nearest Neighbors
#============================
def nearest_neighbors(hs, qcxs, q_params):
    'Plain Nearest Neighbors'
    data_index = q_params.data_index
    nn_params  = q_params.nn_params
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
        print_('.')
        (qfx2_dx, qfx2_dist) = nnfunc(qcx)
        qcx2_nns[qcx] = (qfx2_dx, qfx2_dist)
        nFoundNN += qfx2_dx.size
    print('\n[mf.1] * found %r nearest neighbors' % nFoundNN)
    return qcx2_nns

#============================
# Nearest Neighbor weights
#============================
def weight_neighbors(hs, qcx2_nns, q_params):
    f_params = q_params.f_params
    print('[mf.2] Step 2) Weight neighbors: '+f_params.get_uid())
    nnfilter_list = f_params.nnfilter_list
    filt2_weights = {}
    for nnfilter in nnfilter_list:
        print('[mf.2] * computing %s weights' % nnfilter)
        nnfilter_fn = eval('nn_filters.nn_'+nnfilter+'_weight')
        filt2_weights[nnfilter] = nnfilter_fn(hs, qcx2_nns, q_params)
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

def score_neighbors(hs, qcx2_nns, filt2_weights, q_params):
    print('[mf.3] Step 3) Scoring neighbors')
    qcx2_nnscores = {}
    data_index = q_params.data_index
    K = q_params.nn_params.K
    dx2_cx = data_index.ax2_cx
    filt2_tw = q_params.f_params.filt2_tw
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

#def spatial_verification_raw(hs, qcx2_nns, qcx2_nnscores, q_params):
    #K = q_params.nn_params.K
    #data_index = q_params.data_index
    #dx2_cx = data_index.ax2_cx
    #for qcx in qcx2_nns.iterkeys():
        #(qfx2_dx, _) = qcx2_nns[qcx]
        #(_, qfx2_valid) = qcx2_nnscores[qcx]
        #dx2_cx

#-----
# Scoring Mechanism
#-----
#s2coring_func  = [LNBNN, PlacketLuce, TopK, Borda]
#load_precomputed(cx, q_params)
def score_chipmatch(hs, qcx, chipmatch, score_method, q_params=None):
    print(' * Scoring chipmatch: '+score_method)
    if score_method == 'chipsum':
        (_, cx2_fs, _) = chipmatch
        cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
        return cx2_score
    if score_method == 'placketluce':
        cx2_score, nx2_score = vr2.score_chipmatch_PL(hs, qcx, chipmatch, q_params)
    return cx2_score

#============================
# Conversion qfx2 -> cx2
#============================
def neighbors_to_chipmatch(hs, qcx2_nns, qcx2_nnscores, q_params):
    '''vsmany/vsone counts here. also this is where the filter 
    weights and thershold are applied to the matches. Essientally 
    nearest neighbors are converted into weighted assignments'''
    print('[mf.4] Step 4) Convert (qfx2_dx/dist) to (cx2_fm/fs)')
    data_index = q_params.data_index
    K = q_params.nn_params.K
    dx2_cx = data_index.ax2_cx
    dx2_fx = data_index.ax2_fx
    dcxs   = q_params.dcxs 
    invert_query = q_params.query_type == 'vsone'
    qcx2_chipmatch = {}
    #Vsone
    if invert_query: 
        assert len(q_params.qcxs) == 1
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
        qcx = q_params.qcxs[0]
        qcx2_chipmatch[qcx] = chipmatch
    return qcx2_chipmatch

#============================
# Conversion to cx2 -> qfx2
#============================
def chipmatch2_neighbors(hs, qcx2_chipmatch, q_params):
    qcx2_nns={}
    K = q_params.nn_params.K
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




#-----
# Spatial Verification
#-----
def spatially_verify_matches(hs, qcx2_chipmatch, q_params):
    sv_params = q_params.sv_params
    print('[mf.5] Step 5) Spatial verification: %r' % sv_params.get_uid())
    prescore_method  = sv_params.prescore_method
    nShortlist      = sv_params.nShortlist
    xy_thresh       = sv_params.xy_thresh
    scale1, scale2  = sv_params.scale_thresh
    use_chip_extent = sv_params.use_chip_extent
    min_nInliers = sv_params.min_nInliers
    cx2_rchip_size = hs.get_cx2_rchip_size()
    cx2_kpts  = hs.feats.cx2_kpts
    qcx2_chipmatchSV = {}
    for qcx in qcx2_chipmatch.iterkeys():
        chipmatch = qcx2_chipmatch[qcx]
        cx2_prescore = score_chipmatch(hs, qcx, chipmatch, prescore_method, q_params)
        (cx2_fm, cx2_fs, cx2_fk) = chipmatch
        topx2_cx= cx2_prescore.argsort()[::-1]
        nRerank = min(len(topx2_cx), nShortlist)
        # Precompute output container
        cx2_fm_V, cx2_fs_V, cx2_fk_V = new_fmfsfk(hs)
        # Check the diaglen sizes before doing the homography
        topx2_dlen_sqrd = np.zeros(nRerank)
        for topx in xrange(nRerank):
            cx = topx2_cx[topx]
            rchip_size2 = cx2_rchip_size[cx]
            fm = cx2_fm[cx]
            if len(fm) == 0:
                topx2_dlen_sqrd[topx] = 1
                continue
            if use_chip_extent:
                dlen_sqrd = rchip_size2[0]**2 + rchip_size2[1]**2
            else:
                kpts2 = cx2_kpts[cx]
                x_m = kpts2[fm[:,1],0].T
                y_m = kpts2[fm[:,1],1].T
                dlen_sqrd = sv2.calc_diaglen_sqrd(x_m, y_m)
            topx2_dlen_sqrd[topx] = dlen_sqrd
        # Query Keypoints
        kpts1 = cx2_kpts[qcx]
        # spatially verify the top __NUM_RERANK__ results
        for topx in xrange(nRerank):
            cx = topx2_cx[topx]
            fm = cx2_fm[cx]
            if len(fm) < min_nInliers:
                print_('x')
                continue
            dlen_sqrd = topx2_dlen_sqrd[topx]
            kpts2 = cx2_kpts[cx]
            fs    = cx2_fs[cx]
            fk    = cx2_fk[cx]
            sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh, scale2,
                                            scale1, dlen_sqrd, min_nInliers)
            if sv_tup is None:
                print_('o')
                continue
            # Return the inliers to the homography
            (H, inliers, Aff, aff_inliers) = sv_tup
            cx2_fm_V[cx] = fm[inliers, :]
            cx2_fs_V[cx] = fs[inliers]
            cx2_fk_V[cx] = fk[inliers]
            print_('.')
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

def _fmfs2_QueryResult(hs, qcx, chipmatch, uid, q_params):
    score_method = q_params.score_method
    cx2_score = score_chipmatch(hs, qcx, chipmatch, score_method, q_params)
    res = ds.QueryResult(qcx, uid)
    (res.cx2_fm, res.cx2_fs, res.cx2_fk) = chipmatch
    res.cx2_score = cx2_score
    return res

def chipmatch_to_res(hs, qcx2_chipmatch, q_params, aug=''):
    print('[mf.6] 6) chipmatch -> res')
    # Keep original score method
    score_method_ = q_params.score_method
    # Hacky dev stuff
    if aug == '+ORIG':
        uid = q_params.get_uid(SV=False, filtered=False, scored=True)
    elif aug == '+FILT':
        uid = q_params.get_uid(SV=False, filtered=True, NN=False, scored=True)
    elif aug == '+SVER':
        uid = q_params.get_uid(SV=True, filtered=False, NN=False, scored=True)
    elif aug == '+SVPL':
        q_params.score_method = 'placketluce'
        uid = q_params.get_uid(SV=True, NN=False, filtered=False, scored=True)
    if aug != '':
        aug = ' '+aug
    # Create the result structures for each query.
    qcx2_res = {}
    for qcx in qcx2_chipmatch.iterkeys():
        chipmatch = qcx2_chipmatch[qcx]
        res = _fmfs2_QueryResult(hs, qcx, chipmatch, uid, q_params)
        res.title = uid+aug
        qcx2_res[qcx] = res
    # Retain original score method
    q_params.score_method = score_method_
    return qcx2_res
