from __future__ import division, print_function
import matplotlib
import textwrap
import draw_func2 as df2
import sys
import vizualizations as viz
import matplotlib
import numpy as np
from numpy import linalg
from numpy.linalg import svd
import helpers
import scipy.optimize
import scipy
import params
import match_chips2 as mc2
from itertools import izip
import pandas as pd

def reload_module():
    import imp, sys
    print('[reload] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

# chipmatch = qcx2_chipmatch[qcx]
def score_chipmatch_PL(hs, qcx, chipmatch, q_params):
    K = q_params.nn_params.K
    # Run Placket Luce Model
    qfx2_utilities = _chipmatch2_utilities(hs, qcx, chipmatch, K)
    qfx2_utilities = _filter_utilities(qfx2_utilities)
    PL_matrix, altx2_tnx = _utilities2_weighted_pairwise_breaking(qfx2_utilities)
    gamma = _optimize(PL_matrix)
    altx2_prob = _PL_score(gamma)
    # Use probabilities as scores
    cx2_score, nx2_score = prob2_cxnx2scores(hs, qcx, altx2_prob, altx2_tnx)
    return cx2_score, nx2_score

TMP = []
def _optimize(M):
    global TMP
    print('[vote] optimize')
    (u, s, v) = svd(M)
    x = np.abs(v[-1])
    check = np.abs(M.dot(x)) < 1E-9
    if not all(check):
        raise Exception('SVD method failed miserabley')
    tmp1 = []
    tmp1 += [('[vote] x=%r' % x)]
    tmp1 += [('[vote] M.dot(x).sum() = %r' % M.dot(x).sum())]
    tmp1 += [('[vote] M.dot(np.abs(x)).sum() = %r' % M.dot(np.abs(x)).sum())]
    print(tmp1)
    TMP  += [tmp1]
    return x


def _PL_score(gamma):
    print('[vote] computing probabilities')
    nAlts = len(gamma)
    altx2_prob = np.zeros(nAlts)
    for ax in xrange(nAlts):
        altx2_prob[ax] = gamma[ax] / np.sum(gamma)
    print('[vote] altx2_prob: '+str(altx2_prob))
    print('[vote] sum(prob): '+str(sum(altx2_prob)))
    return altx2_prob

def prob2_cxnx2scores(hs, qcx, altx2_prob, altx2_tnx):
    nx2_score = np.zeros(len(hs.tables.nx2_name))
    cx2_score = np.zeros(hs.num_cx)
    nx2_cxs = hs.get_nx2_cxs()
    for altx, prob in enumerate(altx2_prob):
        tnx = altx2_tnx[altx]
        if tnx < 0: # account for temporary names
            cx2_score[-tnx] = prob
            nx2_score[1] += prob
        else:
            nx2_score[tnx] = prob
            for cx in nx2_cxs[tnx]:
                if cx == qcx: continue
                cx2_score[cx] = prob
    return cx2_score, nx2_score

def _chipmatch2_utilities(hs, qcx, chipmatch, K):
    print('[vote] computing utilities')
    cx2_nx = hs.tables.cx2_nx
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
    max_alts = 200
    tnxs = [util[1] for utils in qfx2_utilities for util in utils]
    tnxs = np.array(tnxs)
    tnxs_min = tnxs.min()
    tnx2_freq = np.bincount(tnxs - tnxs_min)
    nAlts = (tnx2_freq > 0).sum()
    nRemove = max(0, nAlts - max_alts)
    print(' * removing %r/%r alternatives' % (nRemove, nAlts))
    if nRemove > 0: #remove least frequent names
        most_freq_tnxs = tnx2_freq.argsort()[::-1] + tnxs_min
        keep_tnxs = set(most_freq_tnxs[0:max_alts].tolist())
        for qfx in xrange(len(qfx2_utilities)):
            utils = qfx2_utilities[qfx]
            qfx2_utilities[qfx] = [util for util in utils if util[1] in keep_tnxs]
    return qfx2_utilities

def _utilities2_pairwise_breaking(qfx2_utilities):
    print('[vote] building pairwise matrix')
    arr_   = np.array
    hstack = np.hstack
    cartesian = helpers.cartesian
    tnxs = [util[1] for utils in qfx2_utilities for util in utils]
    altx2_tnx = pd.unique(tnxs)
    tnx2_altx = {nx:altx for altx, nx in enumerate(altx2_tnx)}
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
        # partial and compliment order over alternatives
        porder = pd.unique(qfx2_porder[qfx])
        nReport = len(porder) 
        if nReport == 0: continue
        #sys.stdout.write('.')
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
    #print('')
    PLmatrix = pairwise_mat / nVoters     
    # sum(0) gives you the sum over rows, which is summing each column
    # Basically a column stochastic matrix should have 
    # M.sum(0) = 0
    #print('CheckMat = %r ' % all(np.abs(PLmatrix.sum(0)) < 1E-9))
    return PLmatrix, altx2_tnx

def _utilities2_weighted_pairwise_breaking(qfx2_utilities):
    print('[vote] building pairwise matrix')
    arr_   = np.array
    hstack = np.hstack
    cartesian = helpers.cartesian
    # get temp name indexes
    tnxs = [util[1] for utils in qfx2_utilities for util in utils]
    altx2_tnx = pd.unique(tnxs)
    tnx2_altx = {nx:altx for altx, nx in enumerate(altx2_tnx)}
    nUtilities = len(qfx2_utilities)
    nAlts   = len(altx2_tnx)
    altxs   = np.arange(nAlts)
    pairwise_mat = np.zeros((nAlts, nAlts))
    qfx2_porder = [np.array([tnx2_altx[util[1]] for util in utils]) for utils in qfx2_utilities]
    qfx2_worder = [np.array([util[2] for util in utils]) for utils in qfx2_utilities]
    nVoters = 0
    for qfx in xrange(nUtilities):
        # partial and compliment order over alternatives
        porder = qfx2_porder[qfx]
        worder = qfx2_worder[qfx]
        _, idx = np.unique(porder, return_inverse=True)
        idx = np.sort(idx)
        porder = porder[idx]
        worder = worder[idx]
        nReport = len(porder) 
        if nReport == 0: continue
        #sys.stdout.write('.')
        corder = np.setdiff1d(altxs, porder)
        nUnreport = len(corder)
        # pairwise winners and losers
        for r_win in xrange(0, nReport):
            i = porder[r_win]
            wi = worder[r_win]
            for r_lose in xrange(r_win+1, nReport):
                j = porder[r_lose]
                wj = worder[r_lose]
                w = wi
                #w = wi - wj
                pairwise_mat[i,j] += w
                pairwise_mat[j,j] -= w
            for r_lose in xrange(nUnreport):
                j = corder[r_lose]
                wj = 0
                w = wi
                #w = wi - wj
                pairwise_mat[i,j] += w
                pairwise_mat[j,j] -= w
            nVoters += wi
    #print('')
    PLmatrix = pairwise_mat / nVoters     
    # sum(0) gives you the sum over rows, which is summing each column
    # Basically a column stochastic matrix should have 
    # M.sum(0) = 0
    #print('CheckMat = %r ' % all(np.abs(PLmatrix.sum(0)) < 1E-9))
    return PLmatrix, altx2_tnx


if __name__ == '__main__':
    pass

