from __future__ import division, print_function
import matplotlib
import textwrap
import draw_func2 as df2
import sys
import vizualizations as viz
import matplotlib
import numpy as np
from numpy import linalg
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
    PL_matrix, altx2_tnx = _utilities2_pairwise_breaking(qfx2_utilities)
    gamma = _optimize(PL_matrix)
    altx2_prob = _PL_score(gamma)
    # Use probabilities as scores
    cx2_score, nx2_score = prob2_cxnx2scores(hs, qcx, altx2_prob, altx2_tnx)
    return cx2_score, nx2_score

def _optimize(M):
    # http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    print('[vote] running optimization')
    m = M.shape[0]
    x0 = np.ones(m)/np.sqrt(m)
    x0[1] = 1
    f   = lambda x, M:linalg.norm(M.dot(x))
    con = lambda x: int(True - (x[1] == 1))#True - np.all(x > 0)
    cons = {'type':'eq', 'fun': con}
    optres = scipy.optimize.minimize(f, x0, args=(M,), constraints=cons,
                                     method='SLSQP')
    x = optres['x']
    print('[vote] x0 = \n%r\n' % (x0,))
    #print('[vote] M  = \n%r\n' % (M,))
    print('[vote]  x = \n%r\n' % (x,))
    print('[vote] Mx = \n%r\n' % (M.dot(x),))
    print('[vote] sum(Mx) = \n%r\n' % (np.sum(M.dot(x)),))
    if not optres['success']:
        print('[vote] M.sum(0) = %r ' % M.sum(0))
        print('[vote] M.sum(1) = %r ' % M.sum(1))
        print(optres)
        raise Exception(optres['message'])
    # This should equal 0 by Theorem 1
    #xnorm = linalg.norm(x)
    #gamma = np.abs(x / xnorm)
    return x

def _optimize_SequentialLeastSquaresProgramming(M):
    m = M.shape[0]
    x0 = np.ones(m)/np.sqrt(m)
    x0[1] = 1
    f   = lambda x, M:linalg.norm(M.dot(x))
    con = lambda x: int(True - (x[1] == 1))#True - np.all(x > 0)
    cons = {'type':'eq', 'fun': con}
    optres = scipy.optimize.minimize(f, x0, args=(M,), constraints=cons,
                                     method='SLSQP')
    x = optres['x']
    if not optres['success']:
        print(optres)
        raise Exception(optres['message'])
    return x

def _optimize_Mx_is_zero_x_nonzero(M):
    '''
    Test Data:
    votes = [(3,2,1,4), (4,1,2,3), (4, 2, 3, 1), (1, 2, 3, 4)]
    qfx2_utilities = [[(nx, nx, k, k) for k, nx in enumerate(vote)] for vote in votes]
    M, altx2_nx= _utilities2_pairwise_breaking(qfx2_utilities)
    '''
    # from numpy.linalg import svd, inv
    # from numpy import eye, diag, zeros
    # Because s is sorted, and M is rank deficient, the value s[-1] should be 0
    # np.set_printoptions(precision=1, suppress=True, linewidth=80)
    # The svd is: 
    # u * s * v = M
    # u.dot(diag(s)).dot(v) = M
    #
    # u is unitary: 
    # inv(u).dot(u) == eye(len(s))
    # 
    # diag(s).dot(v) == inv(u).dot(M)
    #
    # u.dot(diag(s)) == M.dot(inv(v))
    # And because s[-1] is 0
    # u.dot(diag(s))[:,-1:] == zeros((len(s),1))
    #
    # Because we want to find Mx = 0
    #
    # So flip the left and right sides
    # M.dot(inv(v)[:,-1:]) == u.dot(diag(s))[:,-1:] 
    #
    # And you find
    # M = M
    # x = inv(v)[:,-1:]
    # 0 = u.dot(diag(s))[:,-1:] 
    # 
    # So we have the solution to our problem as x = inv(v)[:,-1:]
    #
    # Furthermore it is true that 
    # inv(v)[:,-1:].T == v[-1:,:]
    # because v is unitary and the last vector in v corresponds to a singular
    # vector because M is rank m-1
    # 
    # ALSO: v.dot(inv(v)) = eye(len(s)) so
    # v[-1].dot(inv(v)[:,-1:]) == 1
    # 
    # this means that v[-1] is non-zero, and v[-1].T == inv(v[:,-1:])
    #
    # So all of this can be done as...
    (u, s, v) = linalg.svd(M)
    x = v[-1]
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
    max_alts = 30
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



if __name__ == '__main__':
    pass

