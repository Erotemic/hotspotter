#========================================
# IMPORTS
#========================================
# Hotspotter Frontend Imports
import drawing_functions2 as df2
# Hotspotter Imports
from cvransac2 import H_homog_from_DELSAC
from helpers import checkpath
from hotspotter.helpers import Timer, get_exec_src, check_path, tic, toc, myprint, printWARN
from hotspotter.other.ConcretePrintable import DynStruct
import algos2
import chip_compute2
import cvransac2
import feature_compute2
import load_data2
import params2
import params2 as params
from params2 import *
import report_results2
# Math and Science Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import pyflann
import scipy.sparse
import sklearn.preprocessing 
# Python Imports
import itertools
import imp
import sys
# Imp Module Reloads
#imp.reload(cvransac2)
#imp.reload(algos2)
#imp.reload(params2)


def myreload():
    import imp
    imp.reload(cvransac2)
    imp.reload(df2)
    imp.reload(algos2)

#========================================
# Parameters and Debugging
#========================================
__K__ = params.__K__
__NUM_RERANK__   = params.__NUM_RERANK__
__RATIO_THRESH__ = params.__RATIO_THRESH__
__FLANN_PARAMS__ = params.__FLANN_PARAMS__
__FEAT_TYPE__    = params.__FEAT_TYPE__ 
__XY_THRESH__    = params.__XY_THRESH__ = .05
__FEAT_TYPE__    = 'HESAFF'

def printDBG(msg):
    pass
#========================================
# Bag-of-Words
#========================================
   
def assign_features_to_bow_vector(vocab):
    pass

bow_norm = 'l2'
def precompute_bag_of_words(hs):
    # Build (or reload) one vs many flann index
    __VOCAB_SIZE__ = 10000
    feat_dir  = hs.dirs.feat_dir
    num_clusters = __VOCAB_SIZE__
    # Compute words
    ax2_cx, ax2_fx, ax2_desc = aggregate_descriptors_1vM(hs)
    ax2_wx, words = algos2.precompute_akmeans(ax2_desc, num_clusters, force_recomp=False)
    # Build a NN index for the words
    flann_words = pyflann.FLANN()
    flann_words_params = flann_words.build_index(words, algorithm='linear')
    print flann_words_params
    # Compute Inverted File
    wx2_axs = [[] for _ in xrange(num_clusters)]
    for ax, wx in enumerate(ax2_wx):
        wx2_axs[wx].append(ax)
    wx2_cxs = [[ax2_cx[ax] for ax in ax_list] for ax_list in wx2_axs]
    wx2_fxs = [[ax2_fx[ax] for ax in ax_list] for ax_list in wx2_axs]
    # Create visual-word-vectors for each chip
    # Build bow using coorindate list coo matrix
    coo_cols = ax2_wx
    coo_rows = ax2_cx
    # The term frequency (TF) is implicit in the coo format
    coo_values = np.ones(len(ax2_cx), dtype=np.uint8)
    coo_format = (coo_values, (coo_rows, coo_cols))
    coo_cx2_bow = scipy.sparse.coo_matrix(coo_format, dtype=np.float, copy=True)
    # Normalize each visual vector
    csr_cx2_bow = scipy.sparse.csr_matrix(coo_cx2_bow, copy=False)
    csr_cx2_bow = sklearn.preprocessing.normalize(csr_cx2_bow, norm=bow_norm, axis=1, copy=False)
    # Calculate inverse document frequency (IDF)
    # chip indexes (cxs) are the documents
    num_chips = np.float(len(hs.tables.cx2_cid))
    wx2_df  = [len(set(cx_list)) for cx_list in wx2_cxs]
    wx2_idf = np.log2(num_chips / np.array(wx2_df, dtype=np.float))
    wx2_idf.shape = (1, wx2_idf.size)
    # Preweight the bow vectors
    idf_sparse = scipy.sparse.csr_matrix(wx2_idf)
    cx2_bow = scipy.sparse.vstack([row.multiply(idf_sparse) for row in csr_cx2_bow], format='csr')
    # Renormalize
    cx2_bow = sklearn.preprocessing.normalize(cx2_bow, norm=bow_norm, axis=1, copy=False)
    # Return vocabulary
    vocab = Vocabulary(words, flann_words, cx2_bow, wx2_idf, wx2_cxs, wx2_fxs)
    vocab.printme()
    return vocab

def truncate_vvec(vvec, thresh):
    shape = vvec.shape
    vvec_flat = vvec.toarray().ravel()
    vvec_flat = vvec_flat * (vvec_flat > thresh)
    vvec_trunc = scipy.sparse.csr_matrix(vvec_flat)
    vvec_trunc = sklearn.preprocessing.normalize(vvec_trunc, norm=bow_norm, axis=1, copy=False)
    return vvec_trunc

def assign_matches_BOW(qcx, cx2_cid, cx2_desc, vocab):
    cx2_bow = vocab.cx2_bow
    wx2_cxs = vocab.wx2_cxs
    wx2_fxs = vocab.wx2_fxs
    words   = vocab.words
    flann_words = vocab.flann_words
    # This is not robust to out of database queries
    vvec = cx2_bow[qcx]
    # Compute distance to every database vector
    cx2_score = (cx2_bow.dot(vvec.T)).toarray()
    # Assign each query descriptor to a word
    qdesc = np.array(cx2_desc[qcx], dtype=words.dtype)
    (qfx2_wx, __word_dist) = flann_words.nn_index(qdesc, 1)
    # Vote for the chips
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    vvec_flat = np.array(vvec.toarray()).ravel()
    qcx2_wx    = np.flatnonzero(vvec_flat)

    cx2_nx = hs.tables.cx2_nx
    qnx = cx2_nx[qcx]
    other_cx, = np.where(cx2_nx == qnx)

    vvec2 = truncate_vvec(vvec, .04)
    vvec2_flat = vvec2.toarray().ravel()

    cx2_score = (cx2_bow.dot(vvec.T)).toarray().ravel()
    top_cxs = cx2_score.argsort()[::-1]

    cx2_score2 = (cx2_bow.dot(vvec2.T)).toarray().ravel()
    top_cxs2 = cx2_score2.argsort()[::-1]

    comp_str = ''
    for t,s,t2,s2 in np.vstack([top_cxs, cx2_score, top_cxs2, cx2_score2]).T:
        m1 = [' ','*'][t in other_cx]
        m2 = [' ','*'][t2 in other_cx] 
        comp_str += m1 + '%4d %4.2f %4d %4.2f' % (t,s,t2,s2) + m2 + '\n'
    print comp_str
    df2.close_all_figures()
    df2.show_signature(vvec_flat, fignum=2)
    df2.show_signature(vvec2_flat, fignum=3)
    df2.present()

    #df2.show_histogram(qcx2_wx, fignum=1)
    _tmp = 0
    _tmp2 = 0
    for qfx, wx in enumerate(qfx2_wx):
        cx_list = wx2_cxs[wx]
        fx_list = wx2_fxs[wx]
        _qfs = vvec_flat[wx]
        for (cx, fx) in zip(cx_list, fx_list): 
            _tmp+=1
            if cx == qcx: continue
            _tmp2+=1
            fm  = (qfx, fx)
            _fs = cx2_bow[cx, wx]
            fs  = _qfs * _fs
            cx2_fm[cx].append(fm)
            cx2_fs[cx].append(fs)
    return cx2_fs, cx2_fm

#========================================
# One-vs-Many 
#========================================

# Feature scoring functions
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist, vdist+1E-8)+1) 
def RATIO_fn(vdist, ndist): return np.divide(ndist, vdist+1E-8)
def LNBNN_fn(vdist, ndist): return ndist - vdist 
score_fn = RATIO_fn

#@profile
def assign_matches_1vM(qcx, cx2_cid, cx2_desc, one_vs_many):
    '''
    Matches desc1 vs all database descriptors using 
    Input:
        qcx        - query chip index
        cx2_cid     - chip ID lookup table (for removing self matches)
        cx2_desc    - chip descriptor lookup table
        one_vs_many - class with FLANN index of database descriptors
    Output: 
        cx2_fm - C x Mx2 array of matching feature indexes
        cx2_fs - C x Mx1 array of matching feature scores
    '''
    print('Assigning 1vM feature matches from qcx=%d to %d chips' % (qcx, len(cx2_cid)))
    flann_1vM = one_vs_many.flann_1vM
    ax2_cx    = one_vs_many.ax2_cx
    ax2_fx    = one_vs_many.ax2_fx
    isQueryIndexed = True
    desc1 = cx2_desc[qcx]
    K = __K__+1 if isQueryIndexed else __K__
    # Find each query descriptor's K+1 nearest neighbors
    (qfx2_ax, qfx2_dists) = flann_1vM.nn_index(desc1, K+1, **__FLANN_PARAMS__)
    vote_dists = qfx2_dists[:, 0:K]
    norm_dists = qfx2_dists[:, K] # K+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([score_fn(_vdist.T, norm_dists) for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    qfx2_cx = ax2_cx[qfx2_ax[:,0:K]]
    qfx2_fx = ax2_fx[qfx2_ax[:,0:K]]
    # Build feature matches
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    num_qf = len(desc1)
    qfx2_qfx = np.tile(np.arange(num_qf).reshape(num_qf,1), (1,K)) 
    iter_matches = iter(zip(qfx2_qfx.flat, qfx2_cx.flat, qfx2_fx.flat, qfx2_score.flat))
    for qfx, cx,fx,score in iter_matches:
        if qcx == cx: continue # dont vote for yourself
        cx2_fm[cx].append((qfx,fx))
        cx2_fs[cx].append(score)
    # Convert to numpy
    for cx in xrange(len(cx2_cid)): cx2_fm[cx] = np.array(cx2_fm[cx])
    for cx in xrange(len(cx2_cid)): cx2_fs[cx] = np.array(cx2_fs[cx])
    return cx2_fm, cx2_fs

def assign_matches_1vM_BINARY(qcx, cx2_cid, cx2_desc):
    return None

def aggregate_descriptors_1vM(hs):
    cx2_cid   = hs.tables.cx2_cid
    cx2_desc  = hs.feats.cx2_desc
    print('Aggregating descriptors for one vs many')
    cx2_nFeats = [len(k) for k in cx2_desc]
    _ax2_cx = [[cx_]*nFeats for (cx_, nFeats) in iter(zip(range(len(cx2_cid)), cx2_nFeats))]
    _ax2_fx = [range(nFeats) for nFeats in iter(cx2_nFeats)]
    ax2_cx  = np.array(list(itertools.chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(itertools.chain.from_iterable(_ax2_fx)))
    ax2_desc = np.vstack(cx2_desc)
    #ax2_desc_white = whiten(ax2_desc)
    #data = ax2_desc_white
    #data = scale_to_byte_range(data)
    return ax2_cx, ax2_fx, ax2_desc

#@profile
def precompute_index_1vM(hs):
    # Build (or reload) one vs many flann index
    feat_dir  = hs.dirs.feat_dir
    feat_type = hs.feats.feat_type
    ax2_cx, ax2_fx, ax2_desc = aggregate_descriptors_1vM(hs)
    flann_1vM = pyflann.FLANN()
    flann_1vM_path = feat_dir + '/flann_1vM_'+feat_type+'.index'
    load_success = False
    if check_path(flann_1vM_path):
        try:
            print('Trying to load FLANN index')
            flann_1vM.load_index(flann_1vM_path, ax2_desc)
            print('...success')
            load_success = True
        except Exception as ex:
            print('...cannot load FLANN index'+repr(ex))
    if not load_success:
        with Timer(msg='rebuilding FLANN index'):
            flann_1vM.build_index(ax2_desc, **__FLANN_PARAMS__)
            flann_1vM.save_index(flann_1vM_path)
    # Return a one-vs-many structure
    one_vs_many = OneVsMany(flann_1vM, ax2_desc, ax2_cx, ax2_fx)
    return one_vs_many

#========================================
# One-vs-One 
#========================================

cv2_matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
def match_1v1_BINARY(desc1, desc2):
    raw_matches = cv2_matcher.knnMatch(desc1, desc2, 1)
    fm = np.array([(m1[0].queryIdx-1, m1[0].trainIdx-1) for m1 in iter(raw_matches)])
    #for (qx, tx) in iter(fm):
        #if tx >= len(desc2) or qx >= len(desc1):
            #print(repr(tx))
            #print(repr(qx))
            #print(repr(len(desc1)))
            #print(repr(len(desc2)))
            #raise Exception('Error')

    fs = np.array([1/m1[0].distance for m1 in iter(raw_matches)])
    return fm, fs

def assign_matches_1v1_BINARY(qcx, cx2_cid, cx2_desc):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    desc1 = cx2_desc[qcx]
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, desc2 in enumerate(cx2_desc):
        sys.stdout.write('.'); sys.stdout.flush()
        if cx == qcx: continue
        fm, fs = match_1v1_BINARY(desc1, desc2)
        #print('fm.shape = %r ' % (repr(fm.shape),))
        #print('fm.max = %r ' % (repr(fm.max(0)),))
        #print('fm.min = %r ' % (repr(fm.min(0)),))
        #print('fs.shape = %r ' % (repr(fs.shape),))
        #print('desc1.shape = %r ' % (repr(desc1.shape),))
        #print('desc2.shape = %r ' % (repr(desc2.shape),))
        cx2_fm[cx] = fm
        cx2_fs[cx] = fs
    sys.stdout.write('DONE')
    return cx2_fm, cx2_fs

def assign_matches_1v1(qcx, cx2_cid, cx2_desc):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    desc1 = cx2_desc[qcx]
    flann_1v1 = pyflann.FLANN()
    flann_1v1.build_index(desc1, **__FLANN_PARAMS__)
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, desc2 in enumerate(cx2_desc):
        sys.stdout.write('.')
        sys.stdout.flush()
        if cx == qcx: continue
        (fm, fs) = match_1v1(desc2, flann_1v1)
        cx2_fm[cx] = fm
        cx2_fs[cx] = fs
    sys.stdout.write('DONE')
    flann_1v1.delete_index()
    return cx2_fm, cx2_fs

def cv2_match(desc1, desc2):
    K = 1
    cv2_matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    raw_matches = cv2_matcher.knnMatch(desc1, desc2, K)
    matches = [(m1.trainIdx, m1.queryIdx) for m1 in raw_matches]

#@profile
def match_1v1(desc2, flann_1v1, ratio_thresh=1.2, burst_thresh=None, DBG=False):
    '''
    Matches desc2 vs desc1 using Lowe's ratio test
    Input:
        desc2         - other descriptors (N2xD)
        flann_1v1     - prebuild FLANN index of desc1 (query descriptors (N1xD)) 
    Thresholds: 
        ratio_thresh = 1.2 - keep match if dist(2)/dist(1) > ratio_thresh
        burst_thresh = 1   - keep match if 0 < matching_freq(desc1) <= burst_thresh 
    Output: 
        fm - Mx2 array of matching feature indexes
        fs - Mx1 array of matching feature scores '''
    # features to their matching query features
    (fx2_qfx, fx2_dist) = flann_1v1.nn_index(desc2, 2, **__FLANN_PARAMS__)
    # RATIO TEST
    fx2_ratio  = np.divide(fx2_dist[:,1], fx2_dist[:,0]+1E-8)
    fx_passratio, = np.where(fx2_ratio > ratio_thresh)
    fx = fx_passratio
    # BURSTINESS TEST
    # Find frequency of descriptor matches
    # Select the query features which only matched < burst_thresh
    # Convert qfx to fx
    # FIXME: there is probably a better way of doing this.
    if not burst_thresh is None:
        qfx2_frequency = np.bincount(fx2_qfx[:,0])
        qfx_occuring   = qfx2_frequency > 0
        qfx_nonbursty  = qfx2_frequency <= burst_thresh
        qfx_nonbursty_unique,= np.where(np.bitwise_and(qfx_occuring, qfx_nonbursty))
        _qfx_set      = set(qfx_nonbursty_unique.tolist())
        fx2_nonbursty = [_qfx in _qfx_set for _qfx in iter(fx2_qfx[:,0])]
        fx_nonbursty, = np.where(fx2_nonbursty)
        fx  = np.intersect1d(fx, fx_nonbursty, assume_unique=True)
    # RETURN 1v1 matches and scores
    qfx = fx2_qfx[fx,0]
    fm  = np.array(zip(qfx, fx))
    fs  = fx2_ratio[fx]
    # DEBUG
    if DBG:
        print('-------------')
        print('Matching 1v1:')
        print(' * Ratio threshold: %r ' % ratio_thresh)
        print(' * Burst threshold: %r ' % burst_thresh)
        print(' * fx_passratio.shape   = %r ' % (  fx_passratio.shape,))
        if not burst_thresh is None:
            print(' * fx_nonbursty.shape   = %r ' % (  fx_nonbursty.shape,))
        print(' * fx.shape   = %r ' % (  fx.shape,))
        print(' * qfx.shape  = %r ' % (  qfx.shape,))
    return (fm, fs)

#========================================
# SPATIAL VERIFIACTION 
#========================================

#@profile
def __spatially_verify(func_homog, kpts1, kpts2, fm, fs, DBG=None):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers '''
    # ugg transpose, I like row first, but ransac seems not to
    if len(fm) == 0: 
        return (np.empty((0,2)),np.empty((0,1)), np.eye(3))
    kpts1_m = kpts1[fm[:,0],:].T
    kpts2_m = kpts2[fm[:,1],:].T
    # -----------------------------------------------
    # TODO: SHOULD THIS HAPPEN HERE? (ISSUE XY_THRESH)
    # Get match threshold 10% of matching keypoint extent diagonal
    img1_extent = (kpts1_m[0:2,:].max(1) - kpts1_m[0:2,:].min(1))[0:2]
    xy_thresh1_sqrd = np.sum(img1_extent**2) * (__XY_THRESH__**2)
    dodbg = False if DBG is None else __DEBUG__
    if dodbg:
        print('---------------------------------------')
        print('INFO: spatially_verify xy threshold:')
        print(' * Threshold is %.1f%% of diagonal length' % (__XY_THRESH__*100))
        print(' * img1_extent = %r '     % img1_extent)
        print(' * img1_diag_len = %.2f ' % np.sqrt(np.sum(img1_extent**2)))
        print(' * xy_thresh1_sqrd=%.2f'  % np.sqrt(xy_thresh1_sqrd))
        print('---------------------------------------')
    # -----------------------------------------------
    hinlier_tup = func_homog(kpts2_m, kpts1_m, xy_thresh1_sqrd) 
    if dodbg:
        print('  * ===')
        print('  * Found '+str(len(inliers))+' inliers')
        print('  * with transform H='+repr(H))
        print('  * fm.shape = %r fs.shape = %r' % (fm.shape, fs.shape))

    if not hinlier_tup is None:
        H, inliers = hinlier_tup
    else:
        H = np.eye(3)
        inliers = []
    if len(inliers) > 0:
        fm_V = fm[inliers,:]
        fs_V = fs[inliers,:]
    else: 
        fm_V = np.array([])
        fs_V = np.array([])
    return fm_V, fs_V, H

def spatially_verify(kpts1, kpts2, fm, fs, DBG=None):
    ''' Concrete implementation of spatial verification
        using the deterministic ellipse based sample conensus'''
    return __spatially_verify(H_homog_from_DELSAC, kpts1, kpts2, fm, fs, DBG)
spatially_verify.__doc__ += '\n'+__spatially_verify.__doc__

#@profile
def spatially_verify_matches(qcx, cx2_kpts, cx2_fm, cx2_fs):
    kpts1     = cx2_kpts[qcx]
    cx2_cscore = np.array([np.sum(fs) for fs in cx2_fs])
    top_cx     = cx2_cscore.argsort()[::-1]
    num_rerank = min(len(top_cx), __NUM_RERANK__)
    # -----------------------------------------------
    # TODO: SHOULD THIS HAPPEN HERE? (ISSUE XY_THRESH)
    #img1_extent = (kpts1[:,0:2].max(0) - qkpts1[:,0:2].min(0))[0:2]
    #xy_thresh1_sqrd = np.sum(img1_extent**2) * __xy_thresh_percent__
    # -----------------------------------------------
    # Precompute output container
    cx2_fm_V = [[] for _ in xrange(len(cx2_fm))]
    cx2_fs_V = [[] for _ in xrange(len(cx2_fs))]
    # spatially verify the top __NUM_RERANK__ results
    for topx in xrange(num_rerank):
        cx    = top_cx[topx]
        kpts2 = cx2_kpts[cx]
        fm    = cx2_fm[cx]
        fs    = cx2_fs[cx]

        fm_V, fs_V, H = spatially_verify(kpts1, kpts2, fm, fs)
        cx2_fm_V[cx] = fm_V
        cx2_fs_V[cx] = fs_V
    return cx2_fm_V, cx2_fs_V

def warp_chip(rchip2, H, rchip1):
    rchip2W = cv2.warpPerspective(rchip2, H, rchip1.shape[0:2][::-1])
    return rchip2W
#========================================
# Wrapper/Data classes
#========================================
class Vocabulary(DynStruct):
    def __init__(self, words, flann_words, cx2_bow, wx2_idf, wx2_cxs, wx2_fxs):
        super(Vocabulary, self).__init__()
        self.words       = words
        self.flann_words = flann_words
        self.cx2_bow     = cx2_bow
        self.wx2_idf     = wx2_idf
        self.wx2_cxs     = wx2_cxs
        self.wx2_fxs     = wx2_fxs

class OneVsMany(DynStruct): # TODO: rename this
    def __init__(self, flann_1vM, ax2_desc, ax2_cx, ax2_fx):
        super(OneVsMany, self).__init__()
        self.flann_1vM = flann_1vM
        self.ax2_desc  = ax2_desc # not used, but needs to maintain scope
        self.ax2_cx = ax2_cx
        self.ax2_fx = ax2_fx

class HotSpotter(DynStruct):
    def __init__(self):
        super(HotSpotter, self).__init__()
        self.tables = None
        self.feats  = None
        self.cpaths = None
        self.dirs   = None

import os
from helpers import ensurepath
__QueryResults_path__ = os.path.realpath('results/queryresults')
__AlgorithmUID__ = 'HESAFF-1vM'
ensurepath(__QueryResults_path__)
class QueryResult(DynStruct):
    def __init__(self, qcx):
        super(QueryResult, self).__init__()
        self.qcx    = qcx
        # Assigned features matches
        self.cx2_fm = []
        self.cx2_fs = []
        # Spatially verified feature matches
        self.cx2_fm_V = []
        self.cx2_fs_V = []

    def get_fpath(self):
        fname = 'result_'+__AlgorithmUID__+'_qcx=%d.npz' % self.qcx
        fpath = os.path.join(__QueryResults_path__, fname)
        return fpath
    
    def save(self):
        fpath = self.get_fpath()
        print('Saving results to '+repr(fpath))
        return self.save_result(fpath)

    def load(self):
        fpath = self.get_fpath()
        if checkpath(fpath):
            return self.load_result(fpath)
        return False

    def save_result(self, fpath):
        'Saves the result to the given database'
        to_save  = self.__dict__.copy()
        np.savez(fpath, **to_save)
        return True

    def load_result(self, fpath):
        'Loads the result from the given database'
        try:
            npz = np.load(fpath)
            for _key in npz.files:
                if _key in ['qcx']:
                    self.__dict__[_key] = npz[_key].tolist()
                else: 
                    self.__dict__[_key] = npz[_key]
            # Numpy saving is werid. gotta cast
            return True
        except Exception as ex:
            os.remove(fpath)
            printWARN('Load Result Exception : ' + str(ex) + 
                    '\nResult was corrupted for qcx=%d' % self.qcx)
            return False

class Matcher(object):
    '''Wrapper class: assigns matches based on matching and feature prefs'''
    def __init__(self, hs, match_type='1vM'):
        feat_type  = hs.feats.feat_type
        self.feat_type  = feat_type
        self.match_type = match_type
        # Possible indexing structures
        self.__one_vs_many    = None
        self.__vocabulary     = None
        # Curry the correct functions
        self.__assign_matches = None
        if   match_type == 'BOW':
            __one_vs_many = precompute_bag_of_words(hs)
            self.__one_vs_many    = __one_vs_many
            self.__assign_matches = assign_matches_BOW
            pass
        elif match_type == 'bBOW':
            pass
        elif match_type == 'b1vM':
            self.__assign_matches = assign_matches_1vM_BINARY
        elif match_type == 'b1v1':
            self.__assign_matches = assign_matches_1v1_BINARY
        elif match_type == '1vM':
            self.__one_vs_many = precompute_index_1vM(hs)
            self.__assign_matches = self.__assign_matches_1vM
        elif match_type == '1v1':
            self.__assign_matches = assign_matches_1v1
        else:
           raise Exception('Unknown match_type: '+repr(match_type))
    def assign_matches(self, qcx, cx2_cid, cx2_desc):
        'Function which calls the correct matcher'
        return self.__assign_matches(qcx, cx2_cid, cx2_desc)
    # class helpers
    def __assign_matches_1vM(self, qcx, cx2_cid, cx2_desc):
        return assign_matches_1vM(qcx, cx2_cid, cx2_desc, self.__one_vs_many)

#========================================
# Work Functions
#========================================
__LAZY_MATCHING__ = True
def run_matching(hs, matcher):
    '''Runs the full matching pipeline using the abstracted classes'''
    cx2_cid  = hs.tables.cx2_cid
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts 
    # qcx2_res
    cx2_res = [QueryResult(qcx) for qcx in xrange(len(cx2_cid))]
    tt_ALL = tic('all queries')
    assign_times = []
    verify_times = []
    skip_list = []
    for qcx, qcid in enumerate(cx2_cid):
        if qcid == 0: 
            skip_list.append(qcx)
            continue
        res = cx2_res[qcx]
        if __LAZY_MATCHING__ and res.load():
            continue
        tt_A = tic('query(qcx=%d)' % qcx)
        # Assign matches with the chosen function (1v1) or (1vM)
        (cx2_fm, cx2_fs) = matcher.assign_matches(qcx, cx2_cid, cx2_desc)
        assign_times.append(toc(tt_A))
        # Spatially verify the assigned matches
        tt_V = tic('verify(qcx=%d)' % qcx)
        (cx2_fm_V, cx2_fs_V) = spatially_verify_matches(qcx, cx2_kpts, cx2_fm, cx2_fs)
        verify_times.append(toc(tt_V))
        # Assign output to a query result
        res.qcx = qcx
        res.cx2_fm    = cx2_fm
        res.cx2_fs    = cx2_fs
        res.cx2_fm_V = cx2_fm_V
        res.cx2_fs_V = cx2_fs_V
        res.save()
    if len(skip_list) > 0:
        print('Skipped more queries than you should have: %r ' % skip_list)
    total_time = toc(tt_ALL)
    # Write results out to disk
    report_results2.write_rank_results(cx2_res, hs, matcher)
    return cx2_res

def runall_match(hs):
    #functools.partial
    hs.printme2()
    cx2_res_1vM = run_matching(hs, Matcher(hs, '1vM'))
    #cx2_res_1v1 = run_matching(hs, Matcher(hs, '1v1'))


#========================================
# DRIVER CODE
#========================================

# TODO, this should go in a more abstracted module
def load_hotspotter(db_dir):
    'Loads a hs class'
    # --- LOAD DATA --- #
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
    # --- LOAD CHIPS --- #
    hs_cpaths = chip_compute2.load_chip_paths(hs_dirs, hs_tables)
    # --- LOAD FEATURES --- #
    hs_feats  = feature_compute2.load_chip_features(hs_dirs, hs_tables, hs_cpaths)
    hs_feats.set_feat_type(__FEAT_TYPE__)
    # --- BUILD HOTSPOTTER --- #
    hs = HotSpotter()
    hs.tables = hs_tables
    hs.feats  = hs_feats
    hs.cpaths = hs_cpaths
    hs.dirs   = hs_dirs
    return hs

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # --- CHOOSE DATABASE --- #
    db_dir = load_data2.MOTHERS
    hs = load_hotspotter(db_dir)
    cx2_kpts = hs.feats.cx2_kpts
    cx2_desc = hs.feats.cx2_desc
    cx2_cid = hs.tables.cx2_cid
    qcx = 1

    __TEST_MODE__ = False or ('test' in sys.argv)
    if __TEST_MODE__:
        runall_match(hs)
        pass

    ## DEV ONLY CODE ##
    __DEV_MODE__ = False
    if __DEV_MODE__: 
        print(hs_cpaths)
        print(hs_dirs)
        print(hs_tables)
        print(hs_feats)
        # Convinent but bad # 
        #exec(hs_cpaths.execstr('hs_cpaths'))
        #exec(hs_feats.execstr('hs_feats'))
        #exec(hs_tables.execstr('hs_tables'))
        #exec(hs_dirs.execstr('hs_dirs'))
        #cx  = 1
        # All of these functions operate on one qcx (except precompute I guess)
        #exec(get_exec_src(precompute_index_1vM))
        #exec(get_exec_src(assign_matches_1vM))
        #exec(get_exec_src(spatially_verify_matches))
        #exec(get_exec_src(precompute_bag_of_words))
        

        try: 
            __IPYTHON__
        except: 
            plt.show()
