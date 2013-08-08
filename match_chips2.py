'''
Module match_chips: 
    Runs vsone, vsmany, and bagofwords matching
'''
#========================================
# IMPORTS
#========================================
# Standard library imports
import itertools
import sys
import os
# Hotspotter Frontend Imports
import drawing_functions2 as df2
# Hotspotter Imports
import helpers
from helpers import Timer, tic, toc, printWARN
from Printable import DynStruct
import algos
import helpers
import spatial_verification
import load_data2
import params
import chip_compute2 as cc2
import feature_compute2 as fc2
import report_results2
# Math and Science Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyflann
import scipy as sp
import scipy.sparse as spsparse
import sklearn.preprocessing 
print ('LOAD_MODULE: match_chips2.py')

#========================================
# Bag-of-Words
#========================================
class BagOfWords(DynStruct):
    def __init__(self, words, words_flann, cx2_vvec, wx2_idf, wx2_cxs, wx2_fxs):
        super(BagOfWords, self).__init__()
        self.words       = words
        self.words_flann = words_flann
        self.cx2_vvec    = cx2_vvec
        self.wx2_idf     = wx2_idf
        self.wx2_cxs     = wx2_cxs
        self.wx2_fxs     = wx2_fxs

# precompute the bag of words model
def precompute_bag_of_words(hs):
    '''Builds a vocabulary with train_sample_cx
    Creates an indexed database with database_sample_cx'''
    print('Precomputing bag of words')
    # Read params
    cache_dir  = hs.dirs.cache_dir
    cx2_desc   = hs.feats.cx2_desc
    vocab_size = params.__NUM_WORDS__
    cx2_desc   = hs.feats.cx2_desc
    train_sample_cx    = range(len(cx2_desc)) if hs.train_sample_cx is None \
                               else hs.train_sample_cx
    database_sample_cx = range(len(cx2_desc)) if hs.database_sample_cx is None \
                               else hs.database_sample_cx
    # Compute vocabulary
    words, words_flann = __compute_vocabulary\
            (cx2_desc, train_sample_cx, vocab_size, cache_dir)
    # Assign visual vectors to the database
    cx2_vvec, wx2_cxs, wx2_fxs, wx2_idf = __index_database_to_vocabulary\
            (cx2_desc, words, words_flann, database_sample_cx)
    bagofwords = BagOfWords(words, words_flann, cx2_vvec, wx2_idf, wx2_cxs, wx2_fxs)
    return bagofwords

# step 1  
def __compute_vocabulary(cx2_desc, train_sample_cx, vocab_size, cache_dir=None):
    '''Computes a vocabulary of size vocab_size given a set of training data'''
    # Make a training set of descriptors to build the vocabulary
    tx2_desc   = cx2_desc[train_sample_cx]
    train_desc = np.vstack(tx2_desc)
    num_train_desc = train_desc.shape[0]
    if vocab_size > num_train_desc:
        helpers.printWARN('Vocab size: %r is less than #train descriptors: %r' %\
                  (vocab_size, num_train_desc))
        vocab_size = num_train_desc / 2
    # Build the vocabualry
    # Cluster descriptors into a visual vocabulary
    _, words = algos.precompute_akmeans(train_desc, vocab_size,
                                        force_recomp=False,
                                        cache_dir=cache_dir)
    # Index the vocabulary for fast nearest neighbor search
    # TODO: Cache the nearest neighbor index
    #algorithm = 'default'
    #algorithm = 'linear'
    #print(' * bag of words is using '+algorithm+' NN search')
    #words_flann = pyflann.FLANN()
    #words_flann_params = words_flann.build_index(words, algorithm=algorithm)
    #print(' * finished building index')
    words_flann = algos.precompute_flann(words, cache_dir, lbl='words')
    return words, words_flann

# step 2
def __index_database_to_vocabulary(cx2_desc, words, words_flann,database_sample_cx):
    '''Assigns each database chip a visual-vector and returns 
       data for the inverted file'''
    # TODO: Save precomputations here
    print('Assigning each database chip a bag-of-words vector')
    sample_cx   = database_sample_cx
    num_database = len(database_sample_cx)
    ax2_cx, ax2_fx, ax2_desc = __aggregate_descriptors(cx2_desc, database_sample_cx)
    # Assign each descriptor to its nearest visual word
    #ax2_desc    = np.array(ax2_desc, dtype=params.__BOW_DTYPE__)
    ax2_wx, _ = words_flann.nn_index(ax2_desc, 1, checks=128)
    # Build inverse word to ax
    wx2_axs = [[] for _ in xrange(len(words))]
    for ax, wx in enumerate(ax2_wx):
        wx2_axs[wx].append(ax)
    # Compute inverted file: words -> database
    wx2_cxs = np.array([[ax2_cx[ax] for ax in ax_list] for ax_list in wx2_axs])
    wx2_fxs = np.array([[ax2_fx[ax] for ax in ax_list] for ax_list in wx2_axs])
    # Build sparse visual vectors with term frequency weights 
    coo_cols = ax2_wx  
    coo_rows = ax2_cx
    coo_values = np.ones(len(ax2_cx), dtype=np.uint8)
    coo_format = (coo_values, (coo_rows, coo_cols))
    coo_cx2_vvec = spsparse.coo_matrix(coo_format, dtype=np.float, copy=True)
    cx2_tf_vvec  = spsparse.csr_matrix(coo_cx2_vvec, copy=False)
    # Compute idf_w = log(Number of documents / Number of docs containing word_j)
    print('Computing tf-idf')
    wx2_df  = np.array([len(set(cxs)) for cxs in wx2_cxs], dtype=np.float)
    wx2_idf = np.array(np.log2(np.float(num_database) / wx2_df))
    # Compute tf-idf
    cx2_tfidf_vvec = algos.sparse_multiply_rows(cx2_tf_vvec, wx2_idf)
    # Normalize
    cx2_vvec = algos.sparse_normalize_rows(cx2_tfidf_vvec)
    return cx2_vvec, wx2_cxs, wx2_fxs, wx2_idf

def __quantize_desc_to_tfidf_vvec(desc, wx2_idf, words, words_flann):
    # Assign each descriptor to its nearest visual word
    #desc = np.array(desc_, params.__BOW_DTYPE__)
    fx2_wx, _ = words_flann.nn_index(desc, 1, checks=128)
    #TODO: soft assignment here
    # Build sparse visual vectors with term frequency weights 
    lil_vvec = spsparse.lil_matrix((len(words),1))
    for wx in iter(fx2_wx):
        lil_vvec[wx, 0] += 1
    tf_vvec = spsparse.csr_matrix(lil_vvec.T, copy=False)
    # Compute tf-idf
    tfidf_vvec = algos.sparse_multiply_rows(tf_vvec, wx2_idf)
    # Normalize
    vvec = algos.sparse_normalize_rows(tfidf_vvec)
    return vvec, fx2_wx

# Used by Matcher class to assign matches to a bag-of-words database
def assign_matches_bagofwords(qcx, cx2_cid, cx2_desc, bagofwords):
    cx2_vvec    = bagofwords.cx2_vvec
    wx2_cxs     = bagofwords.wx2_cxs
    wx2_fxs     = bagofwords.wx2_fxs
    wx2_idf     = bagofwords.wx2_idf
    words       = bagofwords.words
    words_flann = bagofwords.words_flann
    # Assign the query descriptors a visual vector
    vvec, qfx2_wx = __quantize_desc_to_tfidf_vvec(cx2_desc[qcx], wx2_idf, words, words_flann)
    # Compute distance to every database vector
    cx2_score = (cx2_vvec.dot(vvec.T)).toarray().flatten()
    # Assign feature to feature matches (for spatial verification)
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for qfx, wx in enumerate(qfx2_wx):
        cx_list = wx2_cxs[wx]
        fx_list = wx2_fxs[wx]
        fs = wx2_idf[wx]
        for (cx, fx) in zip(cx_list, fx_list): 
            if cx == qcx: continue
            fm = (qfx, fx)
            cx2_fm[cx].append(fm)
            cx2_fs[cx].append(fs)
    # Convert to numpy
    for cx in xrange(len(cx2_cid)): cx2_fm[cx] = np.array(cx2_fm[cx])
    for cx in xrange(len(cx2_cid)): cx2_fs[cx] = np.array(cx2_fs[cx])
    return cx2_fm, cx2_fs, cx2_score

#========================================
# One-vs-Many 
#========================================
class OneVsMany(DynStruct): # TODO: rename this
    '''Contains a one-vs-many index and the 
       inverted information needed for voting'''
    def __init__(self, vsmany_flann, ax2_desc, ax2_cx, ax2_fx):
        super(OneVsMany, self).__init__()
        self.vsmany_flann = vsmany_flann
        self.ax2_desc  = ax2_desc # not used, but needs to maintain scope
        self.ax2_cx = ax2_cx
        self.ax2_fx = ax2_fx

def __aggregate_descriptors(cx2_desc, sample_cx):
    '''Aggregates a sample set of descriptors. 
    Returns descriptors, chipxs, and featxs indexed by ax'''
    # sample the descriptors you wish to aggregate
    sx2_cx   = sample_cx
    sx2_desc = cx2_desc[sx2_cx]
    sx2_numfeat = [len(k) for k in iter(cx2_desc[sx2_cx])]
    cx_numfeat_iter = iter(zip(sx2_cx, sx2_numfeat))
    # create indexes from agg desc back to chipx and featx
    _ax2_cx = [[cx]*num_feats for (cx, num_feats) in cx_numfeat_iter]
    _ax2_fx = [range(num_feats) for num_feats in iter(sx2_numfeat)]
    ax2_cx  = np.array(list(itertools.chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(itertools.chain.from_iterable(_ax2_fx)))
    ax2_desc = np.vstack(cx2_desc[sample_cx])
    return ax2_cx, ax2_fx, ax2_desc

def aggregate_descriptors_vsmany(hs):
    '''aggregates all descriptors for vsmany search'''
    print('Aggregating descriptors for one-vs-many')
    cx2_desc  = hs.feats.cx2_desc
    sample_cx = np.arange(len(cx2_desc))
    return __aggregate_descriptors(cx2_desc, sample_cx)

#@profile
def precompute_index_vsmany(hs):
    # Build (or reload) one vs many flann index
    feat_dir  = hs.dirs.feat_dir
    feat_type = hs.feats.feat_type
    ax2_cx, ax2_fx, ax2_desc = aggregate_descriptors_vsmany(hs)
    vsmany_flann = pyflann.FLANN()
    feat_uid = hs.feat_uid()
    vsmany_flann_path = feat_dir + '/flann_One-vs-Many_'+feat_uid+'.index'
    load_success = False
    if helpers.checkpath(vsmany_flann_path):
        try:
            print('Trying to load FLANN index')
            vsmany_flann.load_index(vsmany_flann_path, ax2_desc)
            print('...success')
            load_success = True
        except Exception as ex:
            print('...cannot load FLANN index'+repr(ex))
    if not load_success:
        with Timer(msg='rebuilding FLANN index'):
            vsmany_flann.build_index(ax2_desc, **params.__FLANN_PARAMS__)
            vsmany_flann.save_index(vsmany_flann_path)
    # Return a one-vs-many structure
    one_vs_many = OneVsMany(vsmany_flann, ax2_desc, ax2_cx, ax2_fx)
    return one_vs_many

# Feature scoring functions
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist, vdist+1E-8)+1) 
def RATIO_fn(vdist, ndist): return np.divide(ndist, vdist+1E-8)
def LNBNN_fn(vdist, ndist): return ndist - vdist 
score_fn = RATIO_fn

#@profile
def assign_matches_vsmany(qcx, cx2_cid, cx2_desc, one_vs_many):
    '''Matches desc1 vs all database descriptors using 
    Input:
        qcx        - query chip index
        cx2_cid     - chip ID lookup table (for removing self matches)
        cx2_desc    - chip descriptor lookup table
        one_vs_many - class with FLANN index of database descriptors
    Output: 
        cx2_fm - C x Mx2 array of matching feature indexes
        cx2_fs - C x Mx1 array of matching feature scores'''
    helpers.println('Assigning vsmany feature matches from qcx=%d to %d chips'\
                    % (qcx, len(cx2_cid)))
    vsmany_flann = one_vs_many.vsmany_flann
    ax2_cx    = one_vs_many.ax2_cx
    ax2_fx    = one_vs_many.ax2_fx
    isQueryIndexed = True
    desc1 = cx2_desc[qcx]
    k_vsmany = params.__K__+1 if isQueryIndexed else params.__K__
    # Find each query descriptor's k+1 nearest neighbors
    checks = params.__FLANN_PARAMS__['checks']
    (qfx2_ax, qfx2_dists) = vsmany_flann.nn_index(desc1, k_vsmany+1, checks=checks)
    vote_dists = qfx2_dists[:, 0:k_vsmany]
    norm_dists = qfx2_dists[:, k_vsmany] # k+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([score_fn(_vdist.T, norm_dists)
                           for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    qfx2_cx = ax2_cx[qfx2_ax[:, 0:k_vsmany]]
    qfx2_fx = ax2_fx[qfx2_ax[:, 0:k_vsmany]]
    # Build feature matches
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    num_qf = len(desc1)
    qfx2_qfx = np.tile(np.arange(num_qf).reshape(num_qf, 1), (1, k_vsmany)) 
    iter_matches = iter(zip(qfx2_qfx.flat, qfx2_cx.flat,
                            qfx2_fx.flat, qfx2_score.flat))
    for qfx, cx, fx, score in iter_matches:
        if qcx == cx: 
            continue # dont vote for yourself
        cx2_fm[cx].append((qfx, fx))
        cx2_fs[cx].append(score)
    # Convert to numpy
    for cx in xrange(len(cx2_cid)): 
        cx2_fm[cx] = np.array(cx2_fm[cx])
    for cx in xrange(len(cx2_cid)): 
        cx2_fs[cx] = np.array(cx2_fs[cx])
    cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
    return cx2_fm, cx2_fs, cx2_score

def assign_matches_vsmany_BINARY(qcx, cx2_cid, cx2_desc):
    return None

#========================================
# One-vs-One 
#========================================
def assign_matches_vsone(qcx, cx2_cid, cx2_desc):
    print('Assigning vsone feature matches from cx=%d to %d chips'\
          % (qcx, len(cx2_cid)))
    desc1 = cx2_desc[qcx]
    vsone_flann = pyflann.FLANN()
    vsone_flann.build_index(desc1, **params.__FLANN_PARAMS__)
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, desc2 in enumerate(cx2_desc):
        sys.stdout.write('.')
        sys.stdout.flush()
        if cx == qcx: continue
        (fm, fs) = match_vsone(desc2, vsone_flann)
        cx2_fm[cx] = fm
        cx2_fs[cx] = fs
    sys.stdout.write('DONE')
    vsone_flann.delete_index()
    cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
    return cx2_fm, cx2_fs, cx2_score

def cv2_match(desc1, desc2):
    K = 1
    cv2_matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    raw_matches = cv2_matcher.knnMatch(desc1, desc2, K)
    matches = [(m1.trainIdx, m1.queryIdx) for m1 in raw_matches]

#@profile
def match_vsone(desc2, vsone_flann, ratio_thresh=1.2, burst_thresh=None):
    '''Matches desc2 vs desc1 using Lowe's ratio test
    Input:
        desc2         - other descriptors (N2xD)
        vsone_flann     - FLANN index of desc1 (query descriptors (N1xD)) 
    Thresholds: 
        ratio_thresh = 1.2 - keep if dist(2)/dist(1) > ratio_thresh
        burst_thresh = 1   - keep if 0 < matching_freq(desc1) <= burst_thresh
    Output: 
        fm - Mx2 array of matching feature indexes
        fs - Mx1 array of matching feature scores '''
    # features to their matching query features
    checks = params.__FLANN_PARAMS__['checks']
    (fx2_qfx, fx2_dist) = vsone_flann.nn_index(desc2, 2, checks=checks)
    # RATIO TEST
    fx2_ratio  = np.divide(fx2_dist[:, 1], fx2_dist[:, 0]+1E-8)
    fx_passratio, = np.where(fx2_ratio > ratio_thresh)
    fx = fx_passratio
    # BURSTINESS TEST
    # Find frequency of descriptor matches. Convert qfx to fx
    # Select the query features which only matched < burst_thresh
    if not burst_thresh is None:
        qfx2_frequency = np.bincount(fx2_qfx[:, 0])
        qfx_occuring   = qfx2_frequency > 0
        qfx_nonbursty  = qfx2_frequency <= burst_thresh
        qfx_nonbursty_unique, = np.where(
            np.bitwise_and(qfx_occuring, qfx_nonbursty))
        _qfx_set      = set(qfx_nonbursty_unique.tolist())
        fx2_nonbursty = [_qfx in _qfx_set for _qfx in iter(fx2_qfx[:, 0])]
        fx_nonbursty, = np.where(fx2_nonbursty)
        fx  = np.intersect1d(fx, fx_nonbursty, assume_unique=True)
    # RETURN vsone matches and scores
    qfx = fx2_qfx[fx, 0]
    fm  = np.array(zip(qfx, fx))
    fs  = fx2_ratio[fx]
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
        return (np.empty((0, 2)), np.empty((0, 1)), np.eye(3))
    xy_thresh = params.__XY_THRESH__
    kpts1_m = kpts1[fm[:, 0], :].T
    kpts2_m = kpts2[fm[:, 1], :].T
    # -----------------------------------------------
    # Get match threshold 10% of matching keypoint extent diagonal
    img1_extent = (kpts1_m[0:2, :].max(1) - kpts1_m[0:2, :].min(1))[0:2]
    xy_thresh1_sqrd = np.sum(img1_extent**2) * (xy_thresh**2)
    # -----------------------------------------------
    hinlier_tup = func_homog(kpts2_m, kpts1_m, xy_thresh1_sqrd) 
    if not hinlier_tup is None:
        H, inliers = hinlier_tup
    else:
        H = np.eye(3)
        inliers = []
    if len(inliers) > 0:
        fm_V = fm[inliers, :]
        fs_V = fs[inliers, :]
    else: 
        fm_V = np.empty((0, 2))
        fs_V = np.array((0, 1))
    return fm_V, fs_V, H

def spatially_verify(kpts1, kpts2, fm, fs, DBG=None):
    ''' Concrete implementation of spatial verification
        using the deterministic ellipse based sample conensus'''
    ransac_func = spatial_verification.H_homog_from_DELSAC
    return __spatially_verify(ransac_func, kpts1, kpts2, fm, fs, DBG)
spatially_verify.__doc__ += '\n'+__spatially_verify.__doc__

#@profile
def spatially_verify_matches(qcx, cx2_kpts, cx2_fm, cx2_fs):
    kpts1     = cx2_kpts[qcx]
    cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
    top_cx     = cx2_score.argsort()[::-1]
    num_rerank = min(len(top_cx), params.__NUM_RERANK__)
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
    cx2_score_V = np.array([np.sum(fs) for fs in cx2_fs_V])
    return cx2_fm_V, cx2_fs_V, cx2_score_V

#=========================
# Query Result Class
#=========================
class QueryResult(DynStruct):
    def __init__(self, qcx):
        super(QueryResult, self).__init__()
        self.qcx    = qcx
        # Assigned features matches
        self.cx2_fm = np.array([])
        self.cx2_fs = np.array([])
        self.cx2_score = np.array([])
        # Spatially verified feature matches
        self.cx2_fm_V = np.array([])
        self.cx2_fs_V = np.array([])
        self.cx2_score_V = np.array([])


    def get_fpath(self, hs):
        query_uid = hs.query_uid()
        qres_dir = hs.dirs.qres_dir 
        fname = 'result_'+query_uid+'_qcx=%d.npz' % self.qcx
        fpath = os.path.join(qres_dir, fname)
        return fpath
    
    def save(self, hs):
        fpath = self.get_fpath(hs)
        if params.__VERBOSE_CACHE__:
            print('caching result: '+repr(fpath))
        else:
            print('caching result: '+repr(os.path.split(fpath)[1]))
        return self.save_result(fpath)

    def load(self, hs):
        fpath = self.get_fpath(hs)
        if helpers.checkpath(fpath):
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
                if _key in ['qcx']: # hack
                    self.__dict__[_key] = npz[_key].tolist()
                else: 
                    self.__dict__[_key] = npz[_key]
            # Numpy saving is werid. gotta cast
            return True
        except Exception as ex:
            os.remove(fpath)
            printWARN('Load Result Exception : ' + repr(ex) + 
                    '\nResult was corrupted for qcx=%d' % self.qcx)
            return False

#=========================
# Matcher Class
#=========================
class Matcher(DynStruct):
    '''Wrapper class: assigns matches based on
       matching and feature prefs'''
    def __init__(self, hs, match_type):
        super(Matcher, self).__init__()
        print('Creating matcher: '+str(match_type))
        self.feat_type  = hs.feats.feat_type
        self.match_type = match_type
        # Possible indexing structures
        self.__one_vs_many    = None
        self.__bag_of_words     = None
        # Curry the correct functions
        self.__assign_matches = None
        if   match_type == 'bagofwords':
            print(' precomputing bag of words')
            self.__bag_of_words   = precompute_bag_of_words(hs)
            self.__assign_matches = self.__assign_matches_bagofwords
        elif match_type == 'vsmany':
            print(' precomputing one vs many')
            self.__one_vs_many = precompute_index_vsmany(hs)
            self.__assign_matches = self.__assign_matches_vsmany
        elif match_type == 'vsone':
            self.__assign_matches = assign_matches_vsone
        else:
            raise Exception('Unknown match_type: '+repr(match_type))
    def assign_matches(self, qcx, cx2_cid, cx2_desc):
        'Function which calls the correct matcher'
        return self.__assign_matches(qcx, cx2_cid, cx2_desc)
    # query helpers
    def __assign_matches_vsmany(self, qcx, cx2_cid, cx2_desc):
        return assign_matches_vsmany(qcx, cx2_cid, cx2_desc, self.__one_vs_many)
    def __assign_matches_bagofwords(self, qcx, cx2_cid, cx2_desc):
        return assign_matches_bagofwords(qcx, cx2_cid, cx2_desc, self.__bag_of_words)

#========================================
# Work Functions
#========================================
def run_matching(hs):
    '''Runs the full matching pipeline using the abstracted classes'''
    matcher  = hs.matcher
    cx2_cid  = hs.tables.cx2_cid
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts 
    qcx2_res = [QueryResult(qcx) for qcx in xrange(len(cx2_cid))]
    test_sample_cx = range(len(cx2_cid)) if hs.test_sample_cx is None else hs.test_sample_cx
    tt_ALL = tic('all queries')
    assign_times = [] # metadata
    verify_times = []
    skip_list    = []
    print('Running matching on: %r' % test_sample_cx)
    for qcx in iter(test_sample_cx):
        qcid = cx2_cid[qcx]
        if qcid == 0: 
            skip_list.append(qcx)
            continue
        helpers.print_ ('query(qcx=%4d)->' % qcx)
        res = qcx2_res[qcx]
        # load query from cache if possible
        cache_load_success = params.__CACHE_QUERY__ and res.load(hs)
        if qcx in params.__FORCE_REQUERY_CX__:
            cache_load_success = False
        # Get what data we have if we are redoing things
        if cache_load_success or\
           params.__RESAVE_QUERY__ or params.__REVERIFY_QUERY__:
            helpers.print_('load_cache->')
            cx2_fm      = res.cx2_fm
            cx2_fs      = res.cx2_fs
            cx2_score   = res.cx2_score
            cx2_fm_V    = res.cx2_fm_V
            cx2_fs_V    = res.cx2_fs_V
            cx2_score_V = res.cx2_score_V
        # Assign matches with the chosen function (vsone) or (vsmany)
        if not cache_load_success:
            tt_A = tic('assign')
            (cx2_fm, cx2_fs, cx2_score) = \
                    matcher.assign_matches(qcx, cx2_cid, cx2_desc)
            assign_times.append(toc(tt_A))
        else: 
            helpers.print_('cache_assign->')
        # Spatially verify the assigned matches
        if not cache_load_success or params.__REVERIFY_QUERY__:
            tt_V = tic('verify')
            (cx2_fm_V, cx2_fs_V, cx2_score_V) = \
                    spatially_verify_matches(qcx, cx2_kpts, cx2_fm, cx2_fs)
            verify_times.append(toc(tt_V))
        else: 
            helpers.print_('cache_verify->')
        # Assign output to the query result 
        res.qcx = qcx
        res.cx2_fm    = cx2_fm
        res.cx2_fs    = cx2_fs
        res.cx2_score = cx2_score
        res.cx2_fm_V = cx2_fm_V
        res.cx2_fs_V = cx2_fs_V
        res.cx2_score_V = cx2_score_V
        # Cache query result
        if not cache_load_success or\
           params.__REVERIFY_QUERY__ or params.__RESAVE_QUERY__:
            helpers.print_('')
            #tt_save = tic('caching query')
            res.save(hs)
            #toc(tt_save)
        helpers.println('endquery;')       
    if len(skip_list) > 0:
        print('Skipped more queries than you should have: %r ' % skip_list)
    #total_time = toc(tt_ALL)
    # Write results out to disk
    report_results2.write_rank_results(hs, qcx2_res)
    return qcx2_res

def run_matching_type(hs, match_type=params.__MATCH_TYPE__):
    matcher = hs.use_matcher(match_type)
    qcx2_res = run_matching(hs)
    return qcx2_res

def runall_match(hs):
    #functools.partial
    hs.printme2()
    match_types = ['vsmany', 'vsone', 'bagofwords']
    qcx2_res_bagofwords  = run_matching_type(hs, 'bagofwords')

#========================================
# DRIVER CODE
#========================================
class HotSpotter(DynStruct):
    '''The HotSpotter main class is a root handle to all relevant data'''
    def __init__(hs, db_dir=None, load_matcher=True):
        super(HotSpotter, hs).__init__()
        hs.tables = None
        hs.feats  = None
        hs.cpaths = None
        hs.dirs   = None
        hs.matcher = None
        hs.train_sample_cx    = None
        hs.test_sample_cx     = None
        hs.database_sample_cx = None
        if not db_dir is None:
            hs.load_database(db_dir, load_matcher)
    def load_database(hs, db_dir, load_matcher=True):
        # Load data
        hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
        hs_cpaths = cc2.load_chip_paths(hs_dirs, hs_tables)
        hs_feats  = fc2.load_chip_features(hs_dirs, hs_tables, hs_cpaths)
        # Build hotspotter structure
        hs.tables  = hs_tables
        hs.feats   = hs_feats
        hs.cpaths  = hs_cpaths
        hs.dirs    = hs_dirs
        hs.load_test_train_database()
        if load_matcher: 
            hs.use_matcher(params.__MATCH_TYPE__)

    def load_test_train_database(hs):
        'tries to load test / train / database sample from internal dir'
        database_sample_fname = hs.dirs.internal_dir+'/database_sample.txt'
        test_sample_fname     = hs.dirs.internal_dir+'/test_sample.txt'
        train_sample_fname    = hs.dirs.internal_dir+'/train_sample.txt'
        hs.database_sample_cx = helpers.eval_from(database_sample_fname, False)
        hs.test_sample_cx     = helpers.eval_from(test_sample_fname, False)
        hs.train_sample_cx    = helpers.eval_from(database_sample_fname, False)
        if hs.database_sample_cx is None and hs.test_sample_cx is None and hs.train_sample_cx is None: 
            hs.database_sample_cx = range(len(hs.feats.cx2_desc))
            hs.test_sample_cx = range(len(hs.feats.cx2_desc))
            hs.train_sample_cx = range(len(hs.feats.cx2_desc))


        db_sample_cx = range(len(cx2_desc)) if hs.database_sample_cx is None \
                               else hs.database_sample_cx

    # TODO: This UID code is repeated in feature_compute2. needs to be better
    # integrated
    def algo_uid(hs):
        return hs.query_uid()
    def query_uid(hs):
        feat_type = hs.feats.feat_type
        match_type = hs.matcher.match_type
        uid_depends = [feat_type,
                       match_type,
                       ['', 'white'][params.__WHITEN_FEATS__]]
        query_uid = '_'.join(uid_depends)
        return query_uid
    def feat_uid(hs):
        feat_type = params.__FEAT_TYPE__
        uid_depends = [feat_type,
                    ['', 'white'][params.__WHITEN_FEATS__]]
        feat_uid = '_'.join(uid_depends)
        return feat_uid
    def chip_uid(hs):
        uid_depends = [['', 'histeq'][params.__HISTEQ__]]
        chip_uid = '_'.join(uid_depends)
        return chip_uid
    def use_matcher(hs, match_type):
        hs.matcher = Matcher(hs, match_type)
        return hs.matcher

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('IN: match_chips2.py: __name__ == '+__name__)

    # --- CHOOSE DATABASE --- #
    db_dir = load_data2.DEFAULT
    hs = HotSpotter(db_dir)
    cx2_kpts = hs.feats.cx2_kpts
    cx2_desc = hs.feats.cx2_desc
    cx2_cid  = hs.tables.cx2_cid
    qcx = 1

    __TEST_MODE__ = False or ('--test' in sys.argv)
    if __TEST_MODE__:
        runall_match(hs)
        pass

    #if 'bow' in sys.argv:
        #exec(helpers.get_exec_src(precompute_bag_of_words)) 

    ## DEV ONLY CODE ##
    __DEV_MODE__ = False
    if __DEV_MODE__: 
        print('DEVMODE IS ON: match_chips2')
        # Convinent but bad # 
        #exec(hs_cpaths.execstr('hs_cpaths'))
        #exec(hs_feats.execstr('hs_feats'))
        #exec(hs_tables.execstr('hs_tables'))
        #exec(hs_dirs.execstr('hs_dirs'))
        #cx  = 1
        # All of these functions operate on one qcx (except precompute I guess)
        #exec(helpers.get_exec_src(precompute_index_vsmany))
        #exec(helpers.get_exec_src(assign_matches_vsmany))
        #exec(helpers.get_exec_src(spatially_verify_matches))
        #exec(helpers.get_exec_src(precompute_bag_of_words))

        debug_compute_bagofwords = False
        if debug_compute_bagofwords:
            naut_train_sample_cx = [1, 3, 5]
            naut_database_sample_cx = [1, 3, 5]
            naut_test_query    = [0, 2, 4]
            train_sample_cx    = naut_train_sample_cx
            database_sample_cx = naut_database_sample_cx
            cache_dir  = hs.dirs.cache_dir
            cx2_desc   = hs.feats.cx2_desc
            vocab_size = params.__NUM_WORDS__
            cx2_desc   = hs.feats.cx2_desc
            #exec(helpers.get_exec_src(precompute_bag_of_words))
        try: 
            __IPYTHON__
        except: 
            plt.show()

    exec(df2.present())
