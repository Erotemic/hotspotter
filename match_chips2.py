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
#print ('LOAD_MODULE: match_chips2.py')

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

BOW_DTYPE = np.uint8
FM_DTYPE  = np.uint32
FS_DTYPE  = np.float32

def fix_res_types(res):
    for cx in xrange(len(res.cx2_fm_V)):
        res.cx2_fm_V[cx] = np.array(res.cx2_fm_V[cx], dtype=FM_DTYPE)
    for cx in xrange(len(res.cx2_fs_V)):
        res.cx2_fs_V[cx] = np.array(res.cx2_fs_V[cx], dtype=FS_DTYPE)

def fix_qcx2_res_types(qcx2_res):
    '''
    Changes data types of cx2_fm_V and cx2_fs_V
    '''
    total_qcx = len(qcx2_res)
    fmt_str = helpers.make_progress_fmt_str(total_qcx)
    for qcx in xrange(total_qcx):
        helpers.print_(fmt_str % (qcx))
        res = qcx2_res[qcx]
        fix_res_types(res)

#=========================
# Query Result Class
#=========================
class QueryResult(DynStruct):
    def __init__(self, qcx):
        super(QueryResult, self).__init__()
        self.qcx    = qcx
        # Assigned features matches
        self.cx2_fm = np.array([], dtype=FM_DTYPE)
        self.cx2_fs = np.array([], dtype=FS_DTYPE)
        self.cx2_score = np.array([])
        # Spatially verified feature matches
        self.cx2_fm_V = np.array([], dtype=FM_DTYPE)
        self.cx2_fs_V = np.array([], dtype=FS_DTYPE)
        self.cx2_score_V = np.array([])

    def get_fpath(self, hs):
        query_uid = params.get_query_uid()
        qres_dir = hs.dirs.qres_dir 
        fname = 'result_'+query_uid+'_qcx=%d.npz' % self.qcx
        fpath = os.path.join(qres_dir, fname)
        return fpath
    
    def save(self, hs):
        # HACK
        self.cx2_fm = np.array([])
        self.cx2_fs = np.array([])
        self.cx2_score = np.array([])
        # Forget non spatial scores
        fpath = self.get_fpath(hs)
        if params.VERBOSE_CACHE:
            print('[mc2] caching result: '+repr(fpath))
        else:
            print('[mc2] caching result: '+repr(os.path.split(fpath)[1]))
        return self.save_result(fpath)

    def has_cache(self, hs):
        fpath = self.get_fpath(hs)
        return os.path.exists(fpath)

    def cache_bytes(self, hs):
        fpath = self.get_fpath(hs)
        return helpers.file_bytes(fpath)

    def load(self, hs):
        fpath = self.get_fpath(hs)
        return self.load_result(fpath)

    def save_result(self, fpath):
        'Saves the result to the given database'
        to_save  = self.__dict__.copy()
        np.savez(fpath, **to_save)
        return True

    def load_result(self, fpath):
        'Loads the result from the given database'
        try:
            with open(fpath, 'rb') as file_:
                npz = np.load(file_)
                for _key in npz.files:
                    if _key in ['qcx']: # hack
                        # Numpy saving is werid. gotta cast
                        self.__dict__[_key] = npz[_key].tolist()
                    else: 
                        self.__dict__[_key] = npz[_key]
                npz.close()
            # HACK
            self.cx2_fm = np.array([])
            self.cx2_fs = np.array([])
            self.cx2_score = np.array([])
            # Forget non spatial scores
            return True
        except Exception as ex:
            os.remove(fpath)
            printWARN('Load Result Exception : ' + repr(ex) + 
                    '\nResult was corrupted for qcx=%d' % self.qcx)
            return False

    def top5_cxs(self):
        cx2_score = self.cx2_score_V
        top_cxs = cx2_score.argsort()[::-1]
        num_top = min(5, len(top_cxs))
        top5_cxs = top_cxs[0:num_top]
        return top5_cxs
    #def __del__(self):
        #print('[mc2] Deleting Query Result')


#========================================
# Work Functions
#========================================
def load_cached_matches(hs):
    print_ = helpers.print_
    test_sample_cx = hs.test_sample_cx
    # Create result containers
    qcx2_res = [QueryResult(qcx) for qcx in xrange(hs.num_cx)]
    #--------------------
    # Read cached queries
    #--------------------
    total_queries = len(test_sample_cx)
    print('[mc2] Total queries: %d' % total_queries)
    dirty_test_sample_cx = []
    clean_test_sample_cx = []
    fmt_str_filter = helpers.make_progress_fmt_str(total_queries, lbl='[mc2] check cache: ')
    
    # Filter queries into dirty and clean sets
    for count, qcx in enumerate(test_sample_cx):
        print_(fmt_str_filter % (count+1))
        if params.CACHE_QUERY and qcx2_res[qcx].has_cache(hs):
            clean_test_sample_cx.append(qcx)
        else:
            dirty_test_sample_cx.append(qcx)
    print('')
    print('[mc2] Num clean queries: %d ' % len(clean_test_sample_cx))
    print('[mc2] Num dirty queries: %d ' % len(dirty_test_sample_cx))
    # Check how much data we are going to load
    num_bytes = 0
    for qcx in iter(clean_test_sample_cx):
        num_bytes += qcx2_res[qcx].cache_bytes(hs)
    print('[mc2] Loading %dMB cached results' % (num_bytes / (2.0 ** 20)))
    # Load clean queries from the cache
    fmt_str_load = helpers.make_progress_fmt_str(len(clean_test_sample_cx),
                                                 lbl='[mc2] load cache: ')
    for count, qcx in enumerate(clean_test_sample_cx):
        print_(fmt_str_load % (count+1))
        qcx2_res[qcx].load(hs)
    print('')
    return qcx2_res, dirty_test_sample_cx


def run_matching(hs, qcx2_res=None, dirty_test_sample_cx=None):
    '''Runs the full matching pipeline using the abstracted classes'''
    print(textwrap.dedent('''
    =============================
    [mc2] Running Matching
    ============================='''))
    # Parameters
    #reverify_query       = params.REVERIFY_QUERY
    #resave_query         = params.RESAVE_QUERY
    verbose_matching      = params.VERBOSE_MATCHING
    # Return if no dirty queries
    if qcx2_res is None:
        qcx2_res, dirty_test_sample_cx = load_cached_matches(hs)
    if len(dirty_test_sample_cx) == 0:
        print('[mc2] No dirty queries')
        return qcx2_res
    #--------------------
    # Execute dirty queries
    #--------------------
    assign_matches  = hs.matcher.assign_matches
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts 
    total_dirty = len(dirty_test_sample_cx)
    print('[mc2] Executing %d queries' % total_dirty)
    for query_num, qcx in enumerate(dirty_test_sample_cx):
        res = qcx2_res[qcx]
        #
        # Assign matches with the chosen function (vsone) or (vsmany)
        num_qdesc = len(cx2_desc[qcx])
        print('[mc2] query(%d/%d)---------------' % (qcx, total_dirty))
        print('[mc2] query(%d/%d) assign %d desc' % (qcx, total_dirty, num_qdesc))
        tt1 = helpers.Timer(verbose=False)
        assign_output = assign_matches(qcx, cx2_desc)
        (cx2_fm, cx2_fs, cx2_score) = assign_output
        assign_time = tt1.toc()
        #
        # Spatially verify the assigned matches
        num_assigned = np.array([len(fm) for fm in cx2_fm]).sum()
        print('[mc2] query(%d/%d) verify %d assigned matches' % (qcx, total_dirty, num_assigned))
        tt2 = helpers.Timer(verbose=False)
        sv_output = spatially_verify_matches(qcx, cx2_kpts, cx2_fm, cx2_fs)
        (cx2_fm_V, cx2_fs_V, cx2_score_V) = sv_output
        num_verified = np.array([len(fm) for fm in cx2_fm_V]).sum()
        print('[mc2] query(%d/%d) verified %d matches' % (qcx, total_dirty, num_verified))
        verify_time = tt2.toc()
        print('...assigned: %.2f seconds' % (assign_time))
        print('...verified: %.2f seconds' % (verify_time))
        #
        # Assign output to the query result 
        res.cx2_fm      = cx2_fm
        res.cx2_fs      = cx2_fs
        res.cx2_score   = cx2_score
        res.cx2_fm_V    = cx2_fm_V
        res.cx2_fs_V    = cx2_fs_V
        res.cx2_score_V = cx2_score_V
        res.save(hs)
    return qcx2_res

#========================================
# Bag-of-Words
#========================================
class BagOfWordsIndex(DynStruct):
    def __init__(self, words, words_flann, cx2_vvec, wx2_idf, wx2_cxs, wx2_fxs):
        super(BagOfWordsIndex, self).__init__()
        self.words       = words
        self.words_flann = words_flann
        self.cx2_vvec    = cx2_vvec
        self.wx2_idf     = wx2_idf
        self.wx2_cxs     = wx2_cxs
        self.wx2_fxs     = wx2_fxs
    def __del__(self):
        print('[mc2] Deleting BagOfWordsIndex')
        self.words_flann.delete_index()

# precompute the bag of words model
def precompute_bag_of_words(hs):
    '''Builds a vocabulary with train_sample_cx
    Creates an indexed database with database_sample_cx'''
    print(textwrap.dedent('''
    =============================
    [mc2] Precompute Bag-of-Words
    ============================='''))
    # Unwrap parameters
    cache_dir  = hs.dirs.cache_dir
    cx2_desc   = hs.feats.cx2_desc
    train_cxs  = hs.train_sample_cx
    train_cxs = range(hs.num_cx) if train_cxs is None else train_cxs
    db_cxs     = hs.database_sample_cx
    db_cxs    = range(hs.num_cx) if db_cxs    is None else db_cxs
    vocab_size = params.__BOW_NUM_WORDS__
    # Compute vocabulary
    print(textwrap.dedent('''
    -----------------------------
    [mc2] precompute_bow(1/2): Build visual vocabulary with %d words
    -----------------------------''' % (vocab_size)))
    _comp_vocab_args   = (cx2_desc, train_cxs, vocab_size, cache_dir)
    words, words_flann = __compute_vocabulary(*_comp_vocab_args)
    # Assign visual vectors to the database
    print(textwrap.dedent('''
    -----------------------------
    [mc2] precompute_bow(2/2): Index database with visual vocabulary
    -----------------------------'''))
    _index_vocab_args = (cx2_desc, words, words_flann, db_cxs, cache_dir)
    _index_vocab_ret  = __index_database_to_vocabulary(*_index_vocab_args)
    cx2_vvec, wx2_cxs, wx2_fxs, wx2_idf = _index_vocab_ret
    # return as a BagOfWordsIndex object
    _bow_args = (words, words_flann, cx2_vvec, wx2_idf, wx2_cxs, wx2_fxs)
    bow_index = BagOfWordsIndex(*_bow_args)
    return bow_index

# step 1
def __compute_vocabulary(cx2_desc, train_cxs, vocab_size, cache_dir=None):
    '''Computes a vocabulary of size vocab_size given a set of training data'''
    # Read params
    akm_flann_params   = params.BOW_AKMEANS_FLANN_PARAMS
    words_flann_params = params.BOW_WORDS_FLANN_PARAMS
    max_iters          = params.AKMEANS_MAX_ITERS
    # Make a training set of descriptors to build the vocabulary
    tx2_desc   = cx2_desc[train_cxs]
    train_desc = np.vstack(tx2_desc)
    num_train_desc = train_desc.shape[0]
    if vocab_size > num_train_desc:
        msg = '[mc2] vocab_size(%r) > #train_desc(%r)' % (vocab_size, num_train_desc)
        helpers.printWARN(msg)
        vocab_size = num_train_desc / 2
    # Cluster descriptors into a visual vocabulary
    matcher_uid = params.get_matcher_uid()
    words_uid   = 'words_'+matcher_uid
    _, words = algos.precompute_akmeans(train_desc, vocab_size, max_iters,
                                        akm_flann_params, cache_dir,
                                        force_recomp=False, same_data=False,
                                        uid=words_uid)
    # Index the vocabulary for fast nearest neighbor search
    words_flann = algos.precompute_flann(words, cache_dir, uid=words_uid,
                                         flann_params=words_flann_params)
    return words, words_flann

# step 2
def __index_database_to_vocabulary(cx2_desc, words, words_flann, db_cxs, cache_dir):
    '''Assigns each database chip a visual-vector and returns 
       data for the inverted file'''
    # TODO: Save precomputations here
    print('[mc2] Assigning each database chip a bag-of-words vector')
    num_database = len(db_cxs)
    ax2_cx, ax2_fx, ax2_desc = __aggregate_descriptors(cx2_desc, db_cxs)
    # Build UID
    matcher_uid  = params.get_matcher_uid()
    data_uid = helpers.hashstr(ax2_desc)
    uid = data_uid + '_' + matcher_uid
    try: 
        #cx2_vvec = helpers.load_cache_npz(ax2_desc, 'cx2_vvec'+matcher_uid, cache_dir, is_sparse=True)
        #wx2_cxs  = helpers.load_cache_npz(ax2_desc, 'wx2_cxs'+matcher_uid, cache_dir)
        #wx2_fxs  = helpers.load_cache_npz(ax2_desc, 'wx2_fxs'+matcher_uid, cache_dir)
        #wx2_idf  = helpers.load_cache_npz(ax2_desc, 'wx2_idf'+matcher_uid, cache_dir)
        cx2_vvec = io.smart_load(cache_dir, 'cx2_vvec', uid, '.cPkl') #sparse
        wx2_cxs  = io.smart_load(cache_dir, 'wx2_cxs',  uid, '.npy')
        wx2_fxs  = io.smart_load(cache_dir, 'wx2_fxs',  uid, '.npy')
        wx2_idf  = io.smart_load(cache_dir, 'wx2_idf',  uid, '.npy')
        print('[mc2] successful cache load: vocabulary indexed databased.')
        return cx2_vvec, wx2_cxs, wx2_fxs, wx2_idf
    #helpers.CacheException as ex:
    except IOError as ex:
        print(repr(ex))

    print('[mc2] quantizing each descriptor to a word')
    # Assign each descriptor to its nearest visual word
    print('[mc2] ...this may take awhile with no indication of progress')
    tt1 = helpers.Timer('quantizing each descriptor to a word')
    ax2_wx, _ = words_flann.nn_index(ax2_desc, 1, checks=128)
    tt1.toc()
    # Build inverse word to ax
    tt2 = helpers.Timer('database_indexing')
    print('[mc2] building inverse word to ax map')
    wx2_axs = [[] for _ in xrange(len(words))]
    for ax, wx in enumerate(ax2_wx):
        wx2_axs[wx].append(ax)
    # Compute inverted file: words -> database
    print('[mc2] building inverted file word -> database')
    wx2_cxs = np.array([[ax2_cx[ax] for ax in ax_list] for ax_list in wx2_axs])
    wx2_fxs = np.array([[ax2_fx[ax] for ax in ax_list] for ax_list in wx2_axs])
    # Build sparse visual vectors with term frequency weights 
    print('[mc2] building sparse visual words')
    coo_cols = ax2_wx  
    coo_rows = ax2_cx
    coo_values = np.ones(len(ax2_cx), dtype=BOW_DTYPE)
    coo_format = (coo_values, (coo_rows, coo_cols))
    coo_cx2_vvec = spsparse.coo_matrix(coo_format, dtype=np.float, copy=True)
    cx2_tf_vvec  = spsparse.csr_matrix(coo_cx2_vvec, copy=False)
    # Compute idf_w = log(Number of documents / Number of docs containing word_j)
    print('[mc2] computing tf-idf')
    wx2_df  = np.array([len(set(cxs))+1 for cxs in wx2_cxs], dtype=np.float)
    wx2_idf = np.array(np.log2(np.float(num_database) / wx2_df))
    # Compute tf-idf
    print('[mc2] preweighting with tf-idf')
    cx2_tfidf_vvec = algos.sparse_multiply_rows(cx2_tf_vvec, wx2_idf)
    # Normalize
    print('[mc2] normalizing')
    cx2_tfidf_vvec = algos.sparse_multiply_rows(cx2_tf_vvec, wx2_idf)
    cx2_vvec = algos.sparse_normalize_rows(cx2_tfidf_vvec)
    tt2.toc()
    # Save to cache
    print('[mc2] saving to cache')
    r'''
    input_data = ax2_desc
    data = cx2_vvec
    uid='cx2_vvec'+matcher_uid
    '''
    io.smart_save(cx2_vvec, cache_dir, 'cx2_vvec', uid, '.cPkl') #sparse
    io.smart_save(wx2_cxs,  cache_dir, 'wx2_cxs',  uid, '.npy')
    io.smart_save(wx2_fxs,  cache_dir, 'wx2_fxs',  uid, '.npy')
    io.smart_save(wx2_idf,  cache_dir, 'wx2_idf',  uid, '.npy')
    #helpers.save_cache_npz(ax2_desc, cx2_vvec, 'cx2_vvec'+matcher_uid, cache_dir, is_sparse=True)
    #helpers.save_cache_npz(ax2_desc, wx2_cxs, 'wx2_cxs'+matcher_uid, cache_dir)
    #helpers.save_cache_npz(ax2_desc, wx2_fxs, 'wx2_fxs'+matcher_uid, cache_dir)
    #helpers.save_cache_npz(ax2_desc, wx2_idf, 'wx2_idf'+matcher_uid, cache_dir)
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
def assign_matches_bagofwords(qcx, cx2_desc, bow_index):
    cx2_vvec    = bow_index.cx2_vvec
    wx2_cxs     = bow_index.wx2_cxs
    wx2_fxs     = bow_index.wx2_fxs
    wx2_idf     = bow_index.wx2_idf
    words       = bow_index.words
    words_flann = bow_index.words_flann
    # Assign the query descriptors a visual vector
    vvec, qfx2_wx = __quantize_desc_to_tfidf_vvec(cx2_desc[qcx], wx2_idf, words, words_flann)
    # Compute distance to every database vector
    #print '---DBG'
    #print type(vvec)
    #print vvec.dtype
    #print type(cx2_vvec)
    #print cx2_vvec.dtype
    #print cx2_vvec
    #import draw_func2 as df2
    #exec(df2.present())
    cx2_score = (cx2_vvec.dot(vvec.T)).toarray().flatten()
    # Assign feature to feature matches (for spatial verification)
    cx2_fm = [[] for _ in xrange(len(cx2_desc))]
    cx2_fs = [[] for _ in xrange(len(cx2_desc))]
    for qfx, wx in enumerate(qfx2_wx):
        cx_list = wx2_cxs[wx]
        fx_list = wx2_fxs[wx]
        fs = wx2_idf[wx] # feature score is the sum of the idf values
        for (cx, fx) in zip(cx_list, fx_list): 
            if cx == qcx: continue
            fm = (qfx, fx)
            cx2_fm[cx].append(fm)
            cx2_fs[cx].append(fs)
    # Convert to numpy
    for cx in xrange(len(cx2_desc)): cx2_fm[cx] = np.array(cx2_fm[cx], dtype=FM_DTYPE)
    for cx in xrange(len(cx2_desc)): cx2_fs[cx] = np.array(cx2_fs[cx], dtype=FS_DTYPE)
    return cx2_fm, cx2_fs, cx2_score

#========================================
# One-vs-Many 
#========================================
class VsManyIndex(DynStruct): # TODO: rename this
    '''Contains a one-vs-many index and the 
       inverted information needed for voting'''
    def __init__(self, vsmany_flann, ax2_desc, ax2_cx, ax2_fx):
        super(VsManyIndex, self).__init__()
        self.vsmany_flann = vsmany_flann
        self.ax2_desc  = ax2_desc # not used, but needs to maintain scope
        self.ax2_cx = ax2_cx
        self.ax2_fx = ax2_fx
    def __del__(self):
        print('[mc2] Deleting VsManyIndex')

def __aggregate_descriptors(cx2_desc, db_cxs):
    '''Aggregates a sample set of descriptors. 
    Returns descriptors, chipxs, and featxs indexed by ax'''
    # sample the descriptors you wish to aggregate
    sx2_cx   = db_cxs
    sx2_desc = cx2_desc[sx2_cx]
    sx2_numfeat = [len(k) for k in iter(cx2_desc[sx2_cx])]
    cx_numfeat_iter = iter(zip(sx2_cx, sx2_numfeat))
    # create indexes from agg desc back to chipx and featx
    _ax2_cx = [[cx]*num_feats for (cx, num_feats) in cx_numfeat_iter]
    _ax2_fx = [range(num_feats) for num_feats in iter(sx2_numfeat)]
    ax2_cx  = np.array(list(itertools.chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(itertools.chain.from_iterable(_ax2_fx)))
    ax2_desc = np.vstack(cx2_desc[sx2_cx])
    return ax2_cx, ax2_fx, ax2_desc

def aggregate_descriptors_vsmany(hs):
    '''aggregates all descriptors for vsmany search'''
    print('[mc2] Aggregating descriptors for one-vs-many')
    cx2_desc  = hs.feats.cx2_desc
    db_cxs    = hs.database_sample_cx
    db_cxs    = range(hs.num_cx) if db_cxs is None else db_cxs
    return __aggregate_descriptors(cx2_desc, db_cxs)

#@profile
def precompute_index_vsmany(hs):
    print(textwrap.dedent('''
    =============================
    [mc2] Building one-vs-many index
    ============================='''))
    # Build (or reload) one vs many flann index
    cache_dir  = hs.dirs.cache_dir
    ax2_cx, ax2_fx, ax2_desc = aggregate_descriptors_vsmany(hs)
    # Precompute flann index
    matcher_uid = params.get_matcher_uid()
    vsmany_flann_params = params.VSMANY_FLANN_PARAMS
    vsmany_flann = algos.precompute_flann(ax2_desc, 
                                          cache_dir=cache_dir,
                                          uid=matcher_uid,
                                          flann_params=vsmany_flann_params)
    # Return a one-vs-many structure
    vsmany_index = VsManyIndex(vsmany_flann, ax2_desc, ax2_cx, ax2_fx)
    return vsmany_index

# Feature scoring functions
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist, vdist+1E-8)+1) 
def RATIO_fn(vdist, ndist): return np.divide(ndist, vdist+1E-8)
def LNBNN_fn(vdist, ndist): return ndist - vdist 
score_fn = RATIO_fn

#@profile
def assign_matches_vsmany(qcx, cx2_desc, vsmany_index):
    '''Matches desc1 vs all database descriptors using 
    Input:
        qcx        - query chip index
        cx2_desc    - chip descriptor lookup table
        vsmany_index - class with FLANN index of database descriptors
    Output: 
        cx2_fm - C x Mx2 array of matching feature indexes
        cx2_fs - C x Mx1 array of matching feature scores'''

    # vsmany_index = hs.matcher._Matcher__vsmany_index
    #helpers.println('Assigning vsmany feature matches from qcx=%d to %d chips'\ % (qcx, len(cx2_desc)))
    vsmany_flann = vsmany_index.vsmany_flann
    ax2_cx    = vsmany_index.ax2_cx
    ax2_fx    = vsmany_index.ax2_fx
    isQueryIndexed = True
    desc1 = cx2_desc[qcx]
    k_vsmany = params.__VSMANY_K__+1 if isQueryIndexed else params.__VSMANY_K__
    # Find each query descriptor's k+1 nearest neighbors
    checks = params.VSMANY_FLANN_PARAMS['checks']
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
    cx2_fm = [[] for _ in xrange(len(cx2_desc))]
    cx2_fs = [[] for _ in xrange(len(cx2_desc))]
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
    for cx in xrange(len(cx2_desc)): 
        cx2_fm[cx] = np.array(cx2_fm[cx], dtype=FM_DTYPE)
    for cx in xrange(len(cx2_desc)): 
        cx2_fs[cx] = np.array(cx2_fs[cx], dtype=FS_DTYPE)
    cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
    return cx2_fm, cx2_fs, cx2_score

def assign_matches_vsmany_BINARY(qcx, cx2_desc):
    return None

#========================================
# One-vs-One 
#========================================
def assign_matches_vsone(qcx, cx2_desc):
    #print('[mc2] Assigning vsone feature matches from cx=%d to %d chips'\ % (qcx, len(cx2_desc)))
    desc1 = cx2_desc[qcx]
    vsone_flann = pyflann.FLANN()
    vsone_flann_params =  params.VSONE_FLANN_PARAMS
    ratio_thresh = params.__VSONE_RATIO_THRESH__
    checks = vsone_flann_params['checks']
    vsone_flann.build_index(desc1, **vsone_flann_params)
    cx2_fm = [[] for _ in xrange(len(cx2_desc))]
    cx2_fs = [[] for _ in xrange(len(cx2_desc))]
    for cx, desc2 in enumerate(cx2_desc):
        sys.stdout.write('.')
        sys.stdout.flush()
        if cx == qcx: continue
        (fm, fs) = match_vsone(desc2, vsone_flann, checks)
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
def match_vsone(desc2, vsone_flann, checks, ratio_thresh=1.2, burst_thresh=None):
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
    fm  = np.array(zip(qfx, fx), dtype=FM_DTYPE)
    fs  = np.array(fx2_ratio[fx], dtype=FS_DTYPE)
    return (fm, fs)

#========================================
# Spatial verifiaction 
#========================================
# Debug data
#@profile
def spatially_verify(kpts1, kpts2, fm, fs, qcx, cx):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers 
       returns fm_V, fs_V, H '''
    xy_thresh         = params.__XY_THRESH__
    scale_thresh_high = params.__SCALE_THRESH_HIGH__
    scale_thresh_low  = params.__SCALE_THRESH_LOW__
    sv_tup = sv2.homography_inliers(kpts1, kpts2, fm, xy_thresh, 
                                    scale_thresh_high, scale_thresh_low)
    if not sv_tup is None:
        H, inliers, Aff, aff_inliers = sv_tup
        fm_V = fm[inliers, :]
        fs_V = fs[inliers, :]
    else:
        H = np.eye(3)
        fm_V = np.empty((0, 2))
        fs_V = np.array((0, 1))
    return fm_V, fs_V, H

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
    bad_consecutive_reranks = 0
    max_bad_consecutive_reranks = 20 # stop if more than 20 bad reranks
    __OVERRIDE__ = False
    for topx in xrange(num_rerank):
        cx    = top_cx[topx]
        kpts2 = cx2_kpts[cx]
        fm    = cx2_fm[cx]
        fs    = cx2_fs[cx]
        fm_V, fs_V, H = spatially_verify(kpts1, kpts2, fm, fs, qcx, cx)
        cx2_fm_V[cx] = fm_V
        cx2_fs_V[cx] = fs_V
        if len(fm_V) == 0 and __OVERRIDE__:
            bad_consecutive_reranks += 1
            if bad_consecutive_reranks > max_bad_consecutive_reranks:
                print('[mc2] Too many bad consecutive spatial verifications')
                break
        else: 
            bad_consecutive_reranks = 0
    cx2_score_V = np.array([np.sum(fs) for fs in cx2_fs_V])
    return cx2_fm_V, cx2_fs_V, cx2_score_V

#=========================
# Matcher Class
#=========================
class Matcher(DynStruct):
    '''Wrapper class: assigns matches based on
       matching and feature prefs'''
    def __init__(self, hs, match_type):
        super(Matcher, self).__init__()
        print('[mc2] Creating matcher: '+str(match_type))
        self.feat_type  = hs.feats.feat_type
        self.match_type = match_type
        # Possible indexing structures
        self.__vsmany_index    = None
        self.__bow_index    = None
        # Curry the correct functions
        self.__assign_matches = None
        if match_type == 'bagofwords':
            print('[mc2] precomputing bag of words')
            self.__bow_index   = precompute_bag_of_words(hs)
            self.__assign_matches = self.__assign_matches_bagofwords
        elif match_type == 'vsmany':
            print('[mc2] precomputing one vs many')
            self.__vsmany_index = precompute_index_vsmany(hs)
            self.__assign_matches = self.__assign_matches_vsmany
        elif match_type == 'vsone':
            self.__assign_matches = assign_matches_vsone
        else:
            raise Exception('Unknown match_type: '+repr(match_type))

    def __del__(self):
        print('[mc2] Deleting Matcher')

    def assign_matches(self, qcx, cx2_desc):
        'Function which calls the correct matcher'
        return self.__assign_matches(qcx, cx2_desc)

    # query helpers
    def __assign_matches_vsmany(self, qcx, cx2_desc):
        return assign_matches_vsmany(qcx, cx2_desc, self.__vsmany_index)

    def __assign_matches_bagofwords(self, qcx, cx2_desc):
        return assign_matches_bagofwords(qcx, cx2_desc, self.__bow_index)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('[mc2] __main__ = match_chips2.py')

    db_dir = load_data2.DEFAULT
    hs = load_data2.HotSpotter(db_dir)
    qcx2_res = run_matching(hs)
    pass

    ## DEV ONLY CODE ##
    __DEV_MODE__ = False
    if __DEV_MODE__: 
        cx2_kpts = hs.feats.cx2_kpts
        cx2_desc = hs.feats.cx2_desc
        qcx = 1
        print('[mc2] DEVMODE IS ON')
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
        #exec(helpers.get_exec_src(precompute_bag_of_words))

    exec(df2.present())
