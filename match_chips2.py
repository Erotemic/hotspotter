'''
Module match_chips: 
    Runs vsone, vsmany, and bagofwords matching
'''
#from numba import autojit
from __future__ import division, print_function
#========================================
# IMPORTS
#========================================
# Standard library imports
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
#print('LOAD_MODULE: match_chips2.py')

def reload_module():
    import imp, sys
    print('[mc2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

BOW_DTYPE = np.uint8
FM_DTYPE  = np.uint32
FS_DTYPE  = np.float32

def debug_cx2_fm_shape(cx2_fm):
    print('-------------------')
    print('Debugging cx2_fm shape')
    print('len(cx2_fm)=%r' % len(cx2_fm))
    last_repr = ''
    for cx in xrange(len(cx2_fm)):
        if type(cx2_fm[cx]) == np.ndarray:
            this_repr = repr(cx2_fm[cx].shape)+repr(type(cx2_fm[cx]))
        else:
            this_repr = repr(len(cx2_fm[cx]))+repr(type(cx2_fm[cx]))
        if last_repr != this_repr:
            last_repr = this_repr
            print(last_repr)
    print('-------------------')


def ensure_array_dtype(arr, dtype):
    if not type(arr) is np.ndarray or arr.dtype != dtype:
        arr = np.array(arr, dtype=dtype)
    return arr

def ensure_2darray_nCols(arr, nCols):
    if len(arr.shape) != 2 or arr.shape[1] != nCols:
        arr = arr.reshape(len(arr), nCols)
    return arr

def fix_fm(fm):
    fm = ensure_array_dtype(fm, FM_DTYPE)
    fm = ensure_2darray_nCols(fm, 2)
    return fm

def fix_fs(fs):
    fs = ensure_array_dtype(fs, FS_DTYPE)
    fs = fs.flatten()
    return fs

def fix_cx2_fm_shape(cx2_fm):
    for cx in xrange(len(cx2_fm)):
        cx2_fm[cx] = fix_fm(cx2_fm[cx])
    cx2_fm = np.array(cx2_fm)
    return cx2_fm

#def fix_cx2_fm_shape(cx2_fm):
    #for cx in xrange(len(cx2_fm)):
        #fm = cx2_fm[cx]
        #if type(fm) != np.ndarray() or fm.dtype != FM_DTYPE:
            #fm = np.array(fm, dtype=FM_DTYPE)
        #if len(fm.shape) != 2 or fm.shape[1] != 2:
            #fm = fm.reshape(len(fm), 2)
        #cx2_fm[cx] = fm
    #cx2_fm = np.array(cx2_fm)
    #return cx2_fm

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

def query_result_fpath(hs, qcx):
    query_uid = params.get_query_uid()
    qres_dir = hs.dirs.qres_dir 
    fname = 'result_'+query_uid+'_qcx=%d.npz' % qcx
    fpath = os.path.join(qres_dir, fname)
    return fpath

def query_result_exists(hs, qcx):
    fpath = query_result_fpath(hs, qcx)
    return os.path.exists(fpath)

def save_npz_from_dict(dict_, fpath):
    if params.VERBOSE_CACHE:
        print('[mc2] caching result: '+repr(fpath))
    else:
        print('[mc2] caching result: '+repr(os.path.split(fpath)[1]))
    with open(fpath, 'wb') as file_:
        np.savez(file_, **dict_.copy())

def load_npz_into_dict(dict_, fpath, scalar_keys=set(['qcx'])):
    with open(fpath, 'rb') as file_:
        npz = np.load(file_)
        for _key in npz.files:
            dict_[_key] = npz[_key].tolist() if _key in scalar_keys else npz[_key]
        npz.close()

class QueryResult(DynStruct):
    def __init__(self, qcx):
        super(QueryResult, self).__init__()
        self.qcx    = qcx
        # Times
        self.assign_time = -1
        self.verify_time = -1
        # Assigned features matches
        self.cx2_fm = np.array([], dtype=FM_DTYPE)
        self.cx2_fs = np.array([], dtype=FS_DTYPE)
        self.cx2_score = np.array([])
        # Spatially verified feature matches
        self.cx2_fm_V = np.array([], dtype=FM_DTYPE)
        self.cx2_fs_V = np.array([], dtype=FS_DTYPE)
        self.cx2_score_V = np.array([])

    def remove_init_assigned(self):
        self.cx2_fm = np.array([], dtype=FM_DTYPE)
        self.cx2_fs = np.array([], dtype=FS_DTYPE)
        self.cx2_score = np.array([])

    def has_cache(self, hs):
        return query_result_exists(hs, self.qcx)

    def has_init_assign(self):
        return not (len(self.cx2_fm) == 0 and len(self.cx2_fm_V) != 0)

    def get_fpath(self, hs):
        return query_result_fpath(hs, self.qcx)
    
    def save(self, hs, remove_init=True):
        if remove_init: self.remove_init_assigned()
        fpath = self.get_fpath(hs)
        save_npz_from_dict(self.__dict__, fpath)
        return True

    def load(self, hs, remove_init=True):
        'Loads the result from the given database'
        fpath = os.path.normpath(self.get_fpath(hs))
        try:
            load_npz_into_dict(self.__dict__, fpath, scalar_keys=set(['qcx']))
            if remove_init: self.remove_init_assigned()
            return True
        except Exception as ex:
            #os.remove(fpath)
            warnmsg = ('Load Result Exception : ' + repr(ex) + 
                    '\nResult was corrupted for qcx=%d' % self.qcx)
            print(warnmsg)
            warnings.warn(warnmsg)
            raise
            #return False

    def cache_bytes(self, hs):
        fpath = self.get_fpath(hs)
        return helpers.file_bytes(fpath)

    def top5_cxs(self):
        return self.topN_cxs(5)

    def topN_cxs(self, N):
        cx2_score = self.cx2_score_V
        top_cxs = cx2_score.argsort()[::-1]
        num_top = min(N, len(top_cxs))
        topN_cxs = top_cxs[0:num_top]
        return topN_cxs




#========================================
# Work Functions
#========================================

def get_dirty_test_cxs(hs):
    test_samp = hs.test_sample_cx
    print('[mc2] checking dirty queries')
    dirty_samp = [qcx for qcx in iter(test_samp) if not query_result_exists(hs, qcx)]
    return dirty_samp

def run_matching2(hs, verbose=params.VERBOSE_MATCHING):
    print(textwrap.dedent('''
    \n=============================
    [mc2] Running Matching 2
    ============================='''))
    dirty_samp = get_dirty_test_cxs(hs)
    total_dirty = len(dirty_samp)
    print('[mc2] There are %d dirty queries' % total_dirty)
    if len(dirty_samp) == 0:
        return
    print_ = helpers.print_
    if hs.matcher is None:
        hs.load_matcher()
    assign_matches  = hs.matcher.assign_matches
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    for qnum, qcx in enumerate(dirty_samp):
        res = QueryResult(qcx)
        build_result(hs, res, cx2_kpts, cx2_desc, cx2_rchip_size,
                     assign_matches, qnum, total_dirty, verbose)

def load_cached_matches(hs):
    print_ = helpers.print_
    test_samp = hs.test_sample_cx
    # Create result containers
    print('[mc2] hs.num_cx = %r ' % hs.num_cx)
    qcx2_res = [QueryResult(qcx) for qcx in xrange(hs.num_cx)]
    #--------------------
    # Read cached queries
    #--------------------
    total_queries = len(test_samp)
    print('[mc2] Total queries: %d' % total_queries)
    dirty_test_sample_cx = []
    clean_test_sample_cx = []
    fmt_str_filter = helpers.make_progress_fmt_str(total_queries, lbl='[mc2] check cache: ')
    
    # Filter queries into dirty and clean sets
    for count, qcx in enumerate(test_samp):
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

def run_matching(hs, qcx2_res=None, dirty_test_sample_cx=None, verbose=params.VERBOSE_MATCHING):
    '''Runs the full matching pipeline using the abstracted classes'''
    print(textwrap.dedent('''
    \n=============================
    [mc2] Running Matching
    ============================='''))
    # Parameters
    #reverify_query       = params.REVERIFY_QUERY
    #resave_query         = params.RESAVE_QUERY
    # Return if no dirty queries
    if qcx2_res is None:
        print('[mc2] qcx2_res was not specified... loading cache')
        qcx2_res, dirty_test_sample_cx = load_cached_matches(hs)
    else:
        print('[mc2] qcx2_res was specified... not loading cache')
    if len(dirty_test_sample_cx) == 0:
        print('[mc2] No dirty queries')
        return qcx2_res
    #--------------------
    # Execute dirty queries
    #--------------------
    assign_matches  = hs.matcher.assign_matches
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    total_dirty = len(dirty_test_sample_cx)
    print('[mc2] Executing %d dirty queries' % total_dirty)
    for qnum, qcx in enumerate(dirty_test_sample_cx):
        if verbose:
            print('[mc2] query(%d/%d)---------------' % (qnum+1, total_dirty))
        res = qcx2_res[qcx]
        build_result(hs, res, cx2_kpts, cx2_desc, cx2_rchip_size,
                     assign_matches, verbose)
    return qcx2_res

def __build_result_assign_step(hs, res, cx2_kpts, cx2_desc, assign_matches, verbose):
    '1) Assign matches with the chosen function (vsone) or (vsmany)'
    if verbose:
        num_qdesc = len(cx2_desc[res.qcx])
        print('[mc2] assign %d desc' % (num_qdesc))
    tt1 = helpers.Timer(verbose=False)
    assign_output = assign_matches(res.qcx, cx2_desc)
    (cx2_fm, cx2_fs, cx2_score) = assign_output
    # Record initial assignments 
    res.assign_time = tt1.toc()
    res.cx2_fm      = np.array(cx2_fm)
    res.cx2_fs      = np.array(cx2_fs)
    res.cx2_score   = cx2_score

def __build_result_verify_step(hs, res, cx2_kpts, cx2_rchip_size, verbose):
    ' 2) Spatially verify the assigned matches'
    cx2_fm    = res.cx2_fm
    cx2_fs    = res.cx2_fs
    cx2_score = res.cx2_score
    if verbose:
        num_assigned = np.array([len(fm) for fm in cx2_fm]).sum()
        print('[mc2] verify %d assigned matches' % (num_assigned))
    tt2 = helpers.Timer(verbose=False)
    sv_output = spatially_verify_matches(res.qcx, cx2_kpts, cx2_rchip_size, cx2_fm, cx2_fs, cx2_score)
    (cx2_fm_V, cx2_fs_V, cx2_score_V) = sv_output
    # Record verified assignments 
    res.verify_time = tt2.toc()
    res.cx2_fm_V    = np.array(cx2_fm_V)
    res.cx2_fs_V    = np.array(cx2_fs_V)
    res.cx2_score_V = cx2_score_V
    if verbose:
        num_verified = np.array([len(fm) for fm in cx2_fm_V]).sum()
        print('[mc2] verified %d matches' % (num_verified))

def build_result(hs, res, cx2_kpts, cx2_desc, cx2_rchip_size, assign_matches,
                 verbose=True, remove_init=True):
    'Calls the actual query calculations and builds the result class'
    __build_result_assign_step(hs, res, cx2_kpts, cx2_desc, assign_matches, verbose)
    __build_result_verify_step(hs, res, cx2_kpts, cx2_rchip_size, verbose)
    if verbose:
        print('...assigned: %.2f seconds' % (res.assign_time))
        print('...verified: %.2f seconds\n' % (res.verify_time))
    else:
        print('...query: %.2f seconds\n' % (res.verify_time + res.assign_time))
    res.save(hs, remove_init=remove_init)

def build_result_qcx(hs, qcx, use_cache=True, remove_init=True):
    'this should be the on-the-fly / Im going to check things function'
    res = QueryResult(qcx)
    if use_cache and res.has_cache(hs):
        res.load(hs, remove_init)
        if not remove_init and res.has_init_assign():
            return res
    verbose = True
    assign_matches = hs.matcher.assign_matches
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    __build_result_assign_step(hs, res, cx2_kpts, cx2_desc, assign_matches, verbose)
    __build_result_verify_step(hs, res, cx2_kpts, cx2_rchip_size, verbose)
    if use_cache:
        res.save(hs, remove_init=remove_init)
    return res


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
    Creates an indexed database with indexed_sample_cx'''
    print(textwrap.dedent('''
    \n=============================
    [mc2] Precompute Bag-of-Words
    ============================='''))
    # Unwrap parameters
    cache_dir  = hs.dirs.cache_dir
    cx2_desc   = hs.feats.cx2_desc
    train_cxs  = hs.train_sample_cx
    train_cxs = range(hs.num_cx) if train_cxs is None else train_cxs
    indexed_cxs = hs.indexed_sample_cx
    indexed_cxs = range(hs.num_cx) if indexed_cxs is None else indexed_cxs
    vocab_size = params.__BOW_NUM_WORDS__
    ndesc_per_word = params.__BOW_NDESC_PER_WORD__
    if not ndesc_per_word is None:
        num_train_desc = sum(map(len, cx2_desc[train_cxs]))
        print('[mc2] there are %d training descriptors: ' % num_train_desc)
        print('[mc2] training vocab with ~%r descriptor per word' % ndesc_per_word)
        vocab_size = int(num_train_desc // ndesc_per_word)
        # oh this is bad, no more globals
        params.__BOW_NUM_WORDS__ = vocab_size
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
    _index_vocab_args = (cx2_desc, words, words_flann, indexed_cxs, cache_dir)
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
    matcher_uid = params.get_matcher_uid(with_train=True, with_indx=False)
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
def __index_database_to_vocabulary(cx2_desc, words, words_flann, indexed_cxs, cache_dir):
    '''Assigns each database chip a visual-vector and returns 
       data for the inverted file'''
    # TODO: Save precomputations here
    print('[mc2] Assigning each database chip a bag-of-words vector')
    num_indexed = len(indexed_cxs)
    ax2_cx, ax2_fx, ax2_desc = __aggregate_descriptors(cx2_desc, indexed_cxs)
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
    print('')
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
    wx2_idf = np.array(np.log2(np.float(num_indexed) / wx2_df))
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
    #print('---DBG')
    #print(type(vvec))
    #print(vvec.dtype)
    #print(type(cx2_vvec))
    #print(cx2_vvec.dtype0
    #print(cx2_vvec)
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
    for cx in xrange(len(cx2_desc)):
        fm = np.array(cx2_fm[cx], dtype=FM_DTYPE)
        fm = fm.reshape(len(fm), 2)
        cx2_fm[cx] = fm
    for cx in xrange(len(cx2_desc)): 
        cx2_fs[cx] = np.array(cx2_fs[cx], dtype=FS_DTYPE)
    cx2_fm = np.array(cx2_fm)
    cx2_fs = np.array(cx2_fs)
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

def __aggregate_descriptors(cx2_desc, indexed_cxs):
    '''Aggregates a sample set of descriptors. 
    Returns descriptors, chipxs, and featxs indexed by ax'''
    # sample the descriptors you wish to aggregate
    sx2_cx   = indexed_cxs
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
    indexed_cxs = hs.indexed_sample_cx
    indexed_cxs = range(hs.num_cx) if indexed_cxs is None else indexed_cxs
    return __aggregate_descriptors(cx2_desc, indexed_cxs)

#@profile
def precompute_index_vsmany(hs):
    print(textwrap.dedent('''
    \n=============================
    [mc2] Building one-vs-many index
    ============================='''))
    # Build (or reload) one vs many flann index
    cache_dir  = hs.dirs.cache_dir
    ax2_cx, ax2_fx, ax2_desc = aggregate_descriptors_vsmany(hs)
    # Precompute flann index
    matcher_uid = params.get_matcher_uid()
    #checks = params.VSMANY_FLANN_PARAMS['checks']
    vsmany_flann_params = params.VSMANY_FLANN_PARAMS
    vsmany_flann = algos.precompute_flann(ax2_desc, 
                                          cache_dir=cache_dir,
                                          uid=matcher_uid,
                                          flann_params=vsmany_flann_params)
    # Return a one-vs-many structure
    vsmany_index = VsManyIndex(vsmany_flann, ax2_desc, ax2_cx, ax2_fx)
    return vsmany_index

# Feature scoring functions
eps = 1E-8
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist, vdist+eps)+1) 
def RATIO_fn(vdist, ndist): return np.divide(ndist, vdist+eps)
def LNBNN_fn(vdist, ndist): return (ndist - vdist) / 1000.0

scoring_func_map = {
    'LNRAT' : LNRAT_fn,
    'RATIO' : RATIO_fn,
    'LNBNN' : LNBNN_fn }

def desc_nearest_neighbors(desc, vsmany_index, K=None):
    vsmany_flann = vsmany_index.vsmany_flann
    ax2_cx       = vsmany_index.ax2_cx
    ax2_fx       = vsmany_index.ax2_fx
    isQueryIndexed = True
    K = params.__VSMANY_K__ if K is None else K
    checks   = params.VSMANY_FLANN_PARAMS['checks']
    # Find each query descriptor's k+1 nearest neighbors
    (qfx2_ax, qfx2_dists) = vsmany_flann.nn_index(desc, K, checks=checks)
    qfx2_cx = ax2_cx[qfx2_ax]
    qfx2_fx = ax2_fx[qfx2_ax]
    return (qfx2_cx, qfx2_fx, qfx2_dists) 

# TODO: Nearest Neighbor Huristrics 
# K-recripricol nearest neighbors
# ROI spatial matching
# Frequency Reranking


def quick_flann_index(data):
    data_flann = pyflann.FLANN()
    flann_params =  params.VSMANY_FLANN_PARAMS
    checks = flann_params['checks']
    data_flann.build_index(data, **flann_params)
    return data_flann

def nearest_neighbors(query, data_flann, K, checks=128):
    (qfx2_dx, qfx2_dists) = data_flann.nn_index(query, K, checks=checks)

def reciprocal_nearest_neighbors(query, data, data_flann, checks):
    nQuery, dim = query.shape
    # Assign query features to K nearest database features
    (qfx2_dx, qfx2_dists) = data_flann.nn_index(query, K, checks=checks)
    # Assign those nearest neighbors to K nearest database features
    qx2_nn = data[qfx2_dx]
    qx2_nn.shape = (nQuery*K, dim)
    (_nn2_dx, nn2_dists) = data_flann.nn_index(qx2_nn, K, checks=checks)
    # Get the maximum distance of the reciprocal neighbors
    nn2_dists.shape = (nQuery, K, K)
    qfx2_maxdist = nn2_dists.max(2)
    # Test if nearest neighbor distance is less than reciprocal distance
    isReciprocal = qfx2_dists < qfx2_maxdist
    return qfx2_dx, qfx2_dists, isReciprocal 


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
    score_fn = scoring_func_map[params.__VSMANY_SCORE_FN__]
    isQueryIndexed = True
    desc1 = cx2_desc[qcx]
    k_vsmany = params.__VSMANY_K__+1 if isQueryIndexed else params.__VSMANY_K__
    checks   = params.VSMANY_FLANN_PARAMS['checks']
    # Find each query descriptor's k+1 nearest neighbors
    (qfx2_ax, qfx2_dists) = vsmany_flann.nn_index(desc1, k_vsmany+1, checks=checks)
    vote_dists = qfx2_dists[:, 0:k_vsmany]
    norm_dists = qfx2_dists[:, k_vsmany] # k+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([score_fn(_vdist.T, norm_dists)
                           for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    ax2_cx       = vsmany_index.ax2_cx
    ax2_fx       = vsmany_index.ax2_fx
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
        fm = np.array(cx2_fm[cx], dtype=FM_DTYPE)
        fm = fm.reshape(len(fm), 2)
        cx2_fm[cx] = fm
    for cx in xrange(len(cx2_desc)): 
        cx2_fs[cx] = np.array(cx2_fs[cx], dtype=FS_DTYPE)
    cx2_fm = np.array(cx2_fm)
    cx2_fs = np.array(cx2_fs)
    cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
    return cx2_fm, cx2_fs, cx2_score

def assign_matches_vsmany_BINARY(qcx, cx2_desc):
    return None

#========================================
# One-vs-One 
#========================================

def get_vsone_flann(desc1):
    vsone_flann = pyflann.FLANN()
    vsone_flann_params =  params.VSONE_FLANN_PARAMS
    ratio_thresh = params.__VSONE_RATIO_THRESH__
    checks = vsone_flann_params['checks']
    vsone_flann.build_index(desc1, **vsone_flann_params)
    return vsone_flann, checks

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
    cx2_fm = np.array(cx2_fm)
    cx2_fs = np.array(cx2_fs)
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
        vsone_flann   - FLANN index of desc1 (query descriptors (N1xD)) 
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
    fm  = fm.reshape(len(fm), 2)
    fs  = np.array(fx2_ratio[fx], dtype=FS_DTYPE)
    return (fm, fs)

#========================================
# Spatial verifiaction 
#========================================
def __default_sv_return():
    'default values returned by bad spatial verification'
    #H = np.eye(3)
    fm_V = np.empty((0, 2))
    fs_V = np.array((0, 1))
    return (fm_V, fs_V)
#@profile
def spatially_verify(kpts1, kpts2, rchip_size2, fm, fs):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers 
       returns fm_V, fs_V, H '''
    # Return if pathological
    min_num_inliers   = 4
    if len(fm) < min_num_inliers:
        return __default_sv_return()
    # Get homography parameters
    xy_thresh         = params.__XY_THRESH__
    scale_thresh_high = params.__SCALE_THRESH_HIGH__
    scale_thresh_low  = params.__SCALE_THRESH_LOW__
    if params.__USE_CHIP_EXTENT__:
        diaglen_sqrd = rchip_size2[0]**2 + rchip_size2[1]**2
    else:
        x_m = kpts2[fm[:,1],0].T
        y_m = kpts2[fm[:,1],1].T
        diaglen_sqrd = sv2.calc_diaglen_sqrd(x_m, y_m)
    # Try and find a homography
    sv_tup = sv2.homography_inliers(kpts1, kpts2, fm,
                                    xy_thresh, 
                                    scale_thresh_high,
                                    scale_thresh_low,
                                    diaglen_sqrd,
                                    min_num_inliers)
    if sv_tup is None:
        return __default_sv_return()
    # Return the inliers to the homography
    (H, inliers, Aff, aff_inliers) = sv_tup
    fm_V = fm[inliers, :]
    fs_V = fs[inliers, :]
    return fm_V, fs_V

def spatially_verify_check_args(xy_thresh=None,
                                scale_min=None,
                                scale_max=None,
                                rchip_size2=None, 
                                min_num_inliers=None):
    ''' Performs argument checks for spatially_verify'''
    if min_num_inliers is None: 
        min_num_inliers = 4
    if xy_thresh is None:
        xy_thresh = params.__XY_THRESH__
    if scale_min is None:
        scale_min = params.__SCALE_THRESH_LOW__
    if scale_max is None:
        scale_max = params.__SCALE_THRESH_HIGH__
    if rchip_size2 is None:
        x_m = kpts2[fm[:,1],0].T
        y_m = kpts2[fm[:,1],1].T
        x_extent_sqrd = (x_m.max() - x_m.min()) ** 2
        y_extent_sqrd = (y_m.max() - y_m.min()) ** 2
        diaglen_sqrd = x_extent_sqrd + y_extent_sqrd
    else:
        diaglen_sqrd = rchip_size2[0]**2 + rchip_size2[1]**2
    return min_num_inliers, xy_thresh, scale_max, scale_min, diaglen_sqrd

def __spatially_verify(kpts1, kpts2, fm, fs, min_num_inliers, xy_thresh,
                       scale_thresh_high, scale_thresh_low, diaglen_sqrd):
    '''Work function for spatial verification'''
    sv_tup = sv2.homography_inliers(kpts1, kpts2, fm,
                                    xy_thresh, 
                                    scale_thresh_high,
                                    scale_thresh_low,
                                    diaglen_sqrd,
                                    min_num_inliers)
    if not sv_tup is None:
        # Return the inliers to the homography
        (H, inliers, Aff, aff_inliers) = sv_tup
        fm_V = fm[inliers, :]
        fs_V = fs[inliers, :]
    else:
        fm_V = np.empty((0, 2))
        fs_V = np.array((0, 1))
        H = np.eye(3)
    return fm_V, fs_V, H


def spatially_verify2(kpts1, kpts2, fm, fs, **kwargs):
    '''Wrapper function for spatial verification with argument checks'''
    min_num_inliers, xy_thresh, scale_min, scale_max, diaglen_sqrd = spatially_verify_check_args(**kwargs)
    fm_V, fs_V, H = __spatially_verify(kpts1, kpts2, fm, fs, min_num_inliers,
                                       xy_thresh, scale_min, scale_max, diaglen_sqrd)
    return fm_V, fs_V


#@profile
def spatially_verify_matches(qcx, cx2_kpts, cx2_rchip_size, cx2_fm, cx2_fs, cx2_score):
    kpts1     = cx2_kpts[qcx]
    #cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
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
        rchip_size2 = cx2_rchip_size[cx]
        fm_V, fs_V = spatially_verify(kpts1, kpts2, rchip_size2, fm, fs)
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
        self.__vsmany_index = None
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
    import load_data2
    import chip_compute2
    import params
    freeze_support()
    print('[mc2] __main__ = match_chips2.py')

    # --- CHOOSE DATABASE --- #
    db_dir = params.DEFAULT
    hs = load_data2.HotSpotter()
    hs.load_tables(db_dir)
    hs.load_chips()
    hs.set_samples()
    hs.load_features()
    hs.load_matcher()

    #qcx = 111
    #cx = 305
    qcx = 0
    cx = hs.get_other_indexed_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)

    res = build_result_qcx(hs, qcx, remove_init=False)
    df2.show_matches_annote_res(res, hs, cx, draw_pts=False, plotnum=(1,2,1))
    df2.show_matches_annote_res(res, hs, cx, draw_pts=False, plotnum=(1,2,2), SV=False)

    if len(sys.argv) > 1:
        try:
            cx = int(sys.argv[1])
            print('cx=%r' % cx)
        except Exception as ex:
            print('exception %r' % ex)
            raise
            print('usage: feature_compute.py [cx]')
            pass

    exec(df2.present())
