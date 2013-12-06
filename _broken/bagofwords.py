from __future__ import division, print_function
import __builtin__
import sys
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
import hotspotter.draw_func2 as df2
# Hotspotter Imports
import hotspotter.fileio as io
import hotspotter.helpers as helpers
from hotspotter.helpers import Timer, tic, toc, printWARN
from hotspotter.Printable import DynStruct
import hotspotter.algos as algos
import hotspotter.spatial_verification2 as sv2
import hotspotter.load_data2 as load_data2
import hotspotter.params as params
# Math and Science Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyflann
import scipy as sp
import scipy.sparse as spsparse
import sklearn.preprocessing 
from itertools import izip
#print('LOAD_MODULE: match_chips2.py')

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass
# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[bow] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

BOW_DTYPE = np.uint8
FM_DTYPE  = np.uint32
FS_DTYPE  = np.float32
#========================================
# Bag-of-Words
#========================================
class BagOfWordsArgs(DynStruct):
    def __init__(self, words, words_flann, cx2_vvec, wx2_idf, wx2_cxs, wx2_fxs):
        super(BagOfWordsArgs, self).__init__()
        self.words       = words
        self.words_flann = words_flann
        self.cx2_vvec    = cx2_vvec
        self.wx2_idf     = wx2_idf
        self.wx2_cxs     = wx2_cxs
        self.wx2_fxs     = wx2_fxs
    def __del__(self):
        print('[mc2] Deleting BagOfWordsArgs')
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
    # return as a BagOfWordsArgs object
    _bow_args = (words, words_flann, cx2_vvec, wx2_idf, wx2_cxs, wx2_fxs)
    bow_args = BagOfWordsArgs(*_bow_args)
    return bow_args

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
def assign_matches_bagofwords(bow_args, qcx, cx2_kpts, cx2_desc, cx2_rchip_size):
    cx2_vvec    = bow_args.cx2_vvec
    wx2_cxs     = bow_args.wx2_cxs
    wx2_fxs     = bow_args.wx2_fxs
    wx2_idf     = bow_args.wx2_idf
    words       = bow_args.words
    words_flann = bow_args.words_flann
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
        fm.shape = (len(fm), 2)
        #fm = fm.reshape(len(fm), 2)
        cx2_fm[cx] = fm
    for cx in xrange(len(cx2_desc)): 
        cx2_fs[cx] = np.array(cx2_fs[cx], dtype=FS_DTYPE)
    cx2_fm = np.array(cx2_fm)
    cx2_fs = np.array(cx2_fs)
    return cx2_fm, cx2_fs, cx2_score
