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
from itertools import izip, chain
import investigate_chip as invest


# Premature optimization is the root of all evil

def main():
    invest.rrr()
    main_locals = invest.main()
    execstr = helpers.execstr_dict(main_locals, 'main_locals')
    exec(execstr)

    return locals()

class NearestNeighborIndex():
    def __init__(self, hs, sample_cxs, **kwargs):

class SpatialVerifyParams():
    def __init__(self):
        self.scale_range_thresh  = (.5, 2)
        self.xy_thresh = .002

class ScoringParams():
    self __init__(self):
        self.vote_weight_fn = LNBNN
        self.voting_rule_fn = PlacketLuce
        self.num_shortlist  = 100

class MatcherParams():
    def __init__(self):
        self.query_cxs    = []
        self.data_cxs     = []
        self.invert_query = False

    def prepare(self, hs):
        if not self.invert_query:
            self.data_index = precompute_database_index()

def precompute_query_index(hs, qcx):
    sx2_cx = [qcx]
    query_index = NNIndex(hs, sx2_cx)
    return query_index

def precompute_database_index(hs):
    sx2_cx = hs.indexed_sample_cx
    data_index = NNIndex(hs, sx2_cx)
    return data_index

class NNIndex(object):
    def __init__(self, hs, sx2_cx):
        cx2_desc  = hs.feats.cx2_desc
        # Make unique id for indexed descriptors
        feat_uid   = params.get_feat_uid()
        sample_uid = helpers.make_sample_id(sx2_cx)
        uid = '_cxs(' + sample_uid + ')' + feat_uid
        # Number of features per sample chip
        sx2_nFeat = [len(cx2_desc[sx]) for sx in iter(sx2_cx)]
        # Inverted index from indexed descriptor to chipx and featx 
        _ax2_cx = [[cx]*nFeat for (cx, nFeat) in izip(sx2_cx, sx2_nFeat)]
        _ax2_fx = [range(nFeat) for nFeat in iter(sx2_nFeat)]
        ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
        ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
        # Aggregate indexed descriptors into continuous structure
        ax2_desc = np.vstack([cx2_desc[cx] for cx in sx2_cx])
        # Build/Load the flann index
        flann_params = {'algorithm':'kdtree', 'trees':4}
        precomp_kwargs = {'cache_dir'    : hs.dirs.cache_dir,
                          'uid'          : uid,
                          'flann_params' : flann_params, }
        flann = algos.precompute_flann(ax2_desc, **precomp_kwargs)
        #----
        # Core
        self.K_nearest = 1
        # Filters
        self.checks = 128
        self.K_reciprocal   = 0 # 0 := off
        self.roidist_thresh = 1 # 1 := off
        self.ratio_thresh   = 1 # 1 := off
        self.freq_thresh    = 1 # 1 := off
        # Agg Data
        self.ax2_cx   = ax2_cx
        self.ax2_fx   = ax2_fx
        self.ax2_data = ax2_desc
        self.flann = flann

def prequery(hs):
    query_params = QueryParams()
    if query_params.invert_query:
        data_index  = precompute_flann(cxs)
        query_cxs   = dcxs
        data_cxs    = cxs
    else:
        data_index  = precompute_flann(dcxs)
        query_cxs   = cxs
        data_cxs    = dcxs

#def execute_query_fast(hs, qcx, query_params):
# fast should be the current sota execute_query that doesn't perform checks and
# need to have precomputation done beforehand. 
# safe should perform all checks and be easilly callable on the fly. 

def execute_query_safe(hs, qcx, query_params):
    # Can we generalize to have more than one query chip?
    cx2_neighbors = {}
    nnfilter_list = [reciprocal, roidist, frexquency, ratiotest, bursty]
    scoring_func  = [LNBNN, PlacketLuce, TopK, Borda]
    load_precomputed(cx, query_params)
    cx2_neighbors[cx] = nearest_neighbors(data_index, query_params, nn_params)
    for nnfilter in nnfilter_list:
        nnfilter(cx2_neighbors)
    scoring_func(cx2_neighbors, scoring_params)
    shortlist_cx, longlist_cx = get_shortlist(query_cxs, cx2_neighbors)
    for cx in shortlist_cx:
        spatial_verify(cx2_neighbors[cx], verify_params)
    for cx in longlist_cx:
        remove_matches(cx2_neighbors[cx])
    scores = scoring_func(cx2_neighbors, scoring_params)
    cache_neighbors(cx2_neighbors)
    return scores, cx2_neighbors

'''
PRIORITY 1: 
* CREATE A SIMPLE TEST DATABASE
* Need simple testcases showing the validity of each step. Do this with a small
* database of three images: Query, TrueMatch, FalseMatch
* Manually remove a selection of keypoints. 

PRIORITY 2: 
* FIX QUERY CACHING
 QueryResult should save each step of the query. 
 * Initial Nearest Neighbors Result,
 * Filter Reciprocal Result
 * Filter Spatial Result
 * Filter Spatial Verification Result 
 You should have the ability to turn the caching of any part off. 

PRIORITY 3: 
 * Unifty vsone and vsmany
 * Just make a query params object
 they are the same process which accepts the parameters: 
     invert_query, qcxs, dcxs


'''
if __name__ == '__main__':
    main_locals = main()
    locals_execstr = helpers.dict_execstr(main_locals, 'main_locals')
    exec(locals_execstr)

