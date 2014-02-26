from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[mc3]', DEBUG=False)
#
import numpy as np
# HotSpotter
from hscom import params
from hscom import helpers as util
import DataStructures as ds
import matching_functions as mf

# TODO INTEGRATE INDENTER 2 better
import HotSpotterAPI as api
from hscom import Parallelize as parallel
modules = [api, api.fc2, api.cc2, parallel, mf, ds]

@profile
@util.indent_decor('[pre_exec]')
def pre_cache_checks(hs, qreq, use_cache=True):
    print(' --- pre cache checks --- ')
    feat_uid = qreq.cfg.feat_cfg.get_uid()
    # Load any needed features or chips into memory
    if hs.feats.feat_uid != feat_uid:
        hs.unload_cxdata('all')
    hs.refresh_features(qreq._dcxs)
    hs.refresh_features(qreq._qcxs)
    return qreq
    # Use the query result cache if possible


@profile
@util.indent_decor('[pre_exec]')
def pre_exec_checks(hs, qreq, use_cache=True):
    print(' --- pre query request execution checks --- ')
    dcxs = qreq.get_dcxs()
    feat_uid = qreq.cfg.feat_cfg.get_uid()
    dcxs_uid = util.hashstr_arr(dcxs, 'dcxs')
    # Ensure the index / inverted index exist
    dftup_uid = (dcxs_uid, feat_uid)
    if not dftup_uid in qreq._dftup2_index:
        print('qreq.flann[dcxs_uid]... nn_index cache miss')
        # Compute the FLANN Index
        data_index = ds.NNIndex(hs, dcxs)
        qreq._dftup2_index[dftup_uid] = data_index
    qreq._data_index = qreq._dftup2_index[dftup_uid]
    return qreq


@profile
@util.indent_decor('[nn_index]')
def ensure_nn_index(hs, qreq):
    print('checking flann')
    # NNIndexes depend on the data cxs AND feature / chip configs
    printDBG('qreq=%r' % (qreq,))
    # Indexed database (is actually qcxs for vsone)
    dcxs = qreq._internal_dcxs
    printDBG('dcxs=%r' % (dcxs,))

    feat_uid = qreq.cfg._feat_cfg.get_uid()
    dcxs_uid = util.hashstr_arr(dcxs, 'dcxs') + feat_uid
    if not dcxs_uid in qreq._dftup2_index:
        # Make sure the features are all computed first
        print('qreq.flann[dcxs_uid]... nn_index cache miss')
        print('dcxs_ is not in qreq cache')
        print('hashstr(dcxs_) = %r' % dcxs_uid)
        print('REFRESHING FEATURES')
        cx_list = np.unique(np.hstack([qreq._internal_dcxs, qreq._internal_qcxs]))
        hs.refresh_features(cx_list)  # Refresh query as well
        # Compute the FLANN Index
        data_index = ds.NNIndex(hs, dcxs)
        qreq._dftup2_index[dcxs_uid] = data_index
    else:
        print('qreq.flann[dcxs_uid]... cache hit')
    qreq._data_index = qreq._dftup2_index[dcxs_uid]


@util.indent_decor('[prep-qreq]')
def prep_query_request(hs, qreq=None, query_cfg=None, qcxs=None, dcxs=None, **kwargs):
    # Get the database chip indexes to query
    if dcxs is None:
        print('given [hs/user] sample? ... hs')
        dcxs = hs.get_indexed_sample()
    else:
        print('given [hs/user] sample? ... user')

    # Get the query request structure
    # Use the hotspotter query data if the user does not provide one
    if qreq is None:
        print('given [new/user] qreq? ... new')
        qreq = ds.QueryRequest()
    else:
        print('given [hs/user] qreq? ... user')

    # Use the hotspotter query_cfg if the user does not provide one
    #hs.assert_prefs()
    if query_cfg is None:
        print('given [hs/user] query_prefs? ... hs')
        query_cfg = hs.prefs.query_cfg
    else:
        print('given [hs/user] query_prefs? ... user')

    # Update any arguments in the query config based on kwargs
    if len(kwargs) > 0:
        print('given kwargs? ... yes')
        print(kwargs)
        query_cfg = query_cfg.deepcopy(**kwargs)
    else:
        print('given kwargs? ... no')

    # Check previously occuring bugs
    assert not isinstance(query_cfg, list)
    qreq.set_cfg(query_cfg, hs=hs)
    if qcxs is None:
        raise AssertionError('please query an index')
    if len(dcxs) == 0:
        raise AssertionError('please select database indexes')

    printDBG('qcxs=%r' % qcxs)
    printDBG('dcxs=%r' % dcxs)

    # The users actual query
    qreq._qcxs = qcxs
    qreq._dcxs = dcxs
    #---------------
    # Flip query / database if using vsone query
    query_type = qreq.cfg.agg_cfg.query_type
    if query_type == 'vsone':
        print('vsone query: swaping q/dcxs')
        (internal_dcxs, internal_qcxs) = (qreq._qcxs, qreq._dcxs)
        if len(qreq._qcxs) != 1:
            print(qreq)
            raise AssertionError('vsone only supports one query')
    elif query_type == 'vsmany':
        print('vsmany query')
        (internal_dcxs, internal_qcxs) = (qreq._dcxs, qreq._qcxs)
    else:
        raise AssertionError('Unknown query_type=%r' % query_type)
    # The flann indexed internel query
    return qreq


def load_cached_query(hs, qreq):
    qcxs = qreq._qcxs
    qcx2_res = mf.load_resdict(hs, qcxs, qreq)
    if qcx2_res is None:
        return None
    print(' ... query result cache hit')
    return qcx2_res


#----------------------
# Main Query Logic
#----------------------

@util.indent_decor('[QL2-qreq]')
def process_query_request(hs, qreq, dochecks=True):
    'wrapper that bypasses all that "qcx2_ map" buisness'
    print('[q3] process_query_request()')
    qcx2_res = execute_cached_query(hs, qreq)
    return qcx2_res


# Query Level 2
@util.indent_decor('[QL2-safe]')
def execute_query_safe(hs, qreq, qcxs, dcxs, use_cache=True):
    '''Executes a query, performs all checks, callable on-the-fly'''
    print('[q2] execute_query_safe()')
    qreq = prep_query_request(hs, qreq=qreq, qcxs=qcxs, dcxs=dcxs)
    return execute_cached_query(hs, qreq, use_cache=use_cache)


# Query Level 1
@util.indent_decor('[QL1]')
def execute_cached_query(hs, qreq, use_cache=True):
    print('[q1] execute_cached_query()')
    # Precache:
    qreq = pre_cache_checks(qreq)
    # Cache Load: Use the query result cache if possible
    if not params.args.nocache_query and use_cache:
        try:
            qcx2_res = mf.load_resdict(hs, qreq)
            return qcx2_res
        except IOError as ex:
            print(ex)
    # Prequery:
    qreq = pre_exec_checks(qreq)
    # Query:
    qcx2_res = execute_query_request_L0(hs, qreq)
    # Cache Save:
    [res.save(hs) for qcx, res in qcx2_res.iteritems()]
    return qcx2_res


# Query Level 1
@util.indent_decor('[QL1]')
def execute_unsafe_cached_query(hs, qreq, use_cache=True):
    # Cache Load: Use the query result cache if possible
    if not params.args.nocache_query and use_cache:
        try:
            qcx2_res = mf.load_resdict(hs, qreq)
            return qcx2_res
        except IOError as ex:
            print(ex)
    print('[q1] execute_unsafe_cached_query()')
    assert qreq._data_index is not None
    # Query:
    qcx2_res = execute_query_request_L0(hs, qreq)
    # Cache Save:
    [res.save(hs) for qcx, res in qcx2_res.iteritems()]
    return qcx2_res


def dbg_qreq(qreq):
    print('ERROR in dbg_qreq()')
    print('[q1] len(qreq._dftup2_index)=%r' % len(qreq._dftup2_index))
    print('[q1] qreq_dftup2_index._dftup2_index=%r' % len(qreq._dftup2_index))
    print('[q1] qreq._dftup2_index.keys()=%r' % qreq._dftup2_index.keys())
    print('[q1] qreq._data_index=%r' % qreq._data_index)


# Query Level 0
@profile
@util.indent_decor('[QL0]')
def execute_query_request_L0(hs, qreq):
    '''
    Driver logic of query pipeline
    Input:
        hs   - HotSpotter database object to be queried
        qreq - QueryRequest Object   # use prep_qreq to create one
    Output:
        qcx2_res - mapping from query indexes to QueryResult Objects
    '''
    # Query Chip Indexes
    # * vsone qcxs/dcxs swapping occurs here
    qcxs = qreq.get_qcxs()
    # Nearest neighbors (qcx2_nns)
    # * query descriptors assigned to database descriptors
    # * FLANN used here
    neighbs = mf.nearest_neighbors(hs, qcxs, qreq)
    # Nearest neighbors weighting and scoring (filt2_weights, filt2_meta)
    # * feature matches are weighted
    weights, filt2_meta = mf.weight_neighbors(hs, neighbs, qreq)
    # Thresholding and weighting (qcx2_nnfilter)
    # * feature matches are pruned
    nnfiltFILT = mf.filter_neighbors(hs, neighbs, weights, qreq)
    # Nearest neighbors to chip matches (qcx2_chipmatch)
    # * Inverted index used to create cx2_fmfsfk (TODO: ccx2_fmfv)
    # * Initial scoring occurs
    # * vsone inverse swapping occurs here
    matchesFILT = mf.build_chipmatches(hs, neighbs, nnfiltFILT, qreq)
    # Spatial verification (qcx2_chipmatch) (TODO: cython)
    # * prunes chip results and feature matches
    matchesSVER = mf.spatial_verification(hs, matchesFILT, qreq)
    # Query results format (qcx2_res) (TODO: SQL / Json Encoding)
    # * Final Scoring. Prunes chip results.
    # * packs into a wrapped query result object
    qcx2_res = mf.chipmatch_to_resdict(hs, matchesSVER, filt2_meta, qreq)
    return qcx2_res
