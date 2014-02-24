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
@util.indent_decor('[nn_index]')
def ensure_nn_index(hs, qdat):
    print('checking flann')
    # NNIndexes depend on the data cxs AND feature / chip configs
    printDBG('qdat=%r' % (qdat,))
    # Indexed database (is actually qcxs for vsone)
    dcxs = qdat._internal_dcxs
    printDBG('dcxs=%r' % (dcxs,))

    feat_uid = qdat.cfg._feat_cfg.get_uid()
    dcxs_uid = util.hashstr_arr(dcxs, 'dcxs') + feat_uid
    if not dcxs_uid in qdat._dcxs2_index:
        # Make sure the features are all computed first
        print('qdat.flann[dcxs_uid]... nn_index cache miss')
        print('dcxs_ is not in qdat cache')
        print('hashstr(dcxs_) = %r' % dcxs_uid)
        print('REFRESHING FEATURES')
        cx_list = np.unique(np.hstack([qdat._internal_dcxs, qdat._internal_qcxs]))
        hs.refresh_features(cx_list)  # Refresh query as well
        # Compute the FLANN Index
        data_index = ds.NNIndex(hs, dcxs)
        qdat._dcxs2_index[dcxs_uid] = data_index
    else:
        print('qdat.flann[dcxs_uid]... cache hit')
    qdat._data_index = qdat._dcxs2_index[dcxs_uid]


@util.indent_decor('[prep-qreq]')
def prep_query_request(hs, qdat=None, query_cfg=None, qcxs=None, dcxs=None, **kwargs):
    printDBG('prep_query_request()')
    printDBG('hs=%r' % hs)
    printDBG('query_cfg=%r' % query_cfg)
    printDBG('qdat=%r' % qdat)
    printDBG('dcxs=%r' % dcxs)
    printDBG('qcxs=%r' % qcxs)

    # Get the database chip indexes to query
    if dcxs is None:
        print('given [hs/user] sample? ... hs')
        dcxs = hs.get_indexed_sample()
    else:
        print('given [hs/user] sample? ... user')

    # Get the query request structure
    # Use the hotspotter query data if the user does not provide one
    if qdat is None:
        qdat = hs.qdat
        qdat._dcxs = dcxs
        print('given [hs/user] qdat? ... hs')
    else:
        print('given [hs/user] qdat? ... user')

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
    qdat.set_cfg(query_cfg, hs=hs)
    if qcxs is None:
        raise AssertionError('please query an index')
    if len(dcxs) == 0:
        raise AssertionError('please select database indexes')

    printDBG('qcxs=%r' % qcxs)
    printDBG('dcxs=%r' % dcxs)

    # The users actual query
    qdat._qcxs = qcxs
    qdat._dcxs = dcxs
    #---------------
    # Flip query / database if using vsone query
    query_type = qdat.cfg.agg_cfg.query_type
    if query_type == 'vsone':
        print('vsone query: swaping q/dcxs')
        (internal_dcxs, internal_qcxs) = (qdat._qcxs, qdat._dcxs)
        if len(qdat._qcxs) != 1:
            print(qdat)
            raise AssertionError('vsone only supports one query')
    elif query_type == 'vsmany':
        print('vsmany query')
        (internal_dcxs, internal_qcxs) = (qdat._dcxs, qdat._qcxs)
    else:
        raise AssertionError('Unknown query_type=%r' % query_type)
    # The flann indexed internel query
    qdat._internal_qcxs = internal_qcxs
    qdat._internal_dcxs = internal_dcxs
    return qdat


@util.indent_decor('[qcheck]')
def prequery_checks(hs, qdat):
    # Checks that happen JUST before querytime
    dcxs = qdat._dcxs
    qcxs = qdat._qcxs
    query_cfg = qdat.cfg
    query_uid = qdat.cfg.get_uid('noCHIP')
    feat_uid = qdat.cfg._feat_cfg.get_uid()
    query_hist_id = (feat_uid, query_uid)

    @util.indent_decor('[refresh]')
    def _refresh(hs, qdat, unload=False):
        print('_refresh, unload=%r' % unload)
        if unload:
            #print('[mc3] qdat._dcxs = %r' % qdat._dcxs)
            hs.unload_cxdata('all')
            # Reload
            qdat = prep_query_request(hs, query_cfg=query_cfg, qcxs=qcxs, dcxs=dcxs)
        ensure_nn_index(hs, qdat)
        return qdat

    print('checking')
    if hs.query_history[-1][0] is None:
        # FIRST LOAD:
        print('FIRST LOAD. Need to reload features')
        qdat = _refresh(hs, qdat, unload=hs.dirty)
    elif hs.query_history[-1][0] != feat_uid:
        print('FEAT_UID is different. Need to reload features')
        print('Old: ' + str(hs.query_history[-1][0]))
        print('New: ' + str(feat_uid))
        qdat = _refresh(hs, qdat, True)
    elif hs.query_history[-1][1] != query_uid:
        print('QUERY_UID is different. Need to refresh features')
        print('Old: ' + str(hs.query_history[-1][1]))
        print('New: ' + str(query_uid))
        qdat = _refresh(hs, qdat, False)
    elif qdat._data_index is None:
        print('qdat._data_index is None. Need to refresh nn_index')
        qdat = _refresh(hs, qdat, False)

    print('checked')
    hs.query_history.append(query_hist_id)
    print('prequery(): feat_uid = %r ' % feat_uid)
    print('prequery(): query_uid = %r ' % query_uid)
    return qdat


def load_cached_query(hs, qdat):
    qcxs = qdat._qcxs
    qcx2_res = mf.load_resdict(hs, qcxs, qdat)
    if qcx2_res is None:
        return None
    print(' ... query result cache hit')
    return qcx2_res


#----------------------
# Main Query Logic
#----------------------
@profile
# Query Level 3
@util.indent_decor('[QL3-dcxs]')
def query_dcxs(hs, qcx, dcxs, qdat, dochecks=True):
    'wrapper that bypasses all that "qcx2_ map" buisness'
    print('[q3] query_dcxs()')
    qcx2_res = execute_query_safe(hs, qdat, [qcx], dcxs)
    res = qcx2_res.values()[0]
    return res


# Query Level 3
@util.indent_decor('[QL2-qreq]')
def process_query_request(hs, qdat, dochecks=True):
    'wrapper that bypasses all that "qcx2_ map" buisness'
    print('[q3] process_query_request()')
    qcx2_res = execute_cached_query(hs, qdat)
    return qcx2_res


# Query Level 2
@util.indent_decor('[QL2-safe]')
def execute_query_safe(hs, qdat, qcxs, dcxs, use_cache=True):
    '''Executes a query, performs all checks, callable on-the-fly'''
    print('[q2] execute_query_safe()')
    qdat = prep_query_request(hs, qdat=qdat, qcxs=qcxs, dcxs=dcxs)
    return execute_cached_query(hs, qdat, use_cache=use_cache)


# Query Level 2
@util.indent_decor('[QL2-list]')
def query_list(hs, qcxs, dcxs=None, **kwargs):
    print('[q2] query_list()')
    qdat = prep_query_request(hs, qcxs=qcxs, dcxs=dcxs, **kwargs)
    qcx2_res = execute_cached_query(hs, qdat)
    return qcx2_res


# Query Level 1
@util.indent_decor('[QL1]')
def execute_cached_query(hs, qdat, use_cache=True):
    # caching
    if not params.args.nocache_query and use_cache:
        qcx2_res = load_cached_query(hs, qdat)
        if qcx2_res is not None:
            return qcx2_res
    print('[q1] execute_cached_query()')
    qdat = prequery_checks(hs, qdat)
    #ensure_nn_index(hs, qdat)
    if qdat._data_index is None:
        print('ERROR in execute_cached_query()')
        print('[q1] len(qdat._dcxs2_index)=%r' % len(qdat._dcxs2_index))
        print('[q1] qdat_dcxs2_index._dcxs2_index=%r' % len(qdat._dcxs2_index))
        print('[q1] qdat._dcxs2_index.keys()=%r' % qdat._dcxs2_index.keys())
        print('[q1] qdat._data_index=%r' % qdat._data_index)
        raise Exception('Data index cannot be None at query time')
    # Do the actually query
    qcx2_res = execute_query_fast(hs, qdat)
    for qcx, res in qcx2_res.iteritems():
        res.save(hs)
    return qcx2_res


# Query Level 1
@util.indent_decor('[QL1]')
def execute_unsafe_cached_query(hs, qdat, use_cache=True):
    # caching
    if not params.args.nocache_query and use_cache:
        qcx2_res = load_cached_query(hs, qdat)
        if qcx2_res is not None:
            return qcx2_res
    print('[q1] execute_unsafe_cached_query()')
    #ensure_nn_index(hs, qdat)
    if qdat._data_index is None:
        print('ERROR in execute_unsafe_cached_query()')
        print('[q1] len(qdat._dcxs2_index)=%r' % len(qdat._dcxs2_index))
        print('[q1] qdat_dcxs2_index._dcxs2_index=%r' % len(qdat._dcxs2_index))
        print('[q1] qdat._dcxs2_index.keys()=%r' % qdat._dcxs2_index.keys())
        print('[q1] qdat._data_index=%r' % qdat._data_index)
        raise Exception('Data index cannot be None at query time')
    # Do the actually query
    qcx2_res = execute_query_fast(hs, qdat)
    for qcx, res in qcx2_res.iteritems():
        res.save(hs)
    return qcx2_res


@profile
# Query Level 0
@util.indent_decor('[QL0]')
def execute_query_fast(hs, qdat):
    '''
    Executes a query and assumes qdat has all precomputed information
    each of the returned objects is a mapping from a query index
    to whatever the information is
    '''
    qcxs = qdat._internal_qcxs
    # Nearest neighbors
    neighbs = mf.nearest_neighbors(hs, qcxs, qdat)
    # Nearest neighbors weighting and scoring
    weights, filt2_meta = mf.weight_neighbors(hs, neighbs, qdat)
    # Thresholding and weighting
    nnfiltFILT = mf.filter_neighbors(hs, neighbs, weights, qdat)
    # Nearest neighbors to chip matches (swaps vsone)
    matchesFILT = mf.build_chipmatches(hs, neighbs, nnfiltFILT, qdat)
    # Spatial verification
    matchesSVER = mf.spatial_verification(hs, matchesFILT, qdat)
    # Query results format
    qcx2_res = mf.chipmatch_to_resdict(hs, matchesSVER, filt2_meta, qdat)
    return qcx2_res
