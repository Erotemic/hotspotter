from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[mc3]', DEBUG=False)
# HotSpotter
from hscom import params
from hscom import helpers as util
import DataStructures as ds
import matching_functions as mf


@util.indent_decor('[quick_ensure]')
def quickly_ensure_qreq(hs, qcxs=None, dcxs=None):
    qreq = hs.qreq
    query_cfg = hs.prefs.query_cfg
    cxs = hs.get_indexed_sample()
    if qcxs is None:
        qcxs = cxs
    if dcxs is None:
        dcxs = cxs
    qreq = prep_query_request(qreq=qreq, query_cfg=query_cfg,
                              qcxs=qcxs, dcxs=dcxs)
    pre_cache_checks(hs, qreq)
    pre_exec_checks(hs, qreq)
    return qreq


@util.indent_decor('[prep_qreq]')
def prep_query_request(qreq=None, query_cfg=None, qcxs=None, dcxs=None, **kwargs):
    # Builds or modifies a query request object
    def loggedif(msg, condition):
        # helper function for logging if statment results
        print(msg + '... ' + ['no', 'yes'][condition])
        return condition
    if loggedif('1) given qreq?', qreq is not None):
        qreq = ds.QueryRequest()
    if loggedif('2) given qcxs?', qcxs is not None):
        qreq._qcxs = qcxs
    if loggedif('3) given dcxs?', dcxs is not None):
        qreq._dcxs = dcxs
    if not loggedif('4) given qcfg?', query_cfg is not None):
        query_cfg = qreq.cfg
    if loggedif('4) given kwargs?', len(kwargs) > 0):
        query_cfg = query_cfg.deepcopy(**kwargs)
    #
    qreq.set_cfg(query_cfg)
    #
    assert (qreq._qcxs is not None and len(qreq._qcxs) > 0), (
        'query request has invalid query chip indexes')
    assert (qreq._dcxs is not None and len(qreq._dcxs) > 0), (
        'query request has invalid database chip indexes')
    assert (qreq.cfg is not None), (
        'query request has invalid query config')
    return qreq


#----------------------
# Query and database checks
#----------------------


@profile
@util.indent_decor('[pre_cache]')
def pre_cache_checks(hs, qreq):
    print(' --- pre cache checks --- ')
    # Ensure hotspotter object is using the right config
    hs.attatch_qreq(qreq)
    feat_uid = qreq.cfg._feat_cfg.get_uid()
    # Load any needed features or chips into memory
    if hs.feats.feat_uid != feat_uid:
        hs.unload_cxdata('all')
    hs.refresh_features(qreq._dcxs)
    hs.refresh_features(qreq._qcxs)
    return qreq


@profile
@util.indent_decor('[pre_exec]')
def pre_exec_checks(hs, qreq):
    print(' --- pre query request execution checks --- ')
    # Get qreq config information
    dcxs = qreq.get_interanl_dcxs()
    feat_uid = qreq.cfg._feat_cfg.get_uid()
    dcxs_uid = util.hashstr_arr(dcxs, 'dcxs')
    # Ensure the index / inverted index exist for this config
    dftup_uid = (dcxs_uid, feat_uid)
    if not dftup_uid in qreq._dftup2_index:
        print('qreq.flann[dcxs_uid]... nn_index cache miss')
        # Compute the FLANN Index
        data_index = ds.NNIndex(hs, dcxs)
        qreq._dftup2_index[dftup_uid] = data_index
    qreq._data_index = qreq._dftup2_index[dftup_uid]
    return qreq


#----------------------
# Main Query Logic
#----------------------

# Query Level 1
@util.indent_decor('[QL1]')
def process_query_request(hs, qreq, use_cache=True):
    print('[q1] process_query_request()')
    qreq = pre_cache_checks(hs, qreq)
    if not params.args.nocache_query and use_cache:
        try:
            qcx2_res = mf.load_resdict(hs, qreq)  # load cache
            return qcx2_res
        except IOError as ex:
            print(ex)
    qreq = pre_exec_checks(hs, qreq)
    qcx2_res = execute_query_request_L0(hs, qreq)  # exec query
    [res.save(hs) for qcx, res in qcx2_res.iteritems()]  # save cache
    return qcx2_res


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
    qcxs = qreq.get_internal_qcxs()
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
