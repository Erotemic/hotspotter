from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[mc3]', DEBUG=False)
# Python
import re
# HotSpotter
from hscom import params
from hscom import helpers
from hscom import helpers as util
import DataStructures as ds
import matching_functions as mf


@profile
def ensure_nn_index(hs, qdat, dcxs, force_refresh=False):
    # NNIndexes depend on the data cxs AND feature / chip configs
    printDBG('qdat=%r' % (qdat,))
    printDBG('dcxs=%r' % (dcxs,))
    printDBG('force_refresh=%r' % (force_refresh,))

    feat_uid = qdat.cfg._feat_cfg.get_uid()
    dcxs_uid = helpers.hashstr_arr(dcxs, 'dcxs') + feat_uid
    if not dcxs_uid in qdat._dcxs2_index:
        # Make sure the features are all computed first
        print('[mc3] qdat._data_index[dcxs_uid]... nn_index cache miss')
        print('[mc3] dcxs_ is not in qdat cache')
        print('[mc3] hashstr(dcxs_) = %r' % dcxs_uid)
        print('[mc3] REFRESHING FEATURES')
        with helpers.Indenter('    [ensure_nn_index]'):
            hs.refresh_features(dcxs)
            # Compute the FLANN Index
            data_index = ds.NNIndex(hs, dcxs)
        qdat._dcxs2_index[dcxs_uid] = data_index
    else:
        print('[mc3] qdat._data_index[dcxs_uid]... cache hit')
    qdat._data_index = qdat._dcxs2_index[dcxs_uid]


def prequery_checks(hs, qdat):
    dcxs = qdat._dcxs
    qcxs = qdat._qcxs
    query_cfg = qdat.cfg
    query_uid = qdat.cfg.get_uid('noCHIP')
    feat_uid = qdat.cfg._feat_cfg.get_uid()
    query_hist_id = (feat_uid, query_uid)
    def _refresh(hs, qdat, unload=True):
        if unload:
            #print('[mc3] qdat._dcxs = %r' % qdat._dcxs)
            with helpers.Indenter('    [prequery_checks]'):
                hs.unload_cxdata('all')
            # Reload
            qdat = prep_query_request(hs, query_cfg=query_cfg, qcxs=qcxs, dcxs=dcxs)
        with helpers.Indenter('    [prequery_checks]'):
            ensure_nn_index(hs, qdat, qdat._dcxs, force_refresh=True)

    if hs.query_history[-1][0] is None:
        # FIRST LOAD:
        print('[mc3] FIRST LOAD. Need to reload features')
        print('[mc3] ensuring nn index')
        _refresh(hs, qdat)

    elif hs.query_history[-1][0] != feat_uid:
        print('[mc3] FEAT_UID is different. Need to reload features')
        print('[mc3] Old: ' + str(hs.query_history[-1][0]))
        print('[mc3] New: ' + str(feat_uid))
        _refresh(hs, qdat)
    elif hs.query_history[-1][1] != query_uid:
        print('[mc3] QUERY_UID is different. Need to refresh features')
        print('[mc3] Old: ' + str(hs.query_history[-1][1]))
        print('[mc3] New: ' + str(query_uid))
        _refresh(hs, False)
    hs.query_history.append(query_hist_id)
    print('[mc3] prequery_checks(): query_uid = %r ' % query_uid)


#----------------------
# Convinience Functions
#----------------------

# QUERY PREP FUNCTIONS

def prep_query_request(hs, qdat=None, query_cfg=None, qcxs=None, dcxs=None, **kwargs):
    printDBG('[mc3]  prep_query_request ---------------')
    printDBG('hs=%r' % hs)
    printDBG('query_cfg=%r' % query_cfg)
    printDBG('qdat=%r' % qdat)
    printDBG('dcxs=%r' % dcxs)
    printDBG('qcxs=%r' % qcxs)
    printDBG('[mc3]---------------')
    if dcxs is None:
        dcxs = hs.get_indexed_sample()
    if qdat is None:
        qdat = hs.qdat
        qdat._dcxs = dcxs
    if query_cfg is None:
        hs.assert_prefs()
        query_cfg = hs.prefs.query_cfg
    if len(kwargs) > 0:
        query_cfg = query_cfg.deepcopy(**kwargs)
    assert not isinstance(query_cfg, list)
    qdat.set_cfg(query_cfg, hs=hs)
    if qcxs is None:
        raise Exception('please query an index')
    if len(dcxs) == 0:
        raise Exception('please select database indexes')
    printDBG('qcxs=%r' % qcxs)
    printDBG('dcxs=%r' % dcxs)
    qdat._qcxs = qcxs
    qdat._dcxs = dcxs
    #---------------
    # Flip if needebe
    query_type = qdat.cfg.agg_cfg.query_type
    if query_type == 'vsone':
        (dcxs, qcxs) = (qdat._qcxs, qdat._dcxs)
    elif query_type == 'vsmany':
        (dcxs, qcxs) = (qdat._dcxs, qdat._qcxs)
    else:
        raise Exception('Unknown query_type=%r' % query_type)
    return qdat


# QUERY EXEC FUNCTIONS

def query_list(hs, qcxs, dcxs=None, **kwargs):
    qdat = prep_query_request(hs, qcxs=qcxs, dcxs=dcxs, **kwargs)
    qcx2_res = execute_query_lazy(hs, qdat, qcxs, dcxs)[0]
    return qcx2_res


def query_dcxs(hs, qcx, dcxs, qdat, dochecks=True):
    'wrapper that bypasses all that "qcx2_ map" buisness'
    result_list = execute_query_safe(hs, qdat, [qcx], dcxs)
    res = result_list[0].values()[0]
    return res


def process_query_request(hs, qdat, dochecks=True):
    'wrapper that bypasses all that "qcx2_ map" buisness'
    print('[mc3] process_query_request()')
    if dochecks:
        prequery_checks(hs, qdat)
    dcxs = qdat._dcxs
    qcxs = qdat._qcxs
    qcxs = qdat._qcxs
    result_list = execute_query_safe(hs, qdat, qcxs, dcxs)
    res = result_list[0]
    return res


# OTHER

def make_nn_index(hs, sx2_cx=None):
    if sx2_cx is None:
        sx2_cx = hs.indexed_sample_cx
    data_index = ds.NNIndex(hs, sx2_cx)
    return data_index


def simplify_test_uid(test_uid):
    # Remove extranious characters from test_uid
    #test_uid = re.sub(r'_trainID\([0-9]*,........\)','', test_uid)
    #test_uid = re.sub(r'_indxID\([0-9]*,........\)','', test_uid)
    test_uid = re.sub(r'_dcxs([^)]*)', '', test_uid)
    #test_uid = re.sub(r'HSDB_zebra_with_mothers','', test_uid)
    #test_uid = re.sub(r'GZ_ALL','', test_uid)
    #test_uid = re.sub(r'_sz750','', test_uid)
    #test_uid = re.sub(r'_FEAT([^(]*)','', test_uid)
    #test_uid = test_uid.strip(' _')
    return test_uid


def load_cached_query(hs, qdat, aug_list=['']):
    qcxs = qdat._qcxs
    result_list = []
    for aug in aug_list:
        qcx2_res = mf.load_resdict(hs, qcxs, qdat, aug)
        if qcx2_res is None:
            return None
        result_list.append(qcx2_res)
    print('[mc3] ... query result cache hit')
    return result_list


#----------------------
# Main Query Logic
#----------------------
@profile
def execute_query_safe(hs, qdat, qcxs, dcxs, use_cache=True):
    '''Executes a query, performs all checks, callable on-the-fly'''
    qdat = prep_query_request(hs, qdat=qdat, qcxs=qcxs, dcxs=dcxs)
    ensure_nn_index(hs, qdat, qdat._dcxs)
    return execute_query_lazy(hs, qdat, qcxs, dcxs, use_cache=use_cache)


def execute_query_lazy(hs, qdat, qcxs, dcxs, use_cache=True):
    # caching
    if not params.args.nocache_query and use_cache:
        result_list = load_cached_query(hs, qdat)
        if not result_list is None:
            return result_list
    print('[mc3] qcxs=%r' % qdat._qcxs)
    print('[mc3] len(dcxs)=%r' % len(qdat._dcxs))
    # Do the actually query
    result_list = execute_query_fast(hs, qdat, qcxs, dcxs)
    for qcx2_res in result_list:
        for qcx, res in qcx2_res.iteritems():
            res.save(hs)
    return result_list


@profile
def execute_query_fast(hs, qdat, qcxs, dcxs):
    '''Executes a query and assumes qdat has all precomputed information'''
    # Nearest neighbors
    neighbs = mf.nearest_neighbors(hs, qcxs, qdat)
    # Nearest neighbors weighting and scoring
    weights, filt2_meta = mf.weight_neighbors(hs, neighbs, qdat)
    # Thresholding and weighting
    nnfiltFILT = mf.filter_neighbors(hs, neighbs, weights, qdat)
    # Nearest neighbors to chip matches
    matchesFILT = mf.build_chipmatches(hs, neighbs, nnfiltFILT, qdat)
    # Spatial verification
    matchesSVER = mf.spatial_verification(hs, matchesFILT, qdat)
    # Query results format
    result_list = [
        mf.chipmatch_to_resdict(hs, matchesSVER, filt2_meta, qdat),
    ]
    return result_list


#if __name__ == '__main__':
    #import multiprocessing
    #multiprocessing.freeze_support()
    #from hsviz import draw_func2 as df2
    #import dev
    ##exec(open('match_chips3.py').read())
    #df2.rrr()
    #df2.reset()
    #mf.rrr()
    #ds.rrr()
    #main_locals = dev.dev_main()
    #execstr = helpers.execstr_dict(main_locals, 'main_locals')
    #exec(execstr)
    ##df2.DARKEN = .5
    #df2.DISTINCT_COLORS = True
    #exec(df2.present())
