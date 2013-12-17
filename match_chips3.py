from __future__ import division, print_function
import __builtin__
# Python
import sys
import re
# HotSpotter
import helpers
from helpers import tic, toc
import tools
import dev
import DataStructures as ds
import matching_functions as mf

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write


def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write


def print_off():
    global print, print_

    def print(*args, **kwargs):
        pass

    def print_(*args, **kwargs):
        pass


def rrr():
    'Dynamic module reloading'
    import imp
    import sys
    print('[mc3] reloading ' + __name__)
    imp.reload(sys.modules[__name__])


#----------------------
# Convinience Functions
#----------------------

def query_dcxs(hs, qcx, dcxs, query_cfg=None, **kwargs):
    'wrapper that bypasses all that "qcx2_ map" buisness'
    if query_cfg is None:
        query_cfg = hs.prefs.query_cfg
    query_uid = ''.join(query_cfg.get_uid('noCHIP'))
    feat_uid = ''.join(query_cfg._feat_cfg.get_uid())
    query_hist_id = (query_uid, feat_uid)
    if hs.query_history[-1][0] != feat_uid:
        print('[mc3] need to reload features')
        hs.unload_cxdata('all')
        hs.refresh_features()
        hs.query_history.append(query_hist_id)
    elif hs.query_history[-1][1] != query_uid:
        print('[mc3] need to refresh features')
        hs.refresh_features()
        hs.query_history.append(query_hist_id)

    query_uid = ''.join(query_cfg.get_uid())
    print('[mc3] query_dcxs(): query_uid = %r ' % query_uid)
    result_list = execute_query_safe(hs, query_cfg, [qcx], dcxs, **kwargs)
    res = result_list[0].values()[0]
    return res


def query_groundtruth(hs, qcx, query_cfg=None, **kwargs):
    'wrapper that restricts query to only known groundtruth'
    print('[mc3] query groundtruth')
    gt_cxs = hs.get_other_indexed_cxs(qcx)
    print('[mc3] len(gt_cxs) = %r' % (gt_cxs,))
    return query_dcxs(hs, qcx, gt_cxs, query_cfg, **kwargs)


def query_database(hs, qcx, query_cfg=None, **kwargs):
    print('\n====================')
    print('[mc3] query database')
    print('====================')
    if hs.indexed_sample_cx is None:
        hs.set_samples()
    dcxs = hs.get_indexed_sample()
    return query_dcxs(hs, qcx, dcxs, query_cfg, **kwargs)


def make_nn_index(hs, sx2_cx=None):
    if sx2_cx is None:
        sx2_cx = hs.indexed_sample_cx
    data_index = ds.NNIndex(hs, sx2_cx)
    return data_index


def unify_cfgs(cfg_list):
    # Super HACK so all query configs share the same nearest neighbor indexes
    GLOBAL_dcxs2_index = cfg_list[0].dcxs2_index
    for query_cfg in cfg_list:
        query_cfg._dcxs2_index = GLOBAL_dcxs2_index


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


#----------------------
# Helper Functions
#----------------------
def ensure_nn_index(hs, query_cfg, dcxs):
    dcxs_ = tuple(dcxs)
    if not dcxs_ in query_cfg._dcxs2_index:
        # Make sure the features are all computed first
        hs.load_chips(dcxs_)
        hs.load_features(dcxs_)
        data_index = ds.NNIndex(hs, dcxs)
        query_cfg._dcxs2_index[dcxs_] = data_index
    query_cfg._data_index = query_cfg._dcxs2_index[dcxs_]


def prequery(hs, query_cfg=None, **kwargs):
    if query_cfg is None:
        query_cfg = ds.QueryConfig(hs, **kwargs)
    if query_cfg.agg_cfg.query_type == 'vsmany':
        dcxs = hs.indexed_sample_cx
        ensure_nn_index(hs, query_cfg, dcxs)
    return query_cfg


def load_cached_query(hs, query_cfg, aug_list=['']):
    qcxs = query_cfg._qcxs
    result_list = []
    for aug in aug_list:
        qcx2_res = mf.load_resdict(hs, qcxs, query_cfg, aug)
        if qcx2_res is None:
            return None
        result_list.append(qcx2_res)
    print('[mc3] ... query result cache hit\n')
    return result_list


#----------------------
# Main Query Logic
#----------------------
def execute_query_safe(hs, query_cfg=None, qcxs=None, dcxs=None, use_cache=True, **kwargs):
    '''Executes a query, performs all checks, callable on-the-fly'''
    print('[mc3] Execute query safe: q%s' % hs.cxstr(qcxs))
    if query_cfg is None:
        query_cfg = ds.QueryConfig(hs, **kwargs)
    if dcxs is None:
        dcxs = hs.indexed_sample_cx
    query_cfg._qcxs = qcxs
    query_cfg._dcxs = dcxs
    #---------------
    # Flip if needebe
    query_type = query_cfg.agg_cfg.query_type
    if query_type == 'vsone':
        (dcxs, qcxs) = (query_cfg._qcxs, query_cfg._dcxs)
    elif query_type == 'vsmany':
        (dcxs, qcxs) = (query_cfg._dcxs, query_cfg._qcxs)
    else:
        raise Exception('Unknown query_type=%r' % query_type)
    # caching
    if not hs.args.nocache_query:
        result_list = load_cached_query(hs, query_cfg)
        if not result_list is None:
            return result_list
    print('[mc3] qcxs=%r' % query_cfg._qcxs)
    print('[mc3] len(dcxs)=%r' % len(query_cfg._dcxs))
    ensure_nn_index(hs, query_cfg, dcxs)
    result_list = execute_query_fast(hs, query_cfg, qcxs, dcxs)
    for qcx2_res in result_list:
        for qcx, res in qcx2_res.iteritems():
            res.save(hs)
    return result_list


def execute_query_fast(hs, query_cfg, qcxs, dcxs):
    '''Executes a query and assumes query_cfg has all precomputed information'''
    # Nearest neighbors
    nn_tt = tic()
    neighbs = mf.nearest_neighbors(hs, qcxs, query_cfg)
    nn_time = toc(nn_tt)
    # Nearest neighbors weighting and scoring
    weight_tt = tic()
    weights  = mf.weight_neighbors(hs, neighbs, query_cfg)
    weight_time = toc(weight_tt)
    # Thresholding and weighting
    filt_tt = tic()
    nnfiltFILT = mf.filter_neighbors(hs, neighbs, weights, query_cfg)
    filt_time = toc(filt_tt)
    # Nearest neighbors to chip matches
    build_tt = tic()
    matchesFILT = mf.build_chipmatches(hs, neighbs, nnfiltFILT, query_cfg)
    build_time = toc(build_tt)
    # Spatial verification
    verify_tt = tic()
    matchesSVER = mf.spatial_verification(hs, matchesFILT, query_cfg)
    verify_time = toc(verify_tt)
    # Query results format
    result_list = [
        mf.chipmatch_to_resdict(hs, matchesSVER, query_cfg),
    ]
    # Add timings to the results
    for res in result_list[0].itervalues():
        res.nn_time     = nn_time
        res.weight_time = weight_time
        res.filt_time   = filt_time
        res.build_time  = build_time
        res.verify_time = verify_time
    return result_list


if __name__ == '__main__':
    import draw_func2 as df2
    #exec(open('match_chips3.py').read())
    df2.rrr()
    df2.reset()
    mf.rrr()
    ds.rrr()
    main_locals = dev.dev_main()
    execstr = helpers.execstr_dict(main_locals, 'main_locals')
    exec(execstr)
    #df2.DARKEN = .5
    df2.DISTINCT_COLORS = True
    exec(df2.present())
