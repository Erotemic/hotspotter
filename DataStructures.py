from __future__ import division, print_function
import params
import helpers
import numpy as np
import pyflann
import algos
from itertools import izip, chain
from Printable import DynStruct
import draw_func2 as df2

def reload_module():
    import imp, sys
    print('reloading '+__name__)
    imp.reload(sys.modules[__name__])

def rrr():
    reload_module()

#=========================
# NN (FLANN) Index Class
#=========================
FM_DTYPE  = np.uint32
FS_DTYPE  = np.float32
class NNIndex(object):
    def __init__(nn_index, hs, sx2_cx):
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
        # Agg Data
        nn_index.ax2_cx   = ax2_cx
        nn_index.ax2_fx   = ax2_fx
        nn_index.ax2_data = ax2_desc
        nn_index.flann = flann
    def __del__(nn_index):
        if not nn_index.flann:
            nn_index.flann.delete_index()
#=========================
# Query Result Class
#=========================
def query_result_fpath(hs, qcx, query_uid=None):
    if query_uid is None: query_uid = params.get_query_uid()

    qres_dir  = hs.dirs.qres_dir 
    fname = 'result_%s_qcx=%d.npz' % (query_uid, qcx)
    fpath = os.path.join(qres_dir, fname)
    return fpath

def query_result_exists(hs, qcx, query_uid=None):
    fpath = query_result_fpath(hs, qcx, query_uid)
    return os.path.exists(fpath)

class QueryResult(DynStruct):
    def __init__(res, qcx, uid):
        super(QueryResult, res).__init__()
        res.qcx       = qcx
        res.query_uid = uid
        # Times
        res.assign_time = -1
        res.verify_time = -1
        # Assigned features matches
        res.cx2_fm = np.array([], dtype=FM_DTYPE)
        res.cx2_fs = np.array([], dtype=FS_DTYPE)
        res.cx2_score = np.array([])
        res.cx2_fm_V = np.array([], dtype=FM_DTYPE)
        res.cx2_fs_V = np.array([], dtype=FS_DTYPE)
        res.cx2_score_V = np.array([])

    def has_cache(res, hs):
        return query_result_exists(hs, res.qcx)

    def get_fpath(res, hs):
        return query_result_fpath(hs, res.qcx, res.query_uid)
    
    def save(res, hs):
        fpath = res.get_fpath(hs)
        if params.VERBOSE_CACHE:
            print('[mc2] caching result: %r' % (fpath,))
        else:
            print('[mc2] caching result: %r' % (os.path.split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            np.savez(file_, **res.__dict__.copy())
        return True

    def load(res, hs):
        'Loads the result from the given database'
        fpath = os.path.normpath(res.get_fpath(hs))
        try:
            with open(fpath, 'rb') as file_:
                npz = np.load(file_)
                for _key in npz.files:
                    res.__dict__[_key] = npz[_key]
                npz.close()
            res.qcx = res.qcx.tolist()
            res.query_uid = str(res.query_uid)
            return True
        except Exception as ex:
            #os.remove(fpath)
            warnmsg = ('Load Result Exception : ' + repr(ex) + 
                    '\nResult was corrupted for qcx=%d' % res.qcx)
            print(warnmsg)
            warnings.warn(warnmsg)
            raise
    def get_SV(res):
        return res.cx2_fm_V.size > 0
    def cache_bytes(res, hs):
        fpath = res.get_fpath(hs)
        return helpers.file_bytes(fpath)
    def top5_cxs(res):
        return res.topN_cxs(5)
    def get_cx2_score(res, SV):
        return res.cx2_score_V if SV else res.cx2_score
    def get_cx2_fm(res, SV):
        return res.cx2_fm_V if SV else res.cx2_fm
    def get_cx2_fs(res, SV):
        return res.cx2_fs_V if SV else res.cx2_fs
    def topN_cxs(res, N, SV=None):
        if SV is None:
            SV = res.get_SV()
        cx2_score = res.get_cx2_score(SV)
        top_cxs = cx2_score.argsort()[::-1]
        num_top = min(N, len(top_cxs))
        topN_cxs = top_cxs[0:num_top]
        return topN_cxs
    def show_query(res, hs, **kwargs):
        df2.show_chip(hs, res=res, **kwargs)
    def show_topN(res, hs, **kwargs):
        df2.show_topN_matches(hs, res, **kwargs)

#=========================
# Param Classes
#=========================
#def udpate_dicts(dict1, dict2):
    #dict1_keys = set(dict1.keys())
    #if key, val in dict2.iteritems():
        #if key in dict1_keys:
            #dict1[key] = val

class NNParams(DynStruct):
    def __init__(nn_params, **kwargs):
        super(NNParams, nn_params).__init__()
        # Core
        nn_params.K = 2
        nn_params.Knorm = 1
        # Filters
        nn_params.checks = 128
        nn_params.update(**kwargs)
    def get_uid(nn_params):
        uid = '_nn(' 
        uid += 'K='+str(nn_params.K)
        uid += ',Kn='+str(nn_params.Knorm)
        uid += ',cks=' + str(nn_params.checks)
        uid += ')'
        return uid

def dict_subset(dict_, keys):
    keys_ = set(keys)
    return {key:val for (key,val) in dict_.iteritems() if key in keys_}
def listrm(list_, item):
    try: 
        list_.remove(item)
    except Exception:
        pass
def listrm_list(list_, items):
    for item in items: listrm(list_, item)


valid_filters = ['recip', 
                 'roidist',
                 'frexquency',
                 'ratio',
                 'bursty', 
                 'lnbnn']

def any_inlist(list_, search_list):
    set_ = set(list_)
    return any([search in set_ for search in search_list])

class ScoreMechanismParams(DynStruct):
    # Rename to scoring mechanism
    def __init__(score_params, **kwargs):
        super(ScoreMechanismParams, score_params).__init__()
        smp = score_params
        smp.aggregation_method = 'ChipSum' # ['NameSum', 'NamePlacketLuce']
        smp.nnfilter_list = ['recip', 'roidist']
        #
        smp.nnfilter_list = ['recip', 'roidist', 'lnbnn']
        smp.roidist_thresh = .5
        smp.ratio_thresh   = 1.2
        smp.burst_thresh   = 1

        smp.filt2_tw = { 
            #tuple(ValidSignThresh, ScoreWeight)
            'roidist'    : ((+1, smp.roidist_thresh), 0),
            'recip'      : ((+1,    0), 0), 
            #'scale'      : ((+1, None), 0),
            'bursty'     : ((+1, smp.burst_thresh), 0),
            'ratio'      : ((-1, smp.ratio_thresh), 1), 
            'lnbnn'      : ((-1, None), 1),
            'lnrat'      : ((-1, None), 1),
        }
        smp.Krecip = 1 # 0 := off
        smp.update(**kwargs)

    def make_feasible(score_params, nn_params):
        nnfilts = score_params.nnfilter_list
        nnp = nn_params
        smp = score_params
        # Knorm
        norm_depends = ['lnbnn', 'ratio', 'lnrat']
        if nnp.Knorm == 0 and not any_inlist(nnfilts, norm_depends):
            listrm_list(nnfilts, norm_depends)
            nnp.Knorm = 0
        # Krecip
        if smp.Krecip == 0 or 'recip' not in nnfilts:
            listrm(nnfilts, 'recip')
            smp.Krecip = 0

    def get_uid(score_params):
        on_filters = dict_subset(score_params.filt2_tw,
                                 score_params.nnfilter_list)
        uid = '_smech(' 
        if score_params.Krecip != 0:
            uid += 'Kr='+str(score_params.Krecip)
        uid += ',' + score_params.aggregation_method
        uid += ',' + signthreshweight_str(on_filters)
        uid += ')'
        return uid

def signthreshweight_str(on_filters):
    stw_list = []
    for key, val in on_filters.iteritems():
        (sign, thresh), weight = val
        stw_str = '('+key
        if not thresh is None:
            sstr = ['<','>'][sign == -1] #actually <=, >=
            stw_str += sstr+str(thresh)
        if weight != 0:
            stw_str += ','+str(weight)
        stw_list.append(stw_str+')')
    return ','.join(stw_list)
    #return helpers.remove_chars(str(dict_), [' ','\'','}','{',':'])

class SpatialVerifyParams(DynStruct):
    def __init__(sv_params, **kwargs):
        super(SpatialVerifyParams, sv_params).__init__()
        sv_params.use_chip_extent = False
        sv_params.scale_thresh  = (.5, 2)
        sv_params.xy_thresh = .002
        sv_params.shortlist_len = 100
        sv_params.update(**kwargs)
    def get_uid(sv_params):
        uid = '_sv(' 
        uid += str(sv_params.shortlist_len)
        uid += ',' + str(sv_params.xy_thresh)
        uid += ',' + str(sv_params.scale_thresh)
        uid += ',cdl' * sv_params.use_chip_extent # chip diag len
        uid += ')'
        return uid

class QueryParams(DynStruct):
    def __init__(query_params, **kwargs):
        super(QueryParams, query_params).__init__()
        query_params.nn_params = NNParams(**kwargs)
        query_params.score_params = ScoreMechanismParams(**kwargs)
        query_params.sv_params = SpatialVerifyParams(**kwargs)
        query_params.query_type = 'vsmany'
        query_params.use_cache  = False
        # Data
        query_params.data_index = None
        query_params.qcxs2_index = {}
        query_params.update(**kwargs)
        query_params.score_params.make_feasible(query_params.nn_params)
    def get_uid(query_params, SV=False, scored=True, long_=False):
        uid = query_params.query_type
        uid += query_params.nn_params.get_uid()
        if SV is True:
            uid += query_params.sv_params.get_uid()
        if scored is True:
            uid += query_params.score_params.get_uid()
        if long_:
            uid += params.get_matcher_uid()
        return uid
