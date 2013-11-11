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
    def __init__(res, qcx, query_params):
        super(QueryResult, res).__init__()
        res.qcx       = qcx
        res.query_uid = query_params.get_uid()
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
    def topN_cxs(res, N, SV):
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
class NNParams(DynStruct):
    def __init__(nn_params, **kwargs):
        super(NNParams, nn_params).__init__()
        # Core
        nn_params.K = 2
        nn_params.Knorm = 1
        nn_params.K_reciprocal = 1 # 0 := off
        # Filters
        nn_params.nnfilter_list = ['reciprocal', 'roidist']
        #['reciprocal', 'roidist', 'frexquency', 'ratiotest', 'bursty']
        nn_params.checks = 128
        nn_params.__dict__.update(**kwargs)
    def get_uid(nn_params):
        uid = '_nn(' 
        uid += ','.join(nn_params.nnfilter_list)
        uid += ','+str(nn_params.K)
        uid += ','+str(nn_params.Knorm)
        uid += ','+str(nn_params.K_reciprocal)
        uid += ',' + str(nn_params.checks)
        uid += ')'
        return uid

class SpatialVerifyParams(DynStruct):
    def __init__(sv_params, **kwargs):
        super(SpatialVerifyParams, sv_params).__init__()
        sv_params.use_chip_extent = False
        sv_params.scale_thresh  = (.5, 2)
        sv_params.xy_thresh = .002
        sv_params.shortlist_len = 100
        sv_params.__dict__.update(kwargs)
    def get_uid(sv_params):
        uid = '_sv(' 
        uid += str(sv_params.shortlist_len)
        uid += ',' + str(sv_params.xy_thresh)
        uid += ',' + str(sv_params.scale_thresh)
        uid += ',cdl' * sv_params.use_chip_extent # chip diag len
        uid += ')'
        return uid

class ScoringParams(DynStruct):
    def __init__(score_params, **kwargs):
        super(ScoringParams, score_params).__init__()
        score_params.aggregation_method = 'ChipSum' # ['NameSum', 'NamePlacketLuce']
        score_params.key2_threshweight = {
            'roidist'    : (.5 , 0),
            'reciprocal' : (0  , 0), 
            'scale'      : (.5 , 0),
            'bursty'     : (1  , 0),
            'ratio'      : (1.2, 0), 
            'lnbnn'      : (0  , 1)
        }
        score_params.__dict__.update(kwargs)
    def get_uid(score_params):
        uid = '_score(' 
        uid += score_params.aggregation_method
        uid += ',' + helpers.remove_chars(str(score_params.key2_threshweight), [' ','\'','}','{',':'])
        uid += ')'
        return uid

class QueryParams(DynStruct):
    def __init__(query_params, **kwargs):
        super(QueryParams, query_params).__init__()
        query_params.nn_params = NNParams(**kwargs)
        query_params.score_params = ScoringParams(**kwargs)
        query_params.sv_params = SpatialVerifyParams( *kwargs)
        query_params.query_type = 'vsmany'
        query_params.use_cache  = False
        # Data
        query_params.data_index = None
        query_params.qcxs2_index = {}
        query_params.__dict__.update(kwargs)
    def get_uid(query_params):
        uid = query_params.query_type
        uid += query_params.sv_params.get_uid()
        uid += query_params.score_params.get_uid()
        uid += query_params.nn_params.get_uid()
        uid += params.get_matcher_uid()
        return uid
