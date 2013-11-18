from __future__ import division, print_function
import __builtin__
import params
import helpers
import numpy as np
import pyflann
import algos
import draw_func2 as df2
import sys
from itertools import izip, chain
from Printable import DynStruct
import warnings
import os
from os.path import exists, split, join, normpath

def print(*arsg, **kwargs): pass
def noprint(*args, **kwargs): pass
def realprint(*args, **kwargs):
    __builtin__.print(*args, **kwargs)
def print_on():
    global print
    print = realprint
def print_off():
    global print
    print = noprint
print_on()

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
FK_DTYPE  = np.int16
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
            nn_index.flann = None
#=========================
# Query Result Class
#=========================
def query_result_fpath(hs, qcx, query_uid=None):
    if query_uid is None: query_uid = params.get_query_uid()
    qres_dir  = hs.dirs.qres_dir 
    fname = 'result_%s_qcx=%d.npz' % (query_uid, qcx)
    fpath = join(qres_dir, fname)
    return fpath

def query_result_exists(hs, qcx, query_uid=None):
    fpath = query_result_fpath(hs, qcx, query_uid)
    return exists(fpath)

class QueryResult(DynStruct):
    def __init__(res, qcx, uid, q_cfg=None):
        super(QueryResult, res).__init__()
        if not q_cfg is None:
            res.true_title = q_cfg.get_uid()
        res.qcx       = qcx
        res.query_uid = uid
        res.uid       = uid
        res.title     = uid
        # Times
        res.assign_time = -1
        res.verify_time = -1
        # Assigned features matches
        res.cx2_fm = np.array([], dtype=FM_DTYPE)
        res.cx2_fs = np.array([], dtype=FS_DTYPE)
        res.cx2_fk = np.array([], dtype=FM_DTYPE)
        res.cx2_score = np.array([])
        # TODO: Remove these
        res.cx2_fm_V = np.array([], dtype=FM_DTYPE)
        res.cx2_fs_V = np.array([], dtype=FS_DTYPE)
        res.cx2_fk_V = np.array([], dtype=FM_DTYPE)
        res.cx2_score_V = np.array([])

    def has_cache(res, hs):
        return query_result_exists(hs, res.qcx)

    def get_fpath(res, hs):
        return query_result_fpath(hs, res.qcx, res.query_uid)
    
    def save(res, hs):
        fpath = res.get_fpath(hs)
        print('[ds] cache result: %r' % (fpath if params.VERBOSE_CACHE
                                         else split(fpath)[1],))
        with open(fpath, 'wb') as file_:
            np.savez(file_, **res.__dict__.copy())
        return True

    def load(res, hs):
        'Loads the result from the given database'
        fpath = res.get_fpath(hs)
        #print('[ds] Load res fpath=%r' % (split(fpath)[1],))
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
            if exists(fpath):
                #os.remove(fpath)
                warnmsg = ('Load Result Exception : ' + repr(ex) + 
                        '\nResult was corrupted for qcx=%d' % res.qcx)
            else:
                #os.remove(fpath)
                warnmsg = ('Load Result Exception : ' + repr(ex) + 
                        '\nResult does not yet exist for qcx=%d' % res.qcx)
            print(warnmsg)
            warnings.warn(warnmsg)
            raise
    def get_SV(res):
        #return res.cx2_fm_V.size > 0
        return len(res.cx2_score_V) > 0
    def cache_bytes(res, hs):
        fpath = res.get_fpath(hs)
        return helpers.file_bytes(fpath)
    def top5_cxs(res):
        return res.topN_cxs(5)
    def get_gt_ranks(res, gt_cxs=None, hs=None):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_cxs is None and hs is None: raise Exception('[res] error')
        if gt_cxs is None: gt_cxs = hs.get_other_cxs(res.qcx)
        cx2_score = res.get_cx2_score()
        top_cxs   = cx2_score.argsort()[::-1]
        foundpos = [np.where(top_cxs == cx)[0] for cx in gt_cxs]
        ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        assert all([len(r) == 1 for r in ranks_])
        gt_ranks = [r[0] for r in ranks_]
        return gt_ranks
    def get_cx2_score(res, SV=None):
        if SV is None:
            SV = res.get_SV()
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
        if not 'SV' in kwargs.keys():
            kwargs['SV'] = res.get_SV()
        df2.show_topN_matches(hs, res, **kwargs)

#=========================
# Param Classes
#=========================
#def udpate_dicts(dict1, dict2):
    #dict1_keys = set(dict1.keys())
    #if key, val in dict2.iteritems():
        #if key in dict1_keys:
            #dict1[key] = val

class NNConfig(DynStruct):
    def __init__(nn_cfg, **kwargs):
        super(NNConfig, nn_cfg).__init__()
        # Core
        nn_cfg.K = 2
        nn_cfg.Knorm = 1
        # Filters
        nn_cfg.checks = 1024#512#128
        nn_cfg.update(**kwargs)
    def get_uid(nn_cfg):
        uid = '_NN(' 
        uid += 'K'+str(nn_cfg.K)+'+'+str(nn_cfg.Knorm)
        uid += ',cks' + str(nn_cfg.checks)
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

#valid_filters = ['recip', 'roidist', 'frexquency', 'ratio', 'bursty', 'lnbnn']
def any_inlist(list_, search_list):
    set_ = set(list_)
    return any([search in set_ for search in search_list])

class FilterConfig(DynStruct):
    # Rename to scoring mechanism
    def __init__(f_cfg, **kwargs):
        super(FilterConfig, f_cfg).__init__()
        fp = f_cfg
        fp.Krecip = 0 # 0 := off
        fp.nnfilter_list = ['recip', 'roidist']
        #
        fp.nnfilter_list = ['recip', 'roidist', 'lnbnn', 'ratio']
        valid_filters = []
        def addfilt(sign, filt, thresh, weight):
            valid_filters.append((sign, filt))
            fp.__dict__[filt+'_thresh'] = thresh
            fp.__dict__[filt+'_weight']  = weight
        #tuple(Sign, Filt, ValidSignThresh, ScoreWeight)
        addfilt(+1, 'roidist', None, 0)
        addfilt(+1, 'recip',      0, 0)
        #addfilt(+1, 'scale' )
        addfilt(+1, 'bursty', None, 0)
        addfilt(-1, 'ratio',  None, 0)
        addfilt(-1, 'lnbnn',  None, 0)
        addfilt(-1, 'lnrat',  None, 1)
        fp.update(**kwargs)
        fp.filt2_tw = {}
        for (sign, filt) in valid_filters:
            stw = ((sign, fp.__dict__[filt+'_thresh']), fp.__dict__[filt+'_weight'])
            fp.filt2_tw[filt] = stw

    def make_feasible(f_cfg, nn_cfg):
        nnfilts = f_cfg.nnfilter_list
        nnp = nn_cfg
        fp = f_cfg
        # Knorm
        if fp.lnbnn_thresh is None and fp.lnbnn_weight == 0:
            listrm(nnfilts, 'lnbnn')
        if fp.ratio_thresh   <= 1:
            listrm(nnfilts, 'ratio')
        norm_depends = ['lnbnn', 'ratio', 'lnrat']
        if nnp.Knorm <= 0 and not any_inlist(nnfilts, norm_depends):
            listrm_list(nnfilts, norm_depends)
            nnp.Knorm = 0
        # Krecip
        if fp.Krecip <= 0 or 'recip' not in nnfilts:
            listrm(nnfilts, 'recip')
            fp.Krecip = 0
        if (fp.roidist_thresh is None or fp.roidist_thresh >= 1) and\
               fp.roidist_weight == 0:
            listrm(nnfilts, 'roidist')
        if fp.bursty_thresh   <= 1:
            listrm(nnfilts, 'bursty')


    def get_uid(f_cfg):
        on_filters = dict_subset(f_cfg.filt2_tw,
                                 f_cfg.nnfilter_list)
        uid = '_FILT(' 
        twstr = signthreshweight_str(on_filters)
        if f_cfg.Krecip != 0:
            uid += 'Kr'+str(f_cfg.Krecip)
            if len(twstr) > 0: uid += ','
        if len(twstr) > 0:
            uid += twstr
        uid += ')'
        return uid

def signthreshweight_str(on_filters):
    stw_list = []
    for key, val in on_filters.iteritems():
        ((sign, thresh), weight) = val
        stw_str = key
        if not thresh is None:
            sstr = ['<','>'][sign == -1] #actually <=, >=
            stw_str += sstr+str(thresh)
        if weight != 0:
            stw_str += '_'+str(weight)
        stw_list.append(stw_str)
    return ','.join(stw_list)
    #return helpers.remove_chars(str(dict_), [' ','\'','}','{',':'])

class SpatialVerifyConfig(DynStruct):
    def __init__(sv_cfg, **kwargs):
        super(SpatialVerifyConfig, sv_cfg).__init__()
        sv_cfg.scale_thresh  = (.5, 2)
        sv_cfg.xy_thresh = .002
        sv_cfg.nShortlist = 500
        sv_cfg.prescore_method = 'csum'
        sv_cfg.use_chip_extent = False
        sv_cfg.min_nInliers = 4
        sv_cfg.update(**kwargs)
    def get_uid(sv_cfg):
        uid = '_SV(' 
        uid += str(sv_cfg.nShortlist)
        uid += ',' + str(sv_cfg.xy_thresh)
        uid += ',' + str(sv_cfg.scale_thresh).replace(' ','')
        uid += ',cdl' * sv_cfg.use_chip_extent # chip diag len
        uid += ','+sv_cfg.prescore_method
        uid += ')'
        return uid

class AggregateConfig(DynStruct):
    def __init__(a_cfg, **kwargs):
        super(AggregateConfig, a_cfg).__init__()
        a_cfg.query_type   = 'vsmany'
        a_cfg.score_method = 'csum' # namesum, placketluce
        a_cfg.update(**kwargs)
    def get_uid(a_cfg):
        uid = '_AGG('
        uid += a_cfg.query_type
        uid += ','+a_cfg.score_method
        uid += ')'
        return uid

class QueryConfig(DynStruct):
    def __init__(q_cfg, **kwargs):
        super(QueryConfig, q_cfg).__init__()
        q_cfg.nn_cfg = NNConfig(**kwargs)
        q_cfg.f_cfg  = FilterConfig(**kwargs)
        q_cfg.sv_cfg = SpatialVerifyConfig(**kwargs)
        q_cfg.a_cfg  = AggregateConfig(**kwargs)
        q_cfg.qcxs = []
        q_cfg.dcxs = []
        q_cfg.use_cache = False
        # Data
        q_cfg.data_index = None # current index
        q_cfg.dcxs2_index = {}  # L1 cached indexes
        q_cfg.update(**kwargs)
        q_cfg.f_cfg.make_feasible(q_cfg.nn_cfg)
    def get_uid(q_cfg, *args, **kwargs):
        uid = ''
        if not 'noNN' in args:
            uid += q_cfg.nn_cfg.get_uid()
        if not 'noFILT' in args:
            uid += q_cfg.f_cfg.get_uid()
        if not 'noSV' in args:
            uid += q_cfg.sv_cfg.get_uid()
        if not 'noAGG' in args:
            uid += q_cfg.a_cfg.get_uid()
        if not 'noCHIP' in args:
            uid += params.get_indexed_uid()
        return uid
