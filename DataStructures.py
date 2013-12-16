from __future__ import division, print_function
import __builtin__
import sys
import os
from itertools import izip, chain
from os.path import exists, split, join
from zipfile import error as BadZipFile  # Screwy naming convention.
#
import helpers
import params
import algos
import vizualizations as viz

import numpy as np
from Printable import DynStruct
from Pref import Pref

FM_DTYPE  = np.uint32
FK_DTYPE  = np.int16
FS_DTYPE  = np.float32

ID_DTYPE = np.int32
X_DTYPE  = np.int32

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
    print('[ds] reloading ' + __name__)
    imp.reload(sys.modules[__name__])


#=========================
# Query Result Class
#=========================
def query_result_fpath(hs, qcx, query_uid):
    qres_dir  = hs.dirs.qres_dir
    qcid  = hs.tables.cx2_cid[qcx]
    fname = 'res_%s_qcid=%d.npz' % (query_uid, qcid)
    if len(fname) > 64:
        hash_id = helpers.hashstr(query_uid, 16)
        fname = 'res_%s_qcid=%d.npz' % (hash_id, qcid)
    fpath = join(qres_dir, fname)
    return fpath


def query_result_exists(hs, qcx, query_uid):
    fpath = query_result_fpath(hs, qcx, query_uid)
    return exists(fpath)


class QueryResult(DynStruct):
    #__slots__ = ['true_uid', 'qcx', 'query_uid', 'uid', 'title', 'nn_time',
                 #'weight_time', 'filt_time', 'build_time', 'verify_time',
                 #'cx2_fm', 'cx2_fs', 'cx2_fk', 'cx2_score', 'cx2_fm_V',
                 #'cx2_fs_V', 'cx2_fk_V', 'cx2_score_V']
    def __init__(res, qcx, uid, query_cfg=None):
        super(QueryResult, res).__init__()
        res.true_uid  = '' if query_cfg is None else query_cfg.get_uid()
        res.qcx       = qcx
        res.query_uid = uid
        res.uid       = uid
        res.title     = uid
        # Times
        res.nn_time     = -1
        res.weight_time = -1
        res.filt_time = -1
        res.build_time  = -1
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
        print('[ds] res.load() fpath=%r' % (split(fpath)[1],))
        try:
            with open(fpath, 'rb') as file_:
                npz = np.load(file_)
                for _key in npz.files:
                    res.__dict__[_key] = npz[_key]
                npz.close()
            res.qcx = res.qcx.tolist()
            res.query_uid = str(res.query_uid)
            return True
        except IOError as ex:
            #print('[ds] Caught IOError: %r' % ex)
            if not exists(fpath):
                #print(fpath)
                #print('[ds] QueryResult(qcx=%d) does not exist' % res.qcx)
                raise
            else:
                msg = ['[ds] QueryResult(qcx=%d) is corrupted' % (res.qcx)]
                msg += ['\n%r' % (ex,)]
                print(''.join(msg))
                raise Exception(msg)
        except BadZipFile as ex:
            print('[ds] Caught other BadZipFile: %r' % ex)
            msg = ['[ds] Attribute Error: QueryResult(qcx=%d) is corrupted' % (res.qcx)]
            msg += ['\n%r' % (ex,)]
            print(''.join(msg))
            if exists(fpath):
                print('[ds] Removing corrupted file: %r' % fpath)
                os.remove(fpath)
                raise IOError(msg)
            else:
                raise Exception(msg)
        except Exception as ex:
            print('Caught other Exception: %r' % ex)
            raise

    def get_SV(res):
        #return res.cx2_fm_V.size > 0
        return len(res.cx2_score_V) > 0

    def cache_bytes(res, hs):
        fpath = res.get_fpath(hs)
        return helpers.file_bytes(fpath)

    def get_gt_ranks(res, gt_cxs=None, hs=None):
        'returns the 0 indexed ranking of each groundtruth chip'
        # Ensure correct input
        if gt_cxs is None and hs is None:
            raise Exception('[res] error')
        if gt_cxs is None:
            gt_cxs = hs.get_other_cxs(res.qcx)
        cx2_score = res.get_cx2_score()
        top_cxs  = cx2_score.argsort()[::-1]
        foundpos = [np.where(top_cxs == cx)[0] for cx in gt_cxs]
        ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        assert all([len(r) == 1 for r in ranks_])
        gt_ranks = [r[0] for r in ranks_]
        return gt_ranks

    def get_cx2_score(res, SV=None):
        if SV is None:
            SV = res.get_SV()
        return res.cx2_score_V if SV else res.cx2_score

    def get_cx2_fm(res, SV=None):
        if SV is None:
            SV = res.get_SV()
        return res.cx2_fm_V if SV else res.cx2_fm

    def get_cx2_fs(res, SV=None):
        if SV is None:
            SV = res.get_SV()
        return res.cx2_fs_V if SV else res.cx2_fs

    def topN_cxs(res, N, query_cfg=None):
        cx2_score = res.get_cx2_score()
        top_cxs = cx2_score.argsort()[::-1]
        if not query_cfg is None:
            top_cxs = np.intersect1d(top_cxs, query_cfg._dcxs)
        nIndexed = len(top_cxs)
        if N == 'all':
            N = nIndexed
        nTop = min(N, nIndexed)
        topN_cxs = top_cxs[0:nTop]
        return topN_cxs

    def show_query(res, hs, **kwargs):
        print('[res] show_query')
        viz.show_chip(hs, res=res, **kwargs)

    def show_analysis(res, hs, *args, **kwargs):
        return viz.res_show_analysis(res, hs, *args, **kwargs)

    def show_top(res, hs, *args, **kwargs):
        return viz.show_top(res, hs, *args, **kwargs)

    def show_gt_matches(res, hs, *args, **kwargs):
        figtitle = ('q%s -- GroundTruth' % (hs.cxstr(res.qcx)))
        gt_cxs = hs.get_other_indexed_cxs(res.qcx)
        viz._show_chip_matches(hs, res, gt_cxs=gt_cxs, figtitle=figtitle,
                               all_kpts=True, *args, **kwargs)

    def plot_matches(res, hs, cx, **kwargs):
        viz.show_matches_annote_res(res, hs, cx, draw_pts=False, **kwargs)


class NNIndex(object):
    'Nearest Neighbor (FLANN) Index Class'
    def __init__(nn_index, hs, cx_list):
        cx2_desc  = hs.feats.cx2_desc
        # Make unique id for indexed descriptors
        feat_uid   = ''.join(hs.prefs.feat_cfg.get_uid())
        sample_uid = helpers.make_sample_id(cx_list)
        uid = '_cxs(' + sample_uid + ')' + feat_uid
        # Number of features per sample chip
        sx2_nFeat = [len(cx2_desc[cx]) for cx in iter(cx_list)]
        # Inverted index from indexed descriptor to chipx and featx
        _ax2_cx = [[cx] * nFeat for (cx, nFeat) in izip(cx_list, sx2_nFeat)]
        _ax2_fx = [range(nFeat) for nFeat in iter(sx2_nFeat)]
        ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
        ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
        # Aggregate indexed descriptors into continuous structure
        ax2_desc = np.vstack([cx2_desc[cx] for cx in cx_list])
        # Build/Load the flann index
        flann_params = {'algorithm': 'kdtree', 'trees': 4}
        precomp_kwargs = {'cache_dir': hs.dirs.cache_dir,
                          'uid': uid,
                          'flann_params': flann_params, }
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
# CONFIG Classes
#=========================
#def udpate_dicts(dict1, dict2):
    #dict1_keys = set(dict1.keys())
    #if key, val in dict2.iteritems():
        #if key in dict1_keys:
            #dict1[key] = val


#ConfigBase = Pref
ConfigBase = DynStruct

class NNConfig(ConfigBase):
    def __init__(nn_cfg, **kwargs):
        super(NNConfig, nn_cfg).__init__()
        # Core
        nn_cfg.K = 2
        nn_cfg.Knorm = 1
        # Filters
        nn_cfg.checks  = 1024  # 512#128
        nn_cfg.update(**kwargs)

    def get_uid(nn_cfg):
        uid  = ['_NN(']
        uid += ['K', str(nn_cfg.K), '+', str(nn_cfg.Knorm)]
        uid += [',cks', str(nn_cfg.checks)]
        uid += [')']
        return uid


def dict_subset(dict_, keys):
    keys_ = set(keys)
    return {key: val for (key, val) in dict_.iteritems() if key in keys_}


def listrm(list_, item):
    try:
        list_.remove(item)
    except Exception:
        pass


def listrm_list(list_, items):
    for item in items:
        listrm(list_, item)


#valid_filters = ['recip', 'roidist', 'frexquency', 'ratio', 'bursty', 'lnbnn']
def any_inlist(list_, search_list):
    set_ = set(list_)
    return any([search in set_ for search in search_list])


class FilterConfig(ConfigBase):
    # Rename to scoring mechanism
    def __init__(filt_cfg, **kwargs):
        super(FilterConfig, filt_cfg).__init__()
        filt_cfg = filt_cfg
        filt_cfg.filt_on = True
        filt_cfg.Krecip = 0  # 0 := off
        filt_cfg._nnfilter_list = []
        #
        #filt_cfg._nnfilter_list = ['recip', 'roidist', 'lnbnn', 'ratio', 'lnrat']
        filt_cfg._valid_filters = []

        def addfilt(sign, filt, thresh, weight):
            filt_cfg._nnfilter_list.append(filt)
            filt_cfg._valid_filters.append((sign, filt))
            filt_cfg.__dict__[filt + '_thresh'] = thresh
            filt_cfg.__dict__[filt + '_weight'] = weight
        #tuple(Sign, Filt, ValidSignThresh, ScoreMetaWeight)
        # thresh test is: sign * score <= sign * thresh
        addfilt(+1, 'roidist', None, 0)  # Lower  scores are better
        addfilt(-1, 'recip',     0, 0)  # Higher scores are better
        addfilt(+1, 'bursty', None, 0)  # Lower  scores are better
        addfilt(-1, 'ratio',  None, 0)  # Higher scores are better
        addfilt(-1, 'lnbnn',  None, 0)  # Higher scores are better
        addfilt(-1, 'lnrat',  None, 0)  # Higher scores are better
        #addfilt(+1, 'scale' )
        filt_cfg._filt2_tw = {}
        filt_cfg.update(**kwargs)

    def make_feasible(filt_cfg, query_cfg):
        '''
        removes invalid parameter settings over all cfgs (move to QueryConfig)
        '''
        nn_cfg = query_cfg.nn_cfg

        # Ensure the list of on filters is valid given the weight and thresh
        if filt_cfg.ratio_thresh <= 1:
            filt_cfg.ratio_thresh = None
        if filt_cfg.roidist_thresh >= 1:
            filt_cfg.roidist_thresh = None
        if filt_cfg.bursty_thresh   <= 1:
            filt_cfg.bursty_thresh = None

        # FIXME: Non-Independent parameters.
        # Need to explicitly model correlation somehow
        if filt_cfg.Krecip == 0:
            filt_cfg.recip_thresh = None
        elif filt_cfg.recip_thresh is None:
            filt_cfg.recip_thresh = 0

        def ensure_filter(filt, sign):
            '''ensure filter in the list if valid else remove
            (also ensure the sign/thresh/weight dict)'''
            thresh = filt_cfg.__dict__[filt + '_thresh']
            weight = filt_cfg.__dict__[filt + '_weight']
            stw = ((sign, thresh), weight)
            filt_cfg._filt2_tw[filt] = stw
            if thresh is None and weight == 0:
                listrm(filt_cfg._nnfilter_list, filt)
            elif not filt in filt_cfg._nnfilter_list:
                filt_cfg._nnfilter_list += [filt]
        for (sign, filt) in filt_cfg._valid_filters:
            ensure_filter(filt, sign)

        # Set Knorm to 0 if there is no normalizing filter on.
        norm_depends = ['lnbnn', 'ratio', 'lnrat']
        if nn_cfg.Knorm <= 0 and not any_inlist(filt_cfg._nnfilter_list, norm_depends):
            #listrm_list(filt_cfg._nnfilter_list , norm_depends)
            # FIXME: Knorm is not independent of the other parameters.
            # Find a way to make it independent.
            nn_cfg.Knorm = 0

    def get_uid(filt_cfg):
        if not filt_cfg.filt_on:
            return ['_FILT()']
        on_filters = dict_subset(filt_cfg._filt2_tw,
                                 filt_cfg._nnfilter_list)
        uid = ['_FILT(']
        twstr = signthreshweight_str(on_filters)
        if filt_cfg.Krecip != 0 and 'recip' in filt_cfg._nnfilter_list:
            uid += ['Kr' + str(filt_cfg.Krecip)]
            if len(twstr) > 0:
                uid += [',']
        if len(twstr) > 0:
            uid += [twstr]
        uid += [')']
        return uid


def signthreshweight_str(on_filters):
    stw_list = []
    for key, val in on_filters.iteritems():
        ((sign, thresh), weight) = val
        stw_str = key
        if thresh is None and weight == 0:
            continue
        if thresh is not None:
            sstr = ['<', '>'][sign == -1]  # actually <=, >=
            stw_str += sstr + str(thresh)
        if weight != 0:
            stw_str += '_' + str(weight)
        stw_list.append(stw_str)
    return ','.join(stw_list)
    #return helpers.remove_chars(str(dict_), [' ','\'','}','{',':'])


class SpatialVerifyConfig(ConfigBase):
    def __init__(sv_cfg, **kwargs):
        super(SpatialVerifyConfig, sv_cfg).__init__()
        sv_cfg.scale_thresh_low = .5
        sv_cfg.scale_thresh_high = 2
        sv_cfg.xy_thresh = .002
        sv_cfg.nShortlist = 1000
        sv_cfg.prescore_method = 'csum'
        sv_cfg.use_chip_extent = False
        sv_cfg.just_affine = False
        sv_cfg.min_nInliers = 4
        sv_cfg.sv_on = True
        sv_cfg.update(**kwargs)

    def get_uid(sv_cfg):
        if not sv_cfg.sv_on or sv_cfg.xy_thresh is None:
            return ['_SV()']
        uid = ['_SV(']
        uid += [str(sv_cfg.nShortlist)]
        uid += [',' + str(sv_cfg.xy_thresh)]
        scale_thresh = (sv_cfg.scale_thresh_low, sv_cfg.scale_thresh_high)
        scale_str = helpers.remove_chars(str(scale_thresh), ' ()')
        uid += [',' + scale_str.replace(',', '_')]
        uid += [',cdl' * sv_cfg.use_chip_extent]  # chip diag len
        uid += [',aff' * sv_cfg.just_affine]  # chip diag len
        uid += [',' + sv_cfg.prescore_method]
        uid += [')']
        return uid


class AggregateConfig(ConfigBase):
    def __init__(agg_cfg, **kwargs):
        super(AggregateConfig, agg_cfg).__init__()
        agg_cfg.query_type   = 'vsmany'
        # chipsum, namesum, placketluce
        agg_cfg.isWeighted = False  # nsum, pl
        agg_cfg.score_method = 'csum'  # nsum, pl
        alt_methods = {
            'topk': 'topk',
            'borda': 'borda',
            'placketluce': 'pl',
            'chipsum': 'csum',
            'namesum': 'nsum',
        }
        # For Placket-Luce
        agg_cfg.max_alts = 1000
        #-----
        # User update
        agg_cfg.update(**kwargs)
        # ---
        key = agg_cfg.score_method.lower()
        # Use w as a toggle for weighted mode
        if key.find('w') == len(key) - 1:
            agg_cfg.isWeighted = True
            key = key[:-1]
            agg_cfg.score_method = key
        # Sanatize the scoring method
        if key in alt_methods:
            agg_cfg.score_method = alt_methods[key]

    def get_uid(agg_cfg):
        uid = []
        uid += ['_AGG(']
        uid += [agg_cfg.query_type]
        uid += [',', agg_cfg.score_method]
        if agg_cfg.isWeighted:
            uid += ['w']
        if agg_cfg.score_method  == 'pl':
            uid += [',%d' % (agg_cfg.max_alts,)]
        uid += [') ']
        return uid


class QueryConfig(ConfigBase):
    def __init__(query_cfg, hs, **kwargs):
        super(QueryConfig, query_cfg).__init__()
        query_cfg.nn_cfg  = NNConfig(**kwargs)
        query_cfg.filt_cfg  = FilterConfig(**kwargs)
        query_cfg.sv_cfg  = SpatialVerifyConfig(**kwargs)
        query_cfg.agg_cfg  = AggregateConfig(**kwargs)
        query_cfg._feat_cfg  = hs.prefs.feat_cfg  # Queries depend on features
        #
        query_cfg.use_cache = False
        # Data TODO: Separate this
        query_cfg._qcxs = []
        query_cfg._dcxs = []
        query_cfg._data_index = None  # current index
        query_cfg._dcxs2_index = {}  # L1 cached indexes
        query_cfg.update(**kwargs)
        query_cfg.filt_cfg.make_feasible(query_cfg)

    def update_cfg(query_cfg, **kwargs):
        query_cfg.nn_cfg.update(**kwargs)
        query_cfg.filt_cfg.update(**kwargs)
        query_cfg.sv_cfg.update(**kwargs)
        query_cfg.agg_cfg.update(**kwargs)
        query_cfg.update(**kwargs)
        query_cfg.filt_cfg.make_feasible(query_cfg)

    def get_uid(query_cfg, *args, **kwargs):
        dcxs_ = repr(tuple(query_cfg._dcxs))
        uids = []
        if not 'noNN' in args:
            uids += query_cfg.nn_cfg.get_uid(**kwargs)
        if not 'noFILT' in args:
            uids += query_cfg.filt_cfg.get_uid(**kwargs)
        if not 'noSV' in args:
            uids += query_cfg.sv_cfg.get_uid(**kwargs)
        if not 'noAGG' in args:
            uids += query_cfg.agg_cfg.get_uid(**kwargs)
        if not 'noCHIP' in args:
            uids += query_cfg._feat_cfg.get_uid()
        # In case you don't search the entire dataset
        uids += ['_dcxs(' + helpers.hashstr(dcxs_) + ')']
        uid = ''.join(uids)
        return uid


class FeatureConfig(ConfigBase):
    def __init__(feat_cfg, hs, **kwargs):
        super(FeatureConfig, feat_cfg).__init__()
        feat_cfg.feat_type = 'hesaff+sift'
        feat_cfg.whiten = False
        feat_cfg.scale_min = 30  # 0    # 30
        feat_cfg.scale_max = 80  # 9001 # 80
        feat_cfg._chip_cfg = hs.prefs.chip_cfg  # Features depend on chips
        if feat_cfg._chip_cfg is None:
            raise Exception('Chip config is required')
        feat_cfg.update(**kwargs)

    def get_dict_args(feat_cfg):
        dict_args = {
            'scale_min': feat_cfg.scale_min,
            'scale_max': feat_cfg.scale_max, }
        return dict_args

    def get_uid(feat_cfg):
        feat_uids = ['_FEAT(']
        feat_uids += feat_cfg.feat_type
        feat_uids += [',white'] * feat_cfg.whiten
        feat_uids += [',%r_%r' % (feat_cfg.scale_min, feat_cfg.scale_max)]
        feat_uids += [')']
        feat_uids += [feat_cfg._chip_cfg.get_uid()]
        return [''.join(feat_uids)]


class ChipConfig(ConfigBase):
    def __init__(cc_cfg, **kwargs):
        super(ChipConfig, cc_cfg).__init__()
        cc_cfg.chip_sqrt_area = 750
        cc_cfg.grabcut         = False
        cc_cfg.histeq          = False
        cc_cfg.region_norm     = False
        cc_cfg.rank_eq         = False
        cc_cfg.local_eq        = False
        cc_cfg.maxcontrast     = False
        cc_cfg.update(**kwargs)

    def get_uid(cc_cfg):
        chip_uid = []
        chip_uid += ['histeq']  * cc_cfg.histeq
        chip_uid += ['grabcut'] * cc_cfg.grabcut
        chip_uid += ['regnorm'] * cc_cfg.region_norm
        chip_uid += ['rankeq']  * cc_cfg.rank_eq
        chip_uid += ['localeq'] * cc_cfg.local_eq
        chip_uid += ['maxcont'] * cc_cfg.maxcontrast
        isOrig = cc_cfg.chip_sqrt_area is None or cc_cfg.chip_sqrt_area  <= 0
        chip_uid += ['szorig'] if isOrig else ['sz%r' % cc_cfg.chip_sqrt_area]
        return '_CHIP(' + (','.join(chip_uid)) + ')'


class HotspotterTables(DynStruct):
    def __init__(self, *args, **kwargs):
        super(HotspotterTables, self).__init__()
        self.init(*args, **kwargs)

    def init(self,
             gx2_gname=[], nx2_name=['____', '____'],
             cx2_cid=[], cx2_nx=[], cx2_gx=[],
             cx2_roi=[], cx2_theta=[], prop_dict={}):
        self.gx2_gname    = np.array(gx2_gname, dtype=str)
        self.nx2_name     = np.array(nx2_name, dtype=str)
        self.cx2_cid      = np.array(cx2_cid, dtype=ID_DTYPE)
        self.cx2_nx       = np.array(cx2_nx, dtype=X_DTYPE)
        self.cx2_gx       = np.array(cx2_gx, dtype=X_DTYPE)
        self.cx2_roi      = np.array(cx2_roi, dtype=np.int32)
        self.cx2_roi.shape = (self.cx2_roi.size // 4, 4)
        self.cx2_theta    = np.array(cx2_theta, dtype=np.float32)
        self.prop_dict    = prop_dict


# ___CLASS HOTSPOTTER DIRS________
class HotspotterDirs(DynStruct):
    def __init__(self, db_dir):
        super(HotspotterDirs, self).__init__()
        import load_data2 as ld2
        # Class variables
        self.db_dir       = db_dir
        self.img_dir      = db_dir + ld2.RDIR_IMG
        self.internal_dir = db_dir + ld2.RDIR_INTERNAL
        self.computed_dir = db_dir + ld2.RDIR_COMPUTED
        self.chip_dir     = db_dir + ld2.RDIR_CHIP
        self.rchip_dir    = db_dir + ld2.RDIR_RCHIP
        self.feat_dir     = db_dir + ld2.RDIR_FEAT
        self.cache_dir    = db_dir + ld2.RDIR_CACHE
        self.result_dir   = db_dir + ld2.RDIR_RESULTS
        self.qres_dir     = db_dir + ld2.RDIR_QRES
        # Make directories if needbe
        helpers.ensure_path(self.internal_dir)
        helpers.ensure_path(self.computed_dir)
        helpers.ensure_path(self.chip_dir)
        helpers.ensure_path(self.rchip_dir)
        helpers.ensure_path(self.feat_dir)
        helpers.ensure_path(self.result_dir)
        helpers.ensure_path(self.rchip_dir)
        helpers.ensure_path(self.qres_dir)
        helpers.ensure_path(self.cache_dir)
        # Shortcut to internals
        #internal_sym = db_dir + '/Shortcut-to-hs_internals'
        #computed_sym = db_dir + '/Shortcut-to-computed'
        #results_sym  = db_dir + '/Shortcut-to-results'
        #helpers.symlink(self.internal_dir, internal_sym, noraise=False)
        #helpers.symlink(self.computed_dir, computed_sym, noraise=False)
        #helpers.symlink(self.result_dir, results_sym, noraise=False)


class HotspotterChipPaths(DynStruct):
    def __init__(self):
        super(HotspotterChipPaths, self).__init__()
        self.cx2_chip_path  = []
        self.cx2_rchip_path = []
        self.chip_uid = ''


class HotspotterChipFeatures(DynStruct):
    def __init__(self):
        super(HotspotterChipFeatures, self).__init__()
        self.cx2_desc = []
        self.cx2_kpts = []
        self.feat_uid = ''


# Convinience
def __dict_default_func(dict_):
    # Sets keys only if they dont exist
    def set_key(key, val):
        if not key in dict_:
            dict_[key] = val
    return set_key


def make_chip_cfg(**kwargs):
    chip_cfg = ChipConfig(**kwargs)
    return chip_cfg


def make_feat_cfg(hs, **kwargs):
    feat_cfg = FeatureConfig(hs, **kwargs)
    return feat_cfg


def make_vsmany_cfg(hs, **kwargs):
    kwargs['query_type'] = 'vsmany'
    kwargs_set = __dict_default_func(kwargs)
    kwargs_set('lnbnn_weight', .001)
    kwargs_set('K', 2)
    kwargs_set('Knorm', 1)
    query_cfg = QueryConfig(hs, **kwargs)
    return query_cfg


def make_vsone_cfg(hs, **kwargs):
    kwargs['query_type'] = 'vsone'
    kwargs_set = __dict_default_func(kwargs)
    kwargs_set('lnbnn_weight', 0)
    kwargs_set('checks', 256)
    kwargs_set('K', 1)
    kwargs_set('Knorm', 1)
    kwargs_set('ratio_weight', 1.0)
    kwargs_set('ratio_thresh', 1.5)
    query_cfg = QueryConfig(hs, **kwargs)
    return query_cfg
