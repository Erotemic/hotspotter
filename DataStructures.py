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

import numpy as np
from Printable import DynStruct
import Config
import vizualizations as viz

FM_DTYPE  = np.uint32
FK_DTYPE  = np.int16
FS_DTYPE  = np.float32

ID_DTYPE = np.int32
X_DTYPE  = np.int32

DEBUG = False

if DEBUG:
    def printDBG(msg):
        print('[DS.DBG] ' + msg)
else:
    def printDBG(msg):
        pass

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
        qcx_good = res.qcx
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
            print('[ds] encountered IOError: %r' % ex)
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
        res.qcx = qcx_good

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
            gt_cxs = hs.get_other_indexed_cxs(res.qcx)
        cx2_score = res.get_cx2_score()
        top_cxs  = cx2_score.argsort()[::-1]
        foundpos = [np.where(top_cxs == cx)[0] for cx in gt_cxs]
        ranks_   = [r if len(r) > 0 else [-1] for r in foundpos]
        assert all([len(r) == 1 for r in ranks_])
        gt_ranks = [r[0] for r in ranks_]
        return gt_ranks

    def get_cx2_score(res, SV=None):
        return res.cx2_score

    def get_cx2_fm(res, SV=None):
        return res.cx2_fm

    def get_cx2_fs(res, SV=None):
        return res.cx2_fs

    def topN_cxs(res, hs, N=None):
        import voting_rules2 as vr2
        cx2_score = np.array(res.get_cx2_score())
        if hs.prefs.display_cfg.name_scoring:
            cx2_chipscore = np.array(cx2_score)
            cx2_score = vr2.enforce_one_name(hs, cx2_score,
                                             cx2_chipscore=cx2_chipscore)
        top_cxs = cx2_score.argsort()[::-1]
        dcxs_ = set(hs.get_indexed_sample()) - set([res.qcx])
        top_cxs = [cx for cx in iter(top_cxs) if cx in dcxs_]
        #top_cxs = np.intersect1d(top_cxs, hs.get_indexed_sample())
        nIndexed = len(top_cxs)
        if N is None:
            N = hs.prefs.display_cfg.N
        if N == 'all':
            N = nIndexed
        #print('[res] cx2_score = %r' % (cx2_score,))
        #print('[res] returning top_cxs = %r' % (top_cxs,))
        nTop = min(N, nIndexed)
        #print('[res] returning nTop = %r' % (nTop,))
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
        import algos
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
        ax2_desc = np.vstack([cx2_desc[cx] for cx in cx_list if len(cx2_desc[cx]) > 0])
        # Build/Load the flann index
        flann_params = {'algorithm': 'kdtree', 'trees': 4}
        precomp_kwargs = {'cache_dir': hs.dirs.cache_dir,
                          'uid': uid,
                          'flann_params': flann_params,
                          'force_recompute': hs.args.nocache_flann}
        flann = algos.precompute_flann(ax2_desc, **precomp_kwargs)
        #----
        # Agg Data
        nn_index.ax2_cx   = ax2_cx
        nn_index.ax2_fx   = ax2_fx
        nn_index.ax2_data = ax2_desc
        nn_index.flann = flann

    def __getstate__(nn_index):
        printDBG('get state NNIndex')
        #if 'flann' in nn_index.__dict__ and nn_index.flann is not None:
            #nn_index.flann.delete_index()
            #nn_index.flann = None
        # This class is not pickleable
        return None

    def __del__(nn_index):
        printDBG('deleting NNIndex')
        if 'flann' in nn_index.__dict__ and nn_index.flann is not None:
            nn_index.flann.delete_index()
            nn_index.flann = None


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


def default_display_cfg(**kwargs):
    display_cfg = Config.DisplayConfig(**kwargs)
    return display_cfg


def default_chip_cfg(**kwargs):
    chip_cfg = Config.ChipConfig(**kwargs)
    return chip_cfg


def default_feat_cfg(hs, **kwargs):
    feat_cfg = Config.FeatureConfig(hs, **kwargs)
    return feat_cfg


def default_vsmany_cfg(hs, **kwargs):
    kwargs['query_type'] = 'vsmany'
    kwargs_set = __dict_default_func(kwargs)
    kwargs_set('lnbnn_weight', .01)
    kwargs_set('xy_thresh', .1)
    kwargs_set('K', 4)
    kwargs_set('Knorm', 1)
    query_cfg = Config.QueryConfig(hs, **kwargs)
    return query_cfg


def default_vsone_cfg(hs, **kwargs):
    kwargs['query_type'] = 'vsone'
    kwargs_set = __dict_default_func(kwargs)
    kwargs_set('lnbnn_weight', 0)
    kwargs_set('checks', 256)
    kwargs_set('K', 1)
    kwargs_set('Knorm', 1)
    kwargs_set('ratio_weight', 1.0)
    kwargs_set('ratio_thresh', 1.5)
    query_cfg = Config.QueryConfig(hs, **kwargs)
    return query_cfg
