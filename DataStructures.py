from __future__ import division, print_function
import __builtin__
import sys
from itertools import izip, chain
#
import helpers

import numpy as np
from Printable import DynStruct
import Config

ID_DTYPE = np.int32  # id datatype
X_DTYPE  = np.int32  # indeX datatype

DEBUG = False  # Debug flag

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
