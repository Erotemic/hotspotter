from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile) = __common__.init(__name__, '[ds]')
# Standard
from itertools import izip, chain
# Scientific
import numpy as np
# HotSpotter
from hscom import helpers
from hscom.Printable import DynStruct

ID_DTYPE = np.int32  # id datatype
X_DTYPE  = np.int32  # indeX datatype

DEBUG = False  # Debug flag

if DEBUG:
    def printDBG(msg):
        print('[DS.DBG] ' + msg)
else:
    def printDBG(msg):
        pass


class QueryData(DynStruct):
    def __init__(qdat):
        super(QueryData, qdat).__init__()
        qdat.cfg = None  # Query Config
        qdat._qcxs = []
        qdat._dcxs = []
        qdat._data_indexes = None  # current index
        qdat._dcxs2_index = {}     # cached indexes

    def set_cfg(qdat, query_cfg):
        qdat.cfg = query_cfg

    def unload_data(qdat):
        # Data TODO: Separate this
        printDBG('[qdat] unload_data()')
        qdat._qcxs = []
        qdat._dcxs = []
        qdat._data_index  = None  # current index
        qdat._dcxs2_index = {}  # cached indexes
        printDBG('[qdat] unload_data(success)')

    def get_uid_list(qdat, *args, **kwargs):
        uid_list = qdat.cfg.get_uid_list(*args, **kwargs)
        if not 'noDCXS' in args:
            # In case you don't search the entire dataset
            dcxs_uid = helpers.hashstr_arr(qdat._dcxs, 'dcxs')
            uid_list += ['_', dcxs_uid]
        return uid_list

    def get_uid(qdat, *args, **kwargs):
        return ''.join(qdat.get_uid_list(*args, **kwargs))


class NNIndex(object):
    'Nearest Neighbor (FLANN) Index Class'
    def __init__(nn_index, hs, cx_list):
        import algos
        cx2_desc  = hs.feats.cx2_desc
        # Make unique id for indexed descriptors
        feat_uid   = hs.prefs.feat_cfg.get_uid()
        sample_uid = helpers.hashstr_arr(cx_list, 'dcxs')
        uid = '_' + sample_uid + feat_uid
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
             gx2_gname=[], gx2_aif=[],
             nx2_name=['____', '____'],
             cx2_cid=[], cx2_nx=[], cx2_gx=[],
             cx2_roi=[], cx2_theta=[], prop_dict={}):
        self.gx2_aif      = np.array(gx2_aif, dtype=bool)
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
        from os.path import join
        # Class variables
        self.db_dir       = db_dir
        self.img_dir      = join(db_dir, ld2.RDIR_IMG)
        self.internal_dir = join(db_dir, ld2.RDIR_INTERNAL)
        self.computed_dir = join(db_dir, ld2.RDIR_COMPUTED)
        self.chip_dir     = join(db_dir, ld2.RDIR_CHIP)
        self.rchip_dir    = join(db_dir, ld2.RDIR_RCHIP)
        self.feat_dir     = join(db_dir, ld2.RDIR_FEAT)
        self.cache_dir    = join(db_dir, ld2.RDIR_CACHE)
        self.result_dir   = join(db_dir, ld2.RDIR_RESULTS)
        self.qres_dir     = join(db_dir, ld2.RDIR_QRES)

    def ensure_dirs(self):
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
        #self.cx2_chip_path  = []  # Unused
        self.cx2_rchip_path = []
        self.cx2_rchip_size = []
        self.chip_uid = ''


class HotspotterChipFeatures(DynStruct):
    def __init__(self):
        super(HotspotterChipFeatures, self).__init__()
        self.cx2_desc = []
        self.cx2_kpts = []
        self.feat_uid = ''
