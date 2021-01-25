
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile, printDBG) = __common__.init(__name__, '[ds]', DEBUG=False)
# Standard
from itertools import chain
# Scientific
import numpy as np
# HotSpotter
from . import algos
from hscom import params
from hscom import helpers as util
from hscom.Printable import DynStruct

ID_DTYPE = np.int32  # id datatype
X_DTYPE  = np.int32  # indeX datatype


class QueryRequest(DynStruct):
    # This will allow for a pipelining structure of requests and results
    def __init__(qreq):
        super(QueryRequest, qreq).__init__()
        qreq.cfg = None  # Query Config
        qreq._qcxs = []
        qreq._dcxs = []
        qreq._data_index = None  # current index
        qreq._dftup2_index = {}   # cached indexes
        qreq.query_uid = None
        qreq.featchip_uid = None
        qreq.vsmany = False
        qreq.vsone = False

    def set_cxs(qreq, qcxs, dcxs):
        qreq._qcxs = qcxs
        qreq._dcxs = dcxs

    def set_cfg(qreq, query_cfg):
        qreq.cfg = query_cfg
        qreq.vsmany = query_cfg.agg_cfg.query_type == 'vsmany'
        qreq.vsone  = query_cfg.agg_cfg.query_type == 'vsone'

    def unload_data(qreq):
        # Data TODO: Separate this
        printDBG('[qreq] unload_data()')
        qreq._data_index  = None  # current index
        qreq._dftup2_index = {}  # cached indexes
        printDBG('[qreq] unload_data(success)')

    def get_uid_list(qreq, *args, **kwargs):
        uid_list = qreq.cfg.get_uid_list(*args, **kwargs)
        if not 'noDCXS' in args:
            if len(qreq._dcxs) == 0:
                raise Exception('QueryRequest has not been populated. len(dcxs)=0')
            # In case you don't search the entire dataset
            dcxs_uid = util.hashstr_arr(qreq._dcxs, '_dcxs')
            uid_list += [dcxs_uid]
        return uid_list

    def get_uid(qreq, *args, **kwargs):
        return ''.join(qreq.get_uid_list(*args, **kwargs))

    def get_query_uid(qreq, hs, qcxs):
        query_uid = qreq.get_uid()
        hs_uid    = hs.get_db_name()
        qcxs_uid  = util.hashstr_arr(qcxs, lbl='_qcxs')
        test_uid  = hs_uid + query_uid + qcxs_uid
        return test_uid

    def get_internal_dcxs(qreq):
        dcxs = qreq._dcxs if qreq.vsmany else qreq._qcxs
        return dcxs

    def get_internal_qcxs(qreq):
        dcxs = qreq._qcxs if qreq.vsmany else qreq._dcxs
        return dcxs


class NNIndex(object):
    'Nearest Neighbor (FLANN) Index Class'
    def __init__(nn_index, hs, cx_list):
        print('[ds] building NNIndex object')
        cx2_desc  = hs.feats.cx2_desc
        assert max(cx_list) < len(cx2_desc)
        # Make unique id for indexed descriptors
        feat_uid   = hs.prefs.feat_cfg.get_uid()
        sample_uid = util.hashstr_arr(cx_list, 'dcxs')
        uid = '_' + sample_uid + feat_uid
        # Number of features per sample chip
        nFeat_iter1 = map(lambda cx: len(cx2_desc[cx]), iter(cx_list))
        nFeat_iter2 = map(lambda cx: len(cx2_desc[cx]), iter(cx_list))
        nFeat_iter3 = map(lambda cx: len(cx2_desc[cx]), iter(cx_list))
        # Inverted index from indexed descriptor to chipx and featx
        _ax2_cx = ([cx] * nFeat for (cx, nFeat) in zip(cx_list, nFeat_iter1))
        _ax2_fx = (range(nFeat) for nFeat in iter(nFeat_iter2))
        ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
        ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
        # Aggregate indexed descriptors into continuous structure
        try:
            # sanatize cx_list
            cx_list = [cx for cx, nFeat in zip(iter(cx_list), nFeat_iter3) if nFeat > 0]
            if isinstance(cx2_desc, list):
                ax2_desc = np.vstack((cx2_desc[cx] for cx in cx_list))
            elif isinstance(cx2_desc, np.ndarray):
                ax2_desc = np.vstack(cx2_desc[cx_list])
        except MemoryError as ex:
            with util.Indenter2('[mem error]'):
                print(ex)
                print('len(cx_list) = %r' % (len(cx_list),))
                print('len(cx_list) = %r' % (len(cx_list),))
            raise
        except Exception as ex:
            with util.Indenter2('[unknown error]'):
                print(ex)
                print('cx_list = %r' % (cx_list,))
            raise
        # Build/Load the flann index
        flann_params = {'algorithm': 'kdtree', 'trees': 4}
        precomp_kwargs = {'cache_dir': hs.dirs.cache_dir,
                          'uid': uid,
                          'flann_params': flann_params,
                          'force_recompute': params.args.nocache_flann}
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
             cx2_roi=[], cx2_theta=[], prop_dict={}, cx2_size=[]):
        #----
        # Image Info
        #----
        self.gx2_aif      = np.array(gx2_aif, dtype=bool)
        self.gx2_gname    = np.array(gx2_gname, dtype=str)

        #----
        # Name Info
        #----
        self.nx2_name     = np.array(nx2_name, dtype=str)

        #----
        # Chip Info
        #----
        # Chip membership info
        self.cx2_cid      = np.array(cx2_cid, dtype=ID_DTYPE)
        self.cx2_nx       = np.array(cx2_nx, dtype=X_DTYPE)
        self.cx2_gx       = np.array(cx2_gx, dtype=X_DTYPE)
        # Chip ROI (in image space)
        self.cx2_roi      = np.array(cx2_roi, dtype=np.int32)
        self.cx2_roi.shape = (self.cx2_roi.size // 4, 4)
        self.cx2_theta    = np.array(cx2_theta, dtype=np.float32)
        # Chip size (in chip space)
        # This cant belong here because it changes based on the
        # algorithm.
        #self.cx2_size     = np.array(cx2_size, dtype=np.int32)
        self.prop_dict    = prop_dict


# ___CLASS HOTSPOTTER DIRS________
class HotspotterDirs(DynStruct):
    def __init__(self, db_dir):
        super(HotspotterDirs, self).__init__()
        from . import load_data2 as ld2
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
        util.ensure_path(self.internal_dir)
        util.ensure_path(self.computed_dir)
        util.ensure_path(self.chip_dir)
        util.ensure_path(self.rchip_dir)
        util.ensure_path(self.feat_dir)
        util.ensure_path(self.result_dir)
        util.ensure_path(self.rchip_dir)
        util.ensure_path(self.qres_dir)
        util.ensure_path(self.cache_dir)
        # Shortcut to internals
        #internal_sym = db_dir + '/Shortcut-to-hs_internals'
        #computed_sym = db_dir + '/Shortcut-to-computed'
        #results_sym  = db_dir + '/Shortcut-to-results'
        #util.symlink(self.internal_dir, internal_sym, noraise=False)
        #util.symlink(self.computed_dir, computed_sym, noraise=False)
        #util.symlink(self.result_dir, results_sym, noraise=False)


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
