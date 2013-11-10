from __future__ import division, print_function
# Standard
from os.path import join
import cv2
import fnmatch
import os
import re
import sys
import textwrap
import types
# Science
import numpy as np
from PIL import Image
# Hotspotter
from Printable import DynStruct
import helpers
import params
from HotSpotter import *

# reloads this module when I mess with it
def reload_module():
    import imp, sys
    print('[ld2] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()
# ___CLASS HOTSPOTTER____
class HotSpotter(DynStruct):
    '''The HotSpotter main class is a root handle to all relevant data'''
    def __init__(hs, db_dir=None, load_basic=False, **kwargs):
        import match_chips2 as mc2
        super(HotSpotter, hs).__init__()
        hs.num_cx = None
        hs.tables = None
        hs.feats  = None
        hs.cpaths = None
        hs.dirs   = None
        hs.train_sample_cx    = None
        hs.test_sample_cx     = None
        hs.indexed_sample_cx = None
        hs.cx2_rchip_size = None
        hs.query_uid = None
        hs.matcher = mc2.Matcher(hs)
        if load_basic:
            hs.load_basic(db_dir)
        elif not db_dir is None:
            hs.load_database(db_dir, **kwargs)
    #---------------
    def load_tables(hs, db_dir):
        import load_data2 as ld2
        hs_dirs, hs_tables = ld2.load_csv_tables(db_dir)
        hs.tables  = hs_tables
        hs.dirs    = hs_dirs
        hs.num_cx = len(hs.tables.cx2_cid)
        if 'vrd' in sys.argv:
            hs.vrd()
        if 'vcd' in sys.argv:
            hs.vcd()

    def load_basic(hs, db_dir):
        hs.load_tables(db_dir)
        hs.load_chips()
        hs.load_features(load_desc=False)

    def load_all(hs, db_dir, matcher=True):
        hs.load_tables(db_dir)
        hs.load_chips()
        hs.load_features()
        hs.set_samples()
        if matcher: 
            hs.load_matcher()

    def db_name(hs):
        db_name = os.path.split(hs.dirs.db_dir)[1]
        return db_name
    #---------------
    def load_chips(hs):
        import chip_compute2 as cc2
        cc2.load_chip_paths(hs)
    #---------------
    def load_features(hs, load_kpts=True, load_desc=True):
        import feature_compute2 as fc2
        hs_feats  = fc2.load_chip_features(hs.dirs, hs.tables, hs.cpaths, load_kpts, load_desc)
        print('The new way is not yet finished and is commented out')
        #hs_feats  = fc2.load_chip_features2(hs, load_kpts, load_desc)
        hs.feats  = hs_feats
    #---------------
    def ensure_matcher_loaded(hs):
        if hs.matcher is None: 
            import match_chips2 as mc2
            hs.matcher = mc2.Matcher(hs)

    def load_matcher(hs, match_type=None):
        hs.ensure_matcher_loaded()
        if match_type is None: 
            match_type = params.__MATCH_TYPE__

    def ensure_match_type(match_type):
        if hs.matcher.match_type != match_type:
            hs.matcher.set_match_type(hs, match_type)

    def ensure_matcher(hs, match_type=None, use_reciprocal=None,
                       use_spatial=None, K=None):
        hs.ensure_matcher_loaded()
        hs.matcher.ensure_match_type(hs, match_type)
        hs.matcher.set_params(use_reciprocal, use_spatial, K)
    #---------------
    def load_database(hs, db_dir,
                      matcher=True,
                      features=True,
                      samples_range=(None, None)):
        hs.load_tables(db_dir)
        hs.load_chips()
        hs.set_sample_range(*samples_range)
        if features:
            hs.load_features()
        if matcher: 
            hs.load_matcher()
    #---------------
    #def get_query_uid(hs):
        #return hs.query_uid
    ##---------------
    #def update_query_uid(hs):
        #hs.query_uid = params.get_query_uid()
    #---------------
    def get_valid_cxs(hs):
        valid_cxs, = np.where(np.array(hs.tables.cx2_cid) > 0)
        return valid_cxs

    def get_valid_cxs_with_indexed_groundtruth(hs):
        return hs.get_valid_cxs_with_name_in_samp(hs.indexed_sample_cx)

    def flag_cxs_with_name_in_sample(hs, cxs, sample_cxs):
        cx2_nx = hs.tables.cx2_nx
        samp_nxs_set = set(cx2_nx[sample_cxs])
        in_sample_flag = np.array([cx2_nx[cx] in samp_nxs_set for cx in cxs])
        return in_sample_flag

    def get_valid_cxs_with_name_in_samp(hs, sample_cxs):
        'returns the valid_cxs which have a correct match in sample_cxs'
        cx2_nx = hs.tables.cx2_nx
        valid_cxs = hs.get_valid_cxs()
        in_sample_flag = hs.flag_cxs_with_name_in_sample(valid_cxs, sample_cxs)
        cxs_in_sample = valid_cxs[in_sample_flag]
        return cxs_in_sample

    def set_sample_split_pos(hs, pos):
        valid_cxs = hs.get_valid_cxs()
        test_samp  = valid_cxs[:pos]
        train_samp = valid_cxs[pos+1:]
        hs.set_samples(test_samp, train_samp)

    def set_sample_range(hs, pos1, pos2):
        valid_cxs = hs.get_valid_cxs()
        test_samp  = valid_cxs[pos1:pos2]
        train_samp = test_samp
        hs.set_samples(test_samp, train_samp)

    def set_samples(hs, test_samp=None,
                        train_samp=None,
                        indx_samp=None):
        ''' This is the correct function to use when setting samples '''
        print('[hs] set_samples():')
        valid_cxs = hs.get_valid_cxs()
        if test_samp is None:
            print('[hs] * default: all chips in testing')
            test_samp = valid_cxs
        else:
            print('[hs] * given: testing chips')
        if train_samp is None:
            print('[hs] * default: all chips in training')
            train_samp = valid_cxs
        else:
            print('[hs] * given: training chips')
        if indx_samp is None:
            print('[hs] * default: training set as database set')
            indx_samp = train_samp
        else:
            print('[hs] * given: indexed chips')

        # Ensure samples are sorted
        test_samp = sorted(test_samp)
        train_samp = sorted(train_samp)
        indx_samp = sorted(indx_samp)

        # Debugging and Info
        test_train_isect = np.intersect1d(test_samp, train_samp)
        indx_train_isect   = np.intersect1d(indx_samp, train_samp)
        indx_test_isect    = np.intersect1d(indx_samp, test_samp)
        lentup = (len(test_samp), len(train_samp), len(indx_samp))
        print('[hs]   ---')
        print('[hs] * num_valid_cxs = %d' % len(valid_cxs))
        print('[hs] * num_test=%d, num_train=%d, num_indx=%d' % lentup)
        print('[hs] * | isect(test, train) |  = %d' % len(test_train_isect))
        print('[hs] * | isect(indx, train) |  = %d' % len(indx_train_isect))
        print('[hs] * | isect(indx, test)  |  = %d' % len(indx_test_isect))
        
        # Unload matcher if database changed
        if hs.train_sample_cx != train_samp or hs.indexed_sample_cx != indx_samp:
            hs.matcher = None

        # Set the sample
        hs.indexed_sample_cx  = indx_samp
        hs.train_sample_cx    = train_samp
        hs.test_sample_cx     = test_samp

        # Hash the samples into sample ids
        train_hash = repr(tuple(train_samp))
        indx_hash  = repr(tuple(indx_samp))
        test_hash  = repr(tuple(test_samp))
        train_indx_hash = repr((tuple(train_samp), tuple(indx_samp)))

        train_indx_id = str(len(indx_samp))+','+helpers.hashstr(train_indx_hash)
        train_id = str(len(train_samp))+','+helpers.hashstr(train_hash)
        indx_id  = str(len(indx_samp))+','+helpers.hashstr(indx_hash)
        test_id  = str(len(test_samp))+','+helpers.hashstr(test_hash) 
 
        print('[hs] set_samples(): train_indx_id=%r' % train_indx_id)
        print('[hs] set_samples(): train_id=%r' % train_id)
        print('[hs] set_samples(): test_id=%r'  % test_id)
        print('[hs] set_samples(): indx_id=%r'  % indx_id)


        params.TRAIN_INDX_SAMPLE_ID = train_indx_id
        params.TRAIN_SAMPLE_ID      = train_id
        params.INDX_SAMPLE_ID       = indx_id
        params.TEST_SAMPLE_ID       = test_id

    #---------------
    def delete_computed_dir(hs):
        computed_dir = hs.dirs.computed_dir
        helpers.remove_files_in_dir(computed_dir, recursive=True)
    #---------------
    def vdd(hs):
        db_dir = os.path.normpath(hs.dirs.db_dir)
        print('[hs] opening db_dir: %r ' % db_dir)
        helpers.vd(db_dir)
    #---------------
    def vcd(hs):
        computed_dir = os.path.normpath(hs.dirs.computed_dir)
        print('[hs] opening computed_dir: %r ' % computed_dir)
        helpers.vd(computed_dir)
    #--------------
    def vrd(hs):
        result_dir = os.path.normpath(hs.dirs.result_dir)
        print('[hs] opening result_dir: %r ' % result_dir)
        helpers.vd(result_dir)
    #--------------
    def get_roi(hs, cx):
        roi = hs.tables.cx2_roi[cx]
        return roi
    #--------------
    def cx2_name(hs, cx):
        cx2_nx = hs.tables.cx2_nx
        nx2_name =hs.tables.nx2_name
        return nx2_name[cx2_nx[cx]]
    #--------------
    def cx2_gname(hs, cx, full=False):
        return hs.get_gname(cx, full)
    #--------------
    def get_gname(hs, cx, full=False):
        gx =  hs.tables.cx2_gx[cx]
        gname = hs.tables.gx2_gname[gx]
        if full:
            gname = join(hs.dirs.img_dir, gname)
        return gname
    #--------------
    def get_image(hs, gx=None, cx=None):
        if not cx is None: 
            return hs.cx2_image(cx)
    #--------------
    def cx2_image(hs, cx):
        img_fpath = hs.get_gname(cx, full=True)
        img = cv2.cvtColor(cv2.imread(img_fpath, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img
    #--------------
    def get_nx2_cxs(hs):
        cx2_nx = hs.tables.cx2_nx
        if len(cx2_nx) == 0:
            return [[],[]]
        max_nx = cx2_nx.max()
        nx2_cxs = [[] for _ in xrange(max_nx+1)]
        for cx, nx in enumerate(cx2_nx):
            if nx > 0:
                nx2_cxs[nx].append(cx)
        return nx2_cxs
    #--------------
    def get_other_indexed_cxs(hs, cx):
        other_cx = hs.get_other_cxs(cx)
        other_indexed_cx = np.intersect1d(other_cx, hs.indexed_sample_cx)
        return other_indexed_cx
    #--------------
    def get_groundtruth_cxs(hs, qcx):
        gt_cxs = hs.get_other_cxs(qcx)
        return gt_cxs
    #--------------
    def is_true_match(hs, qcx, cx):
        cx2_nx  = hs.tables.cx2_nx
        qnx = cx2_nx[qcx]
        nx  = cx2_nx[cx]
        is_true = nx == qnx
        is_unknown = nx <= 1
        return is_true, is_unknown
    #--------------
    UNKNOWN_STR = '???'
    TRUE_STR    = 'TRUE'
    FALSE_STR   = 'FALSE'
    def is_true_match_str(hs, qcx, cx):
        is_true, is_unknown = hs.is_true_match(qcx, cx)
        if is_unknown:
            return hs.UNKNOWN_STR
        elif is_true:
            return hs.TRUE_STR
        else:
            return hs.FALSE_STR
    #--------------
    def vs_str(hs, qcx, cx):
        if False:
            return '(qcx=%r v cx=%r)' % (qcx, cx)
        else: 
            cx2_cid = hs.tables.cx2_cid
            return '(qcid=%r v cid=%r)' % (cx2_cid[qcx], cx2_cid[cx])
    #--------------
    def num_indexed_gt_str(hs, cx):
        num_gt = len(hs.get_other_indexed_cxs(cx))
        return '#gt=%r' % num_gt
    #--------------
    def cxstr(hs, cx, digits=None):
        #return 'cx=%r' % cx
        if not np.iterable(cx):
            if not digits is None:
                return ('cid=%'+str(digits)+'d') % hs.tables.cx2_cid[cx]
            return 'cid=%d' % hs.tables.cx2_cid[cx]
        else: 
            return hs.cx_liststr(cx)
    #--------------
    def cx_liststr(hs, cx_list):
        #return 'cx=%r' % cx
        return 'cid_list=%r' % hs.tables.cx2_cid[cx_list].tolist()
    #--------------
    def cid2_cx(hs, cid):
        try: 
            array_index = helpers.array_index
            cx2_cid = hs.tables.cx2_cid
            if type(cid) is types.IntType:
                return array_index(cx2_cid, cid)
            else:
                return np.array([array_index(cx2_cid, cid_) for cid_ in cid])
        except Exception as ex:
            print('---------')
            print(cid)
            print(cx2_cid)
            print('---------')
            raise
    #--------------
    def get_other_cxs(hs, cx):
        cx2_nx   = hs.tables.cx2_nx
        nx = cx2_nx[cx]
        if nx <= 1:
            return np.array([])
        other_cx_, = np.where(cx2_nx == nx)
        other_cx  = other_cx_[other_cx_ != cx]
        return other_cx
    #--------------
    def get_chip(hs, cx):
        imread = cv2.imread
        cx2_rchip_path = hs.cpaths.cx2_rchip_path
        if not np.iterable(cx):
            return imread(cx2_rchip_path[cx])
        else:
            return [imread(cx2_rchip_path[cx_]) for cx_ in cx]
    #--------------
    def get_chip_pil(hs, cx):
        chip = Image.open(hs.cpaths.cx2_rchip_path[cx])
        return chip
    #--------------
    def get_kpts(hs, cx):
        cx2_kpts = hs.feats.cx2_kpts
        if not np.iterable(cx):
            return cx2_kpts[cx]
        else:
            return [cx2_kpts[cx_] for cx_ in cx]
    #--------------
    def _cx2_rchip_size(hs, cx):
        rchip_path = hs.cpaths.cx2_rchip_path[cx]
        return Image.open(rchip_path).size

    def load_cx2_rchip_size(hs):
        cx2_rchip_path = hs.cpaths.cx2_rchip_path
        cx2_rchip_size = [Image.open(path).size for path in cx2_rchip_path]
        hs.cx2_rchip_size = cx2_rchip_size

    def get_cx2_rchip_size(hs):
        if hs.cx2_rchip_size is None:
            hs.load_cx2_rchip_size()
        return hs.cx2_rchip_size
    #--------------
    def get_features(hs, cx):
        import spatial_verification2 as sv2
        fx2_kp     = hs.feats.cx2_kpts[cx]
        fx2_desc   = hs.feats.cx2_desc[cx]
        fx2_scale  = sv2.keypoint_scale(fx2_kp)
        return (fx2_kp, fx2_desc, fx2_scale)

    def get_feature_fn(hs, cx):
        (fx2_kp, fx2_desc, fx2_scale) = hs.get_features(cx)
        def fx2_feature(fx):
            kp    = fx2_kp[fx:fx+1]
            desc  = fx2_desc[fx]
            scale = fx2_scale[fx]
            radius = 3*np.sqrt(3*scale)
            return (kp, scale, radius, desc)
        return fx2_feature
    #--------------
    def get_assigned_matches(hs, qcx):
        cx2_desc = hs.feats.cx2_desc
        cx2_fm, cx2_fs, cx2_score = hs.matcher.assign_matches(qcx, cx2_desc)
        return cx2_fm, cx2_fs, cx2_score
    #--------------
    def get_assigned_matches_to(hs, qcx, cx):
        cx2_fm, cx2_fs, cx2_score = hs.get_assigned_matches(qcx)
        print(cx2_score.argsort()[::-1])
        fm = cx2_fm[cx]
        fs = cx2_fs[cx]
        score = cx2_score[cx]
        return fm, fs, score
    #--------------
    def free_some_memory(hs):
        print('[hs] Releasing matcher memory')
        import gc
        helpers.memory_profile()
        print("[hs] HotSpotter Referrers: "+str(gc.get_referrers(hs)))
        print("[hs] Matcher Referrers: "+str(gc.get_referrers(hs.matcher)))
        print("[hs] Desc Referrers: "+str(gc.get_referrers(hs.feats.cx2_desc)))
        #reffers = gc.get_referrers(hs.feats.cx2_desc) #del reffers
        del hs.feats.cx2_desc
        del hs.matcher
        gc.collect()
        helpers.memory_profile()
        ans = raw_input('[hs] good?')
