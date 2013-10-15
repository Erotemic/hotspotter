''' Computes feature representations '''
#from __init__ import *
from __future__ import division
# hotspotter
import draw_func2 as df2
import algos
import params
import tpl.extern_feat as extern_feat
import helpers
from Parallelize import parallel_compute
from Printable import DynStruct
import fileio as io
# scientific
from numpy import array, cos, float32, hstack, pi, round, sqrt, uint8, zeros
import numpy as np
import cv2
# python
import sys
import os
from os.path import exists, join

def reload_module():
    import imp, sys
    print('[fc2] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def root_sift(desc):
    ''' Takes the square root of each descriptor and returns the features in the
    range 0 to 255 (as uint8) '''
    desc_ = array([sqrt(d) for d in algos.norm_zero_one(desc)])
    desc_ = array([round(255.0 * d) for d in algos.norm_zero_one(desc_)], dtype=uint8)
    return desc_

# =======================================
# Parallelizable Work Functions          
# =======================================

feat_type2_precompute = {
    ('hesaff','sift') : extern_feat.precompute_hesaff,
    ('mser','sift')   : extern_feat.precompute_mser
}

# =======================================
# Main Script 
# =======================================

class HotspotterChipFeatures(DynStruct):
    def __init__(self):
        super(HotspotterChipFeatures, self).__init__()
        self.is_binary = False
        self.cx2_desc = None
        self.cx2_kpts = None
        self.feat_type = None

# Decorators? 
#def cache_features():
#def uncache_features():

# hs is a handle to whatever data is loaded 
# if you aren't using something
# del hs.something
# EG: del hs.feats
'''
 del.hs.feats['sift']
 ax2_features = hs.feats['sift']
 ax2_features = hs.feats['constructed-sift']
 ax2_features = hs.feats['mser']
'''

def index_features(hs):
    mser_feats = hs.feats['mser']
    sift_feats = hs.feats['sift']

    feats = sift_feats
    ax2_desc = feats.ax2_desc
    ax2_kpts = feats.ax2_kpts
    cx2_nFeats = feats.cx2_nFeats
    cx2_axStart = np.cumsum(cx2_nFeats)
    def cx2_axs(cx):
        nFeats  = cx2_nFeats[cx]
        axStart = cx2_axStart[cx]
        fx2_ax = np.arange(nFeats, axStart+nFeats)
        return fx2_ax

    hs.feats.cx2_axs = cx2_axs

    #ax_in_grid[0,0] = [...]
    #ax_in_part[0]   = [...]
    #valid_ax        = [...]

def test2(hs):
    cx_list = hs.get_valid_cxs()
    #feature_types = params.FEAT_TYPE
    feature_types = [('hesaff','sift'), ('mser','sift')]
    feat_type = 'mser'
    load_features(hs, cx_list, feature_types)

class Features(object):
    def __init__(self, feat_type):
        self.feat_type = feat_type


def load_features(hs, cx_list, feature_types, regions=None):
    feat_dict = {}

    feat_dir       = hs.dirs.feat_dir
    cx2_rchip_path = hs.cpaths.cx2_rchip_path
    cx2_cid        = hs.tables.cx2_cid
    valid_cxs = hs.get_valid_cxs()

    def precompute_feat_type(hs, feat_type, cx_list):
        if not feat_type in feat_dict.keys():
            feat_dict[feat_type] = Features(feat_type)
        feat = feat_dict[feat_type]
        # Build Parallel Jobs, Compute features, saving them to disk.
        # Then Run Parallel Jobs 
        cid_iter = (cx2_cid[cx] for cx in cx_list)
        feat_type_str = repr(feat_type).replace(' ','').replace('(','').replace(')','').replace('\'','')
        cx2_feat_path = [feat_dir+'/CID_%d_%s.npz' % (cid, feat_type_str) for cid in cid_iter]
        precompute_fn = feat_type2_precompute[feat_type]
        parallel_compute(precompute_fn, [cx2_rchip_path, cx2_feat_path])

    for feat_type in feature_types:
        precompute_feat_type(hs, feat_type, cx_list)

def load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid,
                        feat_type, feat_uid, cache_dir, 
                        load_kpts=True, load_desc=True):
    print('[fc2] Loading '+feat_type+' features: UID='+str(feat_uid))
    # args for smart load/save
    dpath = cache_dir
    uid   = feat_uid
    ext   = '.npy'
    #io.debug_smart_load(dpath, fname='*', uid=uid, ext='.*')
    # Try to read cache
    if load_kpts:
        cx2_kpts = io.smart_load(dpath, 'cx2_kpts', uid, ext, can_fail=True)
    else: 
        raise NotImplemented('[fc2] that hack is for desc only')
    if load_desc: 
        cx2_desc = io.smart_load(dpath, 'cx2_desc', uid, ext, can_fail=True)
    elif not cx2_kpts is None: #HACK
        print('[fc2] ! Not loading descriptors')
        cx2_desc = np.array([np.array([])] * len(cx2_kpts))
    else:
        cx2_desc = None

    if (not cx2_kpts is None and not cx2_desc is None):
        # This is pretty dumb. Gotta have a more intelligent save/load
        cx2_desc_ = cx2_desc.tolist()
        cx2_kpts  = cx2_kpts.tolist()
        print('[fc2]  Loaded cx2_kpts and cx2_desc from cache')
        #print all([np.all(desc == desc_) for desc, desc_ in zip(cx2_desc, cx2_desc_)])
    else:
        print('[fc2]  Loading individual '+feat_uid+' features')
        cx2_feat_path = [ feat_dir+'/CID_%d_%s.npz' % (cid, feat_uid) for cid in cx2_cid]

        # Compute features, saving them to disk 
        precompute_fn = feat_type2_precompute[feat_type]
        parallel_compute(precompute_fn, [cx2_rchip_path, cx2_feat_path])

        # Load precomputed features 
        cx2_kpts = []
        cx2_desc = []
        # Debug loading (seems to use lots of memory)
        fmt_str = helpers.make_progress_fmt_str(len(cx2_feat_path),
                                                lbl='Loading feature: ')
        print('\n')
        try: 
            for cx, feat_path in enumerate(cx2_feat_path):
                npz = np.load(feat_path, mmap_mode=None)
                kpts = npz['arr_0']
                desc = npz['arr_1']
                npz.close()
                cx2_kpts.append(kpts)
                cx2_desc.append(desc)
                helpers.print_(fmt_str % cx)
            print('[fc2] Finished load of individual kpts and desc')
            cx2_desc = np.array(cx2_desc)
        except MemoryError as ex:
            print('\n------------')
            print('[fc2] Out of memory')
            print('[fc2] Trying to read: %r' % feat_path)
            print('[fc2] len(cx2_kpts) = %d' % len(cx2_kpts))
            print('[fc2] len(cx2_desc) = %d' % len(cx2_desc))
            raise
        if params.WHITEN_FEATS:
            print('[fc2] * Whitening features')
            ax2_desc = np.vstack(cx2_desc)
            ax2_desc_white = algos.scale_to_byte(algos.whiten(ax2_desc))
            index = 0
            offset = 0
            for cx in xrange(len(cx2_desc)):
                old_desc = cx2_desc[cx]
                print ('[fc2] * '+helpers.info(old_desc, 'old_desc'))
                offset = len(old_desc)
                new_desc = ax2_desc_white[index:(index+offset)]
                cx2_desc[cx] = new_desc
                index += offset
        # Cache all the features
        print('[fc2] Caching cx2_desc and cx2_kpts')
        io.smart_save(cx2_desc, dpath, 'cx2_desc', uid, ext)
        io.smart_save(cx2_kpts, dpath, 'cx2_kpts', uid, ext)
    return cx2_kpts, cx2_desc
    
def load_chip_features2(hs, load_kpts=True, load_desc=True):
    print('\n=============================')
    print('[fc2] Computing and loading features for %r' % hs.db_name())
    print('=============================')
    return load_chip_features(hs.dirs, hs.tables, hs.cpaths, load_kpts, load_desc)

def load_chip_features(hs_dirs, hs_tables, hs_cpaths, load_kpts=True,
                       load_desc=True):
    # --- GET INPUT --- #
    hs_feats = HotspotterChipFeatures()
    # Paths to features
    feat_dir       = hs_dirs.feat_dir
    cache_dir      = hs_dirs.cache_dir
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    cx2_cid        = hs_tables.cx2_cid
    # Load all the types of features
    feat_uid = params.get_feat_uid()
    feat_type = params.__FEAT_TYPE__
    cx2_kpts, cx2_desc = load_chip_feat_type(feat_dir, 
                                             cx2_rchip_path, 
                                             cx2_cid, 
                                             feat_type, 
                                             feat_uid, 
                                             cache_dir,
                                             load_kpts,
                                             load_desc)
    hs_feats.feat_type = params.__FEAT_TYPE__
    hs_feats.cx2_kpts  = cx2_kpts
    hs_feats.cx2_desc  = cx2_desc
    #hs_feats.cx2_feats_sift   = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, 'SIFT')
    #hs_feats.cx2_feats_freak  = load_chip_feat_type(feat_dir, cx2_rchip_path, cx2_cid, 'FREAK')
    return hs_feats

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('[fc2] __main__ = feature_compute2.py')

    __DEV_MODE__ = True
    if __DEV_MODE__ or 'test' in sys.argv:
        import load_data2 as ld2
        import match_chips2 as mc2
        import chip_compute2
        import params
        # --- CHOOSE DATABASE --- #
        db_dir = ld2.DEFAULT 
        hs = ld2.HotSpotter()
        hs.load_tables(db_dir)
        hs.load_chips()
        hs.set_samples()
        test2(hs)
        #exec(hs.execstr('hs'))
        #exec(hs.tables.execstr('hs.tables'))
        #exec(hs.dirs.execstr('hs.tables'))
        #exec(hs.cpaths.execstr('hs.tables'))
        ## Load all the types of features
        #feat_uid = params.get_feat_uid()
        #feat_type = params.__FEAT_TYPE__

        #hs.load_features()
        #cx2_desc = hs.feats.cx2_desc
        #cx2_kpts = hs.feats.cx2_kpts

        #cx = helpers.get_arg_after('--cx', type_=int)
        #if not cx is None:
            #df2.show_chip(hs, cx)
        #else:
            #print('usage: feature_compute.py --cx [cx]')

    exec(df2.present())
