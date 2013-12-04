''' Computes feature representations '''
#from __init__ import *
from __future__ import division, print_function
import __builtin__
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

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[fc2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr(): reload_module()

# =======================================
# Parallelizable Work Functions          
# =======================================

feat_type2_precompute = {
    ('hesaff','sift') : extern_feat.precompute_hesaff,
    ('harris','sift') : extern_feat.precompute_harris,
    ('mser',  'sift') : extern_feat.precompute_mser,
}

# =======================================
# Main Script 
# =======================================

class HotspotterChipFeatures(DynStruct):
    def __init__(self):
        super(HotspotterChipFeatures, self).__init__()
        self.cx2_desc = None
        self.cx2_kpts = None
        self.cfg   = None

class FeatureConfig(DynStruct):
    def __init__(feat_cfg, **kwargs):
        super(FeatureConfig, feat_cfg).__init__()
        feat_cfg.feat_type = ('hesaff', 'sift')
        feat_cfg.whiten = False
        feat_cfg.scale_min = 30 #0    # 30
        feat_cfg.scale_max = 80 #9001 # 80
        feat_cfg.update(**kwargs)
    def get_dict_args(feat_cfg):
        dict_args = {
            'scale_min' : feat_cfg.scale_min,
            'scale_max' : feat_cfg.scale_max, }
        return dict_args
    def get_uid(feat_cfg):
        feat_uids = ['_FEAT(']
        feat_uids += ['%s_%s' % feat_cfg.feat_type]
        feat_uids += [',white'] * feat_cfg.whiten
        feat_uids += [',%r_%r' % (feat_cfg.scale_min, feat_cfg.scale_max)]
        feat_uids += [')']
        feat_uids + [params.get_chip_uid()]
        return feat_uids

def load_cached_feats(dpath, uid, ext, use_cache, load_kpts=True, load_desc=True):
    if not use_cache:
        return None, None
    # Try to load from the cache first
    #io.debug_smart_load(dpath, fname='*', uid=uid, ext='.*')
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
    return cx2_kpts, cx2_desc

def load_feats_from_config(hs, feat_cfg):
    feat_uid  = ''.join(feat_cfg.get_uid())
    print('[fc2] Loading features: UID='+str(feat_uid))
    hs_dirs = hs.dirs
    hs_tables = hs.tables
    hs_cpaths = hs.cpaths
    # Paths to features
    feat_dir       = hs_dirs.feat_dir
    cache_dir      = hs_dirs.cache_dir
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    cx2_cid        = hs_tables.cx2_cid
    # args for smart load/save
    dpath = cache_dir
    uid   = feat_uid
    ext   = '.npy'
    use_cache = not hs.args.nocache_feats
    cx2_kpts, cx2_desc = load_cached_feats(dpath, uid, ext, use_cache)
    if not (cx2_kpts is None or cx2_desc is None):
        # This is pretty dumb. Gotta have a more intelligent save/load
        cx2_desc_ = cx2_desc.tolist()
        cx2_kpts  = cx2_kpts.tolist()
        print('[fc2]  Loaded cx2_kpts and cx2_desc from cache')
        #print all([np.all(desc == desc_) for desc, desc_ in zip(cx2_desc, cx2_desc_)])
    else:
        print('[fc2]  Loading individual '+feat_uid+' features')
        cx2_feat_path = [feat_dir+'/CID_%d%s.npz' % (cid, feat_uid) for cid in cx2_cid]
        # Compute features, saving them to disk 
        precompute_fn   =  feat_type2_precompute[feat_cfg.feat_type]
        cx2_dict_args   = [feat_cfg.get_dict_args()]*len(cx2_rchip_path)
        precompute_args = [cx2_rchip_path, cx2_feat_path, cx2_dict_args]
        parallel_compute(precompute_fn, precompute_args, lazy=use_cache)
        # rchip_fpath=cx2_rchip_path[0]; feat_fpath=cx2_feat_path[0]
        # Load precomputed features sequentially
        cx2_kpts, cx2_desc = sequential_load_features(cx2_feat_path)
        # Cache all the features
        print('[fc2] Caching cx2_desc and cx2_kpts')
        io.smart_save(cx2_desc, dpath, 'cx2_desc', uid, ext)
        io.smart_save(cx2_kpts, dpath, 'cx2_kpts', uid, ext)
    return cx2_kpts, cx2_desc

def sequential_load_features(cx2_feat_path):
    cx2_kpts = []
    cx2_desc = []
    # Debug loading (seems to use lots of memory)
    print('\n')
    try: 
        fmt_str = helpers.make_progress_fmt_str(len(cx2_feat_path),
                                                lbl='Loading feature: ')
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
        cx2_desc = whiten_features(cx2_desc)
    return cx2_kpts, cx2_desc

def whiten_features(cx2_desc):
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

def load_features(hs, feat_cfg=None, **kwargs):
    # --- GET INPUT --- #
    hs_feats = HotspotterChipFeatures()
    if feat_cfg is None: 
        feat_cfg = FeatureConfig(**kwargs)
    feat_cfg.update(**kwargs)
    cx2_kpts, cx2_desc = load_feats_from_config(hs, feat_cfg)
    hs_feats.cx2_kpts  = cx2_kpts
    hs_feats.cx2_desc  = cx2_desc
    hs_feats.cfg = feat_cfg
    return hs_feats

def clear_feature_cache(hs, feat_cfg):
    feat_dir = hs.dirs.feat_dir
    cache_dir = hs.dirs.cache_dir
    feat_uid = ''.join(feat_cfg.get_uid())
    print('[fc2] clearing feature cache: %r' % feat_dir)
    helpers.remove_files_in_dir(feat_dir, '*'+feat_uid+'*', verbose=True, dryrun=False)
    helpers.remove_files_in_dir(cache_dir, '*'+feat_uid+'*', verbose=True, dryrun=False)
    pass

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('[fc2] __main__ = feature_compute2.py')

    __LOAD_FEATURES2__ = True
    if not __LOAD_FEATURES2__:
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
            exec(hs.execstr('hs'))
            exec(hs.tables.execstr('hs.tables'))
            exec(hs.dirs.execstr('hs.dirs'))
            exec(hs.cpaths.execstr('hs.cpaths'))
            # Load all the types of features
            #test2()
            hs.load_features()
            #cx2_desc = hs.feats.cx2_desc
            #cx2_kpts = hs.feats.cx2_kpts

            cx = helpers.get_arg_after('--cx', type_=int)
            nRandKpts = helpers.get_arg_after('--nRandKpts', type_=int)

            if not cx is None:
                df2.show_chip(hs, cx, nRandKpts=nRandKpts)
            else:
                print('usage: feature_compute.py --cx [cx] --nRandKpts [num]')
    elif __LOAD_FEATURES2__:
        import investigate_chip as iv
        import feature_compute2 as fc2
        main_locals = iv.main(load_features=False)
        exec(helpers.execstr_dict(main_locals, 'main_locals'))

        cx = helpers.get_arg_after('--cx', type_=int)
        delete_features = '--delete-features' in sys.argv
        nRandKpts = helpers.get_arg_after('--nRandKpts', type_=int)
        hs.load_features()
        if delete_features:
            fc2.clear_feature_cache(hs)
        if not cx is None:
            df2.show_chip(hs, cx, nRandKpts=nRandKpts)
        else:
            print('usage: feature_compute.py --cx [cx] --nRandKpts [num] [--delete-features]')

    exec(df2.present())
