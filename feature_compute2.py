''' Computes feature representations '''
#from __init__ import *
from __future__ import division, print_function
import __builtin__
# scientific
import numpy as np
# python
import sys
from os.path import join
# hotspotter
import helpers
import fileio as io
import DataStructures as ds
import _tpl.extern_feat as extern_feat
from Parallelize import parallel_compute

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

def whiten_features(desc_list):
    import algos
    print('[fc2] * Whitening features')
    ax2_desc = np.vstack(desc_list)
    ax2_desc_white = algos.scale_to_byte(algos.whiten(ax2_desc))
    index = 0
    offset = 0
    for cx in xrange(len(desc_list)):
        old_desc = desc_list[cx]
        print ('[fc2] * '+helpers.info(old_desc, 'old_desc'))
        offset = len(old_desc)
        new_desc = ax2_desc_white[index:(index+offset)]
        desc_list[cx] = new_desc
        index += offset

# =======================================
# Main Script 
# =======================================

def bigcache_feat_save(cache_dir, uid, ext, kpts_list, desc_list):
    print('[fc2] Caching desc_list and kpts_list')
    io.smart_save(kpts_list, cache_dir, 'kpts_list', uid, ext)
    io.smart_save(desc_list, cache_dir, 'desc_list', uid, ext)

def bigcache_feat_load(cache_dir, uid, ext):
    #io.debug_smart_load(cache_dir, fname='*', uid=uid, ext='.*')
    kpts_list = io.smart_load(cache_dir, 'kpts_list', uid, ext, can_fail=True)
    desc_list = io.smart_load(cache_dir, 'desc_list', uid, ext, can_fail=True)
    if desc_list is None or kpts_list is None: return None
    desc_list = desc_list.tolist()
    kpts_list = kpts_list.tolist()
    print('[fc2]  Loaded kpts_list and desc_list from big cache')
    return kpts_list, desc_list

def sequential_feat_load(feat_cfg, feat_fpath_list):
    kpts_list = []
    desc_list = []
    # Debug loading (seems to use lots of memory)
    print('\n')
    try: 
        make_fmt_str = helpers.make_progress_fmt_str
        fmt_str = make_fmt_str(len(feat_fpath_list), lbl='[fc2] Loading feature: ')
        for count, feat_path in enumerate(feat_fpath_list):
            try:
                npz = np.load(feat_path, mmap_mode=None)
            except IOError:
                print('\n')
                helpers.checkpath(feat_path, verbose=True)
                print('IOError on feat_path=%r' % feat_path)
                raise
            kpts = npz['arr_0']
            desc = npz['arr_1']
            npz.close()
            kpts_list.append(kpts)
            desc_list.append(desc)
            sys.stdout.write(fmt_str % (count+1))
        print('')
        print('[fc2] Finished load of individual kpts and desc')
    except MemoryError as ex:
        print('\n------------')
        print('[fc2] Out of memory')
        print('[fc2] Trying to read: %r' % feat_path)
        print('[fc2] len(kpts_list) = %d' % len(kpts_list))
        print('[fc2] len(desc_list) = %d' % len(desc_list))
        raise
    if feat_cfg.whiten:
        desc_list = whiten_features(desc_list)
    return kpts_list, desc_list

feat_type2_precompute = {
    ('hesaff','sift') : extern_feat.precompute_hesaff,
    ('harris','sift') : extern_feat.precompute_harris,
    ('mser',  'sift') : extern_feat.precompute_mser, }

def _load_features_individualy(hs, cx_list):
    use_cache = not hs.args.nocache_feats
    feat_cfg = hs.feat_cfg
    feat_dir = hs.dirs.feat_dir
    feat_uid = ''.join(feat_cfg.get_uid())
    print('[fc2]  Loading '+feat_uid+' individually')
    # Build feature paths
    rchip_fpath_list = [hs.cpaths.cx2_rchip_path[cx] for cx in iter(cx_list)]
    cid_list = hs.tables.cx2_cid[cx_list]
    feat_fname_fmt = ''.join(('cid%d',feat_uid,'.npz'))
    feat_fpath_list = [join(feat_dir, feat_fname_fmt % cid) for cid in cid_list]
    # Compute features in parallel, saving them to disk 
    kwargs_list = [feat_cfg.get_dict_args()]*len(rchip_fpath_list)
    precompute_args = [rchip_fpath_list, feat_fpath_list, kwargs_list]
    num_procs = hs.args.num_procs
    if len(cx_list) < num_procs / 2:
        num_procs = 1 # Hack for small amount of tasks
    pfc_kwargs = {'num_procs':num_procs, 'lazy':use_cache}
    precompute_fn = feat_type2_precompute[feat_cfg.feat_type]
    parallel_compute(precompute_fn, precompute_args, **pfc_kwargs)
    # Load precomputed features sequentially
    kpts_list, desc_list = sequential_feat_load(feat_cfg, feat_fpath_list)
    return kpts_list, desc_list
    
def _load_features_bigcache(hs, cx_list):
    # args for smart load/save
    feat_cfg = hs.feat_cfg
    feat_uid = ''.join(feat_cfg.get_uid())
    cache_dir  = hs.dirs.cache_dir
    sample_uid = helpers.make_sample_id(cx_list, lbl='cids')
    bigcache_uid = '_'.join((feat_uid, sample_uid))
    ext = '.npy'
    loaded = bigcache_feat_load(cache_dir, bigcache_uid, ext)
    if loaded is not None: # Cache Hit
        kpts_list, desc_list = loaded
    else: # Cache Miss
        kpts_list, desc_list = _load_features_individualy(hs, cx_list)
        # Cache all the features
        bigcache_feat_save(cache_dir, bigcache_uid, ext, kpts_list, desc_list)
    return kpts_list, desc_list

def load_features(hs, cx_list=None, **kwargs):
    print('\n=============================')
    print('[fc2] Precomputing and loading features: %r' % hs.db_name())
    print('=============================')
    #----------------
    # COMPUTE SETUP 
    #----------------
    use_cache = not hs.args.nocache_feats
    use_big_cache = use_cache and cx_list is None
    # Get/Update FeatConfig and ChipFeatures object
    if hs.feats is None: hs.feats = ds.HotspotterChipFeatures()
    if hs.feat_cfg is not None: hs.feat_cfg.update(**kwargs)
    else: hs.feat_cfg = ds.FeatureConfig(hs, **kwargs)
    feat_uid = ''.join(hs.feat_cfg.get_uid())
    print('[fc2] feat_uid = %r' % feat_uid)
    if hs.feats.feat_uid != '' and hs.feats.feat_uid != feat_uid:
        raise Exception('Disagreement: feat_uid = %r' % hs.feats.feat_uid)
    # Get the list of chips to load
    cx_list = hs.get_valid_cxs() if cx_list is None else cx_list
    if cx_list == []: return # HACK
    if not np.iterable(cx_list): cx_list = [cx_list]
    if use_big_cache: # use only if all descriptors requested
        kpts_list, desc_list = _load_features_bigcache(hs, cx_list)
    else:
        kpts_list, desc_list = _load_features_individualy(hs, cx_list)
    # Extend the datastructure if needed
    list_size = max(cx_list)
    helpers.ensure_list_size(hs.feats.cx2_kpts, list_size)
    helpers.ensure_list_size(hs.feats.cx2_desc, list_size)
    # Copy the values into the ChipPaths object
    for lx, cx in enumerate(cx_list):
        hs.feats.cx2_kpts[cx] = kpts_list[lx]
    for lx, cx in enumerate(cx_list):
        hs.feats.cx2_desc[cx] = desc_list[lx]
    hs.feats.feat_uid = feat_uid
    print('[fc2]=============================')

def clear_feature_cache(hs):
    feat_cfg = hs.feat_cfg
    feat_dir = hs.dirs.feat_dir
    cache_dir = hs.dirs.cache_dir
    feat_uid = ''.join(feat_cfg.get_uid())
    print('[fc2] clearing feature cache: %r' % feat_dir)
    helpers.remove_files_in_dir(feat_dir, '*'+feat_uid+'*', verbose=True, dryrun=False)
    helpers.remove_files_in_dir(cache_dir, '*'+feat_uid+'*', verbose=True, dryrun=False)
    pass

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('[fc2] __main__ = feature_compute2.py')
    import main
    import HotSpotter
    import vizualizations as viz
    import feature_compute2 as fc2
    from feature_compute2 import *
    # Debugging vars
    feat_cfg=None
    cx_list=None
    kwargs = {}
    # --- LOAD TABLES --- #
    args = main.parse_arguments(db='NAUTS')
    hs = HotSpotter.HotSpotter(args)
    hs.load_tables()
    hs.set_samples()
    # --- LOAD CHIPS --- #
    hs.load_configs()
    hs.load_chips()
    # --- LOAD FEATURES --- #
    load_features(hs)
    cx = helpers.get_arg_after('--cx', type_=int)
    delete_features = '--delete-features' in sys.argv
    nRandKpts = helpers.get_arg_after('--nRandKpts', type_=int)
    if delete_features:
        fc2.clear_feature_cache(hs)
    if not cx is None:
        viz.show_chip(hs, cx, nRandKpts=nRandKpts)
    else:
        print('usage: feature_compute.py --cx [cx] --nRandKpts [num] [--delete-features]')

    exec(viz.present())
