from __future__ import division, print_function
import __builtin__
import sys
# Standard
import os
# Science
import numpy as np
from PIL import Image
# Hotspotter
import load_data2 as ld2
import chip_compute2 as cc2
import feature_compute2 as fc2
import match_chips3 as mc3
import convert_db
from Printable import DynStruct
import DataStructures as ds
import helpers
import tools
import params
import fileio as io
from os.path import exists, join, split, relpath
from itertools import izip
import shutil

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
    print('[hs] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr(): reload_module()

def _on_loaded_tables(hs):
    'checks relevant arguments after loading tables'
    args = hs.args
    if args.vrd or args.vrdq:
        hs.vrd()
        if args.vrdq: sys.exit(1)
    if args.vcd or args.vcdq:
        hs.vcd()
        if args.vcdq: sys.exit(1)

def is_invalid_path(db_dir): 
    return db_dir is None or not exists(db_dir)

# ___CLASS HOTSPOTTER____
class HotSpotter(DynStruct):
    '''The HotSpotter main class is a root handle to all relevant data'''
    def __init__(hs, args=None, db_dir=None):
        super(HotSpotter, hs).__init__()
        hs.args = args
        hs.num_cx = None
        hs.tables = None
        hs.feats  = None
        hs.cpaths = None
        hs.dirs   = None
        hs.train_sample_cx   = None
        hs.test_sample_cx    = None
        hs.indexed_sample_cx = None
        hs.cx2_rchip_size = None
        hs.query_uid = None
        hs.needs_save = False
        #
        hs.feat_cfg  = None
        hs.chip_cfg  = None
        hs.query_cfg = None
        if db_dir is not None:
            hs.args.dbdir = db_dir
    #---------------
    # Loading Functions
    #---------------
    def load(hs, load_all=False):
        '(current load function) Loads the appropriate database'
        print('[hs] load()')
        hs.load_tables()
        hs.load_configs()
        hs.set_samples()
        if load_all:
            print('[hs] load_all=True')
            hs.load_chips()
            hs.load_features()
        else:
            print('[hs] load_all=False')
            hs.load_chips([])
            hs.load_features([])
        return hs

    def load_configs(hs):
        hs.chip_cfg = ds.get_chip_cfg()
        kwargs_ = {}
        (kwargs_['scale_min'],
         kwargs_['scale_max']) = hs.args.sthresh
        hs.feat_cfg = ds.get_feat_cfg(hs, **kwargs_)

    def load_tables(hs):
        # Check to make sure dbdir is specified correctly
        if hs.args.dbdir is None or not exists(hs.args.dbdir):
            raise ValueError('db_dir=%r does not exist!' % (hs.args.dbdir))
        convert_db.convert_if_needed(hs.args.dbdir)
        hs_dirs, hs_tables = ld2.load_csv_tables(hs.args.dbdir)
        hs.tables  = hs_tables
        hs.dirs    = hs_dirs
        hs.num_cx = len(hs.tables.cx2_cid)
        _on_loaded_tables(hs)

    def load_chips(hs, cx_list=None):
        cc2.load_chips(hs, cx_list)

    def load_features(hs, cx_list=None):
        fc2.load_features(hs, cx_list=cx_list)

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
        DEBUG_SET_SAMPLE = False
        if DEBUG_SET_SAMPLE:
            test_train_isect = np.intersect1d(test_samp, train_samp)
            indx_train_isect = np.intersect1d(indx_samp, train_samp)
            indx_test_isect  = np.intersect1d(indx_samp, test_samp)
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
        train_indx_hash = repr((tuple(train_samp), tuple(indx_samp)))
        train_indx_id = str(len(indx_samp))+','+helpers.hashstr(train_indx_hash)
        train_id = helpers.make_sample_id(train_samp)
        indx_id  = helpers.make_sample_id(indx_samp)
        test_id  = helpers.make_sample_id(test_samp) 
 
        print('[hs] set_samples(): train_indx_id=%r' % train_indx_id)
        print('[hs] set_samples(): train_id=%r' % train_id)
        print('[hs] set_samples(): test_id=%r'  % test_id)
        print('[hs] set_samples(): indx_id=%r'  % indx_id)
        params.TRAIN_INDX_SAMPLE_ID = train_indx_id # Depricate
        params.TRAIN_SAMPLE_ID      = train_id
        params.INDX_SAMPLE_ID       = indx_id
        params.TEST_SAMPLE_ID       = test_id
        # Fix
        hs.train_id = train_id
        hs.indx_id  = indx_id
        # The query_cfg must resample
        hs.query_cfg = None


    #---------------
    # Query Functions
    #---------------
    def query(hs, qcx):
        if hs.query_cfg is None:
            hs.query_cfg = ds.get_vsmany_cfg(hs)
        res = mc3.query_database(hs, qcx, hs.query_cfg)
        return res

    # ---------------
    # Modifying functions
    # ---------------

    def unload_cxdata(hs, cx):
        'unloads features and chips. not tables'
        print('[hs] unload_cxdata(cx=%r)' % cx)
        lists = [hs.feats.cx2_kpts,
                 hs.feats.cx2_desc,
                 hs.cpaths.cx2_rchip_path,
                 hs.cpaths.cx2_chip_path]
        for list_ in lists:
            helpers.ensure_list_size(list_, cx)
            list_[cx] = None

    def delete_cxdata(hs, cx):
        'deletes features and chips. not tables'
        hs.unload_cxdata(cx)
        print('[hs] delete_cxdata(cx=%r)' % cx)
        cid = hs.tables.cx2_cid[cx]
        cid_str_list = ['cid%d_' % cid,
                        'qcid=%d.npz' % cid, ]
        for cid_str in cid_str_list:
            helpers.remove_files_in_dir(hs.dirs.computed_dir, '*'+cid_str+'*',
                                        recursive=True, verbose=True, dryrun=False)

    def on_modification(hs, cx=None, gx=None):
        hs.needs_save = True
    def on_addition(hs, cx=None, gx=None, resample=True):
        hs.needs_save = True
        if resample:
            hs.set_samples(hs)
    def on_deletion(hs, cx=None, gx=None, resample=True):
        hs.needs_save = True
        if resample:
            hs.set_samples()

    def change_roi(hs, cx, new_roi):
        hs.delete_cxdata(cx)
        hs.on_modification(cx=cx)
        hs.tables.cx2_roi[cx] = new_roi

    def change_theta(hs, cx, new_theta):
        hs.delete_cxdata(cx)
        hs.on_modification(cx=cx)
        hs.tables.cx2_theta[cx] = new_theta

    def change_name(hs, cx, new_name):
        hs.on_modification()
        new_nx_ = np.where(hs.tables.nx2_name == new_name)[0]
        if len(new_nx_) == 0:
            new_nx = hs.add_name(new_name)
        else:
            new_nx = new_nx_[0]
        hs.tables.cx2_nx[cx] = new_nx

    # ---------------
    # Adding functions
    # ---------------

    def add_name(hs, name): 
        nx2_name = hs.tables.nx2_name.tolist()
        nx2_name.append(name)
        hs.tables.nx2_name = np.array(nx2_name)
        nx = len(hs.tables.nx2_name)
        hs.on_addition()
        return nx

    def add_chip(hs, gx, roi):
        print('[hs] adding chip to gx=%r' % gx)
        if len(hs.tables.cx2_cid) > 0:
            next_cid = hs.tables.cx2_cid.max() + 1
        else:
            next_cid = 1
        # Allocate space for a new chip
        hs.tables.cx2_cid   = np.concatenate((hs.tables.cx2_cid, [next_cid]))
        hs.tables.cx2_nx    = np.concatenate((hs.tables.cx2_nx,  [0]))
        hs.tables.cx2_gx    = np.concatenate((hs.tables.cx2_gx,  [gx]))
        hs.tables.cx2_roi   = np.vstack((hs.tables.cx2_roi, [roi]))
        hs.tables.cx2_theta = np.concatenate((hs.tables.cx2_theta, [0]))
        for key in hs.tables.prop_dict.keys():
            hs.tables.prop_dict[key] = np.concatenate(hs.tables.prop_dict[key], [''])
        hs.num_cx += 1
        cx = len(hs.tables.cx2_cid)-1
        hs.on_addition()
        return cx

    def add_images(hs, fpath_list, move_images=True):
        nImages = len(fpath_list)
        print('[hs.add_imgs] adding %d images' % nImages)
        img_dir = hs.dirs.img_dir
        copy_list = []
        helpers.ensurepath(img_dir)
        if move_images:
            # Build lists of where the new images will be
            fpath_list2 = [join(img_dir, split(fpath)[1]) for fpath in fpath_list]
            copy_iter = izip(fpath_list, fpath_list2) 
            copy_list = [(src, dst) for src, dst in copy_iter if not exists(dst)]
            nExist = len(fpath_list2) - len(copy_list)
            print('[hs] copying %d images' % len(copy_list))
            print('[hs] %d images already exist' % nExist)
            for src, dst in copy_list:
                shutil.copy(src, dst)
        else:
            print('[hs.add_imgs] using original image paths')
            fpath_list2 = fpath_list
        # Get location of the new images relative to the image dir
        gx2_gname = hs.tables.gx2_gname.tolist()
        relpath_list = [relpath(fpath, img_dir) for fpath in fpath_list2]
        current_gname_set = set(gx2_gname)
        # Check to make sure the gnames are not currently indexed
        new_gnames = [gname for gname in relpath_list if not gname in current_gname_set]
        nNewImages = len(new_gnames)
        nIndexed = nImages - nNewImages
        print('[hs.add_imgs] new_gnames:\n'+'\n'.join(new_gnames))
        print('[hs.add_imgs] %d images already indexed.' % nIndexed)
        print('[hs.add_imgs] Added %d new images.' % nIndexed)
        # Append the new gnames to the hotspotter table
        hs.tables.gx2_gname = np.array(gx2_gname+new_gnames)
        if nNewImages > 0:
            hs.on_addition()
        return nNewImages

    # ---------------
    # Deleting functions
    # ---------------

    def delete_chip(hs, cx, resample=True):
        hs.delete_cxdata(cx)
        hs.tables.cx2_cid[cx] = -1
        hs.tables.cx2_gx[cx]  = -1
        hs.tables.cx2_nx[cx]  = -1
        hs.on_deletion(cx=cx, resample=resample)

    def delete_image(hs, gx):
        cx_list = hs.gx2_cxs(gx)
        for cx in cx_list:
            hs.delete_chip(cx, resample=False)
        hs.tables.gx2_gname[gx] = ''
        hs.on_deletion(gx=gx)

    # ---------------
    # Getting functions
    # ---------------
    def db_name(hs, devmode=False):
        db_name = split(hs.dirs.db_dir)[1]
        if devmode:
            # Grab the dev name insetad
            import params
            dev_databases = params.dev_databases
            db_tups = [(v, k) for k, v in dev_databases.iteritems() if v is not None]
            #print('  \n'.join(map(str,db_tups)))
            dev_dbs = dict((split(v)[1] ,k) for v, k in db_tups)
            db_name = dev_dbs[db_name]
        return db_name
    #---------------
    def get_valid_cxs(hs):
        valid_cxs = np.where(hs.tables.cx2_cid > 0)[0]
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

    def get_indexed_uid(hs, with_train=True, with_indx=True):
        indexed_uid = ''
        if with_train:
            indexed_uid += '_trainID('+hs.train_id+')'
        if with_indx:
            indexed_uid += '_indxID('+hs.indx_id+')'
        # depends on feat
        indexed_uid += hs.feats.cfg.get_uid()
        return indexed_uid
    #---------------
    def save_database(hs):
        print('[hs] save_database')
        ld2.write_csv_tables(hs)
        hs.needs_save = False
    #---------------
    def delete_computed_dir(hs):
        computed_dir = hs.dirs.computed_dir
        [hs.unload_cxdata(cx) for cx in hs.get_valid_cxs()]
        helpers.remove_files_in_dir(computed_dir, recursive=True, verbose=True,
                                    dryrun=False)
    #---------------
    def delete_global_prefs(hs):
        global_cache_dir = io.GLOBAL_CACHE_DIR
        helpers.remove_files_in_dir(global_cache_dir, recursive=True, verbose=True,
                                    dryrun=False)
    #---------------
    def vdd(hs):
        db_dir = os.path.normpath(hs.dirs.db_dir)
        print('[hs] viewing db_dir: %r ' % db_dir)
        helpers.vd(db_dir)
    #---------------
    def vcd(hs):
        computed_dir = os.path.normpath(hs.dirs.computed_dir)
        print('[hs] viewing computed_dir: %r ' % computed_dir)
        helpers.vd(computed_dir)
    #---------------
    def vgd(hs):
        global_dir = io.GLOBAL_CACHE_DIR
        print('[hs] viewing global_dir: %r ' % global_dir)
        helpers.vd(global_dir)
    #--------------
    def vrd(hs):
        result_dir = os.path.normpath(hs.dirs.result_dir)
        print('[hs] viewing result_dir: %r ' % result_dir)
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
        gx =  hs.tables.cx2_gx[cx]
        return hs.gx2_gname(gx, full)
    #--------------
    def gx2_gname(hs, gx, full=False):
        gname = hs.tables.gx2_gname[gx]
        if full:
            gname = join(hs.dirs.img_dir, gname)
        return gname
    #--------------
    def get_image(hs, gx=None, cx=None):
        if not cx is None: 
            return hs.cx2_image(cx)
        if not gx is None: 
            return hs.gx2_image(gx)
    #--------------
    def cx2_image(hs, cx):
        gx =  hs.tables.cx2_gx[cx]
        return hs.gx2_image(gx)
    #--------------
    def gx2_image(hs, gx):
        img_fpath = hs.gx2_gname(gx, full=True)
        img = io.imread(img_fpath)
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
    def gx2_cxs(hs, gx):
        cx_list = np.where(hs.tables.cx2_gx == gx)[0]
        return cx_list 

    def get_gx2_cxs(hs):
        cx2_gx = hs.tables.cx2_gx
        max_gx = len(hs.tables.gx2_gname)
        gx2_cxs = [[] for _ in xrange(max_gx+1)]
        for cx, gx in enumerate(cx2_gx):
            if gx == -1: continue
            gx2_cxs[gx].append(cx)
        return gx2_cxs
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
            #return '(qcx=%r v cx=%r)' % (qcx, cx)
            return 'cx(%r v %r)' % (qcx, cx)
        else: 
            cx2_cid = hs.tables.cx2_cid
            return 'cid(%r v %r)' % (cx2_cid[qcx], cx2_cid[cx])
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
    def cid2_gx(hs, cid):
        'chip_id ==> image_index'
        cx = hs.tables.cid2_cx(cid)
        gx = hs.tables.cx2_gx[cx]
        return gx

    @tools.class_iter_input
    def cid2_cx(hs, cid_input):
        'chip_id ==> chip_index'
        index_of = tools.index_of
        cx2_cid = hs.tables.cx2_cid
        cx_output = [index_of(cid, cx2_cid) for cid in cid_input]
        return cx_output
    #--------------
    @tools.class_iter_input
    def _try_cxlist_get(hs, cx_input, cx2_var):
        ''' Input: cx_input: a vector input, cx2_var: a array mapping cx to a
        variable Returns: list of values corresponding with cx_input '''
        ret = [cx2_var[cx_] for cx_ in cx_input]
        # None is invalid in a cx2_var array
        if any([val is None for val in ret]):
            raise IndexError()
        return ret
    #--------------
    def _onthefly_cxlist_get(hs, cx_input, cx2_var, load_fn):
        '''tries to get from the cx_input indexed list and performs a cx load function
        if unable to get failure'''
        try: 
            ret = hs._try_cxlist_get(cx_input, cx2_var)
        except IndexError as ex:
            load_fn(cx_input)
            ret = hs._try_cxlist_get(cx_input, cx2_var)
        return ret

    #--------------
    def get_desc(hs, cx_input):
        cx2_desc = hs.feats.cx2_desc
        return hs._onthefly_cxlist_get(cx_input, cx2_desc, hs.load_features)
    #--------------
    def get_kpts(hs, cx_input):
        cx2_kpts = hs.feats.cx2_kpts
        return hs._onthefly_cxlist_get(cx_input, cx2_kpts, hs.load_features)
    #--------------
    def get_rchip_path(hs, cx_input):
        cx2_rchip_path = hs.cpaths.cx2_rchip_path
        return hs._onthefly_cxlist_get(cx_input, cx2_rchip_path, hs.load_chips)
    #--------------
    def get_chip(hs, cx):
        rchip_path = hs.get_rchip_path(cx)
        if np.iterable(cx):
            return [io.imread(fpath) for fpath in rchip_path]
        else:
            return io.imread(rchip_path)
    #--------------
    def get_chip_pil(hs, cx):
        chip = Image.open(hs.cpaths.cx2_rchip_path[cx])
        return chip
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
