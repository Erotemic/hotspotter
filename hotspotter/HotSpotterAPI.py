from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[hs]')
import sys
# Standard
import os
from os.path import exists, join, split, relpath
from itertools import izip
import shutil
import datetime
# Science
import numpy as np
from PIL import Image
# Hotspotter
from hscom import fileio as io
from hscom import helpers
from hscom import tools
from hscom.Printable import DynStruct
from hscom.Preferences import Pref
import DataStructures as ds
import Config
import chip_compute2 as cc2
import feature_compute2 as fc2
import load_data2 as ld2
import match_chips3 as mc3
import matching_functions as mf


def _checkargs_onload(hs):
    'checks relevant arguments after loading tables'
    args = hs.args
    if args is None:
        return
    if args.vrd or args.vrdq:
        hs.vrd()
        if args.vrdq:
            sys.exit(1)
    if args.vcd or args.vcdq:
        hs.vcd()
        if args.vcdq:
            sys.exit(1)
    if hs.args.delete_cache:
        hs.delete_cache()
    if hs.args.quit:
        print('[hs] user requested quit.')
        sys.exit(1)


# Logic for hotspotter functions are module level functions for easy reloading.

def _import_scripts(hs):
    import scripts
    scripts.rrr()


@profile
def _get_datatup_list(hs, tblname, index_list, header_order, extra_cols):
    '''
    Used by guiback to get lists of datatuples by internal column names.
    '''
    cols = _datatup_cols(hs, tblname)
    cols.update(extra_cols)
    unknown_header = lambda indexes: ['ERROR!' for gx in indexes]
    get_tup = lambda header: cols.get(header, unknown_header)(index_list)
    unziped_tups = [get_tup(header) for header in header_order]
    datatup_list = [tup for tup in izip(*unziped_tups)]
    return datatup_list


@profile
def _datatup_cols(hs, tblname, cx2_score=None):
    '''
    Returns maps which map which maps internal column names
    to lazy evaluation functions which compute the data (hence the lambdas)
    '''
    # Chips
    cx2_cid   = hs.tables.cx2_cid
    cx2_roi   = hs.tables.cx2_roi
    cx2_theta = hs.tables.cx2_theta
    cx2_nx    = hs.tables.cx2_nx
    cx2_gx    = hs.tables.cx2_gx
    prop_dict = hs.tables.prop_dict
    # Name
    nx2_name  = hs.tables.nx2_name
    nx2_cxs   = hs.get_nx2_cxs()
    # Image
    gx2_gname = hs.tables.gx2_gname
    gx2_aif   = hs.tables.gx2_aif
    # Features
    cx2_kpts  = hs.feats.cx2_kpts
    # Return requested columns
    if tblname == 'nxs':
        cols = {
            'nx':    lambda nxs: nxs,
            'name':  lambda nxs: [nx2_name[nx] for nx in iter(nxs)],
            #'nCxs':  lambda nxs: hs.nx2_cxs(nxs),
            'nCxs':  lambda nxs: [len(nx2_cxs[nx]) for nx in iter(nxs)],
        }
    elif tblname == 'gxs':
        cols = {
            'gx':    lambda gxs: gxs,
            'aif':   lambda gxs: [gx2_aif[gx] for gx in iter(gxs)],
            'gname': lambda gxs: [gx2_gname[gx] for gx in iter(gxs)],
            'nCxs':  lambda gxs: map(len, hs.gx2_cxs(gxs)),
            'exif':  lambda gxs: hs.gx2_exif(gxs),
            'exif.DateTime': lambda gxs: hs.gx2_exif(gxs, tag='DateTime'),
        }
    elif tblname in ['cxs', 'res']:
        # Tau is the future. Unfortunately society is often stuck in the past.
        # (tauday.com)
        FUTURE = False
        tau = (2 * np.pi)
        taustr = 'tau' if FUTURE else '2pi'

        def theta_str(theta):
            'Format theta so it is interpretable in base 10'
            #coeff = (((tau - theta) % tau) / tau)
            coeff = (theta / tau)
            return ('%.2f * ' % coeff) + taustr

        cols = {
            'cid':    lambda cxs: [cx2_cid[cx]           for cx in iter(cxs)],
            'name':   lambda cxs: [nx2_name[cx2_nx[cx]]  for cx in iter(cxs)],
            'gname':  lambda cxs: [gx2_gname[cx2_gx[cx]] for cx in iter(cxs)],
            'nGt':    lambda cxs: [len(gtcxs) for gtcxs in iter(hs.get_gtcxs(cxs))],
            'nKpts':  lambda cxs: [tools.safe_listget(cx2_kpts, cx, len) for cx in iter(cxs)],
            'theta':  lambda cxs: [theta_str(cx2_theta[cx]) for cx in iter(cxs)],
            'roi':    lambda cxs: [str(cx2_roi[cx]) for cx in iter(cxs)],
        }
        prop_iter = prop_dict.iteritems()
        prop_cols = dict([(k, lambda cxs: [v[cx] for cx in iter(cxs)]) for (k, v) in prop_iter])
        cols.update(prop_cols)
        if tblname == 'res':
            cols.update({
                'rank':   lambda cxs:  range(1, len(cxs) + 1),
            })
    else:
        cols = {}
    return cols


@profile
def _delete_image(hs, gx_list):
    for gx in gx_list:
        cx_list = hs.gx2_cxs(gx)
        for cx in cx_list:
            hs.delete_chip(cx, resample=False)
        hs.tables.gx2_gname[gx] = ''

    trash_dir = join(hs.dirs.db_dir, 'deleted-images')
    src_list = hs.gx2_gname(gx_list, full=True)
    dst_list = hs.gx2_gname(gx_list, prefix=trash_dir)
    helpers.ensuredir(trash_dir)

    # Move deleted images into the trash
    move_list = zip(src_list, dst_list)
    mark_progress, end_progress = helpers.progress_func(len(move_list), lbl='Trashing Image')
    for count, (src, dst) in enumerate(move_list):
        shutil.move(src, dst)
        mark_progress(count)
    end_progress()
    hs.update_samples()
    hs.save_database()


def _cx2_exif(hs, cx_list, **kwargs):
    gx_list = hs.cx2_gx(cx_list)
    exif_list = hs.gx2_exif(gx_list, **kwargs)
    return exif_list


def _cx2_unixtime(hs, cx_list):
    gx_list = hs.cx2_gx(cx_list)
    unixtime_list = _gx2_unixtime(hs, gx_list)
    return unixtime_list


def _gx2_unixtime(hs, gx_list):
    datetime_list = hs.gx2_exif(gx_list, tag='DateTime')
    unixtime_list = map(io.exiftime_to_unixtime, datetime_list)
    return unixtime_list


def _cx2_tnx(hs, cx_input):
    'maps chip index to a name index (uses negative chip index if unnamed)'
    cx2_nx = hs.tables.cx2_nx
    tnx_output = cx2_nx[cx_input]
    # Apply temporary labels to any unnamed chip
    is_uniden = tnx_output <= 2
    if isinstance(cx_input, np.ndarray):
        tnx_output[is_uniden] = -cx_input[is_uniden]
    elif is_uniden:
        tnx_output = -cx_input
    return tnx_output


def __define_method(hs, method_name):
    from hotspotter import HotSpotterAPI as api
    api.rrr()
    method_name = 'cx2_tnx'
    hs.__dict__[method_name] = lambda *args: api.__dict__['_' + method_name](hs, *args)
    #hs.cx2_tnx = lambda *args: api._cx2_tnx(hs, *args)


class HotSpotter(DynStruct):
    'The HotSpotter main class is a root handle to all relevant data'
    def __init__(hs, args=None, db_dir=None):
        super(HotSpotter, hs).__init__(child_exclude_list=['prefs', 'args'])
        #printDBG('[\hs] Creating HotSpotter API')
        # TODO Remove args / integrate into prefs
        hs.args = args
        hs.callbacks = {}
        hs.tables = None
        hs.dirs   = None
        hs.feats  = ds.HotspotterChipFeatures()
        hs.cpaths = ds.HotspotterChipPaths()
        #
        hs.train_sample_cx   = None
        hs.test_sample_cx    = None
        hs.indexed_sample_cx = None
        #
        pref_fpath = join(io.GLOBAL_CACHE_DIR, 'prefs')
        hs.prefs = Pref('root', fpath=pref_fpath)
        if hs.args.nocache_prefs:
            hs.default_preferences()
        else:
            hs.load_preferences()
        #if args is not None:
            #hs.prefs.N = args.N if args is not None
            #args_dict = vars(args)
            #hs.update_preferences(**args_dict)
        hs.query_history = [(None, None)]
        hs.qdat = ds.QueryData()  # Query Data
        if db_dir is not None:
            hs.load_tables(db_dir=db_dir)
        #printDBG(r'[/hs] Created HotSpotter API')

    def rrr(hs):
        import HotSpotterAPI
        HotSpotterAPI.rrr()

    def import_scripts(hs):
        return _import_scripts(hs)

    # --------------
    # Preferences functions
    # --------------
    @profile
    def load_preferences(hs):
        print('[hs] load preferences')
        hs.default_preferences()
        prefmsg = hs.prefs.load()
        was_loaded = prefmsg is True
        print('[hs] Able to load prefs? ...%r' % was_loaded)
        if was_loaded:
            hs.fix_prefs()
        else:
            print('[hs]' + prefmsg)
            hs.default_preferences()
        hs.assert_prefs()

    @profile
    def default_preferences(hs):
        print('[hs] defaulting preferences')
        hs.prefs.display_cfg = Config.default_display_cfg()
        hs.prefs.chip_cfg  = Config.default_chip_cfg()
        hs.prefs.feat_cfg  = Config.default_feat_cfg(hs)
        hs.prefs.query_cfg = Config.default_vsmany_cfg(hs)

    def fix_prefs(hs):
        # When loading some pointers may become broken. Fix them.
        hs.prefs.feat_cfg._chip_cfg = hs.prefs.chip_cfg
        hs.prefs.query_cfg._feat_cfg = hs.prefs.feat_cfg

    def assert_prefs(hs):
        try:
            query_cfg = hs.prefs.query_cfg
            feat_cfg  = hs.prefs.feat_cfg
            chip_cfg  = hs.prefs.chip_cfg
            assert query_cfg._feat_cfg is feat_cfg
            assert query_cfg._feat_cfg._chip_cfg is chip_cfg
            assert feat_cfg._chip_cfg is chip_cfg
        except AssertionError:
            print('[hs] preferences dependency tree is broken')
            raise

    def update_preferences(hs, **kwargs):
        print('[hs] updateing preferences')
        hs.prefs.query_cfg.update_cfg(**kwargs)

    # --------------
    # Saving functions
    # --------------
    def save_database(hs):
        print('[hs] save_database')
        ld2.write_csv_tables(hs)

    #---------------
    # Loading Functions
    #---------------
    def load(hs, load_all=False):
        '(current load function) Loads the appropriate database'
        print('[hs] load()')
        hs.unload_all()
        hs.load_tables()
        hs.update_samples()
        if load_all:
            #printDBG('[hs] load_all=True')
            hs.load_chips()
            hs.load_features()
        else:
            #printDBG('[hs] load_all=False')
            hs.load_chips([])
            hs.load_features([])
        return hs

    def load_tables(hs, db_dir=None):
        # Check to make sure db_dir is specified correctly
        if db_dir is None:
            db_dir = hs.args.dbdir
        if db_dir is None or not exists(db_dir):
            raise ValueError('db_dir=%r does not exist!' % (db_dir))
        hs_dirs, hs_tables, db_version = ld2.load_csv_tables(db_dir)
        hs.tables = hs_tables
        hs.dirs = hs_dirs
        if db_version != 'current':
            print('Loaded db_version=%r. Converting...' % db_version)
            hs.save_database()
        _checkargs_onload(hs)

    def load_chips(hs, cx_list=None):
        cc2.load_chips(hs, cx_list)

    def load_features(hs, cx_list=None):
        fc2.load_features(hs, cx_list=cx_list)

    @profile
    def refresh_features(hs, cx_list=None):
        hs.load_chips(cx_list=cx_list)
        hs.load_features(cx_list=cx_list)

    def update_samples_split_pos(hs, pos):
        valid_cxs = hs.get_valid_cxs()
        test_samp  = valid_cxs[:pos]
        train_samp = valid_cxs[pos + 1:]
        hs.update_samples(test_samp, train_samp)

    def update_samples_range(hs, pos1, pos2):
        valid_cxs = hs.get_valid_cxs()
        test_samp  = valid_cxs[pos1:pos2]
        train_samp = test_samp
        hs.update_samples(test_samp, train_samp)

    @profile
    def update_samples(hs, test_samp=None, train_samp=None, indx_samp=None):
        ''' This is the correct function to use when setting samples '''
        print('[hs] update_samples():')
        valid_cxs = hs.get_valid_cxs()
        if test_samp is None:
            test_samp = valid_cxs
        if train_samp is None:
            print('[hs] * default: all chips in training')
            train_samp = valid_cxs
        if indx_samp is None:
            indx_samp = train_samp

        tools.assert_int(test_samp, 'test_samp')
        tools.assert_int(train_samp, 'train_samp')
        tools.assert_int(indx_samp, 'indx_samp')
        # Ensure samples are sorted
        test_samp  = sorted(test_samp)
        train_samp = sorted(train_samp)
        indx_samp  = sorted(indx_samp)
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
        # Set the sample
        hs.indexed_sample_cx  = indx_samp
        hs.train_sample_cx    = train_samp
        hs.test_sample_cx     = test_samp

    #---------------
    # On Modification
    #---------------
    @profile
    def unload_all(hs):
        print('[hs] Unloading all data')
        hs.feats  = ds.HotspotterChipFeatures()
        hs.cpaths = ds.HotspotterChipPaths()
        hs.qdat.unload_data()
        hs.clear_lru_caches()
        print('[hs] finished unloading all data')

    @profile
    def unload_cxdata(hs, cx):
        'unloads features and chips. not tables'
        print('[hs] unload_cxdata(cx=%r)' % cx)
        # HACK This should not really be removed EVERY time you unload any cx
        hs.qdat.unload_data()
        hs.clear_lru_caches()
        lists = []
        if hs.cpaths is not None:
            lists += [hs.cpaths.cx2_rchip_path, hs.cpaths.cx2_rchip_size]
        if hs.feats is not None:
            lists += [hs.feats.cx2_kpts, hs.feats.cx2_desc]
        if cx == 'all':
            hs.unload_all()
            return
        for list_ in lists:
            helpers.ensure_list_size(list_, cx + 1)
            list_[cx] = None

    @profile
    def delete_ciddata(hs, cid):
        cid_str_list = ['cid%d_' % cid, 'qcid=%d.npz' % cid, ]
        hs.clear_lru_caches()
        for cid_str in cid_str_list:
            dpath = hs.dirs.computed_dir
            pat = '*' + cid_str + '*'
            helpers.remove_files_in_dir(dpath, pat, recursive=True,
                                        verbose=True, dryrun=False)

    @profile
    def delete_cxdata(hs, cx):
        'deletes features and chips. not tables'
        hs.unload_cxdata(cx)
        print('[hs] delete_cxdata(cx=%r)' % cx)
        cid = hs.tables.cx2_cid[cx]
        hs.delete_ciddata(cid)
        hs.clear_lru_caches()

    def clear_lru_caches(hs):
        'clears the least recently used caches'
        hs._read_chip.clear_cache()
        hs.gx2_image.clear_cache()
        hs.get_exif.clear_cache()

    #---------------
    # Query Functions
    #---------------
    @profile
    def prequery(hs):
        mc3.prequery(hs)

    @profile
    def query(hs, qcx, *args, **kwargs):
        return hs.query_database(qcx, *args, **kwargs)

    @profile
    def query_database(hs, qcx, query_cfg=None, dochecks=True, **kwargs):
        'queries the entire (sampled) database'
        print('\n====================')
        print('[hs] query database')
        print('====================')
        if query_cfg is None:
            hs.assert_prefs()
            query_cfg = hs.prefs.query_cfg
        if len(kwargs) > 0:
            query_cfg = query_cfg.deepcopy(**kwargs)
        qdat = hs.qdat
        qdat.set_cfg(query_cfg)
        dcxs = hs.get_indexed_sample()
        try:
            res = mc3.query_dcxs(hs, qcx, dcxs, hs.qdat, dochecks=dochecks)
        except mf.QueryException as ex:
            msg = '[hs] Query Failure: %r' % ex
            print(msg)
            if hs.args.strict:
                raise
            return msg
        return res

    @profile
    def query_groundtruth(hs, qcx, query_cfg=None, **kwargs):
        'wrapper that restricts query to only known groundtruth'
        print('\n====================')
        print('[hs] query groundtruth')
        print('====================')
        if query_cfg is None:
            hs.assert_prefs()
            query_cfg = hs.prefs.query_cfg
        if len(kwargs) > 0:
            query_cfg = query_cfg.deepcopy(**kwargs)
        qdat = hs.qdat
        qdat.set_cfg(query_cfg)
        gt_cxs = hs.get_other_indexed_cxs(qcx)
        qdat.dcxs = gt_cxs
        print('[mc3] len(gt_cxs) = %r' % (gt_cxs,))
        return mc3.query_dcxs(hs, qcx, gt_cxs, qdat)

    # ---------------
    # Change functions
    # ---------------
    @profile
    def change_roi(hs, cx, new_roi):
        hs.delete_cxdata(cx)  # Delete old data
        hs.delete_queryresults_dir()  # Query results are now invalid
        hs.tables.cx2_roi[cx] = new_roi

    @profile
    def change_theta(hs, cx, new_theta):
        hs.delete_cxdata(cx)  # Delete old data
        hs.delete_queryresults_dir()  # Query results are now invalid
        hs.tables.cx2_theta[cx] = new_theta

    @profile
    def change_name(hs, cx, new_name):
        new_nx_ = np.where(hs.tables.nx2_name == new_name)[0]
        new_nx  = new_nx_[0] if len(new_nx_) > 0 else hs.add_name(new_name)
        hs.tables.cx2_nx[cx] = new_nx

    @profile
    def change_property(hs, cx, key, val):
        hs.tables.prop_dict[key][cx] = val

    @profile
    def change_aif(hs, gx, val):
        hs.tables.gx2_aif[gx] = np.bool_(val)

    def get_property(hs, cx, key):
        return hs.tables.prop_dict[key][cx]

    def cx2_property(hs, cx, key):
        # TODO: property keys should be case insensitive
        try:
            return hs.tables.prop_dict[key][cx]
        except KeyError:
            return None

    # ---------------
    # Adding functions
    # ---------------
    @profile
    def add_property(hs, key):
        if not isinstance(key, str):
            raise ValueError('[hs] New property %r is a %r, not a string.' % (key, type(key)))
        if key in hs.tables.prop_dict:
            raise UserWarning('[hs] WARNING: Property add an already existing property')
        hs.tables.prop_dict[key] = ['' for _ in xrange(hs.get_num_chips())]

    @profile
    def add_name(hs, name):
        # TODO: Allocate memory better (use python lists)
        nx2_name = hs.tables.nx2_name.tolist()
        nx2_name.append(name)
        hs.tables.nx2_name = np.array(nx2_name)
        nx = len(hs.tables.nx2_name) - 1
        return nx

    # RCOS TODO: Rectify this with add_name and user iter_input
    @profile
    def add_names(hs, name_list):
        # TODO Assert names are unique
        nx2_name = hs.tables.nx2_name.tolist()
        nx2_name.extend(name_list)
        hs.tables.nx2_name = np.array(nx2_name)

    @profile
    def add_chip(hs, gx, roi, nx=0, theta=0, props={}, dochecks=True):
        # TODO: Restructure for faster adding (preallocate and double size)
        # OR just make all the tables python lists
        print('[hs] adding chip to gx=%r' % gx)
        if len(hs.tables.cx2_cid) > 0:
            next_cid = hs.tables.cx2_cid.max() + 1
        else:
            next_cid = 1
        # Remove any conflicts from disk
        if dochecks:
            hs.delete_ciddata(next_cid)
        # Allocate space for a new chip
        hs.tables.cx2_cid   = np.concatenate((hs.tables.cx2_cid, [next_cid]))
        hs.tables.cx2_nx    = np.concatenate((hs.tables.cx2_nx,  [nx]))
        hs.tables.cx2_gx    = np.concatenate((hs.tables.cx2_gx,  [gx]))
        hs.tables.cx2_roi   = np.vstack((hs.tables.cx2_roi, [roi]))
        hs.tables.cx2_theta = np.concatenate((hs.tables.cx2_theta, [theta]))
        prop_dict = hs.tables.prop_dict
        for key in prop_dict.iterkeys():
            prop_dict[key].append(props.get(key, ''))
        #hs.num_cx += 1
        cx = len(hs.tables.cx2_cid) - 1
        if dochecks:
            hs.update_samples()
            # Remove any conflicts from memory
            hs.unload_cxdata(cx)
            hs.delete_queryresults_dir()  # Query results are now invalid
        return cx

    @profile
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
            # RCOS TODO: Copying like this should be a helper function.
            # It appears in multiple places
            # Also there should be the option of parallelization? IDK, these are
            # disk writes, but it still might help.
            mark_progress, end_progress = helpers.progress_func(len(copy_list), lbl='Copying Image')
            for count, (src, dst) in enumerate(copy_list):
                shutil.copy(src, dst)
                mark_progress(count)
            end_progress()
        else:
            print('[hs.add_imgs] using original image paths')
            fpath_list2 = fpath_list
        # Get location of the new images relative to the image dir
        gx2_gname = hs.tables.gx2_gname.tolist()
        gx2_aif   = hs.tables.gx2_aif.tolist()
        relpath_list = [relpath(fpath, img_dir) for fpath in fpath_list2]
        current_gname_set = set(gx2_gname)
        # Check to make sure the gnames are not currently indexed
        new_gnames = [gname for gname in relpath_list if not gname in current_gname_set]
        new_aifs   = [False] * len(new_gnames)
        nNewImages = len(new_gnames)
        nIndexed = nImages - nNewImages
        print('[hs.add_imgs] new_gnames:\n' + '\n'.join(new_gnames))
        print('[hs.add_imgs] %d images already indexed.' % nIndexed)
        print('[hs.add_imgs] Added %d new images.' % nIndexed)
        # Append the new gnames to the hotspotter table
        hs.tables.gx2_gname = np.array(gx2_gname + new_gnames)
        hs.tables.gx2_aif   = np.array(gx2_aif   + new_aifs)
        hs.update_samples()
        return nNewImages

    # ---------------
    # Deleting functions
    # ---------------
    def delete_chip(hs, cx, resample=True):
        hs.delete_cxdata(cx)
        hs.tables.cx2_cid[cx] = -1
        hs.tables.cx2_gx[cx]  = -1
        hs.tables.cx2_nx[cx]  = -1
        #hs.num_cx -= 1
        if resample:
            hs.update_samples()

    @tools.class_iter_input
    def delete_image(hs, gx_list):
        return _delete_image(hs, gx_list)

    def delete_cache(hs):
        print('[hs] DELETE CACHE')
        computed_dir = hs.dirs.computed_dir
        hs.unload_all()
        #[hs.unload_cxdata(cx) for cx in hs.get_valid_cxs()]
        helpers.remove_files_in_dir(computed_dir, recursive=True, verbose=True,
                                    dryrun=False)

    def delete_global_prefs(hs):
        io.delete_global_cache()

    def delete_queryresults_dir(hs):
        qres_dir = hs.dirs.qres_dir
        hs.unload_all()
        #[hs.unload_cxdata(cx) for cx in hs.get_valid_cxs()]
        helpers.remove_files_in_dir(qres_dir, recursive=True, verbose=True,
                                    dryrun=False)

    # ---------------
    # Getting functions
    # ---------------
    def has_property(hs, key):
        return key in hs.tables.prop_dict

    def get_valid_indexes(hs, tblname):
        return {
            'cxs': lambda: hs.get_valid_cxs(),
            'nxs': lambda: hs.get_valid_nxs(unknown=False),
            'gxs': lambda: hs.get_valid_gxs(),
        }[tblname]()

    def get_datatup_list(hs, tblname, index_list, header_order, extra_cols):
        datatup_list = _get_datatup_list(hs, tblname, index_list, header_order, extra_cols)
        return datatup_list

    def get_db_name(hs, devmode=False):
        db_name = split(hs.dirs.db_dir)[1]
        if devmode:
            # Grab the dev name insetad
            from hscom import params
            dev_databases = params.dev_databases
            db_tups = [(v, k) for k, v in dev_databases.iteritems() if v is not None]
            #print('  \n'.join(map(str,db_tups)))
            dev_dbs = dict((split(v)[1], k) for v, k in db_tups)
            db_name = dev_dbs[db_name]
        return db_name

    # -------
    # Get valid index functions
    # -------
    # TODO: Many of these functions should be decorated with iter_input

    def get_indexed_sample(hs):
        dcxs = hs.indexed_sample_cx
        return dcxs

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

    def get_num_chips(hs):
        return len(hs.tables.cx2_cid)

    def get_valid_cxs(hs):
        valid_cxs = np.where(hs.tables.cx2_cid > 0)[0]
        return valid_cxs

    def get_valid_gxs(hs):
        valid_gxs = np.where(hs.tables.gx2_gname != '')[0]
        return valid_gxs

    def get_valid_nxs(hs, unknown=True):
        x = 2 * (not unknown)
        valid_nxs = np.where(hs.tables.nx2_name[x:] != '')[0] + x
        return valid_nxs

    def get_valid_cxs_with_indexed_groundtruth(hs):
        return hs.get_valid_cxs_with_name_in_samp(hs.indexed_sample_cx)

    #----
    # chip index --> property

    def cx2_gx(hs, cx):
        gx = hs.tables.cx2_gx[cx]
        return gx

    def cx2_roi(hs, cx):
        roi = hs.tables.cx2_roi[cx]
        return roi

    def cx2_theta(hs, cx):
        theta = hs.tables.cx2_theta[cx]
        return theta

    def cx2_cid(hs, cx):
        return hs.tables.cx2_cid[cx]

    def cx2_name(hs, cx):
        cx2_nx = hs.tables.cx2_nx
        nx2_name = hs.tables.nx2_name
        return nx2_name[cx2_nx[cx]]

    def cx2_tnx(hs, cx_input):
        return _cx2_tnx(hs, cx_input)

    def cx2_gname(hs, cx, full=False):
        gx =  hs.tables.cx2_gx[cx]
        return hs.gx2_gname(gx, full)

    def cx2_image(hs, cx):
        gx =  hs.tables.cx2_gx[cx]
        return hs.gx2_image(gx)

    @tools.class_iter_input
    def cx2_exif(hs, cx_list, **kwargs):
        return _cx2_exif(hs, cx_list)

    @tools.class_iter_input
    def gx2_unixtime(hs, gx_list):
        return _gx2_unixtime(hs, gx_list)

    def get_unixtime_diff(hs, qcx, cx):
        unixtime1, unixtime2 = hs.cx2_unixtime([qcx, cx])
        if -1 in [unixtime1, unixtime2]:
            return None
        unixtime_diff = unixtime2 - unixtime1
        return unixtime_diff

    def get_timedelta_str(hs, qcx, cx):
        unixtime_diff = hs.get_unixtime_diff(qcx, cx)
        if unixtime_diff is None:
            deltastr = 'NA'
        else:
            sign = '+' if unixtime_diff >= 0 else '-'
            deltastr = sign + str(datetime.timedelta(seconds=abs(unixtime_diff)))
        timedelta_str = 'timedelta(%s)' % (deltastr)
        return timedelta_str

    @tools.class_iter_input
    @profile
    def cx2_unixtime(hs, gx_list):
        return _cx2_unixtime(hs, gx_list)

    #----
    # image index --> property
    @tools.class_iter_input
    @profile
    def gx2_exif(hs, gx_list, **kwargs):
        gname_list = hs.gx2_gname(gx_list, full=True)
        exif_list = io.read_exif_list(gname_list, **kwargs)
        return exif_list

    @profile
    @tools.lru_cache(max_size=100)
    @profile
    def get_exif(hs, **kwargs):
        gx_list = hs.get_valid_gxs()
        exif_list = hs.gx2_exif(gx_list, **kwargs)
        return exif_list

    @tools.class_iter_input
    @profile
    def gx2_gname(hs, gx_input, full=False, prefix=None):
        gx2_gname_ = hs.tables.gx2_gname
        gname_list = [gx2_gname_[gx] for gx in iter(gx_input)]
        if full or prefix is not None:
            img_dir = hs.dirs.img_dir if prefix is None else prefix
            gname_list = [join(img_dir, gname) for gname in iter(gname_list)]
        return gname_list

    @tools.lru_cache(max_size=7)
    @profile
    def gx2_image(hs, gx):
        img_fpath = hs.gx2_gname(gx, full=True)
        img = io.imread(img_fpath)
        return img

    @tools.class_iter_input
    @profile
    def gx2_image_size(hs, gx_input):
        gfpath_list = hs.gx2_gname(gx_input, full=True)
        # RCOS TODO: Do you need to do a .close here? or does gc take care of it?
        gsize_list = [Image.open(gfpath).size for gfpath in iter(gfpath_list)]
        return gsize_list

    @tools.class_iter_input
    @profile
    def gx2_aif(hs, gx_input):
        gx2_aif_ = hs.tables.gx2_aif
        aif_list = [gx2_aif_[gx] for gx in iter(gx_input)]
        return aif_list

    @tools.class_iter_input
    @profile
    def gx2_cxs(hs, gx_input):
        cxs_list = [np.where(hs.tables.cx2_gx == gx)[0] for gx in gx_input]
        return cxs_list

    @tools.class_iter_input
    @profile
    def gx2_nChips(hs, gx_input):
        nChips_list = [len(np.where(hs.tables.cx2_gx == gx)[0]) for gx in gx_input]
        return nChips_list

    # build metaproperty tables
    @profile
    def cid2_gx(hs, cid):
        'chip_id ==> image_index'
        cx = hs.tables.cid2_cx(cid)
        gx = hs.tables.cx2_gx[cx]
        return gx

    @tools.class_iter_input
    def cid2_cx(hs, cid_input):
        'returns chipids belonging to a chip index(s)'
        'chip_id ==> chip_index'
        index_of = tools.index_of
        cx2_cid = hs.tables.cx2_cid
        try:
            cx_output = [index_of(cid, cx2_cid) for cid in cid_input]
        except IndexError as ex:
            print('[hs] ERROR %r ' % ex)
            print('[hs] ERROR a cid in %r does not exist.' % (cid_input,))
            raise
        return cx_output

    def get_nx2_cxs(hs):
        'returns mapping from name indexes to chip indexes'
        cx2_nx = hs.tables.cx2_nx
        if len(cx2_nx) == 0:
            return [[], []]
        max_nx = cx2_nx.max()
        nx2_cxs = [[] for _ in xrange(max_nx + 1)]
        for cx, nx in enumerate(cx2_nx):
            if nx > 0:
                nx2_cxs[nx].append(cx)
        return nx2_cxs

    #def nx2_cxs(hs, nx_list):
        #'returns mapping from name indexes to chip indexes'
        #cx2_nx = hs.tables.cx2_nx
        #if len(cx2_nx) == 0:
            #nx2_cxs_ = [[], []]
        #max_nx = cx2_nx.max()
        #nx2_cxs = [[] for _ in xrange(max_nx + 1)]
        #for cx, nx in enumerate(cx2_nx):
            #if nx > 0:
                #nx2_cxs[nx].append(cx)
        #return nx2_cxs

    def get_gx2_cxs(hs):
        'returns mapping from image indexes to chip indexes'
        cx2_gx = hs.tables.cx2_gx
        max_gx = len(hs.tables.gx2_gname)
        gx2_cxs = [[] for _ in xrange(max_gx + 1)]
        for cx, gx in enumerate(cx2_gx):
            if gx == -1:
                continue
            gx2_cxs[gx].append(cx)
        return gx2_cxs

    @tools.class_iter_input
    def get_other_cxs(hs, cx_input):
        'returns other chips with the same known name'
        cx2_nx = hs.tables.cx2_nx
        nx_list = [cx2_nx[cx] for cx in iter(cx_input)]

        def _2ocxs(cx, nx):
            other_cx_ = np.where(cx2_nx == nx)[0]
            return other_cx_[other_cx_ != cx]
        others_list = [_2ocxs(cx, nx) if nx > 1 else np.array([], ds.X_DTYPE)
                       for nx, cx in izip(nx_list, cx_input)]
        return others_list

    @tools.class_iter_input
    def get_other_indexed_cxs(hs, cx_input):
        'returns other indexed chips with the same known name'
        other_list_ = hs.get_other_cxs(cx_input)
        indx_samp  = hs.indexed_sample_cx
        other_list = [np.intersect1d(ocxs, indx_samp) for
                      ocxs in iter(other_list_)]
        return other_list

    def get_gtcxs(hs, cx_input):
        return hs.get_other_indexed_cxs(cx_input)

    # Strings
    def is_true_match(hs, qcx, cx):
        cx2_nx  = hs.tables.cx2_nx
        qnx = cx2_nx[qcx]
        nx  = cx2_nx[cx]
        is_true = nx == qnx
        is_unknown = nx <= 1 or qnx <= 1
        return is_true, is_unknown

    def vs_str(hs, qcx, cx):
        if False:
            #return '(qcx=%r v cx=%r)' % (qcx, cx)
            return 'cx(%r v %r)' % (qcx, cx)
        else:
            cx2_cid = hs.tables.cx2_cid
            return 'cid(%r v %r)' % (cx2_cid[qcx], cx2_cid[cx])

    def num_indexed_gt_str(hs, cx):
        num_gt = len(hs.get_other_indexed_cxs(cx))
        return '#gt=%r' % num_gt

    def cidstr(hs, cx, digits=None, notes=False):
        cx2_cid = hs.tables.cx2_cid
        if not np.iterable(cx):
            int_fmt = '%d' if digits is None else ('%' + str(digits) + 'd')
            cid_str = 'cid=' + int_fmt % cx2_cid[cx]
        else:
            cid_str = 'cids=[%s]' % ', '.join(['%d' % cx2_cid[cx_] for cx_ in cx])
        if notes:

            cid_str += ' - ' + str(hs.cx2_property(cx, 'Notes'))
        return cid_str

    # Precomputed properties
    #@tools.debug_exception
    @tools.class_iter_input
    @profile
    def _try_cxlist_get(hs, cx_input, cx2_var):
        ''' Input: cx_input: a vector input, cx2_var: a array mapping cx to a
        variable Returns: list of values corresponding with cx_input '''
        ret = [cx2_var[cx] for cx in cx_input]
        # None is invalid in a cx2_var array
        if any([val is None for val in ret]):
            none_index = ret.index(None)
            raise IndexError('ret[%r] == None' % none_index)
        return ret

    @profile
    def _onthefly_cxlist_get(hs, cx_input, cx2_var, load_fn):
        '''tries to get from the cx_input indexed list and performs a cx load function
        if unable to get failure'''
        try:
            ret = hs._try_cxlist_get(cx_input, cx2_var)
        except IndexError:
            try:
                load_fn(cx_input)
                ret = hs._try_cxlist_get(cx_input, cx2_var)
            except Exception as ex:
                print('[hs] Caught Exception ex=%r' % ex)
                msg = ['[hs] Data was not loaded/unloaded propertly']
                msg += ['[hs] cx_input=%r' % cx_input]
                msg += ['[hs] cx2_var=%r' % cx2_var]
                msg += ['[hs] load_fn=%r' % load_fn]
                msg_ = '\n'.join(msg)
                print(msg_)
                raise
        return ret

    @profile
    def get_desc(hs, cx_input):
        cx2_desc = hs.feats.cx2_desc
        return hs._onthefly_cxlist_get(cx_input, cx2_desc, hs.load_features)

    # cx2_kpts
    @profile
    def get_kpts(hs, cx_input):
        cx2_kpts = hs.feats.cx2_kpts
        return hs._onthefly_cxlist_get(cx_input, cx2_kpts, hs.load_features)

    @profile
    def get_rchip_path(hs, cx_input):
        cx2_rchip_path = hs.cpaths.cx2_rchip_path
        return hs._onthefly_cxlist_get(cx_input, cx2_rchip_path, hs.load_chips)

    @profile
    def get_chip_pil(hs, cx):
        chip = Image.open(hs.cpaths.cx2_rchip_path[cx])
        return chip

    @tools.lru_cache(max_size=7)
    @profile
    def _read_chip(hs, fpath):
        return io.imread(fpath)

    @profile
    def get_chip(hs, cx_input):
        rchip_path = hs.get_rchip_path(cx_input)
        if np.iterable(cx_input):
            return [hs._read_chip(fpath) for fpath in rchip_path]
        else:
            return hs._read_chip(rchip_path)

    @profile
    def cx2_rchip_size(hs, cx_input):
        #cx_input = hs.get_valid_cxs()
        cx2_rchip_size = hs.cpaths.cx2_rchip_size
        return hs._onthefly_cxlist_get(cx_input, cx2_rchip_size, hs.load_chips)

    @profile
    def get_arg(hs, argname, default=None):
        try:
            val = eval('hs.args.' + argname)
            result = default if val is None else val
        except KeyError:
            result = default
        return result

    #---------------
    # Print Tables

    def print_name_table(hs):
        print(ld2.make_name_csv(hs))

    def print_image_table(hs):
        print(ld2.make_image_csv(hs))

    def print_chip_table(hs):
        print(ld2.make_chip_csv(hs))

    #---------------
    # View Directories
    def vdd(hs):
        db_dir = os.path.normpath(hs.dirs.db_dir)
        print('[hs] viewing db_dir: %r ' % db_dir)
        helpers.vd(db_dir)

    def vcd(hs):
        computed_dir = os.path.normpath(hs.dirs.computed_dir)
        print('[hs] viewing computed_dir: %r ' % computed_dir)
        helpers.vd(computed_dir)

    def vgd(hs):
        global_dir = io.GLOBAL_CACHE_DIR
        print('[hs] viewing global_dir: %r ' % global_dir)
        helpers.vd(global_dir)

    def vrd(hs):
        result_dir = os.path.normpath(hs.dirs.result_dir)
        print('[hs] viewing result_dir: %r ' % result_dir)
        helpers.vd(result_dir)

    #---------------
    def get_cache_uid(hs, cx_list=None, lbl='cxs'):
        query_cfg = hs.prefs.query_cfg
        # Build query big cache uid
        hs_uid    = 'HSDB(%s)' % hs.get_db_name()
        uid_list = [hs_uid] + query_cfg.get_uid_list()
        if cx_list is not None:
            cxs_uid = helpers.hashstr_arr(cx_list, 'cxs')
            uid_list.append('_' + cxs_uid)
        cache_uid = ''.join(uid_list)
        return cache_uid

    #---------------
    # Callbacks

    def select_nx(hs, nx):
        hs.callbacks['select_nx'](nx)

    def select_gx(hs, gx):
        hs.callbacks['select_gx'](gx)

    def select_cx(hs, cx):
        hs.callbacks['select_cx'](cx)

    def register_backend(hs, back):
        hs.back = back
        hs.callbacks['select_cx'] = back.select_cx
        hs.callbacks['select_nx'] = back.select_nx
        hs.callbacks['select_gx'] = back.select_gx

    #----------------------
    # Debug Consistency Checks
    def dbg_cx2_kpts(hs):
        cx2_cid = hs.tables.cx2_cid
        cx2_kpts = hs.feats.cx2_kpts
        bad_cxs = [cx for cx, kpts in enumerate(cx2_kpts) if kpts is None]
        passed = True
        if len(bad_cxs) > 0:
            print('[hs.dbg] cx2_kpts has %d None positions:' % len(bad_cxs))
            print('[hs.dbg] bad_cxs = %r' % bad_cxs)
            passed = False
        if len(cx2_kpts) != len(cx2_cid):
            print('[hs.dbg] len(cx2_kpts) != len(cx2_cid): %r != %r' % (len(cx2_kpts), len(cx2_cid)))
            passed = False
        if passed:
            print('[hs.dbg] cx2_kpts is OK')
