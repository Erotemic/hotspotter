'''
Module: load_data
    Loads the paths and table information from which all other data is computed.
    This is the first script run in the loading pipeline. 
'''
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

# reloads this module when I mess with it
def reload_module():
    import imp, sys
    print('[ld2] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def printDBG(msg, lbl=''):
    print('DBG: '+lbl+str(msg))

# paths relative to dbdir
RDIR_IMG      = '/images'
RDIR_INTERNAL = '/.hs_internals'
RDIR_COMPUTED = '/.hs_internals/computed'
RDIR_CHIP     = '/.hs_internals/computed/chips'
RDIR_RCHIP    = '/.hs_internals/computed/temp'
RDIR_CACHE    = '/.hs_internals/computed/cache'
RDIR_FEAT     = '/.hs_internals/computed/feats'
RDIR_RESULTS  = '/.hs_internals/computed/results'
RDIR_QRES     = '/.hs_internals/computed/query_results'

#========================================
# DRIVER CODE
#========================================

# ___CLASS HOTSPOTTER____
class HotSpotter(DynStruct):
    '''The HotSpotter main class is a root handle to all relevant data'''
    def __init__(hs, db_dir=None, load_basic=False, **kwargs):
        super(HotSpotter, hs).__init__()
        hs.num_cx = None
        hs.tables = None
        hs.feats  = None
        hs.cpaths = None
        hs.dirs   = None
        hs.matcher = None
        hs.train_sample_cx    = None
        hs.test_sample_cx     = None
        hs.indexed_sample_cx = None
        if load_basic:
            hs.load_basic(db_dir)
        elif not db_dir is None:
            hs.load_database(db_dir, **kwargs)
    #---------------
    def load_tables(hs, db_dir):
        hs_dirs, hs_tables = load_csv_tables(db_dir)
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

    def load_chips(hs):
        import chip_compute2 as cc2
        cc2.load_chip_paths(hs)

    def load_features(hs, load_kpts=True, load_desc=True):
        import feature_compute2 as fc2
        hs_feats  = fc2.load_chip_features(hs.dirs, hs.tables, hs.cpaths, load_kpts, load_desc)
        print('The new way is not yet finished and is commented out')
        #hs_feats  = fc2.load_chip_features2(hs, load_kpts, load_desc)
        hs.feats  = hs_feats

    def load_matcher(hs):
        import match_chips2 as mc2
        hs.matcher = mc2.Matcher(hs, params.__MATCH_TYPE__)
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
    def cx2_gname(hs, cx):
        gx =  hs.tables.cx2_gx[cx]
        gname = hs.tables.gx2_gname[gx]
        return gname
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
        return cv2.imread(hs.cpaths.cx2_rchip_path[cx])
    #--------------
    def get_chip_pil(hs, cx):
        chip = Image.open(hs.cpaths.cx2_rchip_path[cx])
        return chip
    #--------------
    def get_kpts(hs, cx):
        return hs.feats.cx2_kpts[cx]
    #--------------
    def cx2_rchip_size(hs, cx):
        rchip_path = hs.cpaths.cx2_rchip_path[cx]
        return Image.open(rchip_path).size

    def get_cx2_rchip_size(hs):
        cx2_rchip_path = hs.cpaths.cx2_rchip_path
        cx2_rchip_size = [Image.open(path).size for path in cx2_rchip_path]
        return cx2_rchip_size
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
        assign_matches = hs.matcher.assign_matches
        cx2_fm, cx2_fs, cx2_score = assign_matches(qcx, cx2_desc)
        return cx2_fm, cx2_fs, cx2_score
    #--------------
    def get_assigned_matches_to(hs, qcx, cx):
        cx2_fm, cx2_fs, cx2_score = hs.get_assigned_matches(qcx)
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

# Testing helper functions
def get_test_data(qcx=0, cx=None, db_dir=None):
    import load_data2 as ld2
    hs = helpers.search_stack_for_localvar('hs')
    if db_dir is None:
        db_dir = ld2.DEFAULT
    if hs is None:
        hs = ld2.HotSpotter(db_dir)
    if cx is None:
        cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    print('(hs, qcx, cx, fm, fs, rchip1, rchip2, kpts1, kpts2)')
    return (hs, qcx, cx, fm, fs, rchip1, rchip2, kpts1, kpts2)

@helpers.__DEPRICATED__
def get_sv_test_data(qcx=0, cx=None):
    return get_test_data(qcx, cx)

# ______________________________

# ___CLASS HOTSPOTTER TABLES____
class HotspotterTables(DynStruct):
    def __init__(self, 
                 gx2_gname = [],
                 nx2_name  = ['____','____'],
                 cx2_cid   = [],
                 cx2_nx    = [],
                 cx2_gx    = [],
                 cx2_roi   = [],
                 cx2_theta = [],
                 prop_dict = {}):
        super(HotspotterTables, self).__init__()
        self.gx2_gname    = np.array(gx2_gname)
        self.nx2_name     = np.array(nx2_name)
        self.cx2_cid      = np.array(cx2_cid)
        self.cx2_nx       = np.array(cx2_nx)
        self.cx2_gx       = np.array(cx2_gx)
        self.cx2_roi      = np.array(cx2_roi)
        self.cx2_theta    = np.array(cx2_theta)
        self.prop_dict    = prop_dict

# ___CLASS HOTSPOTTER DIRS________
class HotspotterDirs(DynStruct):
    def __init__(self, db_dir):
        super(HotspotterDirs, self).__init__()
        # Class variables
        self.db_dir       = db_dir
        self.img_dir      = db_dir + RDIR_IMG
        self.internal_dir = db_dir + RDIR_INTERNAL
        self.computed_dir = db_dir + RDIR_COMPUTED
        self.chip_dir     = db_dir + RDIR_CHIP
        self.rchip_dir    = db_dir + RDIR_RCHIP
        self.feat_dir     = db_dir + RDIR_FEAT
        self.cache_dir    = db_dir + RDIR_CACHE
        self.result_dir   = db_dir + RDIR_RESULTS
        self.qres_dir     = db_dir + RDIR_QRES
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
        internal_sym = db_dir + '/Shortcut-to-hs_internals'
        computed_sym = db_dir + '/Shortcut-to-computed'
        results_sym  = db_dir + '/Shortcut-to-results'

        helpers.symlink(self.internal_dir, internal_sym, noraise=False)
        helpers.symlink(self.computed_dir, computed_sym, noraise=False)
        helpers.symlink(self.result_dir, results_sym, noraise=False)

def tryindex(list, val):
    try: 
        return list.index(val)
    except ValueError as ex:
        return -1

def load_csv_tables(db_dir):
    '''
    Big function which loads the csv tables from a datatabase directory
    Returns HotspotterDirs and HotspotterTables
    '''
    print('\n=============================')
    print('[ld2] Loading hotspotter csv tables: '+str(db_dir))
    print('=============================')
    hs_dirs = HotspotterDirs(db_dir)
    #exec(hs_dirs.execstr('hs_dirs'))
    #print(hs_dirs.execstr('hs_dirs'))
    feat_dir     = hs_dirs.feat_dir
    img_dir      = hs_dirs.img_dir
    rchip_dir    = hs_dirs.rchip_dir
    chip_dir     = hs_dirs.chip_dir
    internal_dir = hs_dirs.internal_dir
    db_dir       = hs_dirs.db_dir
    # --- Table Names ---
    chip_table   = internal_dir + '/chip_table.csv'
    name_table   = internal_dir + '/name_table.csv'
    image_table  = internal_dir + '/image_table.csv' # TODO: Make optional
    # --- CHECKS ---
    has_dbdir   = helpers.checkpath(db_dir)
    has_imgdir  = helpers.checkpath(img_dir)
    has_chiptbl = helpers.checkpath(chip_table)
    has_nametbl = helpers.checkpath(name_table)
    has_imgtbl  = helpers.checkpath(image_table)
    if not all([has_dbdir, has_imgdir, has_chiptbl, has_nametbl, has_imgtbl]):
        errmsg  = ''
        errmsg += ('\n\n!!!!!\n\n')
        errmsg += ('  ! The data tables seem to not be loaded')
        errmsg += (' Files in internal dir: '+repr(internal_dir))
        for fname in os.listdir(internal_dir):
            errmsg += ('   ! fname') 
        errmsg += ('\n\n!!!!!\n\n')
        print(errmsg)
        raise Exception(errmsg)
    print('-------------------------')
    print('[ld2] Loading database tables: ')
    cid_lines  = [] 
    line_num   = 0
    csv_line   = ''
    csv_fields = []
    try:
        # ------------------
        # --- READ NAMES --- 
        # ------------------
        print('[ld2] Loading name table: '+name_table)
        nx2_name = ['____', '____']
        nid2_nx  = { 0:0, 1:1}
        name_lines = open(name_table,'r')
        for line_num, csv_line in enumerate(name_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            nid = int(csv_fields[0])
            name = csv_fields[1]
            nid2_nx[nid] = len(nx2_name)
            nx2_name.append(name)
        name_lines.close()
        if params.VERBOSE_LOAD_DATA:
            print('[ld2] * Loaded '+str(len(nx2_name)-2)+' names (excluding unknown names)')
            print('[ld2] * Done loading name table')

        # -------------------
        # --- READ IMAGES --- 
        # -------------------
        gx2_gname = []
        print('[ld2] Loading images')
        # Load Image Table 
        # <LEGACY CODE>
        if params.VERBOSE_LOAD_DATA:
            print('[ld2] Loading image table: '+image_table)
        gid2_gx = {}
        gid_lines = open(image_table,'r').readlines()
        for line_num, csv_line in enumerate(gid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            gid = int(csv_fields[0])
            if len(csv_fields) == 3: 
                gname = csv_fields[1]
            if len(csv_fields) == 4: 
                gname = csv_fields[1:3]
            gid2_gx[gid] = len(gx2_gname)
            gx2_gname.append(gname)
        nTableImgs = len(gx2_gname)
        fromTableNames = set(gx2_gname)
        if params.VERBOSE_LOAD_DATA:
            print('[ld2] table specified '+str(nTableImgs)+' images')
            # </LEGACY CODE>
            # Load Image Directory
            print('[ld2] Loading image directory: '+img_dir)
        nDirImgs = 0
        nDirImgsAlready = 0
        for fname in os.listdir(img_dir):
            if len(fname) > 4 and fname[-4:].lower() in ['.jpg', '.png', '.tiff']:
                if fname in fromTableNames: 
                    nDirImgsAlready += 1
                    continue
                gx2_gname.append(fname)
                nDirImgs += 1
        if params.VERBOSE_LOAD_DATA:
            print('[ld2] dir specified '+str(nDirImgs)+' images')
            print('[ld2] '+str(nDirImgsAlready)+' were already specified in the table')
            print('[ld2] Loaded '+str(len(gx2_gname))+' images')
            print('[ld2] Done loading images')

        # ------------------
        # --- READ CHIPS --- 
        # ------------------
        print('[ld2] Loading chip table: '+chip_table)
        # Load Chip Table Header
        cid_lines = open(chip_table,'r').readlines()
        # Header Markers
        header_numdata   = '# NumData '
        header_csvformat_re = '# *ChipID,'
        # Default Header Variables
        chip_csv_format = ['ChipID', 'ImgID',  'NameID',   'roi[tl_x  tl_y  w  h]',  'theta']
        num_data   = -1
        # Parse Chip Table Header
        for line_num, csv_line in enumerate(cid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0:
                continue
            csv_line = csv_line.strip('\n')
            if csv_line.find('#') != 0:
                break # Break after header
            if not re.search(header_csvformat_re, csv_line) is None:
                chip_csv_format = [_.strip() for _ in csv_line.strip('#').split(',')]
            if csv_line.find(header_numdata) == 0:
                num_data = int(csv_line.replace(header_numdata,''))
        if params.VERBOSE_LOAD_DATA:
            print('[ld2] num_chips: '+str(num_data))
            print('[ld2] chip_csv_format: '+str(chip_csv_format))
        cid_x   = tryindex(chip_csv_format, 'ChipID')
        gid_x   = tryindex(chip_csv_format, 'ImgID')
        nid_x   = tryindex(chip_csv_format, 'NameID')
        roi_x   = tryindex(chip_csv_format, 'roi[tl_x  tl_y  w  h]')
        theta_x = tryindex(chip_csv_format, 'theta')
        # new fields
        gname_x = tryindex(chip_csv_format, 'Image')
        name_x  = tryindex(chip_csv_format, 'Name')
        required_x = [cid_x, gid_x, gname_x, nid_x, name_x, roi_x, theta_x]
        # Hotspotter Chip Tables
        cx2_cid   = []
        cx2_nx    = []
        cx2_gx    = []
        cx2_roi   = []
        cx2_theta = []
        # x is a csv field index in this context
        # get csv indexes which are unknown properties
        prop_x_list  = np.setdiff1d(range(len(chip_csv_format)), required_x).tolist()
        px2_prop_key = [chip_csv_format[x] for x in prop_x_list]
        prop_dict = {}
        for prop in iter(px2_prop_key):
            prop_dict[prop] = []
        if params.VERBOSE_LOAD_DATA:
            print('[ld2] num_user_properties: '+str(len(prop_dict.keys())))
        # Parse Chip Table
        for line_num, csv_line in enumerate(cid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            # Load Chip ID
            cid = int(csv_fields[cid_x])
            # Load Chip Image Info
            if gid_x != -1:
                gid = int(csv_fields[gid_x])
                gx  = gid2_gx[gid]
            elif gname_x != -1:
                gname = csv_fields[gname_x]
                gx = gx2_name.index(gname)
            # Load Chip Name Info
            if nid_x != -1:
                nid = int(csv_fields[nid_x])
                nx = nid2_nx[nid]
            elif name_x != -1:
                name = csv_fields[name_x]
                nx = nx2_name.index(name)
            # Load Chip ROI Info
            roi_str = csv_fields[roi_x].strip('[').strip(']')
            roi = [int(_) for _ in roi_str.split()]
            # Load Chip theta Info
            if theta_x != -1:
                theta = float(csv_fields[theta_x])
            else:
                theta = 0
            # Append info to cid lists
            cx2_cid.append(cid)
            cx2_gx.append(gx)
            cx2_nx.append(nx)
            cx2_roi.append(roi)
            cx2_theta.append(theta)
            for px, x in enumerate(prop_x_list): 
                prop = px2_prop_key[px]
                prop_val = csv_fields[x]
                prop_dict[prop].append(prop_val)
    except Exception as ex:
        print('[ld2] Failed parsing: '+str(''.join(cid_lines)))
        print('[ld2] Failed on line number:  '+str(line_num))
        print('[ld2] Failed on line:         '+repr(csv_line))
        print('[ld2] Failed on fields:       '+repr(csv_fields))
        raise

    if params.VERBOSE_LOAD_DATA:
        print('[ld2] Loaded: '+str(len(cx2_cid))+' chips')
        print('[ld2] Done loading chip table')
    # Return all information from load_tables
    #hs_tables.gid2_gx = gid2_gx
    #hs_tables.nid2_nx  = nid2_nx
    hs_tables =  HotspotterTables(gx2_gname, nx2_name, cx2_cid, cx2_nx,
                                  cx2_gx, cx2_roi, cx2_theta, prop_dict)
    print('[ld2] Done Loading hotspotter csv tables: '+str(db_dir))

    if 'vdd' in sys.argv:
        helpers.vd(hs_dirs.db_dir)
    if 'vcd' in sys.argv:
        helpers.vd(hs_dirs.computed_dir)
    return hs_dirs, hs_tables
    

def __print_chiptableX(hs_tables):
    #print(hs_tables.execstr('hs_tables'))
    #exec(hs_tables.execstr('hs_tables'))
    cx2_gx    = hs_tables.cx2_gx
    cx2_cid   = hs_tables.cx2_cid
    cx2_nx    = hs_tables.cx2_nx
    cx2_theta = hs_tables.cx2_theta
    cx2_roi   = hs_tables.cx2_roi
    #prop_names = ','.join(px2_propname)
    print('=======================================================')
    print('# Begin ChipTableX')
    print('# ChipID, NameX,  ImgX,     roi[tl_x  tl_y  w  h],  theta')
    chip_iter = iter(zip(cx2_cid, cx2_nx, cx2_gx, cx2_roi, cx2_theta))
    for (cid, nx, gx, roi, theta) in chip_iter:
        print('%8d, %5d, %5d, %25s, %6.3f' % (cid, nx, gx, str(roi).replace(',',''), theta))
    print('# End ChipTableX')
    print('=======================================================')

def print_chiptable(hs_tables):
    #exec(hs_tables.execstr('hs_tables'))
    #print(hs_tables.execstr('hs_tables'))
    #prop_names = ','.join(px2_propname)
    print('=======================================================')
    print('# Begin ChipTable')
    # Get length of the max vals for formating
    cx2_cid   = hs_tables.cx2_cid
    cx2_theta = hs_tables.cx2_theta
    cx2_gname = [hs_tables.gx2_gname[gx] for gx in  hs_tables.cx2_gx]
    cx2_name  = [hs_tables.nx2_name[nx]  for nx in  hs_tables.cx2_nx]
    cx2_stroi = [str(roi).replace(',','') for roi in  hs_tables.cx2_roi]
    max_gname = max([len(gname) for gname in iter( cx2_gname)])
    max_name  = max([len(name)  for name  in iter( cx2_name) ])
    max_stroi = max([len(stroi) for stroi in iter( cx2_stroi)])
    _mxG = str(max([max_gname+1, 5]))
    _mxN = str(max([max_name+1, 4]))
    _mxR = str(max([max_stroi+1, 21]))

    fmt_str = '%8d, %'+_mxN+'s, %'+_mxG+'s, %'+_mxR+'s, %6.3f'

    c_head = '# ChipID'
    n_head = ('%'+_mxN+'s') %  'Name'
    g_head = ('%'+_mxG+'s') %  'Image'
    r_head = ('%'+_mxR+'s') %  'roi[tl_x  tl_y  w  h]'
    t_head = ' theta'
    header = ', '.join([c_head,n_head,g_head,r_head,t_head])
    print(header)

    # Build the table
    chip_iter = iter(zip( cx2_cid, cx2_name, cx2_gname, cx2_stroi, cx2_theta))
    for (cid, name, gname, stroi, theta) in chip_iter:
        _roi  = str(roi).replace(',',' ') 
        print(fmt_str % (cid, name, gname, stroi, theta))

    print('# End ChipTable')
    print('=======================================================')


def make_csv_table(column_labels=None, column_list=[], header='', column_type=None):
    if len(column_list) == 0: 
        print('[ld2] No columns')
        return header
    column_len  = [len(col) for col in column_list]
    num_data =  column_len[0] 
    if num_data == 0:
        print('[ld2] No data')
        return header
    if any([num_data != clen for clen in column_len]):
        print('[ld2] inconsistent column length')
        return header

    if column_type is None:
        column_type = [type(col[0]) for col in column_list]

    csv_rows = []
    csv_rows.append(header)
    csv_rows.append('# NumData '+str(num_data))

    column_maxlen = []
    column_str_list = []

    if column_labels is None:
        column_labels = ['']*len(column_list)

    def _toint(c):
        try: 
            if np.isnan(c):
                return 'nan'
        except TypeError as ex:
            print('------')
            print('[ld2] _toint(c) failed')
            print('[ld2] c = %r ' % c)
            print('[ld2] type(c) = %r ' % type(c))
            print('------')
            raise
        return ('%d') % int(c)
    
    for col, lbl, coltype in iter(zip(column_list, column_labels, column_type)):
        if coltype == types.ListType:
            col_str  = [str(c).replace(',',' ') for c in iter(col)]
        elif coltype == types.FloatType:
            col_str = [('%.2f') % float(c) for c in iter(col)]
        elif coltype == types.IntType:
            col_str = [_toint(c) for c in iter(col)]
        elif coltype == types.StringType:
            col_str = [str(c) for c in iter(col)]
        else:
            col_str  = [str(c) for c in iter(col)]
        col_lens = [len(s) for s in iter(col_str)]
        max_len  = max(col_lens)
        max_len  = max(len(lbl), max_len)
        column_maxlen.append(max_len)
        column_str_list.append(col_str)

    fmtstr = ','.join(['%'+str(maxlen+2)+'s' for maxlen in column_maxlen])
    csv_rows.append('# '+fmtstr % tuple(column_labels))
    for row in zip(*column_str_list):
        csv_rows.append('  ' + fmtstr % row)

    csv_text = '\n'.join(csv_rows)
    return csv_text

# Common databases I use
FROGS     = params.WORK_DIR+'/Frogs'
JAGUARS   = params.WORK_DIR+'/JAG_Jaguar_Data'
NAUTS     = params.WORK_DIR+'/NAUT_Dan'
GZ_ALL    = params.WORK_DIR+'/GZ_ALL'
WS_HARD   = params.WORK_DIR+'/WS_hard'
MOTHERS   = params.WORK_DIR+'/HSDB_zebra_with_mothers'
OXFORD    = params.WORK_DIR+'/Oxford_Buildings'
PARIS     = params.WORK_DIR+'/Paris_Buildings'
SONOGRAMS = params.WORK_DIR+'/sonograms'

#if sys.platform == 'linux2':
    #DEFAULT = MOTHERS
#else:
DEFAULT = params.DEFAULT

@helpers.unit_test
def test_load_csv():
    db_dir = params.DEFAULT
    hs_dirs, hs_tables = load_csv_tables(db_dir)
    print_chiptable(hs_tables)
    __print_chiptableX(hs_tables)
    print(hs_tables.nx2_name)
    print(hs_tables.gx2_gname)
    hs_tables.printme2(val_bit=True, max_valstr=10)
    return hs_dirs, hs_tables

# Test load csv tables
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    import draw_func2 as df2
    if '--test' in sys.argv:
        helpers.PRINT_CHECKS = True #might as well
        hs_dirs, hs_tables = test_load_csv()
    else:
        db_dir = params.DEFAULT
        hs_dirs, hs_tables = load_csv_tables(db_dir)
    exec(df2.present())
