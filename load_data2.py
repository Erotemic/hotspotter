'''
Module: load_data
    Loads the paths and table information from which all other data is computed.
    This is the first script run in the loading pipeline. 
'''
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
    import imp
    import sys
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
    def __init__(hs, db_dir=None, **kwargs):
        super(HotSpotter, hs).__init__()
        hs.num_cx = None
        hs.tables = None
        hs.feats  = None
        hs.cpaths = None
        hs.dirs   = None
        hs.matcher = None
        hs.train_sample_cx    = None
        hs.test_sample_cx     = None
        hs.database_sample_cx = None
        if not db_dir is None:
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

    def load_chips(hs):
        import chip_compute2 as cc2
        hs_cpaths = cc2.load_chip_paths(hs.dirs, hs.tables)
        hs.cpaths  = hs_cpaths

    def load_features(hs, load_kpts=True, load_desc=True):
        import feature_compute2 as fc2
        hs_feats  = fc2.load_chip_features(hs.dirs, hs.tables, hs.cpaths, load_kpts, load_desc)
        hs.feats  = hs_feats

    def load_matcher(hs):
        import match_chips2 as mc2
        hs.matcher = mc2.Matcher(hs, params.__MATCH_TYPE__)
    #---------------

    def load_database(hs, db_dir,
                      load_matcher=True,
                      load_features=True,
                      samples_subset=(None, None)):
        # Load data
        hs.load_tables(db_dir)
        hs.load_chips()
        # Load sample sets
        hs.set_sample_subset(*samples_subset)
        if load_features:
            hs.load_features()
        else: 
            print('[hs] Not Loading Features!!')
        # Load Matcher
        if load_matcher: 
            hs.load_matcher()
    #---------------
    def get_valid_cxs(hs):
        valid_cxs, = np.where(np.array(hs.tables.cx2_cid) > 0)
        return valid_cxs

    def split_test_train_at(hs, pos):
        valid_cxs = hs.get_valid_cxs()
        test_samp  = valid_cxs[:pos]
        train_samp = valid_cxs[pos+1:]
        hs.set_samples(test_samp, train_samp)

    def set_sample_subset(hs, pos1, pos2):
        valid_cxs = hs.get_valid_cxs()
        test_samp  = valid_cxs[pos1:pos2]
        train_samp = test_samp
        hs.set_samples(test_samp, train_samp)

    def set_samples(hs, test_samp=None,
                        train_samp=None,
                        db_samp=None):
        ''' This is the correct function to use when setting samples '''
        print('[hs] set_samples():')
        valid_cxs = hs.get_valid_cxs()
        if test_samp is None:
            print('[hs] * using all chips in testing')
            test_samp = valid_cxs
        if train_samp is None:
            print('[hs] * using all chips in training')
            train_samp = valid_cxs
        if db_samp is None:
            print('[hs] * using training set as database set')
            db_samp = train_samp


        # Debugging and Info
        test_train_isect = np.intersect1d(test_samp, train_samp)
        db_train_isect = np.intersect1d(db_samp, train_samp)
        db_test_isect = np.intersect1d(db_samp, test_samp)
        print('[hs]   ---')
        print('[hs] * | isect(test, train) |  = %d' % len(test_train_isect))
        print('[hs] * | isect(db, train)   |  = %d' % len(db_train_isect))
        print('[hs] * | isect(db, test)    |  = %d' % len(db_test_isect))
        print('[hs] * num_valid_cxs = %d' % len(valid_cxs))
        lentup = (len(test_samp), len(train_samp), len(db_samp))
        print('[hs] * num_test=%d, num_train=%d, num_db=%d' % lentup)

        hs.database_sample_cx = db_samp
        hs.train_sample_cx    = train_samp
        hs.test_sample_cx     = test_samp

        train_hash = repr(tuple(train_samp))
        train_id = helpers.hashstr(train_hash)
        print('[hs] set_samples(): train_id=%r' % train_id)
        params.TRAIN_SAMPLE_ID = train_id
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
    def get_nx2_cxs(hs):
        cx2_nx = hs.tables.cx2_nx
        max_nx = cx2_nx.max()
        nx2_cxs = [[] for _ in xrange(max_nx+1)]
        for cx, nx in enumerate(cx2_nx):
            if nx > 0:
                nx2_cxs[nx].append(cx)
        return nx2_cxs

    def get_other_cxs(hs, cx):
        cx2_nx   = hs.tables.cx2_nx
        nx = cx2_nx[cx]
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
def get_sv_test_data(qcx=0, cx=None):
    import load_data2
    hs = helpers.search_stack_for_localvar('hs')
    if hs is None:
        db_dir = load_data2.DEFAULT
        hs = load_data2.HotSpotter(db_dir)
    if cx is None:
        cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    return (hs, qcx, cx, fm, fs, rchip1, rchip2, kpts1, kpts2)

# ______________________________

# ___CLASS HOTSPOTTER TABLES____
class HotspotterTables(DynStruct):
    def __init__(self, 
                 gx2_gname = [],
                 nx2_name  = [],
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
        if not os.path.islink(internal_sym):
            helpers.symlink(self.internal_dir, internal_sym, noraise=False)

        results_sym = db_dir + '/Shortcut-to-results'
        if not os.path.islink(results_sym):
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
    import draw_func2 as df2
    if '--test' in sys.argv:
        helpers.PRINT_CHECKS = True #might as well
        hs_dirs, hs_tables = test_load_csv()
    else:
        db_dir = params.DEFAULT
        hs_dirs, hs_tables = load_csv_tables(db_dir)
    exec(df2.present())
