'''
Module: load_data
    Loads the paths and table information from which all other data is computed.
    This is the first script run in the loading pipeline. 
'''
import re
import os
import sys
from os.path import join
import fnmatch
import cv2
from PIL import Image
import types
import numpy as np
import helpers
import params
import textwrap
from helpers import checkpath, unit_test, ensure_path, symlink, remove_files_in_dir
from helpers import myprint
from Printable import DynStruct
# rename: data_managers? 
#print ('LOAD_MODULE: load_data2.py')

# reloads this module when I mess with it
def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

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

def NewHotSpotter():
    hs_tables = HotspotterTables()

# ___CLASS HOTSPOTTER____
class HotSpotter(DynStruct):
    '''The HotSpotter main class is a root handle to all relevant data'''
    def __init__(hs, db_dir=None,
                 load_matcher=True,
                 load_features=True,
                 samples_from_file=False):
        super(HotSpotter, hs).__init__()
        hs.tables = None
        hs.feats  = None
        hs.cpaths = None
        hs.dirs   = None
        hs.matcher = None
        hs.train_sample_cx    = None
        hs.test_sample_cx     = None
        hs.database_sample_cx = None
        if not db_dir is None:
            hs.load_database(db_dir, load_matcher, load_features)
    #---------------
    def load_database(hs, db_dir,
                      load_matcher=True,
                      load_features=True,
                      samples_from_file=False):
        import chip_compute2 as cc2
        import feature_compute2 as fc2
        # Load data
        hs_dirs, hs_tables = load_csv_tables(db_dir)
        hs_cpaths = cc2.load_chip_paths(hs_dirs, hs_tables)
        hs.tables  = hs_tables
        hs.cpaths  = hs_cpaths
        hs.dirs    = hs_dirs
        hs.feats   = None
        if load_features:
            hs_feats  = fc2.load_chip_features(hs_dirs, hs_tables, hs_cpaths)
            hs.feats  = hs_feats
        else: 
            print('Not Loading Features!!')
        # Load sample sets
        hs.database_sample_cx = None
        hs.test_sample_cx     = None
        hs.train_sample_cx    = None
        if samples_from_file:
            hs.default_test_train_database_samples()
        else:
            hs.default_test_train_database_samples()
        # Load Matcher
        if load_matcher: 
            hs.load_matcher()
    #---------------
    def load_matcher(hs):
        import match_chips2 as mc2
        hs.matcher = mc2.Matcher(hs, params.__MATCH_TYPE__)
    #---------------
    def default_test_train_database_samples(hs):
        print(textwrap.dedent('''
        =============================
        Using all data as sample set
        ============================='''))
        hs.database_sample_cx = range(len(hs.tables.cx2_cid))
        hs.test_sample_cx     = range(len(hs.tables.cx2_cid))
        hs.train_sample_cx    = range(len(hs.tables.cx2_cid))

    def load_test_train_database_samples_from_file(hs,
                                                   db_sample_fname='database_sample.txt',
                                                   test_sample_fname='test_sample.txt',
                                                   train_sample_fname='train_sample.txt',):
        'tries to load test / train / database sample from internal dir'
        print(textwrap.dedent('''
        =============================
        Loading sample sets from disk relative to internal_dir
        ============================='''))
        internal_dir = hs.dirs.internal_dir
        db_sample_fpath    = join(internal_dir, db_sample_fname)
        test_sample_fpath  = join(internal_dir, test_sample_fname)
        train_sample_fpath = join(internal_dir, train_sample_fname)

        hs.database_sample_cx = np.array(helpers.eval_from(db_sample_fpath, False))
        hs.test_sample_cx     = np.array(helpers.eval_from(test_sample_fpath, False))
        hs.train_sample_cx    = np.array(helpers.eval_from(train_sample_fpath, False))
        if hs.database_sample_cx is None and hs.test_sample_cx is None and hs.train_sample_cx is None: 
            hs.default_test_train_database_samples()
        #hs.test_sample_cx = np.array([0,2,3])
        #db_sample_cx = range(len(cx2_desc)) if hs.database_sample_cx is None \
                               #else hs.database_sample_cx
    #---------------
    def delete_computed_dir(hs):
        computed_dir = hs.dirs.computed_dir
        helpers.remove_files_in_dir(computed_dir, recursive=True)
    #---------------
    def vdd(hs):
        db_dir = os.path.normpath(hs.dirs.db_dir)
        print('opening db_dir: %r ' % db_dir)
        helpers.vd(db_dir)
    #---------------
    def vcd(hs):
        computed_dir = os.path.normpath(hs.dirs.computed_dir)
        print('opening computed_dir: %r ' % computed_dir)
        helpers.vd(computed_dir)
    #--------------
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

# Testing helper functions
def get_sv_test_data():
    import load_data2
    db_dir = load_data2.DEFAULT
    hs = load_data2.HotSpotter(db_dir)
    qcx = 0
    cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    return (hs, qcx, cx, fm, rchip1, rchip2, kpts1, kpts2)

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
        ensure_path(self.internal_dir)
        ensure_path(self.computed_dir)
        ensure_path(self.chip_dir)
        ensure_path(self.rchip_dir)
        ensure_path(self.feat_dir)
        ensure_path(self.result_dir)
        ensure_path(self.rchip_dir)
        ensure_path(self.qres_dir)
        ensure_path(self.cache_dir)
        # Shortcut to internals
        internal_sym = db_dir + '/Shortcut-to-hs_internals'
        if not os.path.islink(internal_sym):
            symlink(self.internal_dir, internal_sym, noraise=True)
        results_sym = db_dir + '/Shortcut-to-results'
        if not os.path.islink(internal_sym):
            symlink(self.result_dir, results_sym, noraise=True)

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
    print('Loading hotspotter csv tables: '+str(db_dir))
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
    has_dbdir   = checkpath(db_dir)
    has_imgdir  = checkpath(img_dir)
    has_chiptbl = checkpath(chip_table)
    has_nametbl = checkpath(name_table)
    has_imgtbl  = checkpath(image_table)
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
    print('Loading database tables: ')
    cid_lines  = [] 
    line_num   = 0
    csv_line   = ''
    csv_fields = []
    try:
        # ------------------
        # --- READ NAMES --- 
        # ------------------
        print('... Loading name table: '+name_table)
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
        if params.__VERBOSE_LOAD_DATA__:
            print('      * Loaded '+str(len(nx2_name)-2)+' names (excluding unknown names)')
            print('      * Done loading name table')

        # -------------------
        # --- READ IMAGES --- 
        # -------------------
        gx2_gname = []
        print('... Loading images')
        # Load Image Table 
        # <LEGACY CODE>
        if params.__VERBOSE_LOAD_DATA__:
            print('    ... Loading image table: '+image_table)
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
        if params.__VERBOSE_LOAD_DATA__:
            print('          * table specified '+str(nTableImgs)+' images')
            # </LEGACY CODE>
            # Load Image Directory
            print('    ... Loading image directory: '+img_dir)
        nDirImgs = 0
        nDirImgsAlready = 0
        for fname in os.listdir(img_dir):
            if len(fname) > 4 and fname[-4:].lower() in ['.jpg', '.png', '.tiff']:
                if fname in fromTableNames: 
                    nDirImgsAlready += 1
                    continue
                gx2_gname.append(fname)
                nDirImgs += 1
        if params.__VERBOSE_LOAD_DATA__:
            print('          * dir specified '+str(nDirImgs)+' images')
            print('          * '+str(nDirImgsAlready)+' were already specified in the table')
            print('  * Loaded '+str(len(gx2_gname))+' images')
            print('  * Done loading images')

        # ------------------
        # --- READ CHIPS --- 
        # ------------------
        print('... Loading chip table: '+chip_table)
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
        if params.__VERBOSE_LOAD_DATA__:
            print('  * num_chips: '+str(num_data))
            print('  * chip_csv_format: '+str(chip_csv_format))
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
        if params.__VERBOSE_LOAD_DATA__:
            print('  * num_user_properties: '+str(len(prop_dict.keys())))
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
        print('Failed parsing: '+str(''.join(cid_lines)))
        print('Failed on line number:  '+str(line_num))
        print('Failed on line:         '+repr(csv_line))
        print('Failed on fields:       '+repr(csv_fields))
        raise

    if params.__VERBOSE_LOAD_DATA__:
        print('  * Loaded: '+str(len(cx2_cid))+' chips')
        print('  * Done loading chip table')
    # Return all information from load_tables
    #hs_tables.gid2_gx = gid2_gx
    #hs_tables.nid2_nx  = nid2_nx
    hs_tables =  HotspotterTables(gx2_gname, nx2_name, cx2_cid, cx2_nx,
                                  cx2_gx, cx2_roi, cx2_theta, prop_dict)
    print('===============================')
    print('Done Loading hotspotter csv tables: '+str(db_dir))
    print('===============================\n')

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
        print('No columns')
        return header
    column_len  = [len(col) for col in column_list]
    num_data =  column_len[0] 
    if num_data == 0:
        print('No data')
        return header
    if any([num_data != clen for clen in column_len]):
        print('inconsistent column length')
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
            print('_toint(c) failed')
            print('c = %r ' % c)
            print('type(c) = %r ' % type(c))
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

# for mothers dataset
viewpoint_pairs = [
    (19, 20),
    (110, 108),
    (108, 112), # 108 is very dark
    (16, 17),
    (73,71),
    (75,78)
]

image_quality = [
    (27, 26), # minor viewpoint
    (67,68), #stupid hard case (query from 68 to 67 direction is better (start with foal)
    (52,53),
    (73,71)
]

lighting_pairs = [
    (49, 50), #brush occlusion on legs
    (93, 94),
    (105,104)
]

confused_pairs = []

occlusion= [
    (64,65)
]


# MODULE GLOBAL VARIABLES
WORK_DIR = 'D:/data/work'
if sys.platform == 'linux2':
    WORK_DIR = '/media/Store/data/work'

# Common databases I use
FROGS     = WORK_DIR+'/FROG_tufts'
JAGUARS   = WORK_DIR+'/JAG_Jaguar_Data'
NAUTS     = WORK_DIR+'/NAUT_Dan'
GZ_ALL    = WORK_DIR+'/GZ_ALL'
WS_HARD   = WORK_DIR+'/WS_hard'
MOTHERS   = WORK_DIR+'/HSDB_zebra_with_mothers'
OXFORD    = WORK_DIR+'/Oxford_Buildings'
PARIS     = WORK_DIR+'/Paris_Buildings'
SONOGRAMS = WORK_DIR+'/sonograms'


#if sys.platform == 'linux2':
    #DEFAULT = MOTHERS
#else:
    #DEFAULT = NAUTS
DEFAULT = MOTHERS

dev_databases = {
    'SONOGRAMS' : SONOGRAMS,
    'JAG'       : JAGUARS,
    'FROGS'     : FROGS,
    'NAUTS'     : NAUTS,
    'GZ_ALL'    : GZ_ALL,
    'WS_HARD'   : WS_HARD,
    'MOTHERS'   : MOTHERS,
    'OXFORD'    : OXFORD,
    'PARIS'     : PARIS}

for argv in iter(sys.argv):
    if argv.upper() in dev_databases.keys():
        print('\n'.join([' * Default Database set to:'+argv.upper(),
                         ' * Previously: '+str(DEFAULT)]))
        DEFAULT = dev_databases[argv.upper()]
#print(' * load_data: Default database is: '+str(DEFAULT))

@unit_test
def test_load_csv():
    db_dir = DEFAULT
    hs_dirs, hs_tables = load_csv_tables(db_dir)
    print_chiptable(hs_tables)
    __print_chiptableX(hs_tables)
    print(hs_tables.nx2_name)
    print(hs_tables.gx2_gname)
    hs_tables.printme2(val_bit=True, max_valstr=10)
    return hs_dirs, hs_tables

# Test load csv tables
if __name__ == '__main__':
    import drawing_functions2 as df2
    if '--test' in sys.argv:
        helpers.__PRINT_CHECKS__ = True #might as well
        hs_dirs, hs_tables = test_load_csv()
    else:
        db_dir = DEFAULT
        hs_dirs, hs_tables = load_csv_tables(db_dir)
    exec(df2.present())
