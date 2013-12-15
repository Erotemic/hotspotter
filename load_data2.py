'''
Module: load_data
    Loads the paths and table information from which all other data is computed.
    This is the first script run in the loading pipeline. 
'''
from __future__ import division, print_function
import __builtin__
import sys
# Standard
from os.path import join, exists, splitext
import cv2
import fnmatch
import os
import shutil
import re
import sys
import textwrap
import types
# Science
import numpy as np
from PIL import Image
# Hotspotter
from Printable import DynStruct
import DataStructures as ds
import helpers
import params

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
    print('[ld2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr(): reload_module()

def printDBG(msg, lbl=''):
    print('DBG: '+lbl+str(msg))

CHIP_TABLE_FNAME = 'chip_table.csv'
NAME_TABLE_FNAME = 'name_table.csv'
IMAGE_TABLE_FNAME = 'image_table.csv'

# TODO: Allow alternative internal directories
RDIR_INTERNAL_ALTS = ['/hs_internals']
RDIR_INTERNAL2 = '.hs_internals'
RDIR_IMG2 = 'images'

# paths relative to dbdir
RDIR_IMG      = '/'+RDIR_IMG2
RDIR_INTERNAL = '/'+RDIR_INTERNAL2
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


# Testing helper functions
def get_test_data(qcx=0, cx=None, db_dir=None):
    import params
    import HotSpotter
    hs = helpers.search_stack_for_localvar('hs')
    if db_dir is None:
        db_dir = params.DEFAULT
    if hs is None:
        hs = HotSpotter.HotSpotter(db_dir)
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

# ______________________________

def tryindex(list, val):
    try: 
        return list.index(val)
    except ValueError as ex:
        return -1

def load_csv_tables(db_dir, allow_new_dir=True):
    '''
    Big function which loads the csv tables from a datatabase directory
    Returns HotspotterDirs and HotspotterTables
    '''
    if 'vdd' in sys.argv:
        helpers.vd(hs_dirs.db_dir)
    print('\n=============================')
    print('[ld2] Loading hotspotter csv tables: '+str(db_dir))
    print('=============================')
    hs_dirs = ds.HotspotterDirs(db_dir)
    hs_tables = ds.HotspotterTables()
    #exec(hs_dirs.execstr('hs_dirs'))
    #print(hs_dirs.execstr('hs_dirs'))
    feat_dir     = hs_dirs.feat_dir
    img_dir      = hs_dirs.img_dir
    rchip_dir    = hs_dirs.rchip_dir
    chip_dir     = hs_dirs.chip_dir
    internal_dir = hs_dirs.internal_dir
    db_dir       = hs_dirs.db_dir
    # --- Table Names ---
    chip_table   = join(internal_dir, CHIP_TABLE_FNAME)
    name_table   = join(internal_dir, NAME_TABLE_FNAME)
    image_table  = join(internal_dir, IMAGE_TABLE_FNAME) # TODO: Make optional
    # --- CHECKS ---
    has_dbdir   = helpers.checkpath(db_dir)
    has_imgdir  = helpers.checkpath(img_dir)
    has_chiptbl = helpers.checkpath(chip_table)
    has_nametbl = helpers.checkpath(name_table)
    has_imgtbl  = helpers.checkpath(image_table)

    if not all([has_dbdir, has_imgdir, has_chiptbl, has_nametbl, has_imgtbl]):
        if allow_new_dir:
            return hs_dirs, hs_tables
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
        nid2_nx  = {0:0, 1:1}
        name_lines = open(name_table,'r')
        for line_num, csv_line in enumerate(name_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0: continue
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
            if len(csv_fields) == 3: gname = csv_fields[1]
            if len(csv_fields) == 4: gname = csv_fields[1:3]
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
        header_numdata = '# NumData '
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
    hs_tables.init(gx2_gname, nx2_name, cx2_cid, cx2_nx, cx2_gx, 
                   cx2_roi, cx2_theta, prop_dict)

    print('[ld2] Done Loading hotspotter csv tables: '+str(db_dir))
    if 'vcd' in sys.argv:
        helpers.vd(hs_dirs.computed_dir)
    return hs_dirs, hs_tables
   
def make_csv_table(column_labels=None, column_list=[], header='', column_type=None):
    if len(column_list) == 0: 
        print('[ld2] No columns')
        return header
    column_len = [len(col) for col in column_list]
    num_data = column_len[0] 
    if num_data == 0:
        print('[ld2.make_csv_table()] No data. (header=%r)' % (header,))
        return header
    if any([num_data != clen for clen in column_len]):
        print('[lds] column_labels = %r ' % (column_labels,))
        print('[lds] column_len = %r ' % (column_len,))
        print('[ld2] inconsistent column lengths')
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
            col_str  = [str(c).replace(',',' ').replace('.','') for c in iter(col)]
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

def backup_csv_tables(hs, force_backup=False):
    internal_dir = hs.dirs.internal_dir
    backup_dir = join(internal_dir, 'backup_v0.1.0')
    if not exists(backup_dir) or force_backup:
        helpers.ensuredir(backup_dir)
        timestamp = helpers.get_timestamp(use_second=True)
        def do_backup(fname):
            src = join(internal_dir, fname)
            dst_fname = ('%s_bak-'+timestamp+'%s') % splitext(fname)
            dst = join(backup_dir, dst_fname)
            if exists(src): shutil.copy(src, dst)
        do_backup(CHIP_TABLE_FNAME)
        do_backup(NAME_TABLE_FNAME)
        do_backup(IMAGE_TABLE_FNAME)

def write_chip_table(internal_dir, cx2_cid, cx2_gid, cx2_nid,
                     cx2_roi, cx2_theta, prop_dict=None):
    print('[ld2] Writing Chip Table')
    # Make chip_table.csv
    header = '# chip table'
    column_labels = ['ChipID', 'ImgID', 'NameID', 'roi[tl_x  tl_y  w  h]', 'theta']
    column_list   = [cx2_cid, cx2_gid, cx2_nid, cx2_roi, cx2_theta]
    column_type   = [int, int, int, list, float]
    if not prop_dict is None:
        for key, val in prop_dict.iteritems():
            column_labels.append(key)
            column_list.append(val)
            column_type.append(str)

    chip_table = make_csv_table(column_labels, column_list, header, column_type)
    chip_table_fpath  = join(internal_dir, CHIP_TABLE_FNAME)
    helpers.write_to(chip_table_fpath, chip_table)

def write_name_table(internal_dir, nx2_nid, nx2_name):
    print('[ld2] Writing Name Table')
    # Make name_table.csv
    column_labels = ['nid', 'name']
    column_list   = [nx2_nid[2:], nx2_name[2:]] # dont write ____ for backcomp
    header = '# name table'
    name_table = make_csv_table(column_labels, column_list, header)
    name_table_fpath  = join(internal_dir, NAME_TABLE_FNAME)
    helpers.write_to(name_table_fpath, name_table)

def write_image_table(internal_dir, gx2_gid, gx2_gname):
    print('[ld2] Writing Image Table')
    # Make image_table.csv 
    column_labels = ['gid', 'gname', 'aif'] # do aif for backwards compatibility
    gx2_aif = np.ones(len(gx2_gid), dtype=np.uint32)
    column_list   = [gx2_gid, gx2_gname, gx2_aif]
    header = '# image table'
    image_table = make_csv_table(column_labels, column_list, header)
    image_table_fpath = join(internal_dir, IMAGE_TABLE_FNAME)
    helpers.write_to(image_table_fpath, image_table)

def write_csv_tables(hs):
    'Saves the tables to disk'
    import convert_db
    internal_dir = hs.dirs.internal_dir
    CREATE_BACKUP = True
    if CREATE_BACKUP:
        backup_csv_tables(hs, force_backup=True)

    # Valid indexes
    valid_cx = np.where(hs.tables.cx2_cid > 0)[0]
    valid_nx = np.where(hs.tables.nx2_name != '')[0]
    valid_gx = np.where(hs.tables.gx2_gname != '')[0]

    # Valid chip tables
    cx2_cid   = hs.tables.cx2_cid[valid_cx]
    # Use the indexes as ids (FIXME: Just go back to g/n-ids)
    cx2_gid   = hs.tables.cx2_gx[valid_cx]+1 # FIXME
    cx2_nid   = hs.tables.cx2_nx[valid_cx]   # FIXME
    cx2_roi   = hs.tables.cx2_roi[valid_cx]
    cx2_theta = hs.tables.cx2_theta[valid_cx]
    prop_dict = {propkey:cx2_propval[valid_cx] 
                 for (propkey, cx2_propval) in hs.tables.prop_dict}

    # Valid name tables
    nx2_nid   = valid_nx # FIXME
    nx2_name  = hs.tables.nx2_name[valid_nx]

    # Image Tables
    gx2_gid   = valid_gx+1 # FIXME
    gx2_gname = hs.tables.gx2_gname[valid_gx]

    # Do write
    # these were stolen from convert chips. TODO: Remove them there.
    write_chip_table(internal_dir, cx2_cid, cx2_gid, cx2_nid,
                     cx2_roi, cx2_theta, prop_dict)
    write_name_table(internal_dir, nx2_nid, nx2_name)
    write_image_table(internal_dir, gx2_gid, gx2_gname)
    


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
