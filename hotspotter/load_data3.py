'''
Module: load_data
    Loads the paths and table information from which all other data is computed.
    This is the first script run in the loading pipeline.
'''
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[ld3]', DEBUG=False)
# Standard
from os.path import join, exists, splitext
import os
import re
import shutil
import sys
# Science
import numpy as np
from PIL import Image
# Hotspotter
import DataStructures as ds
from hscom import helpers
from hscom import helpers as util
from hscom import tools
import db_info

#========================================
# GLOBALS
#========================================

VERBOSE_LOAD_DATA = True
VERBOSE_DETERMINE_VERSION = False

CHIP_TABLE_FNAME = 'chip_table.csv'
NAME_TABLE_FNAME = 'name_table.csv'
IMAGE_TABLE_FNAME = 'image_table.csv'

# TODO: Allow alternative internal directories
RDIR_INTERNAL_ALTS = ['.hs_internals']
RDIR_INTERNAL2 = '_hsdb'
RDIR_IMG2 = 'images'

# paths relative to dbdir
RDIR_IMG      = RDIR_IMG2
RDIR_INTERNAL = RDIR_INTERNAL2
RDIR_COMPUTED = join(RDIR_INTERNAL2, 'computed')
RDIR_CHIP     = join(RDIR_COMPUTED, 'chips')
RDIR_RCHIP    = join(RDIR_COMPUTED, 'temp')
RDIR_CACHE    = join(RDIR_COMPUTED, 'cache')
RDIR_FEAT     = join(RDIR_COMPUTED, 'feats')
RDIR_RESULTS  = join(RDIR_COMPUTED, 'results')
RDIR_QRES     = join(RDIR_COMPUTED, 'query_results')

UNKNOWN_NAME = '____'
VALID_UNKNOWN_NAMES = ['UNIDENTIFIED', UNKNOWN_NAME]


def detect_checkpath(dir_):
    return helpers.checkpath(dir_, verbose=VERBOSE_DETERMINE_VERSION)


def detect_version(db_dir):
    '''
    Attempt to detect the version of the database
    Input: db_dir - the directory to the database
    Output:
    '''
    printDBG('[ld3] detect_version(%r)' % db_dir)
    hs_dirs = ds.HotspotterDirs(db_dir)
    # --- Directories ---
    db_dir       = hs_dirs.db_dir
    img_dir      = hs_dirs.img_dir
    internal_dir = hs_dirs.internal_dir

    # --- Table File Names ---
    chip_table   = join(internal_dir, CHIP_TABLE_FNAME)
    name_table   = join(internal_dir, NAME_TABLE_FNAME)
    image_table  = join(internal_dir, IMAGE_TABLE_FNAME)  # TODO: Make optional

    # --- CHECKS ---
    has_dbdir   = detect_checkpath(db_dir)
    has_imgdir  = detect_checkpath(img_dir)
    has_chiptbl = detect_checkpath(chip_table)
    has_nametbl = detect_checkpath(name_table)
    has_imgtbl  = detect_checkpath(image_table)

    # ChipTable Header Markers and ChipTable Header Variables
    header_numdata = '# NumData '
    header_csvformat_re = '# *ChipID,'
    chip_csv_format = ['ChipID', 'ImgID',  'NameID',   'roi[tl_x  tl_y  w  h]',  'theta']
    vss_csvformat_re = '#imgindex,'
    v12_csvformat_re = r'#[ 0-9]*\) '
    v12_csv_format = ['instance_id', 'image_id', 'name_id', 'roi']

    db_version = 'current'
    isCurrentVersion = all([has_dbdir, has_imgdir, has_chiptbl, has_nametbl, has_imgtbl])
    printDBG('[ld3] isCurrentVersion=%r' % isCurrentVersion)

    if not isCurrentVersion:
        def assign_alternate(tblname, optional=False):
            # Checks several places for target file
            path = join(db_dir, tblname)
            if detect_checkpath(path):
                return path
            path = join(db_dir, '.hs_internals', tblname)
            if detect_checkpath(path):
                return path
            if optional:
                return None
            else:
                raise AssertionError('bad state=%r' % tblname)

        # Assign the following:
        # db_version : database version,
        # header_csvformat_re : Header format regex (to locate the # header)
        # chip_cvs_format : Default header order
        # chip_table, name_table, image_table

        # HOTSPOTTER VERSION 2
        if db_info.has_v2_gt(db_dir):
            db_version = 'hotspotter-v2'
            header_csvformat_re = v12_csvformat_re
            chip_csv_format = 'MULTILINE'
            chip_table  = assign_alternate('instance_table.csv')
            name_table  = assign_alternate('name_table.csv')
            image_table = assign_alternate('image_table.csv')
        # HOTSPOTTER VERSION 1
        elif db_info.has_v1_gt(db_dir):
            db_version = 'hotspotter-v1'
            header_csvformat_re = v12_csvformat_re
            chip_csv_format = 'MULTILINE'
            chip_table  = assign_alternate('animal_info_table.csv')
            name_table  = assign_alternate('name_table.csv', optional=True)
            image_table = assign_alternate('image_table.csv', optional=True)
        # STRIPESPOTTER VERSION
        elif db_info.has_ss_gt(db_dir):
            db_version = 'stripespotter'
            header_csvformat_re = vss_csvformat_re
            chip_csv_format = ['imgindex', 'original_filepath', 'roi', 'animal_name']
            chip_table = join(db_dir, 'SightingData.csv')
            name_table  = None
            image_table = None
            if not detect_checkpath(chip_table):
                msg = 'chip_table=%r must exist to convert stripespotter db' % chip_table
                raise AssertionError(msg)
        else:
            try:
                # ALTERNATIVE CURRENT VERSION
                db_version = 'current'  # Well almost
                chip_table  = assign_alternate(CHIP_TABLE_FNAME)
                name_table  = assign_alternate(NAME_TABLE_FNAME)
                image_table = assign_alternate(IMAGE_TABLE_FNAME)
            except AssertionError:
                # CORRUPTED CURRENT VERSION
                if db_info.has_partial_gt(db_dir):
                    db_version = 'partial'
                    chip_table =  join(db_dir, 'flat_table.csv')
                    name_table  = None
                    image_table = None
                # XLSX VERSION
                elif db_info.has_xlsx_gt(db_dir):
                    db_version = 'xlsx'
                    chip_table  = None
                    name_table  = None
                    image_table = None
                # NEW DATABASE
                else:
                    db_version = 'newdb'
                    chip_table  = None
                    name_table  = None
                    image_table = None
        version_info = {
            'db_version':          db_version,
            'chip_csv_format':     chip_csv_format,
            'header_csvformat_re': header_csvformat_re,
            'tables_fnames':       (chip_table, name_table, image_table)
        }
        print('[ld3] has %s database format' % db_version)
        return version_info


def load_csv_tables(db_dir, allow_new_dir=True):
    # Detect the version info
    version_info = detect_version(db_dir)
    db_version          = version_info['db_version']
    chip_csv_format     = version_info['chip_csv_format']
    header_csvformat_re = version_info['header_csvformat_re']
    (chip_table, name_table, image_table) = version_info['tables_fnames']

    unimplemented_formats = ['partial', 'xlsx']
    if db_version in unimplemented_formats:
        raise NotImplementedError('cannot parse db_version=%r' % db_version)
    if db_version == 'newdb':
        if not allow_new_dir:
            raise AssertionError('assert not newdatabase')


def tryindex(list, valid_items):
    # Return the index of the first valid item
    for item in valid_items:
        try:
            return list.index(item)
        except ValueError:
            pass
    return None


def parse_csv_tables():
    # The goal of this function is to load in tables with specified properties
    # as well as arbitrary properties
    nx2_name = [UNKNOWN_NAME, UNKNOWN_NAME]
    nid2_nx  = {0: 0, 1: 1}
    def add_name(name, nid=None):
        nx = len(nx2_name)
        if nid is not None:
            if nid < 2:
                assert name in VALID_UNKNOWN_NAMES
                name = UNKNOWN_NAME
            nid2_nx[nid] = nx
        nx2_name.append(name)
        return nx

    name_csv_format = ['NameID', 'Name']
    nid_x  = tryindex(name_csv_format, ['NameID'])
    name_x = tryindex(name_csv_format, ['Name'])
    csv_columns = [(nid_x, int), (name_x, str),]
    parse_csv_table_data(nx2_name, csv_columns, add_name)


def parse_csv_table_data(csv_fname, csv_columns, add_fn):
    # Open CSV file
    with open(csv_fname, 'r') as csv_file:
        # For each line
        for line_num, csv_line in enumerate(csv_file):
            # Remove leading and trailing whitespace
            csv_line = csv_line.strip('\n\r\t ')
            # Skip blank and commented lines
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            # Parse CSV fields
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            fields = [type(csv_fields[fieldx]) for fieldx, type in csv_columns]
            add_fn(*fields)
