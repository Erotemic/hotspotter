'''
Module: load_data
    Loads the paths and table information from which all other data is computed.
    This is the first script run in the loading pipeline.
'''
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[ld2]', DEBUG=False)
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

# GLOBALS

VERBOSE_LOAD_DATA = True


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

# DRIVER CODE


#@profile  # This is perfectly fast .2 seconds on GZ
@util.indent_decor('[ld2.load_csv]')
def load_csv_tables(db_dir, allow_new_dir=True):
    '''
    Big function which loads the csv tables from a datatabase directory
    Returns HotspotterDirs and HotspotterTables
    '''
    if 'vdd' in sys.argv:
        helpers.vd(db_dir)
    print('=============================')
    print('[ld2] Loading hotspotter csv tables: %r' % db_dir)
    hs_dirs = ds.HotspotterDirs(db_dir)
    hs_tables = ds.HotspotterTables()
    #exec(hs_dirs.execstr('hs_dirs'))
    #print(hs_dirs.execstr('hs_dirs'))
    img_dir      = hs_dirs.img_dir
    internal_dir = hs_dirs.internal_dir
    db_dir       = hs_dirs.db_dir
    # --- Table Names ---
    chip_table   = join(internal_dir, CHIP_TABLE_FNAME)
    name_table   = join(internal_dir, NAME_TABLE_FNAME)
    image_table  = join(internal_dir, IMAGE_TABLE_FNAME)  # TODO: Make optional
    # --- CHECKS ---
    has_dbdir   = helpers.checkpath(db_dir)
    has_imgdir  = helpers.checkpath(img_dir)
    has_chiptbl = helpers.checkpath(chip_table)
    has_nametbl = helpers.checkpath(name_table)
    has_imgtbl  = helpers.checkpath(image_table)

    # ChipTable Header Markers
    header_numdata = '# NumData '
    header_csvformat_re = '# *ChipID,'
    v12_csvformat_re = r'#[ 0-9]*\) '
    # Default ChipTable Header Variables
    chip_csv_format = ['ChipID', 'ImgID',  'NameID',   'roi[tl_x  tl_y  w  h]',  'theta']
    v12_csv_format = ['instance_id', 'image_id', 'name_id', 'roi']

    # TODO DETECT OLD FORMATS HERE
    db_version = 'current'
    isCurrentVersion = all([has_dbdir, has_imgdir, has_chiptbl, has_nametbl, has_imgtbl])
    print('[ld2] isCurrentVersion=%r' % isCurrentVersion)
    IS_VERSION_1_OR_2 = False

    if not isCurrentVersion:
        helpers.checkpath(db_dir, verbose=True)
        helpers.checkpath(img_dir, verbose=True)
        helpers.checkpath(chip_table, verbose=True)
        helpers.checkpath(name_table, verbose=True)
        helpers.checkpath(image_table, verbose=True)
        import db_info

        def assign_alternate(tblname, optional=False):
            path = join(db_dir, tblname)
            if helpers.checkpath(path, verbose=True):
                return path
            path = join(db_dir, '.hs_internals', tblname)
            if helpers.checkpath(path, verbose=True):
                return path
            if optional:
                return None
            else:
                raise Exception('bad state=%r' % tblname)
        #
        if db_info.has_v2_gt(db_dir):
            IS_VERSION_1_OR_2 = True
            db_version = 'hotspotter-v2'
            print('[ld2] has %s database format' % db_version)
            chip_csv_format = []
            header_csvformat_re = v12_csvformat_re
            chip_table  = assign_alternate('instance_table.csv')
            name_table  = assign_alternate('name_table.csv')
            image_table = assign_alternate('image_table.csv')
        #
        elif db_info.has_v1_gt(db_dir):
            IS_VERSION_1_OR_2 = True
            db_version = 'hotspotter-v1'
            print('[ld2] has %s database format' % db_version)
            chip_csv_format = []
            header_csvformat_re = v12_csvformat_re
            chip_table  = assign_alternate('animal_info_table.csv')
            name_table  = assign_alternate('name_table.csv', optional=True)
            image_table = assign_alternate('image_table.csv', optional=True)
        #
        elif db_info.has_ss_gt(db_dir):
            db_version = 'stripespotter'
            print('[ld2] has %s database format' % db_version)
            chip_table = join(db_dir, 'SightingData.csv')

            chip_csv_format = ['imgindex', 'original_filepath', 'roi', 'animal_name']
            header_csvformat_re = '#imgindex,'
            #raise NotImplementedError('stripe spotter conversion')
            if not helpers.checkpath(chip_table, verbose=True):
                raise Exception('bad state chip_table=%r' % chip_table)
        else:
            try:
                db_version = 'current'  # Well almost
                chip_table  = assign_alternate(CHIP_TABLE_FNAME)
                name_table  = assign_alternate(NAME_TABLE_FNAME)
                image_table = assign_alternate(IMAGE_TABLE_FNAME)
            except Exception:
                if db_info.has_partial_gt(db_dir):
                    print('[ld2] detected incomplete database')
                    raise NotImplementedError('partial database recovery')
                elif allow_new_dir:
                    print('[ld2] detected new dir')
                    hs_dirs.ensure_dirs()
                    return hs_dirs, hs_tables, 'newdb'
                else:
                    import traceback
                    print(traceback.format_exc())
                    print('[ld2] I AM IN A BAD STATE!')
                    errmsg  = ''
                    errmsg += ('\n\n!!!!!\n\n')
                    errmsg += ('  ! The data tables seem to not be loaded')
                    errmsg += (' Files in internal dir: %r' % internal_dir)
                    for fname in os.listdir(internal_dir):
                        errmsg += ('   ! fname')
                    errmsg += ('\n\n!!!!!\n\n')
                    print(errmsg)
                    raise Exception(errmsg)
    if not helpers.checkpath(chip_table):
        raise Exception('bad state chip_table=%r' % chip_table)
    print('[ld2] detected %r' % db_version)
    hs_dirs.ensure_dirs()
    print('-------------------------')
    print('[ld2] Loading database tables: ')
    cid_lines  = []
    line_num   = 0
    csv_line   = ''
    csv_fields = []

    # RCOS TODO: We need a more general csv read function
    # which can handle all of these little corner cases delt with here.
    # ------------------
    # --- READ NAMES ---
    # ------------------
    nx2_name = [UNKNOWN_NAME, UNKNOWN_NAME]
    nid2_nx  = {0: 0, 1: 1}

    def add_name(name, nid=None):
        nx = len(nx2_name)
        nx2_name.append(name)
        if nid is not None:
            nid2_nx[nid] = nx
        return nx
    try:
        if name_table is None:
            raise IOError('name_table will be given in chip table.')
        print('[ld2] Loading name table: %r' % name_table)
        name_lines = open(name_table, 'r')
        for line_num, csv_line in enumerate(name_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            nid = int(csv_fields[0])
            name = csv_fields[1]
            add_name(name, nid)
        name_lines.close()
        if VERBOSE_LOAD_DATA:
            print('[ld2] * Loaded %r names (excluding unknown names)' % (len(nx2_name) - 2))
            print('[ld2] * Done loading name table')
    except IOError as ex:
        print('IOError: %r' % ex)
        print('[ld2.name] loading without name table')
        #raise
    except Exception as ex:
        print('[ld2.name] ERROR %r' % ex)
        #print('[ld2.name] ERROR name_tbl parsing: %s' % (''.join(cid_lines)))
        print('[ld2.name] ERROR on line number:  %r' % (line_num))
        print('[ld2.name] ERROR on line:         %r' % (csv_line))
        print('[ld2.name] ERROR on fields:       %r' % (csv_fields))

    # -------------------
    # --- READ IMAGES ---
    # -------------------
    gx2_gname = []
    gx2_aif   = []
    gid2_gx = {}  # this is not used. It can probably be removed

    def add_image(gname, aif, gid):
        gx = len(gx2_gname)
        gx2_gname.append(gname)
        gx2_aif.append(aif)
        if gid is not None:
            # this is not used. It can probably be removed
            gid2_gx[gid] = gx
        return gx
    try:
        print('[ld2] Loading images')
        # Load Image Table
        # <LEGACY CODE>
        if image_table is None:
            raise IOError('image_table will be given in chip table')
        if VERBOSE_LOAD_DATA:
            print('[ld2] * Loading image table: %r' % image_table)
        gid_lines = open(image_table, 'r').readlines()
        for line_num, csv_line in enumerate(gid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]
            gid = int(csv_fields[0])
            # You have 3 csv files. Format == gid, gname.ext, aif
            if len(csv_fields) == 3:
                gname = csv_fields[1]
                aif   = csv_fields[2].lower() in ['true', '1']  # convert to bool correctly
            # You have 4 csv fields. Format == gid, gname, ext, aif
            elif len(csv_fields) == 4:
                gname = '.'.join(csv_fields[1:3])
                aif   =  csv_fields[3].lower() in ['true', '1']
            else:
                gname = 'ERROR'
                aif = False
            add_image(gname, aif, gid)
        nTableImgs = len(gx2_gname)
        fromTableNames = set(gx2_gname)
        if VERBOSE_LOAD_DATA:
            print('[ld2] * table specified %r images' % nTableImgs)
            # </LEGACY CODE>
            # Load Image Directory
            print('[ld2] * Loading image directory: %r' % img_dir)
        nDirImgs = 0
        nDirImgsAlready = 0
        for fname in os.listdir(img_dir):
            if len(fname) > 4 and fname[-4:].lower() in ['.jpg', '.png', '.tiff']:
                if fname in fromTableNames:
                    nDirImgsAlready += 1
                    continue
                add_image(fname, False, None)
                nDirImgs += 1
        if VERBOSE_LOAD_DATA:
            print('[ld2] * dir specified %r images' % nDirImgs)
            print('[ld2] * %r were already specified in the table' % nDirImgsAlready)
            print('[ld2] * Loaded %r images' % len(gx2_gname))
            print('[ld2] * Done loading images')
    except IOError as ex:
        print('IOError: %r' % ex)
        print('[ld2.img] loading without image table')
        #if '--strict' in sys.argv:
            #raise
    except Exception as ex:
        print('[ld2!.img] ERROR %r' % ex)
        #print('[ld2.img] ERROR image_tbl parsing: %s' % (''.join(cid_lines)))
        print('[ld2!.img] ERROR on line number:  %r' % (line_num))
        print('[ld2!.img] ERROR on line:         %r' % (csv_line))
        print('[ld2!.img] ERROR on fields:       %r' % (csv_fields))
        raise

    try:
        # ------------------
        # --- READ CHIPS ---
        # ------------------
        print('[ld2] Loading chip table: %r' % chip_table)
        # Load Chip Table Header
        cid_lines = open(chip_table, 'r').readlines()
        num_data   = -1
        # Parse Chip Table Header
        for line_num, csv_line in enumerate(cid_lines):
            #print('[LINE %4d] %r' % (line_num, csv_line))
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0:
                #print('[LINE %4d] BROKEN' % (line_num))
                continue
            csv_line = csv_line.strip('\n')
            if csv_line.find('#') != 0:
                #print('[LINE %4d] BROKEN' % (line_num))
                break  # Break after header
            if re.search(header_csvformat_re, csv_line) is not None:
                #print('[LINE %4d] SEARCH' % (line_num))
                # Specified Header Variables
                if IS_VERSION_1_OR_2:
                    #print(csv_line)
                    end_ = csv_line.find('-')
                    if end_ != -1:
                        end_ = end_ - 1
                        #print('end_=%r' % end_)
                        fieldname = csv_line[5:end_]
                    else:
                        fieldname = csv_line[5:]
                    #print(fieldname)
                    fieldname = fieldname.strip(' ')
                    chip_csv_format += [fieldname]

                else:
                    chip_csv_format = [_.strip() for _ in csv_line.strip('#').split(',')]
                #print('[ld2] read chip_csv_format: %r' % chip_csv_format)
            if csv_line.find(header_numdata) == 0:
                #print('[LINE %4d] NUM_DATA' % (line_num))
                num_data = int(csv_line.replace(header_numdata, ''))
        if IS_VERSION_1_OR_2 and len(chip_csv_format) == 0:
            chip_csv_format = v12_csv_format
        if VERBOSE_LOAD_DATA:
            print('[ld2] * num_chips: %r' % num_data)
            print('[ld2] * chip_csv_format: %r ' % chip_csv_format)
        #print('[ld2.chip] Header Columns: %s\n    ' % '\n   '.join(chip_csv_format))

        def tryindex(list, valid_items):
            # Return the index of the first valid item
            for item in valid_items:
                try:
                    return list.index(item)
                except ValueError:
                    pass
            return None

        #class AllocedList(object):
            #def __init__(self, list_=[], nAlloc=0):
                #self._len = 0

            #def __add__(self, other):
                #pass

        # Hotspotter Chip Tables
        cx2_cid   = []
        cx2_nx    = []
        cx2_gx    = []
        cx2_roi   = []
        cx2_theta = []

        # Define standard properties
        # The following variables specify the column index of the property
        # if it exists in the CSV table.
        cid_x   = tryindex(chip_csv_format, ['ChipID', 'imgindex', 'instance_id'])
        gid_x   = tryindex(chip_csv_format, ['ImgID', 'image_id'])
        nid_x   = tryindex(chip_csv_format, ['NameID', 'name_id'])
        roi_x   = tryindex(chip_csv_format, ['roi[tl_x  tl_y  w  h]', 'roi'])
        theta_x = tryindex(chip_csv_format, ['theta'])
        # new fields
        gname_x = tryindex(chip_csv_format, ['Image', 'image_name', 'original_filepath'])
        name_x  = tryindex(chip_csv_format, ['Name', 'animal_name'])
        # A list of the standard
        standard_xs_ = [cid_x, gid_x, gname_x, nid_x, name_x, roi_x, theta_x]
        standard_xs = [x for x in standard_xs_ if x is not None]

        # Make sure all required fields are there
        if cid_x is None:
            raise Exception('Improper format. Cannot find cid_x')
        if gid_x is None and gname_x is None:
            raise Exception('Improper format. Cannot find gid_x or gname_x')
        if nid_x is None and name_x is None:
            raise Exception('Improper format. Cannot find nid_x or name_x')

        # All nonstandard properties are unknown and belong in the prop_dict
        # get csv indexes which are unknown properties
        prop_x_list  = np.setdiff1d(range(len(chip_csv_format)), standard_xs).tolist()
        px2_prop_key = [chip_csv_format[x] for x in prop_x_list]
        prop_dict = {}
        for prop in iter(px2_prop_key):
            prop_dict[prop] = []

        # Print header parsing status
        if VERBOSE_LOAD_DATA:
            print('[ld2] * num_user_properties: %d' % (len(prop_dict.keys())))
            print('[ld2] * num_standard_properties: %d / %d' %
                  (len(standard_xs), len(standard_xs_)))

        #def parse_integer(csv_fields, field_x):
            #try:
                #int_ = int(csv_fields[field_x])
                #return int_
            #except ValueError:
                #print('[ld2!] field_x = %r' % field_x)
                #print('[ld2!] csv_fields = %r' % csv_fields)
                #print('[ld2!] csv_fields[field_x] = %r' % csv_fields[field_x])
                #print(chip_csv_format)
                #raise

        def parse_intlist_field(csv_field):
            list_str = csv_field.strip('[').strip(']')
            list_ = [int(round(float(_))) for _ in list_str.split()]
            return list_

        # Read each noncommnted line in the chip table
        for line_num, csv_line in enumerate(cid_lines):
            csv_line = csv_line.strip('\n\r\t ')
            if len(csv_line) == 0 or csv_line.find('#') == 0:
                continue
            # Split each line into a list of fields
            csv_fields = [_.strip(' ') for _ in csv_line.strip('\n\r ').split(',')]

            # Parse nonstandard properties first
            for px, x in enumerate(prop_x_list):
                prop = px2_prop_key[px]
                prop_val = csv_fields[x]
                prop_dict[prop].append(prop_val)

            # Parse Chip ID
            cid = int(csv_fields[cid_x])

            # Parse Chip ROI
            if roi_x is not None:
                roi = parse_intlist_field(csv_fields[roi_x])
            else:
                assert gname_x is not None, 'roi_x AND gname_x cannot be None'

            # Parse Chip theta
            if theta_x is not None:
                theta = float(csv_fields[theta_x])
            else:
                theta = 0

            # Parse Image ID/X
            if gid_x is not None:
                gid = int(csv_fields[gid_x])
                gx  = gid2_gx[gid]
            elif gname_x is not None:
                gname = csv_fields[gname_x]
                # LEGACY HACK
                if db_version != 'current':
                    if db_version == 'stripespotter':
                        if not exists(gname):
                            gname = 'img-%07d.jpg' % cid
                    else:
                        # This is so hacky. The gpath in hospotter-v2 doesnt have
                        # extensions
                        name, ext = splitext(gname)
                        if ext not in helpers.IMG_EXTENSIONS:
                            if 'img_extension' in prop_dict:
                                gname = gname + '.' + prop_dict['img_extension'][-1]
                # /LEGACY HACK
                try:
                    gx = gx2_gname.index(gname)
                except ValueError:
                    gx = add_image(gname, False, None)

            # LEGACY HACK
            if len(roi) == 0:
                # Entire image is the roi
                print('[ld2] Converting %s database' % db_version)
                gpath = join(db_dir, RDIR_IMG, gname)
                w, h = Image.open(gpath).size
                roi = [1, 1, w, h]
            # /LEGACY HACK

            # Parse Name ID/X
            if nid_x is not None:
                nid = int(csv_fields[nid_x])
                nx = nid2_nx[nid]
            elif name_x is not None:
                name = csv_fields[name_x]
                try:
                    nx = nx2_name.index(name)
                except ValueError:
                    nx = add_name(name)
            # Append info to cid lists
            cx2_cid.append(cid)
            cx2_gx.append(gx)
            cx2_nx.append(nx)
            cx2_roi.append(roi)
            cx2_theta.append(theta)
    except Exception as ex:
        print('[chip.ld2] ERROR %r' % ex)
        #print('[chip.ld2] ERROR parsing: %s' % (''.join(cid_lines)))
        print('[chip.ld2] ERROR reading header:  %r' % (line_num))
        print('[chip.ld2] ERROR on line number:  %r' % (line_num))
        print('[chip.ld2] ERROR on line:         %r' % (csv_line))
        print('[chip.ld2] ERROR on fields:       %r' % (csv_fields))
        raise

    if VERBOSE_LOAD_DATA:
        print('[ld2] * Loaded: %r chips' % (len(cx2_cid)))
        print('[ld2] * Done loading chip table')

    # Return all information from load_tables
    #hs_tables.gid2_gx = gid2_gx
    #hs_tables.nid2_nx  = nid2_nx
    hs_tables.init(gx2_gname, gx2_aif,
                   nx2_name,
                   cx2_cid, cx2_nx, cx2_gx,
                   cx2_roi, cx2_theta, prop_dict)

    print('[ld2] Done Loading hotspotter csv tables: %r' % (db_dir))
    if 'vcd' in sys.argv:
        helpers.vd(hs_dirs.computed_dir)
    return hs_dirs, hs_tables, db_version


# Make Table Functions
# Returns the formated csv table text


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
    csv_rows.append('# NumData %r' % num_data)

    column_maxlen = []
    column_str_list = []

    if column_labels is None:
        column_labels = [''] * len(column_list)

    def _toint(c):
        try:
            if np.isnan(c):
                return 'nan'
        except TypeError as ex:
            print('------')
            print('[ld2] TypeError %r ' % ex)
            print('[ld2] _toint(c) failed')
            print('[ld2] c = %r ' % c)
            print('[ld2] type(c) = %r ' % type(c))
            print('------')
            raise
        return ('%d') % int(c)

    for col, lbl, coltype in iter(zip(column_list, column_labels, column_type)):
        if tools.is_list(coltype):
            col_str  = [str(c).replace(',', ' ').replace('.', '') for c in iter(col)]
        elif tools.is_float(coltype):
            col_str = [('%.2f') % float(c) for c in iter(col)]
        elif tools.is_int(coltype):
            col_str = [_toint(c) for c in iter(col)]
        elif tools.is_str(coltype):
            col_str = [str(c) for c in iter(col)]
        else:
            col_str  = [str(c) for c in iter(col)]
        col_lens = [len(s) for s in iter(col_str)]
        max_len  = max(col_lens)
        max_len  = max(len(lbl), max_len)
        column_maxlen.append(max_len)
        column_str_list.append(col_str)

    _fmtfn = lambda maxlen: ''.join(['%', str(maxlen + 2), 's'])
    fmtstr = ','.join([_fmtfn(maxlen) for maxlen in column_maxlen])
    csv_rows.append('# ' + fmtstr % tuple(column_labels))
    for row in zip(*column_str_list):
        csv_rows.append('  ' + fmtstr % row)

    csv_text = '\n'.join(csv_rows)
    return csv_text


def make_flat_table(hs, cx_list):
    # Valid chip tables
    if len(cx_list) == 0:
        return ''
    cx2_cid   = hs.tables.cx2_cid[cx_list]
    # Use the indexes as ids (FIXME: Just go back to g/n-ids)
    cx2_gname = hs.cx2_gname(cx_list)
    cx2_name  = hs.cx2_name(cx_list)
    try:
        cx2_roi   = hs.tables.cx2_roi[cx_list]
    except IndexError as ex:
        print(ex)
        cx2_roi = np.array([])
    cx2_theta = hs.tables.cx2_theta[cx_list]
    prop_dict = {propkey: [cx2_propval[cx] for cx in iter(cx_list)]
                 for (propkey, cx2_propval) in hs.tables.prop_dict.iteritems()}
    # Turn the chip indexes into a DOCUMENTED csv table
    header = '# flat table'
    column_labels = ['ChipID', 'Image', 'Name', 'roi[tl_x  tl_y  w  h]', 'theta']
    column_list   = [cx2_cid, cx2_gname, cx2_name, cx2_roi, cx2_theta]
    column_type   = [int, int, int, list, float]
    if not prop_dict is None:
        for key, val in prop_dict.iteritems():
            column_labels.append(key)
            column_list.append(val)
            column_type.append(str)

    chip_table = make_csv_table(column_labels, column_list, header, column_type)
    return chip_table


def make_chip_csv(hs, cx_list):
    # Valid chip tables
    if len(cx_list) == 0:
        return ''
    cx2_cid   = hs.tables.cx2_cid[cx_list]
    # Use the indexes as ids (FIXME: Just go back to g/n-ids)
    cx2_gx   = hs.tables.cx2_gx[cx_list] + 1  # FIXME
    cx2_nx   = hs.tables.cx2_nx[cx_list]   # FIXME
    try:
        cx2_roi   = hs.tables.cx2_roi[cx_list]
    except IndexError as ex:
        print(ex)
        cx2_roi = np.array([])
    cx2_theta = hs.tables.cx2_theta[cx_list]
    prop_dict = {propkey: [cx2_propval[cx] for cx in iter(cx_list)]
                 for (propkey, cx2_propval) in hs.tables.prop_dict.iteritems()}
    # Turn the chip indexes into a DOCUMENTED csv table
    header = '# chip table'
    column_labels = ['ChipID', 'ImgID', 'NameID', 'roi[tl_x  tl_y  w  h]', 'theta']
    column_list   = [cx2_cid, cx2_gx, cx2_nx, cx2_roi, cx2_theta]
    column_type   = [int, int, int, list, float]
    if not prop_dict is None:
        for key, val in prop_dict.iteritems():
            column_labels.append(key)
            column_list.append(val)
            column_type.append(str)

    chip_table = make_csv_table(column_labels, column_list, header, column_type)
    return chip_table


def make_image_csv(hs, gx_list):
    'return an image table csv string'
    if len(gx_list) == 0:
        return ''
    gx2_gid   = np.array(gx_list) + 1  # FIXME
    gx2_gname = hs.tables.gx2_gname[gx_list]
    try:
        gx2_aif   = hs.tables.gx2_aif[gx_list]
    except Exception as ex:
        print(ex)
        gx2_aif = np.zeros(len(gx2_gid), dtype=np.uint32)
    # Make image_table.csv
    header = '# image table'
    column_labels = ['gid', 'gname', 'aif']  # do aif for backwards compatibility
    column_list   = [gx2_gid, gx2_gname, gx2_aif]
    image_table = make_csv_table(column_labels, column_list, header)
    return image_table


def make_name_csv(hs, nx_list):
    'returns an name table csv string'
    if len(nx_list) == 0:
        return ''
    nx_list_  = np.setdiff1d(nx_list, [0, 1])   # dont write ____ for backcomp
    nx2_name  = hs.tables.nx2_name[nx_list_]
    # Make name_table.csv
    header = '# name table'
    column_labels = ['nid', 'name']
    column_list   = [nx_list_, nx2_name]
    name_table = make_csv_table(column_labels, column_list, header)
    return name_table


# Write Table Functions
# Makes and writes csv files to disk

def write_csv_tables(hs):
    'Saves the tables to disk'
    print('[ld2] Writing csv tables')
    # Output directories
    internal_dir = hs.dirs.internal_dir
    # Create backup # RCOS TODO: Should be a preference
    CREATE_BACKUP = True
    if CREATE_BACKUP:
        backup_csv_tables(hs, force_backup=True)
    # Get valid indexes
    valid_cx = hs.get_valid_cxs()
    valid_gx = hs.get_valid_gxs()
    valid_nx = hs.get_valid_nxs()
    # Make table from indexes
    chip_table  = make_chip_csv(hs, valid_cx)
    image_table = make_image_csv(hs, valid_gx)
    name_table  = make_name_csv(hs, valid_nx)
    # Make csv filenames
    chip_table_fpath  = join(internal_dir, CHIP_TABLE_FNAME)
    name_table_fpath  = join(internal_dir, NAME_TABLE_FNAME)
    image_table_fpath = join(internal_dir, IMAGE_TABLE_FNAME)
    # write csv files
    print('[ld2] Writing chip table')
    helpers.write_to(chip_table_fpath, chip_table)
    print('[ld2] Writing name table')
    helpers.write_to(name_table_fpath, name_table)
    print('[ld2] Writing image table')
    helpers.write_to(image_table_fpath, image_table)


def write_flat_table(hs):
    dbdir = hs.dirs.db_dir
    # Make flat table
    valid_cx = hs.get_valid_cxs()
    flat_table  = make_flat_table(hs, valid_cx)
    flat_table_fpath  = join(dbdir, 'flat_table.csv')
    # Write flat table
    print('[ld2] Writing flat table')
    helpers.write_to(flat_table_fpath, flat_table)


def backup_csv_tables(hs, force_backup=False):
    internal_dir = hs.dirs.internal_dir
    backup_dir = join(internal_dir, 'backup_v0.1.0')
    if not exists(backup_dir) or force_backup:
        helpers.ensuredir(backup_dir)
        timestamp = helpers.get_timestamp(use_second=True)

        def do_backup(fname):
            src = join(internal_dir, fname)
            dst_fname = ('%s_bak-' + timestamp + '%s') % splitext(fname)
            dst = join(backup_dir, dst_fname)
            if exists(src):
                shutil.copy(src, dst)
        do_backup(CHIP_TABLE_FNAME)
        do_backup(NAME_TABLE_FNAME)
        do_backup(IMAGE_TABLE_FNAME)
