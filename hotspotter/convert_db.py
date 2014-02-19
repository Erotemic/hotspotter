from __future__ import division, print_function
from hscom import __common__
(print, print_,
 print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[convert]')
import sys
from os.path import join, relpath, normpath, exists
import collections
import os
import parse
# Science
from PIL import Image
import numpy as np
# Hotspotter
from hscom import helpers
from hscom import helpers as util
import load_data2 as ld2
import db_info


# BUGS: TODO:
# Orientation within chip

def try_autoconvert(db_dir):
    if db_info.has_v2_gt(db_dir):
        raise NotImplementedError('hotspotter v2 conversion')
    if db_info.has_v1_gt(db_dir):
        raise NotImplementedError('hotspotter v1 conversion')
    if db_info.has_ss_gt(db_dir):
        raise NotImplementedError('stripe spotter conversion')
    if db_info.has_partial_gt(db_dir):
        raise NotImplementedError('partial database recovery')
    return False


def is_current(db_dir):
    return db_info.has_internal_tables(db_dir)


def try_user_guided(db_dir):
    if db_info.is_imgdir(db_dir):
        img_dpath = join(db_dir, ld2.RDIR_IMG2)
        gt_format = None
        return init_database_from_images(db_dir, img_dpath, gt_format=gt_format,
                                         allow_unknown_chips=False)
    pass


def try_new_database(db_dir):
    return exists(db_dir)  # and len(os.listdir(db_dir)) == 0


def convert_if_needed(db_dir):
    if is_current(db_dir):
        return db_dir
    elif try_autoconvert(db_dir):
        return db_dir
    elif try_user_guided(db_dir):
        pass
    elif try_new_database(db_dir):
        pass
    else:
        raise Exception('unknown and non-empty database directory')


# Port of Philbin07 code to python
def compute_ap(groundtruth_query, ranked_list):
    good_set = set(open(groundtruth_query + '_good.txt').readlines())
    ok_set   = set(open(groundtruth_query + '_ok.txt').readlines())
    junk_set = set(open(groundtruth_query + '_junk.txt').readlines())
    pos_set  = set.union(good_set, ok_set)
    ap = compute_ap(pos_set, junk_set, ranked_list)
    return ap


def compute_ap2(pos, amb, ranked_list):
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0
    j = 0
    for i in xrange(ranked_list):
        if amb.count(ranked_list[i]):
            continue
        if pos.count(ranked_list[i]):
            intersect_size += 1

        recall    = intersect_size / float(len(pos))
        precision = intersect_size / (j + 1.0)

        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision
        j += 1
    return ap

# what I think needs to be done:
# all images become chips
# just read the query images
# use philbins thing to compute ap

# It looks like images can have multiple labels.
# in oxford its either 5 of the same or 10 of two.


def __oxgtfile2_oxsty_gttup(gt_fname):
    # num is an id not a number of chips
    '''parases the groundtruth filename for: groundtruth name, quality_lbl, and num'''
    gt_format = '{}_{:d}_{:D}.txt'
    name, num, quality = parse.parse(gt_format, gt_fname)
    return (name, num, quality)


def __read_oxsty_gtfile(gt_fpath, name, quality, img_dpath, corrupted_gname_set):
    oxsty_chip_info_list = []
    # read the individual ground truth file
    with open(gt_fpath, 'r') as file:
        line_list = file.read().splitlines()
        for line in line_list:
            if line == '':
                continue
            fields = line.split(' ')
            gname = fields[0].replace('oxc1_', '') + '.jpg'
            # >:( Because PARIS just cant keep paths consistent
            if gname.find('paris_') >= 0:
                paris_hack = gname[6:gname.rfind('_')]
                gname = join(paris_hack, gname)
            if gname in corrupted_gname_set:
                continue
            if len(fields) > 1:  # if has roi
                roi =  map(int, map(round, map(float, fields[1:])))
            else:
                gpath = join(img_dpath, gname)
                (w, h) = Image.open(gpath).size
                roi = [0, 0, w, h]
            oxsty_chip_info = (gname, roi)
            oxsty_chip_info_list.append(oxsty_chip_info)
    return oxsty_chip_info_list


def convert_from_oxford_style(db_dir):
    # Get directories for the oxford groundtruth
    oxford_gt_dpath      = join(db_dir, 'oxford_style_gt')
    helpers.assertpath(oxford_gt_dpath)
    # Check for corrupted files (Looking at your Paris Buildings Dataset)
    corrupted_file_fpath = join(oxford_gt_dpath, 'corrupted_files.txt')
    corrupted_gname_set = set([])
    if helpers.checkpath(corrupted_file_fpath):
        with open(corrupted_file_fpath) as f:
            corrupted_gname_list = f.read().splitlines()
        corrupted_gname_set = set(corrupted_gname_list)

    # Recursively get relative path of all files in img_dpath
    print('Loading Oxford Style Images from: ' + db_dir)
    img_dpath  = join(db_dir, 'images')
    helpers.assertpath(img_dpath)
    gname_list_ = [join(relpath(root, img_dpath), fname).replace('\\', '/').replace('./', '')
                   for (root, dlist, flist) in os.walk(img_dpath)
                   for fname in iter(flist)]
    gname_list = [gname for gname in iter(gname_list_)
                  if not gname in corrupted_gname_set and helpers.matches_image(gname)]
    print(' * num_images = %d ' % len(gname_list))

    # Read the Oxford Style Groundtruth files
    print('Loading Oxford Style Names and Chips')
    gt_fname_list = os.listdir(oxford_gt_dpath)
    num_gt_files = len(gt_fname_list)
    query_chips  = []
    gname2_chips_raw = collections.defaultdict(list)
    name_set = set([])
    print(' * num_gt_files = %d ' % num_gt_files)
    sys.stdout.write('parsed: 0000/%4d' % (num_gt_files))
    for gtx, gt_fname in enumerate(gt_fname_list):
        sys.stdout.write(('\b' * 9) + '%4d/%4d' % (gtx + 1, num_gt_files))
        if gtx % 10 - 1 == 0:
            sys.stdout.flush()
        if gt_fname == 'corrupted_files.txt':
            continue
        #Get name, quality, and num from fname
        (name, num, quality) = __oxgtfile2_oxsty_gttup(gt_fname)
        gt_fpath = join(oxford_gt_dpath, gt_fname)
        name_set.add(name)
        oxsty_chip_info_sublist = __read_oxsty_gtfile(gt_fpath, name,
                                                      quality, img_dpath,
                                                      corrupted_gname_set)
        if quality == 'query':
            for (gname, roi) in iter(oxsty_chip_info_sublist):
                query_chips.append((gname, roi, name, num))
        else:
            for (gname, roi) in iter(oxsty_chip_info_sublist):
                gname2_chips_raw[gname].append((name, roi, quality))
    sys.stdout.write('\n')
    print(' * num_query images = %d ' % len(query_chips))
    # Remove duplicates img.jpg : (*1.txt, *2.txt, ...) -> (*.txt)
    gname2_chips     = collections.defaultdict(list)
    multinamed_gname_list = []
    for gname, val in gname2_chips_raw.iteritems():
        val_repr = map(repr, val)
        unique_reprs = set(val_repr)
        unique_indexes = [val_repr.index(urep) for urep in unique_reprs]
        for ux in unique_indexes:
            gname2_chips[gname].append(val[ux])
        if len(gname2_chips[gname]) > 1:
            multinamed_gname_list.append(gname)
    # print some statistics
    query_gname_list = [tup[0] for tup in query_chips]
    gname_with_groundtruth_list = gname2_chips.keys()
    gname_without_groundtruth_list = np.setdiff1d(gname_list, gname_with_groundtruth_list)
    print(' * num_images = %d ' % len(gname_list))
    print(' * images with groundtruth    = %d ' % len(gname_with_groundtruth_list))
    print(' * images without groundtruth = %d ' % len(gname_without_groundtruth_list))
    print(' * images with multi-groundtruth = %d ' % len(multinamed_gname_list))
    #make sure all queries have ground truth and there are no duplicate queries
    assert len(query_gname_list) == len(np.intersect1d(query_gname_list, gname_with_groundtruth_list))
    assert len(query_gname_list) == len(set(query_gname_list))
    # build hotspotter tables
    print('adding to table: ')
    gx2_gname = gname_list
    nx2_name  = ['____', '____'] + list(name_set)
    nx2_nid   = [1, 1] + range(2, len(name_set) + 2)
    gx2_gid   = range(1, len(gx2_gname) + 1)

    cx2_cid     = []
    cx2_theta   = []
    cx2_quality = []
    cx2_roi     = []
    cx2_nx      = []
    cx2_gx      = []
    prop_dict   = {'oxnum': []}

    def add_to_hs_tables(gname, name, roi, quality, num=''):
        cid = len(cx2_cid) + 1
        nx = nx2_name.index(name)
        if nx == 0:
            nx = 1
        gx = gx2_gname.index(gname)
        cx2_cid.append(cid)
        cx2_roi.append(roi)
        cx2_quality.append(quality)
        cx2_nx.append(nx)
        cx2_gx.append(gx)
        cx2_theta.append(0)
        prop_dict['oxnum'].append(num)
        sys.stdout.write(('\b' * 10) + 'cid = %4d' % cid)
        return cid

    for gname, roi, name, num in query_chips:
        add_to_hs_tables(gname, name, roi, 'query', num)
    for gname in gname2_chips.keys():
        if len(gname2_chips[gname]) == 1:
            (name, roi, quality) = gname2_chips[gname][0]
            add_to_hs_tables(gname, name, roi, quality)
        else:
            # just take the first name. This is foobar
            names, rois, qualities = zip(*gname2_chips[gname])
            add_to_hs_tables(gname, names[0], rois[0], qualities[0])
    for gname in gname_without_groundtruth_list:
        gpath = join(img_dpath, gname)
        try:
            (w, h) = Image.open(gpath).size
            roi = [0, 0, w, h]
            add_to_hs_tables(gname, '____', roi, 'unknown')
        except Exception as ex:
            print('Exception ex=%r' % ex)
            print('Not adding gname=%r' % gname)
            print('----')
    cx2_nid = np.array(nx2_nid)[cx2_nx]
    cx2_gid = np.array(gx2_gid)[cx2_gx]
    #
    # Write tables
    internal_dir      = join(db_dir, ld2.RDIR_INTERNAL2)
    helpers.ensurepath(internal_dir)
    write_chip_table(internal_dir, cx2_cid, cx2_gid, cx2_nid, cx2_roi, cx2_theta, prop_dict)
    write_name_table(internal_dir, nx2_nid, nx2_name)
    write_image_table(internal_dir, gx2_gid, gx2_gname)


# Converts the name_num.jpg image format into a database
def convert_named_chips(db_dir, img_dpath=None):
    print('\n --- Convert Named Chips ---')
    # --- Initialize ---
    gt_format = '{}_{:d}.jpg'
    print('gt_format (name, num) = %r' % gt_format)
    if img_dpath is None:
        img_dpath = db_dir + '/images'
    print('Converting db_dir=%r and img_dpath=%r' % (db_dir, img_dpath))
    # --- Build Image Table ---
    helpers.print_('Building name table: ')
    gx2_gname = helpers.list_images(img_dpath)
    gx2_gid   = range(1, len(gx2_gname) + 1)
    print('There are %d images' % len(gx2_gname))
    # ---- Build Name Table ---
    helpers.print_('Building name table: ')
    name_set = set([])
    for gx, gname in enumerate(gx2_gname):
        name, num = parse.parse(gt_format, gname)
        name_set.add(name)
    nx2_name  = ['____', '____'] + list(name_set)
    nx2_nid   = [1, 1] + range(2, len(name_set) + 2)
    print('There are %d names' % (len(nx2_name) - 2))
    # ---- Build Chip Table ---
    print('[converdb] Building chip table: ')
    cx2_cid     = []
    cx2_theta   = []
    cx2_roi     = []
    cx2_nx      = []
    cx2_gx      = []
    cid = 1

    def add_to_hs_tables(gname, name, roi, theta=0):
        cid = len(cx2_cid) + 1
        nx = nx2_name.index(name)
        gx = gx2_gname.index(gname)
        cx2_cid.append(cid)
        cx2_roi.append(roi)
        cx2_nx.append(nx)
        cx2_gx.append(gx)
        cx2_theta.append(theta)
        return cid

    for gx, gname in enumerate(gx2_gname):
        name, num = parse.parse(gt_format, gname)
        img_fpath = join(img_dpath, gname)
        (w, h) = Image.open(img_fpath).size
        roi = [1, 1, w, h]
        cid = add_to_hs_tables(gname, name, roi)
    cx2_nid = np.array(nx2_nid)[cx2_nx]
    cx2_gid = np.array(gx2_gid)[cx2_gx]
    print('There are %d chips' % (cid - 1))

    # Write tables
    internal_dir = join(db_dir, ld2.RDIR_INTERNAL2)
    helpers.ensurepath(internal_dir)
    write_chip_table(internal_dir, cx2_cid, cx2_gid, cx2_nid, cx2_roi, cx2_theta)
    write_name_table(internal_dir, nx2_nid, nx2_name)
    write_image_table(internal_dir, gx2_gid, gx2_gname)


def init_database_from_images(db_dir, img_dpath=None, gt_format=None,
                              allow_unknown_chips=False):
    # --- Initialize ---
    if img_dpath is None:
        img_dpath = db_dir + '/images'
    print('Converting db_dir=%r and img_dpath=%r' % (db_dir, img_dpath))
    gx2_gid, gx2_gname = imagetables_from_img_dpath(img_dpath)
    name_set = groundtruth_from_imagenames(gx2_gname, gt_format)
    nx2_name, nx2_nid = nametables_from_nameset(name_set)
    # ---- Build Chip Table ---
    helpers.print_('Building chip table: ')
    cx2_cid     = []
    cx2_theta   = []
    cx2_roi     = []
    cx2_nx      = []
    cx2_gx      = []
    cid = 1

    def add_to_hs_tables(gname, name, roi, theta=0):
        cid = len(cx2_cid) + 1
        nx = nx2_name.index(name)
        gx = gx2_gname.index(gname)
        cx2_cid.append(cid)
        cx2_roi.append(roi)
        cx2_nx.append(nx)
        cx2_gx.append(gx)
        cx2_theta.append(theta)
        return cid
    for gx, gname in enumerate(gx2_gname):
        if gt_format is None:
            name = '____'
        else:
            name, num = parse.parse(gt_format, gname)
        if name == '____' and not allow_unknown_chips:
            continue
        img_fpath = join(img_dpath, gname)
        roi = roi_from_imgsize(img_fpath)
        if not roi is None:
            cid = add_to_hs_tables(gname, name, roi)
    cx2_nid = np.array(nx2_nid)[cx2_nx]
    cx2_gid = np.array(gx2_gid)[cx2_gx]
    print('There are %d chips' % (cid - 1))

    # Write tables
    internal_dir      = join(db_dir, ld2.RDIR_INTERNAL2)
    helpers.ensurepath(internal_dir)
    write_chip_table(internal_dir, cx2_cid, cx2_gid, cx2_nid, cx2_roi, cx2_theta)
    write_name_table(internal_dir, nx2_nid, nx2_name)
    write_image_table(internal_dir, gx2_gid, gx2_gname)
    return True


def read_xlsx_file(xlsx_fpath):
    import openpyxl
    wb = openpyxl.load_workbook(filename=xlsx_fpath)
    active_sheet  = wb.get_active_sheet()
    header_cells  = active_sheet.rows[0]
    column_cells  = active_sheet.columns
    column_labels = [cell.value for cell in header_cells]
    column_list   = [[cell.value for cell in column[1:]] for column in column_cells]
    return column_labels, column_list


#csv_fpath = params.WY_TOADS+'/WY_TOAD_MATCHES.csv'
def read_csv_file(csv_fpath):
    '''reads a csv file into column_labels containing header info and
       column_list containing data'''
    def parse_csv_line(csv_line):
        csv_line_ = csv_line.strip('\n\r\t ')
        csv_fields = [_.strip(' ') for _ in csv_line_.strip('\n\r ').split(', ')]
        return csv_fields
    csv_file = open(csv_fpath, 'r')
    csv_lines = csv_file.readlines()
    csv_file.close()
    csv_iter = iter(csv_lines)
    # Read first line (header)
    csv_line = csv_iter.next()
    column_labels = parse_csv_line(csv_line)
    csv_rows = [parse_csv_line(csv_line_) for csv_line_ in csv_iter]
    column_list = zip(*csv_rows)
    return column_labels, column_list


def wildid_xlsx_to_tables(db_dir):
    'finds any xlsx files in db_dir and transforms them into an image table'
    import glob
    db_dir = normpath(db_dir)
    img_dpath = normpath(join(db_dir, 'images'))
    xlsx_files = glob.glob(join(db_dir, ' * .xlsx'))
    if len(xlsx_files) != 1:
        raise Exception('non-unique xlsx files')
    xlsx_fpath = normpath(xlsx_files[0])
    #'research/testdata/WILDEBEEST_MORRISON_B.xlsx'
    print('[convert] Converting db_dir  = %r' % db_dir)
    print('[convert] with img_dpath     = %r' % img_dpath)
    print('[convert] Reading xlsx_fpath = %r' % (xlsx_fpath,))
    column_labels, column_list = read_xlsx_file(xlsx_fpath)
    return wildid_to_tables(db_dir, img_dpath, column_labels, column_list)


#db_dir = params.TOADS
def wildid_csv_to_tables(db_dir):
    import glob
    db_dir = normpath(db_dir)
    img_dpath = normpath(join(db_dir, 'images'))
    csv_files = glob.glob(join(db_dir, ' * .csv'))
    if len(csv_files) != 1:
        raise Exception('non-unique csv files = %r' % csv_files)
    csv_fpath = normpath(csv_files[0])
    print('[convert] Converting db_dir  = %r' % db_dir)
    print('[convert] with img_dpath     = %r' % img_dpath)
    print('[convert] Reading csv_fpath = %r' % (csv_fpath,))
    column_labels, column_list = read_csv_file(csv_fpath)
    return wildid_to_tables(db_dir, img_dpath, column_labels, column_list)


def wildid_to_tables(db_dir, img_dpath, column_labels, column_list):
    row_lengths = [len(col) for col in column_list]
    num_rows = row_lengths[0]
    assert all([num_rows == rowlen for rowlen in row_lengths]), 'number of rows in xlsx file must be consistent'
    #header = 'Converted from: '+repr(xlsx_fpath)
    #csv_string = ld2.make_csv_table(column_labels, column_list, header)
    # Get Image set
    print('[convert] Building image table')
    gx2_gid, gx2_gname = imagetables_from_img_dpath(img_dpath)
    # Get name set
    print('[convert] Building name table')

    def get_lbl_pos(column_labels, valid_labels):
        for lbl in valid_labels:
            index = helpers.listfind(column_labels, lbl)
            if index is not None:
                return index
        raise Exception('There is no valid label')
    name_colx = get_lbl_pos(column_labels, ['ANIMAL_ID', 'AnimalID'])
    name_set = set(column_list[name_colx])
    nx2_name, nx2_nid = nametables_from_nameset(name_set)
    # Get chip set
    print('[convert] build chip table')
    # ---------
    # This format has multiple images per row
    chips_per_name = 2  # this is apparently always 2

    def get_multiprop_colx_list(prefix):
        colx_list = []
        for num in xrange(chips_per_name):
            lbl = prefix + str(num + 1)
            colx = get_lbl_pos(column_labels, [lbl])
            colx_list.append(colx)
        return colx_list
    # ---------
    # Essential properties
    #prop2_colx_list = {}
    try:
        image_colx_list = get_multiprop_colx_list('IMAGE_')
    except Exception:
        image_colx_list = get_multiprop_colx_list('Image')
    # ---------
    # Nonessential multi-properties
    try_multiprops = ['DATE_NO']
    multiprop2_colx = {}
    for key in try_multiprops:
        try:
            multiprop2_colx[key]  = get_multiprop_colx_list(key)
        except Exception:
            pass
    # ---------
    # Nonessential single-properties
    try_props = ['SEX']
    prop2_colx = {}
    for key in try_props:
        try:
            other_colx = get_lbl_pos(column_labels, [key])
            prop2_colx[key] = other_colx
        except Exception:
            pass
    # ---------
    # Nonessential pairwise-properties
    try_match_props = ['matches', 'WildID_score']
    pairprop2_colx = {}
    for key in try_match_props:
        try:
            other_colx = get_lbl_pos(column_labels, [key])
            pairprop2_colx[key] = other_colx
        except Exception:
            pass
    # ---------
    # Build tables
    cx2_cid     = []
    cx2_theta   = []
    cx2_roi     = []
    cx2_nx      = []
    cx2_gx      = []
    prop_dict       = {}
    pairwise_dict   = {}
    gnameroi_to_cid = {}
    for key in prop2_colx.keys():
        prop_dict[key] = []
    for key in multiprop2_colx.keys():
        prop_dict[key] = []
    cid = 1

    def wildid_add_to_hs_tables(gname, name, roi, theta=0, **kwargs):
        cid = len(cx2_cid) + 1
        nx = nx2_name.index(name)
        gx = gx2_gname.index(gname)
        cx2_cid.append(cid)
        cx2_roi.append(roi)
        cx2_nx.append(nx)
        cx2_gx.append(gx)
        for key, val in kwargs.iteritems():
            prop_dict[key].append(val)
        cx2_theta.append(theta)
        sys.stdout.write(('\b' * 10) + 'cid = %4d' % cid)
        return cid
    # ---------
    # Wildid parsing
    bad_rows = 0
    for rowx in xrange(num_rows):
        name        = column_list[name_colx][rowx]
        tbl_kwargs2 = {key: column_list[val][rowx] for key, val in prop2_colx.iteritems()}
        pairwise_vals = [column_list[colx][rowx] for colx in pairprop2_colx.values()]
        cid_tup = []
        for num in xrange(chips_per_name):  # TODO: This is always just pairwise
            img_colx = image_colx_list[num]
            gname = column_list[img_colx][rowx]
            tbl_kwargs1 = {key: column_list[val[num]][rowx] for key, val in multiprop2_colx.iteritems()}
            tbl_kwargs = dict(tbl_kwargs1.items() + tbl_kwargs2.items())
            roi      = roi_from_imgsize(join(img_dpath, gname), silent=True)
            if roi is None:
                img_fpath = join(img_dpath, gname)
                bad_rows += 1
                if not exists(img_fpath):
                    print('nonexistant image: %r' % gname)
                else:
                    print('corrupted image: %r' % gname)
                continue
            gnameroi = (gname, tuple(roi))
            if gnameroi in gnameroi_to_cid.keys():
                cid = gnameroi_to_cid[gnameroi]
                cid_tup.append(cid)
                continue
            cid = wildid_add_to_hs_tables(gname, name, roi, **tbl_kwargs)
            gnameroi_to_cid[gnameroi] = cid
            cid_tup.append(cid)
        pairwise_dict[tuple(cid_tup)] = pairwise_vals

    print('bad_rows = %r ' % bad_rows)
    print('num_rows = %r ' % num_rows)
    print('chips_per_name = %r ' % chips_per_name)
    print('cid = %r ' % cid)

    print('num pairwise properties: %r' % len(pairwise_dict))
    print('implementation of pairwise properties does not yet exist')

    num_known_chips = len(cx2_cid)
    print('[convert] Added %r known chips.' % num_known_chips)
    # Add the rest of the nongroundtruthed chips
    print('[convert] Adding unknown images to table')

    # Check that images were unique
    unique_gx = np.unique(np.array(cx2_gx))
    print('len(cx2_gx)=%r'    % len(cx2_gx))
    print('len(unique_gx)=%r' % len(unique_gx))
    assert len(cx2_gx) == len(unique_gx), \
        'There are images specified twice'

    # Check that cids were unique
    cx2_cid_arr = np.array(cx2_cid)
    valid_cids  = cx2_cid_arr[np.where(cx2_cid_arr > 0)[0]]
    unique_cids = np.unique(valid_cids)
    print('len(cx2_cid)     = %r' % len(cx2_cid))
    print('len(valid_cids)  = %r' % len(valid_cids))
    print('len(unique_cids) = %r' % len(unique_cids))
    assert len(valid_cids) == len(unique_cids), \
        'There are chipids specified twice'

    known_gx_set = set(cx2_gx)
    for gx, gname in enumerate(gx2_gname):
        if gx in known_gx_set:
            continue
        name     = '____'
        roi      = roi_from_imgsize(join(img_dpath, gname), silent=False)
        tbl_kwargs1 = {key: 'NA' for key, val in multiprop2_colx.iteritems()}
        tbl_kwargs2 = {key: 'NA' for key, val in prop2_colx.iteritems()}
        tbl_kwargs = dict(tbl_kwargs1.items() + tbl_kwargs2.items())
        if not roi is None:
            cid = wildid_add_to_hs_tables(gname, name, roi, **tbl_kwargs)
    num_unknown_chips = len(cx2_cid) - num_known_chips
    print('[convert] Added %r more unknown chips.' % num_unknown_chips)
    cx2_nid = np.array(nx2_nid)[cx2_nx]
    cx2_gid = np.array(gx2_gid)[cx2_gx]
    print('[convert] There are %d chips' % (cid - 1))
    #
    # Write tables
    internal_dir      = join(db_dir, ld2.RDIR_INTERNAL2)
    helpers.ensurepath(internal_dir)
    write_chip_table(internal_dir, cx2_cid, cx2_gid, cx2_nid, cx2_roi, cx2_theta, prop_dict)
    write_name_table(internal_dir, nx2_nid, nx2_name)
    write_image_table(internal_dir, gx2_gid, gx2_gname)
    print('[convert] finished conversion')


#------------------------
# New modular function below
def roi_from_imgsize(img_fpath, silent=False):
    try:
        (w, h) = Image.open(img_fpath).size
        roi = [1, 1, w, h]
        return roi
    except Exception as ex:
        if not silent:
            print('Exception ex=%r' % ex)
            print('Not adding img_fpath=%r' % img_fpath)
            print('----')
        return None


def imagetables_from_img_dpath(img_dpath=None):
    gx2_gname = helpers.list_images(img_dpath)
    gx2_gid   = range(1, len(gx2_gname) + 1)
    print('There are %d images' % len(gx2_gname))
    return gx2_gid, gx2_gname


def groundtruth_from_imagenames(gx2_gname, gt_format):
    if gt_format is None:
        print('There is no image ground truth')
        return set([])
    print('Parsing image names for groundtruth. gt_format=%r' % gt_format)
    name_set = set([])
    for gx, gname in enumerate(gx2_gname):
        name, num = parse.parse(gt_format, gname)
        name_set.add(name)
    return name_set


def nametables_from_nameset(name_set):
    nx2_name  = ['____', '____'] + list(name_set)
    nx2_nid   = [1, 1] + range(2, len(name_set) + 2)
    print('There are %d names' % (len(nx2_name) - 2))
    return nx2_name, nx2_nid


DIFF_CHECK = True


def diff_strings(str1, str2):
    'kinda hacky and specific to wildebeast'
    s1 = str1.splitlines(1)
    s2 = str2.splitlines(1)
    print('len(str1) == len(str2) | %r == %r' % (len(str1), len(str2)))
    print('len(s1)   == len(s2)   | %r == %r' % (len(s1), len(s2)))
    if len(s1) == len(s2):
        for linex, (line1, line2) in enumerate(zip(s1, s2)):
            line1 = line1.replace(' ', '')
            line1 = line1.replace('test_', '')
            line1 = line1.replace('testA_', '')
            line2 = line2.replace(' ', '')
            if line1 != line2:
                print('--- Line: %r ---' % linex)
                sys.stdout.write(line1)
                sys.stdout.write(line2)
    print('')


def write_to_wrapper(csv_fpath, csv_string):
    if exists(csv_fpath) and DIFF_CHECK:
        print('table already exists: %r' % csv_fpath)
        with open(csv_fpath) as file:
            csv_string2 = file.read()
            if csv_string2 == csv_string:
                print('No difference!')
            else:
                print('difference!')
                #print('--------')
                #print(csv_string2)
                #print('--------')
                #print(csv_string)
                #print('--------')
                #diff_str = diff_strings(csv_string2, csv_string)
                #print(diff_str)
                #print('--------')
    else:
        helpers.write_to(csv_fpath, csv_string)


def write_chip_table(internal_dir, cx2_cid, cx2_gid, cx2_nid,
                     cx2_roi, cx2_theta, prop_dict=None):
    helpers.__PRINT_WRITES__ = True
    print('Writing Chip Table')
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

    chip_table = ld2.make_csv_table(column_labels, column_list, header, column_type)
    chip_table_fpath  = join(internal_dir, ld2.CHIP_TABLE_FNAME)
    write_to_wrapper(chip_table_fpath, chip_table)


def write_name_table(internal_dir, nx2_nid, nx2_name):
    helpers.__PRINT_WRITES__ = True
    # Make name_table.csv
    column_labels = ['nid', 'name']
    column_list = [nx2_nid[2:], nx2_name[2:]]  # dont write ____ for backcomp
    header = '# name table'
    name_table = ld2.make_csv_table(column_labels, column_list, header)
    name_table_fpath  = join(internal_dir, ld2.NAME_TABLE_FNAME)
    write_to_wrapper(name_table_fpath, name_table)


def write_image_table(internal_dir, gx2_gid, gx2_gname):
    helpers.__PRINT_WRITES__ = True
    # Make image_table.csv
    column_labels = ['gid', 'gname', 'aif']  # do aif for backwards compatibility
    gx2_aif = np.ones(len(gx2_gid), dtype=np.uint32)
    column_list   = [gx2_gid, gx2_gname, gx2_aif]
    header = '# image table'
    image_table = ld2.make_csv_table(column_labels, column_list, header)
    image_table_fpath = join(internal_dir, ld2.IMAGE_TABLE_FNAME)
    write_to_wrapper(image_table_fpath, image_table)
