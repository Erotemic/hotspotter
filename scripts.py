from __future__ import print_function, division
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[scripts]')
# Python
from os.path import dirname, join, splitext
import shutil
from itertools import izip
# Science
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
# HotSpotter
from hscom import fileio as io
from hscom import helpers as util
from hotspotter import load_data2 as ld2

#from dbgimport import *  # NOQA


def extract_encounter(hs, eid):
    work_dir = dirname(hs.dirs.db_dir)
    newdb_name = hs.get_db_name() + '_encounter_%s' % (eid)
    new_dbdir = join(work_dir, newdb_name)
    gx_list = np.where(np.array(hs.tables.gx2_eid) == eid)[0]
    if len(gx_list) == 0:
        raise Exception('no images to export')
    export_subdb_locals = export_subdatabase(hs, gx_list, new_dbdir)
    return locals()


def export_subdatabase(hs, gx_list, new_dbdir):
    # New database dirs
    new_imgdir = join(new_dbdir, ld2.RDIR_IMG)
    new_internal = join(new_dbdir, ld2.RDIR_INTERNAL)
    print('[scripts] Exporting into %r' % new_dbdir)

    # Ensure new database
    util.ensuredir(new_dbdir)
    util.ensuredir(new_imgdir)
    util.ensuredir(new_internal)

    gname_list = hs.gx2_gname(gx_list)
    src_gname_list = hs.gx2_gname(gx_list, full=True)
    dst_gname_list = map(lambda gname: join(new_imgdir, gname), gname_list)

    copy_list = [(src, dst) for (src, dst) in zip(src_gname_list, dst_gname_list)]

    mark_progress, end_prog = util.progress_func(len(copy_list), lbl='Copy Images')
    for count, (src, dst) in enumerate(copy_list):
        shutil.copy(src, dst)
        mark_progress(count)
    end_prog()

    cx_list = [cx for cxs in hs.gx2_cxs(gx_list) for cx in cxs.tolist()]
    nx_list = np.unique(hs.tables.cx2_nx[cx_list])

    image_table = ld2.make_image_csv(hs, gx_list)
    chip_table  = ld2.make_chip_csv(hs, cx_list)
    name_table  = ld2.make_name_csv(hs, nx_list)
    # csv filenames
    chip_table_fpath  = join(new_internal, ld2.CHIP_TABLE_FNAME)
    name_table_fpath  = join(new_internal, ld2.NAME_TABLE_FNAME)
    image_table_fpath = join(new_internal, ld2.IMAGE_TABLE_FNAME)
    # write csv files
    util.write_to(chip_table_fpath, chip_table)
    util.write_to(name_table_fpath, name_table)
    util.write_to(image_table_fpath, image_table)
    return locals()


def import_database(hs, other_dbdir):
    '''
    %run dbgimport.py
    workdir = expanduser('~/data/work')
    other_dbdir = join(workdir, 'hsdb_exported_138_185_encounter_eid=1 nGxs=43')
    '''
    import HotSpotterAPI as api

    dbdir1 = hs.dirs.db_dir
    dbdir2 = other_dbdir

    print('[scripts] Importing %r into %r' % (dbdir2, dbdir1))

    hs1 = hs
    hs2 = api.HotSpotter(hs1.args, db_dir=dbdir2)

    names1 = hs1.tables.nx2_name[:]
    names2 = hs2.tables.nx2_name[:]
    print('num_names1 = %r' % len(names1))
    print('num_names2 = %r' % len(names2))
    common_names = np.setdiff1d(np.intersect1d(names1, names2), [ld2.UNKNOWN_NAME])
    unique_names2 = np.setdiff1d(np.setdiff1d(names2, common_names), [ld2.UNKNOWN_NAME])
    if len(common_names) > 0:
        print('warning: the following names are used by both databases.')
        print('         I hope they are consistent.')
        print(common_names)

    gnames1 = hs1.tables.gx2_gname[:]
    gnames2 = hs2.tables.gx2_gname[:]
    print('num_gnames1 = %r' % len(gnames1))
    print('num_gnames2 = %r' % len(gnames2))
    common_gnames = np.intersect1d(gnames1, gnames2)
    unique_gnames2 = np.setdiff1d(gnames2, common_gnames)

    if len(unique_gnames2) > 0:
        msg = ('I havent been programmed to handle len(unique_gnames) > 0')
        print(msg)
        raise NotImplementedError(msg)

    # RCOS TODO: Rectify this with add_name and user iter_input
    def add_names(hs, name_list):
        # TODO Assert names are unique
        nx2_name = hs.tables.nx2_name.tolist()
        nx2_name.extend(name_list)
        hs.tables.nx2_name = np.array(nx2_name)

    # Add new names to database1
    add_names(hs1, unique_names2)

    # Build mapings from database2 to database1 indexes
    gx_map = {}
    for gx2, gname in enumerate(gnames2):
        gx1 = np.where(hs1.tables.gx2_gname == gname)[0][0]
        gx_map[gx2] = gx1

    nx_map = {}
    for nx2, name in enumerate(names2):
        nx1 = np.where(hs1.tables.nx2_name == name)[0][0]
        nx_map[nx2] = nx1

    for key in hs2.tables.prop_dict.keys():
        try:
            hs1.add_property(key)
        except UserWarning as ex:
            print(ex)
            pass

    # Build lists using database1 indexes
    cx_list2   = hs2.get_valid_cxs()
    change_cxs = []
    add_cxs = []
    # Find all chips which are in the same image and have the same roi
    for cx2 in cx_list2:
        gx2 = hs2.cx2_gx(cx2)
        gx1 = gx_map[gx2]
        cxs1 = hs1.gx2_cxs(gx1)
        rois1 = hs1.cx2_roi(cxs1)
        roi2 = hs2.cx2_roi(cx2)
        found = np.where(map(np.all, roi2 == rois1))[0]
        if len(found) == 1:
            cx1 = cxs1[found[0]]
            change_cxs.append((cx2, cx1))
        else:
            add_cxs.append(cx1)

    for cx2, cx1 in change_cxs:
        name2 = hs2.cx2_name(cx2)
        name1 = hs1.cx2_name(cx1)
        if name1 != name2:
            if name1 != ld2.UNKNOWN_NAME:
                print('conflict')
            hs1.change_name(cx1, name2)
            for key, vals in hs2.tables.prop_dict.iteritems():
                hs1.change_property(cx1, key, vals[cx2])

    gx_list2    = [gx_map[hs2.tables.cx2_gx[cx]] for cx in cx_list2]
    nx_list2    = [nx_map[hs2.tables.cx2_nx[cx]] for cx in cx_list2]
    roi_list2   = hs.tables.cx2_roi[cx_list2]
    theta_list2 = hs.tables.cx2_theta[cx_list2]
    prop_dict2  = {propkey: [cx2_propval[cx] for cx in iter(cx_list2)]
                   for (propkey, cx2_propval) in hs2.tables.prop_dict.iteritems()}

    for key in prop_dict2.keys():
        try:
            hs1.add_property(key)
        except UserWarning as ex:
            print(ex)
            pass

    # RCOS FIXME: This is a bad way of preallocing data.
    # Need to do it better. Modify add_chip to do things correctly

    def zip_dict(dict_):
        return [{k: v for k, v in zip(dict_.keys(), tup)} for tup in izip(*dict_.values())]

    # RCOS: FIXME: This script actually doesn't work correctly.
    # It works when all you need to do is update names though.
    #gx_list    = gx_list2
    #roi_list   = roi_list2
    #nx_list    = nx_list2
    #theta_list = theta_list2
    #props_dict = prop_dict2

    # TODO: Replace add_chips with a better version of this
    def add_chips(hs, gx_list, roi_list, nx_list, theta_list, props_dict, dochecks=True):
        if len(hs.tables.cx2_cid) > 0:
            next_cid = hs.tables.cx2_cid.max() + 1
        else:
            next_cid = 1
        num_new = len(gx_list)
        next_cids = np.arange(next_cid, next_cid + num_new)
        # Check to make sure lengths are consitent
        list_lens = map(len, [next_cids, gx_list, roi_list, nx_list, theta_list])
        prop_lens = map(len, props_dict.values())
        sizes_agree = all([len_ == num_new for len_ in list_lens + prop_lens])
        assert sizes_agree, 'sizes do not agree'
        # Remove any conflicts from disk
        if dochecks:
            for cid in next_cids:
                hs.delete_ciddata(cid)
        # Allocate space for a new chip
        hs.tables.cx2_cid   = np.concatenate((hs.tables.cx2_cid, next_cids))
        hs.tables.cx2_nx    = np.concatenate((hs.tables.cx2_nx,  nx_list))
        hs.tables.cx2_gx    = np.concatenate((hs.tables.cx2_gx,  gx_list))
        hs.tables.cx2_roi   = np.vstack((hs.tables.cx2_roi, roi_list))
        hs.tables.cx2_theta = np.concatenate((hs.tables.cx2_theta, theta_list))
        prop_dict = hs.tables.prop_dict
        for key in prop_dict.iterkeys():
            try:
                prop_dict[key].extend(props_dict[key])
            except KeyError:
                default = ['' for _ in xrange(num_new)]
                prop_dict[key].extend(default)
        #hs.num_cx += 1
        new_len = len(hs.tables.cx2_cid)
        cxs = np.arange(new_len - num_new, new_len)
        hs.update_samples()
        # Remove any conflicts from memory
        for cx in cxs:
            hs.unload_cxdata(cx)
        return cx

    add_chips(hs1, gx_list2, roi_list2, nx_list2, theta_list2, prop_dict2, dochecks=False)
    #back.populate_tables()


def delete_suffixed_images(hs, back):
    remove_cands = []
    gx2_gname = hs.tables.gx2_gname

    # Check to see if the image is a copy of another
    for gx, gname in enumerate(gx2_gname):
        name, ext = splitext(gname)
        components = name.split('_')
        if len(components) == 2:
            orig_name, copynum = components
            orig_gname = orig_name + ext
            copyof = np.where(gx2_gname == orig_gname)[0]
            if len(copyof) > 0:
                remove_cands.append((gx, copyof))

    # Make sure the images are actually duplicates
    remove_gxs = []
    orphaned_cxs = []
    for copy_gx, orig_gx in remove_cands:
        if isinstance(orig_gx, np.ndarray):
            orig_gx = orig_gx[0]
        if np.all(hs.gx2_image(copy_gx) == hs.gx2_image(orig_gx)):
            print('[script] duplicate found copy_gx=%r, orig_gx=%r' % (copy_gx, orig_gx))
            remove_gxs.append(copy_gx)
            copy_cxs = hs.gx2_cxs(copy_gx)
            orphaned_cxs.append((copy_cxs, orig_gx))

    # THESE ACTUALLY MODIFY THE DATABASE

    # Move all chips to the original
    for cx_list, orig_gx in orphaned_cxs:
        for cx in cx_list:
            print('[script] relocate cx=%r to gx=%r' % (cx, orig_gx))
            hs.tables.cx2_gx[cx] = orig_gx

    # Move deleted images into the trash
    trash_dir = join(hs.dirs.db_dir, 'deleted-images')
    src_list = hs.gx2_gname(remove_gxs, full=True)
    dst_list = hs.gx2_gname(remove_gxs, prefix=trash_dir)
    util.ensuredir(trash_dir)

    move_list = zip(src_list, dst_list)
    mark_progress, end_prog = util.progress_func(len(move_list), lbl='Trashing Image')
    for count, (src, dst) in enumerate(move_list):
        shutil.move(src, dst)
        mark_progress(count)
    end_prog()

    for gx in remove_gxs:
        print('[script] remove gx=%r' % (gx,))
        hs.tables.gx2_gname[gx] = ''

    # Update and save
    hs.update_samples()
    back.populate_image_table()

    hs.save_database()
    return locals()


# 138.185
def compute_encounters(hs, back, seconds_thresh=15):
    '''
    clusters encounters togethers (by time, not space)

    An encounter is a meeting, localized in time and space between a camera and
    a group of animals.

    Animals are identified within each encounter.
    '''
    if not 'seconds_thresh' in vars():
        seconds_thresh = 15
    gx_list = hs.get_valid_gxs()
    datetime_list = hs.gx2_exif(gx_list, tag='DateTime')

    unixtime_list = [io.exiftime_to_unixtime(datetime_str) for datetime_str in datetime_list]

    unixtime_list = np.array(unixtime_list)
    X = np.vstack([unixtime_list, np.zeros(len(unixtime_list))]).T
    print('[scripts] clustering')

    # Build a mapping from clusterxs to member gxs
    gx2_clusterid = fclusterdata(X, seconds_thresh, criterion='distance')
    clusterx2_gxs = [[] for _ in xrange(gx2_clusterid.max())]
    for gx, clusterx in enumerate(gx2_clusterid):
        clusterx2_gxs[clusterx - 1].append(gx)  # IDS are 1 based

    clusterx2_nGxs = np.array(map(len, clusterx2_gxs))
    print('cluster size stats: %s' % util.printable_mystats(clusterx2_nGxs))

    # Change IDs such that higher number = more gxs
    gx2_ex = [None] * len(gx2_clusterid)
    gx2_eid = [None] * len(gx2_clusterid)
    ex2_clusterx = clusterx2_nGxs.argsort()
    ex2_gxs = [None] * len(ex2_clusterx)
    for ex in xrange(len(ex2_clusterx)):
        clusterx = ex2_clusterx[ex]
        gxs = clusterx2_gxs[clusterx]
        ex2_gxs[ex] = gxs
        for gx in gxs:
            nGx = len(gxs)
            USE_STRING_ID = True
            if USE_STRING_ID:
                # String ID
                eid = 'ex=%r_nGxs=%d' % (ex, nGx)
            else:
                # Float ID
                eid = ex + (nGx / 10 ** np.ceil(np.log(nGx) / np.log(10)))
            gx2_eid[gx] = eid
            gx2_ex[gx] = ex

    hs.tables.gx2_ex  = np.array(gx2_ex)
    hs.tables.gx2_eid = np.array(gx2_eid)

    # Give info to GUI
    extra_cols = {'eid': lambda gx_list: [gx2_eid[gx] for gx in iter(gx_list)]}
    back.append_header('gxs', 'eid')
    back.populate_image_table(extra_cols=extra_cols)
    return locals()


def plot_time(unixtime_list):
    import draw_func2 as df2
    unixtime_list = np.array(unixtime_list)
    fixed_time = unixtime_list[unixtime_list > 0]
    df2.plot(sorted(unixtime_list))
    ax = df2.gca()
    ax.set_ylim(fixed_time.min(), fixed_time.max())
