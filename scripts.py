from __future__ import print_function, division
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[scripts]')
# Python
from os.path import dirname, join, splitext
import shutil
# Science
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
# HotSpotter
import fileio as io
import helpers
import load_data2 as ld2

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
    helpers.ensuredir(new_dbdir)
    helpers.ensuredir(new_imgdir)
    helpers.ensuredir(new_internal)

    gname_list = hs.gx2_gname(gx_list)
    src_gname_list = hs.gx2_gname(gx_list, full=True)
    dst_gname_list = map(lambda gname: join(new_imgdir, gname), gname_list)

    copy_list = [(src, dst) for (src, dst) in zip(src_gname_list, dst_gname_list)]

    mark_progress = helpers.progress_func(len(copy_list), lbl='Copy Images')
    for count, (src, dst) in enumerate(copy_list):
        shutil.copy(src, dst)
        mark_progress(count)

    cx_list = [cx for cxs in hs.gx2_cxs(gx_list) for cx in cxs.tolist()]
    nx_list = np.unique(hs.tables.cx2_nx[cx_list])

    image_table = ld2.make_image_csv2(hs, gx_list)
    chip_table  = ld2.make_chip_csv2(hs, cx_list)
    name_table  = ld2.make_name_csv2(hs, nx_list)
    # csv filenames
    chip_table_fpath  = join(new_internal, ld2.CHIP_TABLE_FNAME)
    name_table_fpath  = join(new_internal, ld2.NAME_TABLE_FNAME)
    image_table_fpath = join(new_internal, ld2.IMAGE_TABLE_FNAME)
    # write csv files
    helpers.write_to(chip_table_fpath, chip_table)
    helpers.write_to(name_table_fpath, name_table)
    helpers.write_to(image_table_fpath, image_table)
    return locals()


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
    helpers.ensuredir(trash_dir)

    move_list = zip(src_list, dst_list)
    mark_progress = helpers.progress_func(len(move_list), lbl='Trashing Image')
    for count, (src, dst) in enumerate(move_list):
        shutil.move(src, dst)
        mark_progress(count)

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
    print('cluster size stats: %s' % helpers.printable_mystats(clusterx2_nGxs))

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
                eid = 'eid=%r nGxs=%d' % (ex, nGx)
            else:
                # Float ID
                eid = ex + (nGx / 10 ** np.ceil(np.log(nGx) / np.log(10)))
            gx2_eid[gx] = eid
            gx2_ex[gx] = ex

    hs.tables.gx2_ex = gx2_ex
    hs.tables.gx2_eid = np.array(gx2_eid)

    # Give info to GUI
    extra_cols = {'Encounter Id': lambda gx_list: [gx2_eid[gx] for gx in iter(gx_list)]}
    try:
        back.imgtbl_headers.index('Encounter Id')
    except ValueError:
        back.imgtbl_headers.append('Encounter Id')
    back.populate_image_table(extra_cols=extra_cols)
    return locals()


def plot_time(unixtime_list):
    import draw_func2 as df2
    unixtime_list = np.array(unixtime_list)
    fixed_time = unixtime_list[unixtime_list > 0]
    df2.plot(sorted(unixtime_list))
    ax = df2.gca()
    ax.set_ylim(fixed_time.min(), fixed_time.max())
