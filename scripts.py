from __future__ import print_function, division
# Python
import datetime
import time
# Science
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
# HotSpotter
import helpers
import load_data2 as ld2
from os.path import dirname, join
import shutil


def export_subdatabase(hs, gx_list, new_dbdir):
    # New database dirs
    if not 'new_dbdir' in vars():
        gx_list = newx2_gxs[138]
        new_dbdir = join(dirname(hs.dirs.db_dir), 'hsdb_exported_138_185')
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

    mark_progress = helpers.progress_func(len(copy_list))
    for count, (src, dst) in enumerate(copy_list):
        shutil.copy(src, dst)
        mark_progress(count)

    cx_list = [item for arr in hs.gx2_cxs(gx_list) for item in arr.tolist()]
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

# 138.185

def cluster_images_by_exif_datetime(hs, back, seconds_thresh=15):
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
    strptime = datetime.datetime.strptime

    def convert_exif_datetime(datetime_str):
        if datetime_str is None:
            return -1
        dt = strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
        return time.mktime(dt.timetuple())

    unixtime_list = [convert_exif_datetime(datetime_str) for datetime_str in datetime_list]

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
    gx2_newid = [None] * len(gx2_clusterid)
    newx2_clusterx = clusterx2_nGxs.argsort()
    newx2_gxs = [None] * len(newx2_clusterx)
    for newx in xrange(len(newx2_clusterx)):
        clusterx = newx2_clusterx[newx]
        gxs = clusterx2_gxs[clusterx]
        newx2_gxs[newx] = gxs
        for gx in gxs:
            nGx = len(gxs)
            #strid = 'nGxs=%d newx=%r' % (nGx, newx)
            floatid = newx + (nGx / 10 ** np.ceil(np.log(nGx) / np.log(10)))
            gx2_newid[gx] = floatid

    # Give info to GUI
    extra_cols = {'cluster': [gx2_newid[gx] for gx in iter(gx_list)]}
    try:
        back.imgtbl_headers.index('cluster')
    except ValueError:
        back.imgtbl_headers.append('cluster')
    back.populate_image_table(extra_cols=extra_cols)


def plot_time(unixtime_list):
    import draw_func2 as df2
    unixtime_list = np.array(unixtime_list)
    fixed_time = unixtime_list[unixtime_list > 0]
    df2.plot(sorted(unixtime_list))
    ax = df2.gca()
    ax.set_ylim(fixed_time.min(), fixed_time.max())
