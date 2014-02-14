from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[dev_consist]')
# Standard
from os.path import relpath
# Science
import numpy as np
# Hotspotter
from hscom import fileio as io


#----------------------
# Debug Consistency Checks
def check_keypoint_consistency(hs):
    cx2_cid = hs.tables.cx2_cid
    cx2_kpts = hs.feats.cx2_kpts
    bad_cxs = [cx for cx, kpts in enumerate(cx2_kpts) if kpts is None]
    passed = True
    if len(bad_cxs) > 0:
        print('[dev_consist] cx2_kpts has %d None positions:' % len(bad_cxs))
        print('[dev_consist] bad_cxs = %r' % bad_cxs)
        passed = False
    if len(cx2_kpts) != len(cx2_cid):
        print('[dev_consist] len(cx2_kpts) != len(cx2_cid): %r != %r' % (len(cx2_kpts), len(cx2_cid)))
        passed = False
    if passed:
        print('[dev_consist] cx2_kpts is OK')


def detect_duplicate_images(hs):
    # TODO: Finish this function
    img_dir = hs.dirs.img_dir
    valid_gxs = hs.get_valid_gxs()
    gx2_gpath = hs.gx2_gname(valid_gxs, full=True)
    imgpath_list = gx2_gpath

    # Find which images are duplicates using hashing
    duplicates = io.detect_duplicate_images(imgpath_list)

    # Convert output paths to indexes
    nDuplicates = 0
    dup_gxs = []
    for hashstr, gpath_list in duplicates.iteritems():
        if len(gpath_list) != 1:
            gname_list = [relpath(gpath, img_dir) for gpath in gpath_list]
            gx_list = np.array(hs.gname2_gx(gname_list))
            dup_gxs.append(gx_list)
            nDuplicates += len(dup_gxs)
    print('[dev_consist] There are %d duplicate sets, and %d duplicate images' % (len(dup_gxs), nDuplicates))

    # Detect which images can be autotrashed
    keep_gxs = []
    conflict_gxs = []
    remove_gxs = []
    for gx_list in dup_gxs:
        cxs_list = np.array(hs.gx2_cxs(gx_list))
        nCxs_list = np.array(map(len, cxs_list))
        nonzeros = nCxs_list != 0
        # Check to see if no image was populated
        populated_gxs = gx_list[nonzeros]
        nonpopulated_gxs = gx_list[True - nonzeros]
        print('-----')
        print('[dev_consist] Nonpopulated gxs = %r'  % (nonpopulated_gxs.tolist(),))
        print('[dev_consist] Populated gxs = %r ' % (populated_gxs.tolist(),))
        if not np.any(nonzeros):
            # There are no chips in these duplicates
            keep = gx_list[0]
            remove = np.setdiff1d(gx_list, [keep])
            keep_gxs.append(keep)
            remove_gxs.append(remove)
            print('[dev_consist] No chips. Can safely remove: %r' % (gx_list,))
            continue
        sorted_nCxs_list = nCxs_list[nonzeros]
        # Check to see if we only one image was populated
        if len(sorted_nCxs_list) == 1:
            print('[dev_consist] Only one image populated. Can remove others')
            keep = gx_list[nonzeros][0]
            remove = np.setdiff1d(gx_list, [keep])
            keep_gxs.append(keep)
            remove_gxs.append(remove)
            continue
        # Check to see if they are all the same
        if np.all(sorted_nCxs_list[0] == sorted_nCxs_list):
            # These might have all the same chip info
            rois_list = [hs.cx2_roi(cxs) for cxs in cxs_list[nonzeros]]
            name_list  = [hs.cx2_name(cxs) for cxs in cxs_list[nonzeros]]

            props_list = [rois_list, name_list]
            num_unique = lambda list_: len(set(map(repr, _list)))
            if all([num_unique(_list) == 1 for _list in props_list]):
                print('[dev_consist] all chips appear to be the same')
                keep = gx_list[nonzeros][0]
                remove = np.setdiff1d(gx_list, [keep])
                keep_gxs.append(keep)
                remove_gxs.append(remove)
            else:
                conflict_gxs.append(gx_list)

    print('[dev_consist] %d can be kept. %d can be removed. %d conflicting sets' % (len(keep_gxs), len(remove_gxs), len(conflict_gxs),))
