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
from hscom import helpers as util


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


def check_qcx2_res(hs, qcx2_res):
    from collections import defaultdict
    if isinstance(qcx2_res, dict):
        print('[consist] qcx2_res is in dict format')
        for qcx, res in qcx2_res.iteritems():
            assert qcx == res.qcx
        res_list = qcx2_res.values()
    elif isinstance(qcx2_res, list):
        print('[consist] qcx2_res is in list format')
        for qcx, res in enumerate(qcx2_res):
            assert res is None or qcx == res.qcx
        res_list = [res for res in qcx2_res if res is not None]
    else:
        msg = ('[consist] UNKNOWN TYPE type(qcx2_res) = %r' % type(qcx2_res))
        print(msg)
        raise AssertionError(msg)
    print('[consist] structure is consistent')

    nUniqueRes = len(np.unique(map(id, res_list)))
    assert nUniqueRes == len(res_list)
    print('[consist] results are unique objects')

    test_vars = defaultdict(list)

    for res in res_list:
        test_vars['res.uid'].append(res.uid)
        test_vars['fmlen'].append(len(res.cx2_fm))
        test_vars['fslen'].append(len(res.cx2_fs))
        test_vars['fklen'].append(len(res.cx2_fk))
        test_vars['scorelen'].append(len(res.cx2_score))

    assert util.list_eq(test_vars['fmlen'])
    assert util.list_eq(test_vars['fslen'])
    assert util.list_eq(test_vars['fklen'])
    assert util.list_eq(test_vars['scorelen'])
    assert util.list_eq(test_vars['res.uid'])

    print('[consist] passed length tests')

    for res in res_list:
        test_vars['qcx_list'].append(res.qcx)
        test_vars['score_list'].append(res.cx2_score)
        test_vars['score_sum'].append(res.cx2_score.sum())
        test_vars['nMatches'].append(sum(map(len, res.cx2_fm)))

    assert len(np.unique(test_vars['qcx_list'])) == len(test_vars['qcx_list'])

    if len(np.unique(test_vars['nMatches'])) < len(test_vars['nMatches']) / 10:
        print('[consist] nMatches = %r' % (test_vars['nMatches'],))

    print('[consist] passed entropy test')


def dbg_check_query_result(hs, res, strict=False):
    print('[qr] Debugging result')
    fpath = res.get_fpath(hs)
    print(res)
    print('fpath=%r' % fpath)

    qcx = res.qcx
    chip_str = 'q%s' % hs.cidstr(qcx)
    kpts = hs.get_kpts(qcx)
    #
    # Check K are all in bounds
    fk_maxmin = np.array([(fk.max(), fk.min())
                          for fk in res.cx2_fk if len(fk) > 0])
    K = hs.prefs.query_cfg.nn_cfg.K
    assert fk_maxmin.max() < K
    assert fk_maxmin.min() >= 0
    #
    # Check feature indexes are in boundsS
    fx_maxmin = np.array([(fm[:, 0].max(), fm[:, 0].min())
                          for fm in res.cx2_fm if len(fm) > 0])
    nKpts = len(kpts)
    if fx_maxmin.max() >= nKpts:
        msg = ('DBG ERROR: ' + chip_str + ' nKpts=%d max_kpts=%d' % (nKpts, fx_maxmin.max()))
        print(msg)
        if strict:
            raise AssertionError(msg)
    assert fx_maxmin.min() >= 0


def dbg_qreq(qreq):
    print('ERROR in dbg_qreq()')
    print('[q1] len(qreq._dftup2_index)=%r' % len(qreq._dftup2_index))
    print('[q1] qreq_dftup2_index._dftup2_index=%r' % len(qreq._dftup2_index))
    print('[q1] qreq._dftup2_index.keys()=%r' % qreq._dftup2_index.keys())
    print('[q1] qreq._data_index=%r' % qreq._data_index)
