from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile, printDBG) = __common__.init(__name__, '[dev_stats]', DEBUG=False)
# Science
import numpy as np
# HotSpotter
from hotspotter import db_info
from hscom import latex_formater
from hscom import helpers as util


def dbstats(hs):
    # Chip / Name / Image stats
    dbinfo_locals = db_info.db_info(hs)
    db_name = hs.get_db_name()
    #num_images = dbinfo_locals['num_images']
    num_chips = dbinfo_locals['num_chips']
    num_names = len(dbinfo_locals['valid_nxs'])
    num_singlenames = len(dbinfo_locals['singleton_nxs'])
    num_multinames = len(dbinfo_locals['multiton_nxs'])
    num_multichips = len(dbinfo_locals['multiton_cxs'])
    multiton_nx2_nchips = dbinfo_locals['multiton_nx2_nchips']

    #tex_nImage = latex_formater.latex_scalar(r'\# images', num_images)
    tex_nChip = latex_formater.latex_scalar(r'\# chips', num_chips)
    tex_nName = latex_formater.latex_scalar(r'\# names', num_names)
    tex_nSingleName = latex_formater.latex_scalar(r'\# singlenames', num_singlenames)
    tex_nMultiName  = latex_formater.latex_scalar(r'\# multinames', num_multinames)
    tex_nMultiChip  = latex_formater.latex_scalar(r'\# multichips', num_multichips)
    tex_multi_stats = latex_formater.latex_mystats(r'\# multistats', multiton_nx2_nchips)

    tex_kpts_scale_thresh = latex_formater.latex_multicolumn('Scale Threshold (%d %d)' %
                                                             (hs.prefs.feat_cfg.scale_min,
                                                              hs.prefs.feat_cfg.scale_max)) + r'\\' + '\n'

    (tex_nKpts, tex_kpts_stats, tex_scale_stats) = db_info.get_keypoint_stats(hs)
    tex_title = latex_formater.latex_multicolumn(db_name + ' database statistics') + r'\\' + '\n'
    tabular_body_list = [
        tex_title,
        tex_nChip,
        tex_nName,
        tex_nSingleName,
        tex_nMultiName,
        tex_nMultiChip,
        tex_multi_stats,
        '',
        tex_kpts_scale_thresh,
        tex_nKpts,
        tex_kpts_stats,
        tex_scale_stats,
    ]
    tabular = latex_formater.tabular_join(tabular_body_list)
    print('[dev stats]')
    print(tabular)


def cache_memory_stats(hs, qcx_list, fnum=None):
    from hotspotter import feature_compute2 as fc2
    from hotspotter import algos
    from hotspotter import DataStructures as ds
    from hscom import latex_formater
    print('[dev stats] cache_memory_stats()')
    kpts_list = hs.get_kpts(qcx_list)
    desc_list = hs.get_desc(qcx_list)
    nFeats_list = map(len, kpts_list)
    gx_list = np.unique(hs.cx2_gx(qcx_list))

    # Flann info
    ax2_cx, ax2_fx, ax2_desc, flann, precomp_kwargs = ds.build_flann_inverted_index(hs, qcx_list, return_info=True)
    del precomp_kwargs['force_recompute']
    flann_fpath = algos.get_flann_fpath(ax2_desc, **precomp_kwargs)

    file_bytes = util.file_bytes

    bytes_map = {
        'chip dbytes': [file_bytes(fpath) for fpath in hs.get_rchip_path(qcx_list)],
        'feat dbytes': [file_bytes(fpath) for fpath in fc2._cx2_feat_fpaths(hs, qcx_list)],
        'img dbytes':  [file_bytes(gpath) for gpath in hs.gx2_gname(gx_list, full=True)],
        'flann dbytes':  file_bytes(flann_fpath),
    }

    byte_units = {
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
    }

    tabular_body_list = [
    ]

    convert_to = 'KB'
    for key, val in bytes_map.iteritems():
        key2 = key.replace('bytes', convert_to)
        if isinstance(val, list):
            val2 = [bytes_ / byte_units[convert_to] for bytes_ in val]
            tex_str = latex_formater.latex_mystats(key2, val2)
        else:
            val2 = val / byte_units[convert_to]
            tex_str = latex_formater.latex_scalar(key2, val2)
        tabular_body_list.append(tex_str)

    tabular = latex_formater.tabular_join(tabular_body_list)

    print(tabular)
    latex_formater.render(tabular)

    if fnum is None:
        fnum = 0

    util.embed()
    return fnum + 1
