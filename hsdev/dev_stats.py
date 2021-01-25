
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile, printDBG) = __common__.init(__name__, '[dev_stats]', DEBUG=False)
# Standard
import textwrap
# HotSpotter
from hotspotter import db_info
from hscom import latex_formater


def dbstats(hs):
    # Chip / Name / Image stats
    dbinfo_locals = db_info.db_info(hs)
    db_name = hs.get_db_name(True)
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
    dedent = textwrap.dedent

    tabular_head = dedent(r'''
    \begin{tabular}{|l|l|}
    ''')
    tabular_tail = dedent(r'''
    \end{tabular}
    ''')
    hline = ''.join([r'\hline', '\n'])
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
    tabular_body = hline.join(tabular_body_list)
    tabular = hline.join([tabular_head, tabular_body, tabular_tail])
    print('[dev stats]')
    print(tabular)
