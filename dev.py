#!/usr/bin/env python
#exec(open('__init__.py').read())
#exec(open('_research/dev.py').read())
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile, printDBG) = __common__.init(__name__, '[dev]', DEBUG=False)
# Matplotlib
import matplotlib
matplotlib.use('Qt4Agg')
# Standard
import textwrap
import sys
from os.path import join
import multiprocessing
# Scientific
import numpy as np
#import cv2
# HotSpotter
#from hotspotter import spatial_verification2 as sv2
from hotspotter import DataStructures as ds
from hotspotter import match_chips3 as mc3
from hotspotter import matching_functions as mf
from hotspotter import report_results2 as rr2
from hotspotter import voting_rules2 as vr2
from hscom import helpers
from hscom import latex_formater as pytex
from hsviz import draw_func2 as df2
from hsviz import viz


def dev_reload():
    print('===========================')
    print('[dev] performing dev_reload')
    print('---------------------------')
    from hotspotter import algos
    from hotspotter import chip_compute2 as cc2
    from hotspotter import feature_compute2 as fc2
    from hsviz import interaction
    #from scripts
    from hscom import fileio as io
    from hotspotter import nn_filters
    rrr()
    io.rrr()
    ds.rrr()
    mf.rrr()
    nn_filters.rrr()
    mc3.rrr()
    viz.rrr()
    interaction.rrr()
    #scripts.rrr()
    vr2.rrr()
    helpers.rrr()
    cc2.rrr()
    fc2.rrr()
    algos.rrr()
    df2.rrr()
    print('---------------------------')
    print('df2 reset')
    df2.reset()
    print('---------------------------')
    print('[dev] finished dev_reload()')
    print('===========================')


def history_entry(database='', cid=-1, ocids=[], notes='', cx=-1):
    return (database, cid, ocids, notes)

# A list of poster child examples. (curious query cases)
GZ_greater1_cid_list = [140, 297, 306, 311, 425, 441, 443, 444, 445, 450, 451,
                        453, 454, 456, 460, 463, 465, 501, 534, 550, 662, 786,
                        802, 838, 941, 981, 1043, 1046, 1047]
HISTORY = [
    history_entry('TOADS', cx=32),
    history_entry('NAUTS', 1,    [],               notes='simple eg'),
    history_entry('WDOGS', 1,    [],               notes='simple eg'),
    history_entry('MOTHERS', 69, [68],             notes='textured foal (lots of bad matches)'),
    history_entry('MOTHERS', 28, [27],             notes='viewpoint foal'),
    history_entry('MOTHERS', 53, [54],             notes='image quality'),
    history_entry('MOTHERS', 51, [50],             notes='dark lighting'),
    history_entry('MOTHERS', 44, [43, 45],         notes='viewpoint'),
    history_entry('MOTHERS', 66, [63, 62, 64, 65], notes='occluded foal'),
]

#MANUAL_GZ_HISTORY = [
    #history_entry('GZ', 662,     [262],            notes='viewpoint / shadow (circle)'),
    #history_entry('GZ', 1046,    [],               notes='extreme viewpoint #gt=2'),
    #history_entry('GZ', 838,     [801, 980],       notes='viewpoint / quality'),
    #history_entry('GZ', 501,     [140],            notes='dark lighting'),
    #history_entry('GZ', 981,     [802],            notes='foal extreme viewpoint'),
    #history_entry('GZ', 306,     [112],            notes='occlusion'),
    #history_entry('GZ', 941,     [900],            notes='viewpoint / quality'),
    #history_entry('GZ', 311,     [289],            notes='quality'),
    #history_entry('GZ', 1047,    [],               notes='extreme viewpoint #gt=4'),
    #history_entry('GZ', 297,     [301],            notes='quality'),
    #history_entry('GZ', 786,     [787],            notes='foal #gt=11'),
    #history_entry('GZ', 534,     [411, 727],       notes='LNBNN failure'),
    #history_entry('GZ', 463,     [173],            notes='LNBNN failure'),
    #history_entry('GZ', 460,     [613, 460],       notes='background match'),
    #history_entry('GZ', 465,     [589, 460],       notes='background match'),
    #history_entry('GZ', 454,     [198, 447],       notes='forground match'),
    #history_entry('GZ', 445,     [702, 435],       notes='forground match'),
    #history_entry('GZ', 453,     [682, 453],       notes='forground match'),
    #history_entry('GZ', 550,     [551, 452],       notes='forground match'),
    #history_entry('GZ', 450,     [614],            notes='other zebra match'),
#]

                                                                                ##csum, pl, plw, borda
#AUTO_GZ_HISTORY = map(lambda tup: tuple(['GZ'] + list(tup)), [
    #(662,   [263],                              'viewpoint / shadow (circle) ranks = [16 20 20 20]'),
    #(1046,  [],                                 'extreme viewpoint #gt=2 ranks     = [592 592 592 592]'),
    #(838,   [802, 981],                         'viewpoint / quality ranks         = [607 607 607 607]'),
    #(501,   [141],                              'dark lighting ranks               = [483 483 483 483]'),
    #(802,   [981],                              'viewpoint / quality /no matches   = [722 722 722 722]'),
    #(907,   [828, 961],                         'occluded but (spatial verif)      = [645 645 645 645]'),
    #(1047,  [],                                 'extreme viewpoint #gt=4 ranks     = [582 582 582 582]'),
    #(16,    [635],                              'NA ranks                          = [839 839 839 839]'),
    #(140,   [501],                              'NA ranks                          = [194 194 194 194]'),
    #(981,   [803],                              'foal extreme viewpoint ranks      = [ 8  9  9 11]'),
    #(425,   [662],                              'NA ranks                          = [21 33 30 34]'),
    #(681,   [198, 454, 765],                    'NA ranks                          = [2 6 6 6]'),
    #(463,   [174],                              'LNBNN failure ranks               = [3 0 3 0]'),
    #(306,   [113],                              'occlusion ranks                   = [1 1 1 1]'),
    #(311,   [290],                              'quality ranks                     = [1 2 1 2]'),
    #(460,   [614, 461],                         'background match ranks            = [2 1 2 1]'),
    #(465,   [590, 461],                         'background match ranks            = [3 0 3 0]'),
    #(454,   [199, 448],                         'forground match ranks             = [5 3 3 2]'),
    #(445,   [703, 436],                         'forground match ranks             = [1 2 2 2]'),
    #(453,   [683, 454],                         'forground match ranks             = [2 3 4 0]'),
    #(550,   [552, 453],                         'forground match ranks             = [5 5 5 4]'),
    #(450,   [615],                              'other zebra match ranks           = [3 4 4 4]'),
    #(95,    [255],                              'NA ranks                          = [2 5 5 5]'),
    #(112,   [306],                              'NA ranks                          = [1 2 2 2]'),
    #(183,   [178],                              'NA ranks                          = [1 2 2 2]'),
    #(184,   [34, 39, 227, 619],                 'NA ranks                          = [1 1 1 1]'),
    #(253,   [343],                              'NA ranks                          = [1 1 1 1]'),
    #(276,   [45, 48],                           'NA ranks                          = [1 0 1 0]'),
    #(277,   [113, 124],                         'NA ranks                          = [1 0 1 0]'),
    #(289,   [311],                              'NA ranks                          = [2 1 2 1]'),
    #(339,   [315],                              'NA ranks                          = [1 1 1 1]'),
    #(340,   [317],                              'NA ranks                          = [1 0 1 0]'),
    #(415,   [408],                              'NA ranks                          = [1 3 2 4]'),
    #(430,   [675],                              'NA ranks                          = [1 0 1 0]'),
    #(436,   [60, 61, 548, 708, 760],            'NA ranks                          = [1 0 0 0]'),
    #(441,   [421],                              'NA ranks                          = [5 5 6 5]'),
    #(442,   [693, 777],                         'NA ranks                          = [1 0 1 0]'),
    #(443,   [420, 478],                         'NA ranks                          = [5 4 6 4]'),
    #(444,   [573],                              'NA ranks                          = [5 3 5 3]'),
    #(446,   [565, 678, 705],                    'NA ranks                          = [1 0 0 0]'),
    #(451,   [541, 549],                         'NA ranks                          = [2 0 1 0]'),
    #(456,   [172, 174, 219, 637],               'NA ranks                          = [3 1 2 0]'),
    #(661,   [59],                               'NA ranks                          = [0 4 4 4]'),
    #(720,   [556, 714],                         'NA ranks                          = [1 0 0 0]'),
    #(763,   [632],                              'NA ranks                          = [0 6 0 6]'),
    #(1044,  [845, 878, 927, 1024, 1025, 1042],  'NA ranks                          = [1 0 0 0]'),
    #(1045,  [846, 876],                         'NA ranks                          = [1 0 1 0]'),
#])
#HISTORY += AUTO_GZ_HISTORY


#def mothers_problem_pairs():
    #'''MOTHERS Dataset: difficult (qcx, cx) query/result pairs'''
    #viewpoint = [( 16, 17), (19, 20), (73, 71), (75, 78), (108, 112), (110, 108)]
    #quality = [(27, 26),  (52, 53), (67, 68), (73, 71), ]
    #lighting = [(105, 104), ( 49,  50), ( 93,  94), ]
    #confused = []
    #occluded = [(64, 65), ]
    #return locals()


# Just put in PL
def top_matching_features(res, axnum=None, match_type=''):
    cx2_fs = res.cx2_fs_V
    cx_fx_fs_list = []
    for cx in xrange(len(cx2_fs)):
        fx2_fs = cx2_fs[cx]
        for fx in xrange(len(fx2_fs)):
            fs = fx2_fs[fx]
            cx_fx_fs_list.append((cx, fx, fs))

    cx_fx_fs_sorted = np.array(sorted(cx_fx_fs_list, key=lambda x: x[2])[::-1])

    sorted_score = cx_fx_fs_sorted[:, 2]
    df2.figure(0)
    df2.plot(sorted_score)


def vary_query_cfg(hs, qcx_list, query_cfg=None, vary_cfg=None, fnum=1):
    # Ground truth matches
    for qcx in qcx_list:
        gt_cxs = hs.get_other_indexed_cxs(qcx)
        for cx in gt_cxs:
            fnum = vary_two_cfg(hs, qcx, cx, query_cfg, vary_cfg, fnum)
    return fnum


def vary_two_cfg(hs, qcx, cx, query_cfg, vary_cfg, fnum=1):
    if len(vary_cfg) > 2:
        raise Exception('can only vary at most two cfgeters')
    print('[dev] vary_two_cfg: q' + hs.vs_str(qcx, cx))
    cfg_keys = vary_cfg.keys()
    cfg_vals = vary_cfg.values()
    cfg1_name = cfg_keys[0]
    cfg2_name = cfg_keys[1]
    cfg1_steps = cfg_vals[0]
    cfg2_steps = cfg_vals[1]
    nRows = len(cfg1_steps)
    nCols = len(cfg2_steps)

    print('[dev] Varying configs: nRows=%r, nCols=%r' % (nRows, nCols))
    print('[dev] %r = %r ' % (cfg1_name, cfg1_steps))
    print('[dev] %r = %r ' % (cfg2_name, cfg2_steps))
    ylabel_args = dict(rotation='horizontal',
                       verticalalignment='bottom',
                       horizontalalignment='right',
                       fontproperties=df2.FONTS.medbold)
    xlabel_args = dict(fontproperties=df2.FONTS.medbold)
    #ax = df2.gca()
    # Vary cfg1
    #df2..gcf().clf()
    print_lock_ = helpers.ModulePrintLock(mc3, df2)
    assign_alg = query_cfg.agg_cfg.query_type
    vert = not hs.args.horiz
    plt_match_args = dict(fnum=fnum, show_gname=False, showTF=False, vert=vert)
    for rowx, cfg1_value in enumerate(cfg1_steps):
        query_cfg.update_cfg(**{cfg1_name: cfg1_value})
        y_title = cfg1_name + '=' + helpers.format(cfg1_value, 3)
        # Vary cfg2
        for colx, cfg2_value in enumerate(cfg2_steps):
            query_cfg.update_cfg(**{cfg2_name: cfg2_value})
            pnum = (nRows, nCols, rowx * nCols + colx + 1)
            # HACK
            #print(pnum)
            #print(query_cfg)
            # query only the chips of interest (groundtruth) when doing vsone
            if assign_alg == 'vsone':
                res = hs.query_groundtruth(qcx, query_cfg)
            # query the entire database in vsmany (just as fast as vgroundtruth)
            elif assign_alg == 'vsmany':
                res = hs.query(qcx, query_cfg)
            res.plot_single_match(hs, cx, pnum=pnum, **plt_match_args)
            x_title = cfg2_name + '=' + helpers.format(cfg2_value, 3)  # helpers.commas(cfg2_value, 3)
            ax = df2.gca()
            if rowx == len(cfg1_steps) - 1:
                ax.set_xlabel(x_title, **xlabel_args)
            if colx == 0:
                ax.set_ylabel(y_title, **ylabel_args)
    del print_lock_
    vary_title = '%s vary %s and %s' % (assign_alg, cfg1_name, cfg2_name)
    figtitle =  '%s %s %s' % (vary_title, hs.vs_str(qcx, cx), str(hs.cx2_property(qcx, 'Notes')))
    subtitle = mc3.simplify_test_uid(query_cfg.get_uid())
    df2.set_figtitle(figtitle, subtitle)
    df2.adjust_subplots_xylabels()
    fnum += 1
    viz.save_if_requested(hs, vary_title)
    return fnum


def show_name(hs, qcx, fnum=1, **kwargs):
    print('[dev] Plotting name')
    viz.show_name_of(hs, qcx, fnum=fnum, **kwargs)
    return fnum + 1


def show_names(hs, qcx_list, fnum=1):
    '''The most recent plot names function, works with qcx_list'''
    print('[dev] show_names()')
    result_dir = hs.dirs.result_dir
    names_dir = join(result_dir, 'show_names')
    helpers.ensuredir(names_dir)
    # NEW:
    print(qcx_list)
    nx_list = np.unique(hs.tables.cx2_nx[qcx_list])
    print(nx_list)
    for nx in nx_list:
        viz.show_name(hs, nx, fnum=fnum)
        df2.save_figure(fpath=names_dir, usetitle=True)
    # OLD:
    #for (qcx) in qcx_list:
        #print('Showing q%s - %r' % (hs.cidstr(qcx, notes=True)))
        #notes = hs.cx2_property(qcx, 'Notes')
        #fnum = show_name(hs, qcx, fnum, subtitle=notes, annote=not hs.args.noannote)
        #if hs.args.save_figures:
            #df2.save_figure(fpath=names_dir, usetitle=True)
    return fnum


def vary_vsone_cfg(hs, qcx_list, fnum, vary_dicts, **kwargs):
    vary_cfg = helpers.dict_union(*vary_dicts)
    query_cfg = ds.get_vsone_cfg(hs, **kwargs)
    return vary_query_cfg(hs, qcx_list, query_cfg, vary_cfg, fnum)


def vary_vsmany_cfg(hs, qcx_list, vary_dicts, fnum, **kwargs):
    vary_cfg = helpers.dict_union(*vary_dicts)
    query_cfg = ds.get_vsmany_cfg(hs, **kwargs)
    return vary_query_cfg(hs, qcx_list, query_cfg, vary_cfg, fnum)


def plot_keypoint_scales(hs, fnum=1):
    print('[dev] plot_keypoint_scales()')
    cx2_kpts = hs.feats.cx2_kpts
    cx2_nFeats = map(len, cx2_kpts)
    kpts = np.vstack(cx2_kpts)
    print('[dev] --- LaTeX --- ')
    _printopts = np.get_printoptions()
    np.set_printoptions(precision=3)
    print(pytex.latex_scalar(r'\# keypoints, ', len(kpts)))
    print(pytex.latex_mystats(r'\# keypoints per image', cx2_nFeats))
    acd = kpts[:, 2:5].T
    scales = np.sqrt(acd[0] * acd[2])
    scales = np.array(sorted(scales))
    print(pytex.latex_mystats(r'keypoint scale', scales))
    np.set_printoptions(**_printopts)
    print('[dev] ---/LaTeX --- ')
    #
    df2.figure(fnum=fnum, docla=True, title='sorted scales')
    df2.plot(scales)
    df2.adjust_subplots_safe()
    #ax = df2.gca()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #
    fnum += 1
    df2.figure(fnum=fnum, docla=True, title='hist scales')
    df2.show_histogram(scales, bins=20)
    df2.adjust_subplots_safe()
    #ax = df2.gca()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    return fnum


def get_qcx_list(hs):
    print('[dev] get_qcx_list()')
    # Sample a large pool of query indexes
    histids = None if hs.args.histid is None else np.array(hs.args.histid)
    if hs.args.all_cases:
        print('[dev] all cases')
        qcx_all = get_cases(hs, with_gt=True, with_nogt=True)
    elif hs.args.all_gt_cases:
        print('[dev] all gt cases')
        qcx_all = get_cases(hs, with_hard=True, with_gt=True, with_nogt=False)
    elif hs.args.qcid is None:
        qcx_all = get_cases(hs, with_hard=True, with_gt=False, with_nogt=False)
    else:
        print('[dev] Chosen qcid=%r' % hs.args.qcid)
        qcx_all =  helpers.ensure_iterable(hs.cid2_cx(hs.args.qcid))
    # Filter only the ones you want from the large pool
    if histids is None:
        qcx_list = qcx_all
    else:
        histids = helpers.ensure_iterable(histids)
        print('[dev] Chosen histids=%r' % histids)
        qcx_list = [qcx_list[id_] for id_ in histids]

    if len(qcx_list) == 0:
        if hs.args.strict:
            raise Exception('no qcx_list history')
        qcx_list = [0]
    return qcx_list


def investigate_vsone_groundtruth(hs, qcx_list, fnum=1):
    print('--------------------------------------')
    print('[dev] investigate_vsone_groundtruth')
    query_cfg = ds.get_vsone_cfg(sv_on=True, ratio_thresh=1.5)
    for qcx in qcx_list:
        res = hs.query_groundtruth(hs, qcx, query_cfg)
        #print(query_cfg)
        #print(res)
        #res.show_query(hs, fnum=fnum)
        fnum += 1
        res.show_topN(hs, fnum=fnum, query_cfg=query_cfg)
        fnum += 1
    return fnum


def investigate_chip_info(hs, qcx_list, fnum=1):
    for qcx in qcx_list:
        chip_info(hs, qcx)
    return fnum


def chip_info(hs, cx, notes=''):
    nx = hs.tables.cx2_nx[cx]
    gx = hs.tables.cx2_gx[cx]
    name = hs.tables.nx2_name[nx]
    gname = hs.tables.gx2_gname[gx]
    indexed_gt_cxs = hs.get_other_indexed_cxs(cx)
    gt_cxs = hs.get_other_indexed_cxs(cx)
    kpts = hs.get_kpts(cx)
    cidstr = hs.cidstr(cx)
    print('------------------')
    print('[dev] Chip Info ')
    infostr_list = [
        cidstr,
        'notes=%r' % notes,
        'cx=%r' % cx,
        'gx=%r' % gx,
        'nx=%r' % nx,
        'name=%r' % name,
        'gname=%r' % gname,
        'len(kpts)=%r' % len(kpts),
        'nGroundTruth = %s ' % str(len(gt_cxs)),
        'nIndexedGroundTruth = %s ' % str(len(indexed_gt_cxs)),
        'Ground Truth: %s' % (hs.cidstr(gt_cxs),),
        'IndexedGroundTruth = %s' % (hs.cidstr(indexed_gt_cxs),),
    ]
    print(helpers.indent('\n'.join(infostr_list), '    '))
    return locals()


def intestigate_keypoint_interaction(hs, qcx_list, fnum=1, **kwargs):
    import hstpl
    for qcx in qcx_list:
        rchip = hs.get_chip(qcx)
        kpts  = hs.feats.cx2_kpts[qcx]
        desc  = hs.feats.cx2_desc[qcx]
        hstpl.extern_feat.keypoint_interaction(rchip, kpts, desc, fnum=fnum, **kwargs)
        fnum += 1
    return fnum


def dbstats(hs):
    import db_info
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

    #tex_nImage = pytex.latex_scalar(r'\# images', num_images)
    tex_nChip = pytex.latex_scalar(r'\# chips', num_chips)
    tex_nName = pytex.latex_scalar(r'\# names', num_names)
    tex_nSingleName = pytex.latex_scalar(r'\# singlenames', num_singlenames)
    tex_nMultiName  = pytex.latex_scalar(r'\# multinames', num_multinames)
    tex_nMultiChip  = pytex.latex_scalar(r'\# multichips', num_multichips)
    tex_multi_stats = pytex.latex_mystats(r'\# multistats', multiton_nx2_nchips)

    tex_kpts_scale_thresh = pytex.latex_multicolumn('Scale Threshold (%d %d)' %
                                                    (hs.prefs.feat_cfg.scale_min,
                                                     hs.prefs.feat_cfg.scale_max)) + r'\\' + '\n'

    (tex_nKpts, tex_kpts_stats, tex_scale_stats) = db_info.get_keypoint_stats(hs)
    tex_title = pytex.latex_multicolumn(db_name + ' database statistics') + r'\\' + '\n'
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
    print(tabular)

# ^^^^^^^^^^^^^^^^^
# Tests

#===========
# Main Script
# exec(open('dev.py').read())


def dev_main(defaultdb='NAUTS', **kwargs):
    'Developer main script. Contains all you need to quickly start tests'
    import main
    print('[dev] main()')
    # Create Hotspotter API
    hs = main.main(defaultdb=defaultdb)
    # Get the query/others/notes list
    # this contains a list of cannonical test examples
    # FIXME: This is specific to one machine right now
    qcx_list = get_qcx_list(hs)
    qcx   = qcx_list[0]
    return locals()
#---end main script


def get_cases(hs, with_hard=True, with_gt=True, with_nogt=True, with_notes=False):
    qcx_list = []
    valid_cxs = hs.get_valid_cxs()
    if with_hard:
        if 'hard' in hs.tables.prop_dict:
            for cx in iter(valid_cxs):
                if hs.cx2_property(cx, 'hard') == 'True':
                    qcx_list += [cx]
    if with_hard:
        if 'Notes' in hs.tables.prop_dict:
            for cx in iter(valid_cxs):
                if hs.cx2_property(cx, 'Notes') != '':
                    qcx_list += [cx]
    if with_gt and not with_nogt:
        for cx in iter(valid_cxs):
            gt_cxs = hs.get_other_indexed_cxs(cx)
            if len(gt_cxs) > 0:
                qcx_list += [cx]
    if with_gt and with_nogt:
        qcx_list = valid_cxs
    return qcx_list


allres_ptr = [None]


def get_allres(hs):
    global allres_ptr
    allres = allres_ptr[0]
    if allres is None:
        allres_ptr[0] = rr2.get_allres(hs)
    return allres_ptr[0]


def report_results(hs):
    from hotspotter import report_results2 as rr2
    if '--list' in sys.argv:
        #listpos = sys.argv.index('--list')
        #if listpos < len(sys.argv) - 1:
        rr2.print_result_summaries_list()
        sys.exit(1)

    #allres = helpers.search_stack_for_localvar('allres')
    #if allres is None:
    allres = get_allres(hs)

    #Helper drawing functions
    #gt_matches = lambda cx: viz.show_chip(allres, cx, 'gt_matches')
    #top5 = lambda cx: viz.show_chip(allres, cx, 'top5')
    #selc = lambda cx: viz.show_chip(allres, cx, 'kpts')
    #matchd = lambda: rr2.viz_db_match_distances(allres)

    print(allres)
    #matchd()
    #exec(df2.present())


# Driver Function
def run_investigations(hs, qcx_list):
    import experiment_harness
    args = hs.args
    qcx = qcx_list[0]
    print('[dev] Running Investigation: ' + hs.cidstr(qcx))
    fnum = 1
    #view_all_history_names_in_db(hs, 'MOTHERS')
    #fnum = compare_matching_methods(hs, qcx, fnum)
    #xy_  = {'xy_thresh':     [None, .2, .02, .002]}
    #xy_  = {'xy_thresh':     [None, .02, .002, .001, .0005]}
    #rat_ = {'ratio_thresh':  [None, 1.4, 1.6, 1.8]}
    xy_  = {'xy_thresh':     [None, .02, .002]}
    rat_ = {'ratio_thresh':  [None, 1.5, 1.7]}
    K_   = {'K':             [2, 5, 10]}
    #Kr_  = {'Krecip':        [0, 2, 5, 10]}

    tests = args.tests[:]

    def intest(*args):
        for testname in args:
            ret = testname in tests
            if ret:
                tests.remove(testname)
        return ret

    if intest('print-hs'):
        print(hs)
    if intest('show-names'):
        show_names(hs, qcx_list)
    if intest('vary-vsone-rat-xy'):
        fnum = vary_vsone_cfg(hs, qcx_list, fnum, [rat_, xy_])
    if intest('vary-vsmany-k-xy'):
        fnum = vary_vsmany_cfg(hs, qcx_list, fnum, [K_, xy_])
    if intest('dbstats'):
        fnum = dbstats(hs)
    if intest('scale'):
        fnum = plot_keypoint_scales(hs)
    if intest('vsone-gt'):
        fnum = investigate_vsone_groundtruth(hs, qcx_list, fnum)
    if intest('chip-info'):
        fnum = investigate_chip_info(hs, qcx_list, fnum)
    if intest('kpts-interact'):
        fnum = intestigate_keypoint_interaction(hs, qcx_list)
    if intest('interact'):
        from hsgui import interaction
        fnum = interaction.interact1(hs, qcx_list, fnum)
    if intest('list'):
        print(experiment_harness.get_valid_testcfg_names())
    if intest('dists'):
        allres = get_allres(hs)
        rr2.viz_db_match_distances(allres)
    if intest('report_results', 'rr'):
        report_results(hs)

    # Allow any testcfg to be in tests like:
    # vsone_1 or vsmany_3
    import experiment_configs as _testcfgs
    testcfg_keys = vars(_testcfgs).keys()
    testcfg_locals = [key for key in testcfg_keys if key.find('_') != 0]
    for test_cfg_name in testcfg_locals:
        if intest(test_cfg_name):
            fnum = experiment_harness.test_configurations(hs, qcx_list, [test_cfg_name], fnum)
    if len(tests) > 0:
        raise Exception('Unknown tests: %r ' % tests)


def export_qon_list(hs, qcx_list):
    print('[dev] Exporting query-object-notes to property tables')
    if not hs.has_property('Notes'):
        hs.add_property('Notes')
    for qcx, ocxs, notes in qcx_list:
        print('----')
        old_prop = hs.get_property(qcx, 'Notes')
        print('old = ' + old_prop)
        print(notes)
        if old_prop.find(notes) == -1:
            new_prop = notes if old_prop == '' else old_prop + '; ' + notes
            print('new: ' + new_prop)
            hs.change_property(qcx, 'Notes', new_prop)
        print(hs.get_property(qcx, 'Notes'))
    hs.save_database()


def all_printoff():
    from hotspotter import fileio as io
    from hotspotter import HotSpotterAPI as api
    ds.print_off()
    mf.print_off()
    io.print_off()
    api.print_off()
    mc3.print_off()
    vr2.print_off()
    #algos.print_off()
    #cc2.print_off()
    #fc2.print_off()
    #ld2.print_off()
    #helpers.print_off()
    #parallel.print_off()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('[dev] __main__ ')
    # useful when copy and pasting into ipython
    main_locals = dev_main()
    hs = main_locals['hs']
    qcx_list = main_locals['qcx_list']
    exec(helpers.execstr_dict(main_locals, 'main_locals'))
    print('[dev]====================')
    if hs.args.printoff:
        all_printoff()
    mf.print_off()  # Make testing slightly faster
    # Big test function. Should be replaced with something
    # not as ugly soon.
    run_investigations(hs, qcx_list)
    # A redundant query argument. Again, needs to be replaced.
    if hs.args.query is not None and len(hs.args.query) > 0:
        qcx = hs.cid2_cx(hs.args.query[0])
        res = hs.query(qcx)
        res.show_top(hs)
    print('[dev]====================')
    kwargs = {}
    dcxs = None
    query_cfg = None
    if hs.args.nopresent:
        print('...not presenting')
        sys.exit(0)
    allres = allres_ptr[0]
    exec(df2.present())  # **df2.OooScreen2()
