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
import sys
from os.path import join
import multiprocessing
# Scientific
import numpy as np
#import cv2
# HotSpotter
#from hotspotter import spatial_verification2 as sv2
from hotspotter import DataStructures as ds
from hotspotter import chip_compute2 as cc2
from hotspotter import feature_compute2 as fc2
from hotspotter import load_data2 as ld2
from hotspotter import matching_functions as mf
from hotspotter import report_results2 as rr2
from hscom import helpers as util
from hscom import latex_formater
from hscom import params
from hsdev import dev_stats
from hsdev import experiment_configs
from hsdev import experiment_harness
from hsdev import test_api
from hsgui import guitools
from hsviz import allres_viz
from hsviz import draw_func2 as df2
from hsviz import interact
from hsviz import viz
import hstpl
#from hscom import fileio as io
#from hotspotter import HotSpotterAPI as api
#from hotspotter import QueryResult as qr
#from hotspotter import match_chips3 as mc3
#from hotspotter import voting_rules2 as vr2


def myexcepthook(type, value, tb):
    #https://stackoverflow.com/questions/14775916/coloring-exceptions-from-python-on-a-terminal
    import traceback
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter

    tbtext = ''.join(traceback.format_exception(type, value, tb))
    lexer = get_lexer_by_name("pytb", stripall=True)
    formatter = TerminalFormatter(bg="dark")
    sys.stderr.write(highlight(tbtext, lexer, formatter))

sys.excepthook = myexcepthook


def export_qon_list(hs, qcx_list):
    " Populates the Notes field with test results "
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
    assign_alg = query_cfg.agg_cfg.query_type
    vert = not params.args.horiz
    plt_match_args = dict(fnum=fnum, show_gname=False, showTF=False, vert=vert)
    for rowx, cfg1_value in enumerate(cfg1_steps):
        query_cfg.update_cfg(**{cfg1_name: cfg1_value})
        y_title = cfg1_name + '=' + util.format(cfg1_value, 3)
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
            x_title = cfg2_name + '=' + util.format(cfg2_value, 3)  # util.commas(cfg2_value, 3)
            ax = df2.gca()
            if rowx == len(cfg1_steps) - 1:
                ax.set_xlabel(x_title, **xlabel_args)
            if colx == 0:
                ax.set_ylabel(y_title, **ylabel_args)
    vary_title = '%s vary %s and %s' % (assign_alg, cfg1_name, cfg2_name)
    figtitle =  '%s %s %s' % (vary_title, hs.vs_str(qcx, cx), str(hs.cx2_property(qcx, 'Notes')))
    df2.set_figtitle(figtitle)
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
    util.ensuredir(names_dir)
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
        #fnum = show_name(hs, qcx, fnum, subtitle=notes, annote=not params.args.noannote)
        #if params.args.save_figures:
            #df2.save_figure(fpath=names_dir, usetitle=True)
    return fnum


def vary_vsone_cfg(hs, qcx_list, fnum, vary_dicts, **kwargs):
    vary_cfg = util.dict_union(*vary_dicts)
    query_cfg = ds.get_vsone_cfg(hs, **kwargs)
    return vary_query_cfg(hs, qcx_list, query_cfg, vary_cfg, fnum)


def vary_vsmany_cfg(hs, qcx_list, vary_dicts, fnum, **kwargs):
    vary_cfg = util.dict_union(*vary_dicts)
    query_cfg = ds.get_vsmany_cfg(hs, **kwargs)
    return vary_query_cfg(hs, qcx_list, query_cfg, vary_cfg, fnum)


def plot_keypoint_scales(hs, fnum=1):
    print('[dev] plot_keypoint_scales()')
    cx2_kpts = hs.feats.cx2_kpts
    if len(cx2_kpts) == 0:
        hs.refresh_features()
        cx2_kpts = hs.feats.cx2_kpts
    cx2_nFeats = map(len, cx2_kpts)
    kpts = np.vstack(cx2_kpts)
    print('[dev] --- LaTeX --- ')
    _printopts = np.get_printoptions()
    np.set_printoptions(precision=3)
    print(latex_formater.latex_scalar(r'\# keypoints, ', len(kpts)))
    print(latex_formater.latex_mystats(r'\# keypoints per image', cx2_nFeats))
    acd = kpts[:, 2:5].T
    scales = np.sqrt(acd[0] * acd[2])
    scales = np.array(sorted(scales))
    print(latex_formater.latex_mystats(r'keypoint scale', scales))
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
    print(util.indent('\n'.join(infostr_list), '    '))
    return locals()


def intestigate_keypoint_interaction(hs, qcx_list, fnum=1, **kwargs):
    for qcx in qcx_list:
        rchip = hs.get_chip(qcx)
        kpts  = hs.feats.cx2_kpts[qcx]
        desc  = hs.feats.cx2_desc[qcx]
        hstpl.extern_feat.keypoint_interaction(rchip, kpts, desc, fnum=fnum, **kwargs)
        fnum += 1
    return fnum


# ^^^^^^^^^^^^^^^^^
# Tests

#===========
# Main Script
# exec(open('dev.py').read())
def dev_main(defaultdb='NAUTS', **kwargs):
    'Developer main script. Contains all you need to quickly start tests'
    print('[dev] main()')
    # Create Hotspotter API
    hs = test_api.main(defaultdb='NAUTS')
    print('')
    print('==========================')
    print('   **** DEV SCRIPT ***    ')
    print('==========================')
    print('[dev] dev_main()')
    print('==========================')

    # Get the query/others/notes list
    # this contains a list of cannonical test examples
    # FIXME: This is specific to one machine right now
    qcx_list = test_api.get_qcx_list(hs)
    qcx   = qcx_list[0]
    return locals()
#---end main script


ALLRES_DICT = {}


def get_allres(hs, qcx_list):
    global ALLRES_DICT
    qcxs_ = tuple(qcx_list)
    if not qcxs_ in ALLRES_DICT:
        ALLRES_DICT[qcxs_] = rr2.get_allres(hs, qcx_list)
    allres = ALLRES_DICT[qcxs_]
    return allres


def get_qcx2_res(hs, qcx_list):
    allres = get_allres(hs, qcx_list)
    qcx2_res = {qcx: res for qcx, res in enumerate(allres.qcx2_res) if res is not None}
    return qcx2_res


def report_results(hs, qcx_list):
    if '--list' in sys.argv:
        #listpos = sys.argv.index('--list')
        #if listpos < len(sys.argv) - 1:
        rr2.print_result_summaries_list()
        sys.exit(1)

    allres = get_allres(hs, qcx_list)
    print(allres)


def plot_feature_distances(allres, orgres_list=None, fnum=1):
    print('[dev] plot_feature_distances()')
    orgres2_distance = allres.get_orgres2_distances(orgres_list=orgres_list)
    db_name = allres.hs.get_db_name()
    allres_viz.show_descriptors_match_distances(orgres2_distance,
                                                db_name=db_name, fnum=fnum)
    fnum += 1
    return fnum


YSCALE = util.get_arg('--yscale', default='symlog')  # 'symlog'
XSCALE = 'linear'


def plot_seperability(hs, qcx_list, fnum=1):
    print('[dev] plot_seperability(fnum=%r)' % fnum)
    qcx2_res = get_qcx2_res(hs, qcx_list)
    qcx2_separability = get_seperatbility(hs, qcx2_res)
    sep_score_list = qcx2_separability.values()
    df2.figure(fnum=fnum, doclf=True, docla=True)
    print('[dev] seperability stats: ' + util.pstats(sep_score_list))
    sorted_sepscores = sorted(sep_score_list)
    df2.plot(sorted_sepscores, color=df2.DEEP_PINK, label='seperation score',
             yscale=YSCALE)
    df2.set_xlabel('true chipmatch index (%d)' % len(sep_score_list))
    df2.set_logyscale_from_data(sorted_sepscores)
    df2.dark_background()
    true_uid = qcx2_res.itervalues().next().true_uid
    df2.set_figtitle('seperability\n' + true_uid)
    df2.legend()
    fnum += 1
    return fnum


def plot_scores(hs, qcx_list, fnum=1):
    print('[dev] plot_scores(fnum=%r)' % fnum)
    qcx2_res = get_qcx2_res(hs, qcx_list)
    all_score_list = []
    gtscore_ys = []
    gtscore_xs = []
    gtscore_ranks = []
    EXCLUDE_ZEROS = True
    N = 1
    # Append all scores to a giant list
    for res in qcx2_res.itervalues():
        cx2_score = res.cx2_score
        # Get gt scores first
        #gt_cxs = hs.get_other_indexed_cxs(res.qcx)
        gt_cxs = np.array(res.topN_cxs(hs, N=N, only_gt=True))
        gt_ys = cx2_score[gt_cxs]
        if EXCLUDE_ZEROS:
            nonzero_cxs = np.where(cx2_score != 0)[0]
            gt_cxs = gt_cxs[gt_ys != 0]
            gt_ranks = res.get_gt_ranks(gt_cxs)
            gt_cxs = np.array(util.list_index(nonzero_cxs, gt_cxs))
            gt_ys  = gt_ys[gt_ys != 0]
            score_list = cx2_score[nonzero_cxs].tolist()
        else:
            score_list = cx2_score.tolist()
            gt_ranks = res.get_gt_ranks(gt_cxs)
        gtscore_ys.extend(gt_ys)
        gtscore_xs.extend(gt_cxs + len(all_score_list))
        gtscore_ranks.extend(gt_ranks)
        # Append all scores
        all_score_list.extend(score_list)
    all_score_list = np.array(all_score_list)
    gtscore_ranks = np.array(gtscore_ranks)
    gtscore_ys = np.array(gtscore_ys)

    # Sort all chipmatch scores
    allx_sorted = all_score_list.argsort()  # mapping from sortedallx to allx
    allscores_sorted = all_score_list[allx_sorted]
    # Change the groundtruth positions to correspond to sorted cmatch scores
    # Find position of gtscore_xs in allx_sorted
    gtscore_sortxs = util.list_index(allx_sorted, gtscore_xs)
    gtscore_sortxs = np.array(gtscore_sortxs)
    # Draw and info
    rank_bounds = [
        (0, 1),
        (1, 5),
        (5, None)
    ]
    rank_colors = [
        df2.TRUE_GREEN,
        df2.UNKNOWN_PURP,
        df2.FALSE_RED
    ]
    print('[dev] matching chipscore stats: ' + util.pstats(all_score_list))
    df2.figure(fnum=fnum, doclf=True, docla=True)
    # Finds the knee
    df2.plot(allscores_sorted, color=df2.ORANGE, label='all scores')

    # get positions which are within rank bounds
    for count, ((low, high), rankX_color) in reversed(list(enumerate(zip(rank_bounds, rank_colors)))):
        rankX_flag_low = gtscore_ranks >= low
        if high is not None:
            rankX_flag_high = gtscore_ranks < high
            rankX_flag = np.logical_and(rankX_flag_low, rankX_flag_high)
        else:
            rankX_flag = rankX_flag_low
        rankX_allgtx = np.where(rankX_flag)[0]
        rankX_gtxs = gtscore_sortxs[rankX_allgtx]
        rankX_gtys = gtscore_ys[rankX_allgtx]
        rankX_label = '%d <= gt rank' % low
        if high is not None:
            rankX_label += ' < %d' % high
        if len(rankX_gtxs) > 0:
            df2.plot(rankX_gtxs, rankX_gtys, 'o', color=rankX_color, label=rankX_label)

    true_uid = qcx2_res.itervalues().next().true_uid

    df2.set_logyscale_from_data(allscores_sorted)
    df2.set_xlabel('chipmatch index')
    df2.dark_background()
    df2.set_figtitle('matching scores\n' + true_uid)
    df2.legend(loc='upper left')
    df2.update()
    fnum += 1
    return fnum


def get_seperatbility(hs, qcx2_res):
    qcx2_separability = {qcx: res.compute_seperability(hs) for qcx, res in qcx2_res.iteritems()}
    qcx2_separability = {qcx: sepscore for qcx, sepscore in qcx2_separability.iteritems() if sepscore is not None}
    return qcx2_separability


# Driver Function
def run_investigations(hs, qcx_list):
    print('\n\n')
    print('==========================')
    print('RUN INVESTIGATIONS %s' % hs.get_db_name())
    print('==========================')
    input_test_list = params.args.tests[:]
    print('[dev] input_test_list = %r' % (input_test_list,))
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

    valid_test_list = []  # build list for printing in case of failure

    def intest(*args):
        for testname in args:
            valid_test_list.append(testname)
            ret = testname in input_test_list
            if ret:
                input_test_list.remove(testname)
                print('[dev] ===================')
                print('[dev] running testname=%s' % testname)
                return ret
        return False

    if intest('print-hs'):
        print(hs)
    if intest('show-names'):
        show_names(hs, qcx_list)
    if intest('vary-vsone-rat-xy'):
        fnum = vary_vsone_cfg(hs, qcx_list, fnum, [rat_, xy_])
    if intest('vary-vsmany-k-xy'):
        fnum = vary_vsmany_cfg(hs, qcx_list, fnum, [K_, xy_])
    if intest('dbstats'):
        fnum = dev_stats.dbstats(hs)
    if intest('scale'):
        fnum = plot_keypoint_scales(hs)
    if intest('vsone-gt'):
        fnum = investigate_vsone_groundtruth(hs, qcx_list, fnum)
    if intest('chip-info'):
        fnum = investigate_chip_info(hs, qcx_list, fnum)
    if intest('kpts-interact'):
        fnum = intestigate_keypoint_interaction(hs, qcx_list)
    if intest('interact'):
        fnum = interact.interact1(hs, qcx_list, fnum)
    if intest('list'):
        print(experiment_harness.get_valid_testcfg_names())
    if intest('matrix'):
        allres = get_allres(hs, qcx_list)
        allres_viz.plot_score_matrix(allres)
    if intest('report_results', 'rr'):
        report_results(hs, qcx_list)
    if intest('custom'):
        fnum = experiment_harness.test_configurations(hs, qcx_list, 'custom', fnum)
    if intest('seperability', 'sep'):
        fnum = plot_seperability(hs, qcx_list, fnum)
    if intest('scores', 'score'):
        fnum = plot_scores(hs, qcx_list, fnum)
    if intest('dists', 'dist'):
        allres = get_allres(hs, qcx_list)
        fnum = plot_feature_distances(allres, orgres_list=None, fnum=fnum)

    # Allow any testcfg to be in tests like:
    # vsone_1 or vsmany_3
    testcfg_keys = vars(experiment_configs).keys()
    testcfg_locals = [key for key in testcfg_keys if key.find('_') != 0]
    for test_cfg_name in testcfg_locals:
        if intest(test_cfg_name):
            fnum = experiment_harness.test_configurations(hs, qcx_list, [test_cfg_name], fnum)

    if intest('help'):
        print('valid tests are:')

        print(''.join(util.indent_list('\n -t ', valid_test_list)))
        return

    if len(input_test_list) > 0:
        print('valid tests are: \n')
        print('\n'.join(valid_test_list))
        raise Exception('Unknown tests: %r ' % input_test_list)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('[dev] __main__ ')
    QUIET = util.get_flag('--quiet', False)
    VERBOSE = util.get_flag('--verbose', False)
    #if QUIET:
        #print_off()
        #all_printoff()
    QUIET_LOAD = not VERBOSE
    QUIET_QUERY = not VERBOSE
    if not VERBOSE:
        ld2.print_off()
        fc2.print_off()
        cc2.print_off()
    #if not VERBOSE:
        #mf.print_off()

    # useful when copy and pasting into ipython
    guitools.init_qtapp()
    main_locals = dev_main()
    hs = main_locals['hs']
    qcx_list = main_locals['qcx_list']
    exec(util.execstr_dict(main_locals, 'main_locals'))
    print('[dev]====================')
    #mf.print_off()  # Make testing slightly faster
    # Big test function. Should be replaced with something
    # not as ugly soon.
    fnum = 1
    run_investigations(hs, qcx_list)
    # A redundant query argument. Again, needs to be replaced.
    if params.args.query is not None and len(params.args.query) > 0:
        hs.prefs.display_cfg.showanalysis = True
        qcx = hs.cid2_cx(params.args.query[0])
        res = hs.query(qcx)
        res.show_top(hs)
    print('[dev]====================')
    kwargs = {}
    dcxs = None
    query_cfg = None
    if params.args.nopresent:
        print('...not presenting')
        sys.exit(0)
    exec(df2.present(wh=1000))
