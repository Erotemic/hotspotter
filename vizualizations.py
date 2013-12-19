from __future__ import division, print_function
import __builtin__
import matplotlib
matplotlib.use('Qt4Agg')
import draw_func2 as df2
# Python
from os.path import realpath, join
import multiprocessing
import os
#import re
import sys
import warnings
# Scientific
import numpy as np
# Hotspotter
import fileio as io
import helpers

# Global variables
BROWSE = True
DUMP = False
FIGNUM = 1

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write


def print_on():
    global print, print_
    print  = __builtin__.print
    print_ = sys.stdout.write


def print_off():
    global print, print_

    def print(*args, **kwargs):
        pass

    def print_(*args, **kwargs):
        pass


def rrr():
    'Dynamic module reloading'
    import imp
    import sys
    print('[viz] reloading ' + __name__)
    imp.reload(sys.modules[__name__])


def printDBG(msg):
    pass


def plot_name_of_cx(hs, cx, **kwargs):
    nx = hs.tables.cx2_nx[cx]
    plot_name(hs, nx, hl_cxs=[cx], **kwargs)


def plot_name(hs, nx, nx2_cxs=None, fignum=0, hl_cxs=[], subtitle='',
              annote=True, **kwargs):
    print('[viz] plot_name nx=%r' % nx)
    nx2_name = hs.tables.nx2_name
    cx2_nx   = hs.tables.cx2_nx
    name = nx2_name[nx]
    if not nx2_cxs is None:
        cxs = nx2_cxs[nx]
    else:
        cxs = np.where(cx2_nx == nx)[0]
    print('[viz] plot_name %r' % hs.cidstr(cxs))
    ncxs  = len(cxs)
    #nCols = int(min(np.ceil(np.sqrt(ncxs)), 5))
    nCols = int(min(ncxs, 5))
    nRows = int(np.ceil(ncxs / nCols))
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    #gs2 = gridspec.GridSpec(nRows, nCols)
    pnum = lambda px: (nRows, nCols, px + 1)
    fig = df2.figure(fignum=fignum, plotnum=pnum(0), **kwargs)
    fig.clf()
    for px, cx in enumerate(cxs):
        show_chip(hs, cx=cx, plotnum=pnum(px), draw_kpts=annote, kpts_alpha=.2)
        if cx in hl_cxs:
            ax = df2.gca()
            df2.draw_border(ax, df2.GREEN, 4)
        #plot_cx3(hs, cx)
    title = 'nx=%r -- name=%r' % (nx, name)
    if not annote:
        title += ' noannote'
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    #df2.set_figtitle(title, subtitle)


def plot_rank_stem(allres, orgres_type='true'):
    print('[viz] plotting rank stem')
    # Visualize rankings with the stem plot
    hs = allres.hs
    title = orgres_type + 'rankings stem plot\n' + allres.title_suffix
    orgres = allres.__dict__[orgres_type]
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    x_data = orgres.qcxs
    y_data = orgres.ranks
    df2.draw_stems(x_data, y_data)
    slice_num = int(np.ceil(np.log10(len(orgres.qcxs))))
    df2.set_xticks(hs.test_sample_cx[::slice_num])
    df2.set_xlabel('query chip indeX (qcx)')
    df2.set_ylabel('groundtruth chip ranks')
    #df2.set_yticks(list(seen_ranks))
    __dump_or_browse(allres.hs, 'rankviz')


def plot_rank_histogram(allres, orgres_type):
    print('[viz] plotting %r rank histogram' % orgres_type)
    ranks = allres.__dict__[orgres_type].ranks
    label = 'P(rank | ' + orgres_type + ' match)'
    title = orgres_type + ' match rankings histogram\n' + allres.title_suffix
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    df2.draw_histpdf(ranks, label=label)  # FIXME
    df2.set_xlabel('ground truth ranks')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.hs, 'rankviz')


def plot_score_pdf(allres, orgres_type, colorx=0.0, variation_truncate=False):
    print('[viz] plotting ' + orgres_type + ' score pdf')
    title  = orgres_type + ' match score frequencies\n' + allres.title_suffix
    scores = allres.__dict__[orgres_type].scores
    print('[viz] len(scores) = %r ' % (len(scores),))
    label  = 'P(score | %r)' % orgres_type
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    df2.draw_pdf(scores, label=label, colorx=colorx)
    if variation_truncate:
        df2.variation_trunctate(scores)
    #df2.variation_trunctate(false.scores)
    df2.set_xlabel('score')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.hs, 'scoreviz')


def plot_score_matrix(allres):
    print('[viz] plotting score matrix')
    score_matrix = allres.score_matrix
    title = 'Score Matrix\n' + allres.title_suffix
    # Find inliers
    #inliers = helpers.find_std_inliers(score_matrix)
    #max_inlier = score_matrix[inliers].max()
    # Trunate above 255
    score_img = np.copy(score_matrix)
    score_img[score_img < 0] = 0
    score_img[score_img > 255] = 255
    #dim = 0
    #score_img = helpers.norm_zero_one(score_img, dim=dim)
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    df2.imshow(score_img, fignum=FIGNUM)
    df2.set_xlabel('database')
    df2.set_ylabel('queries')
    __dump_or_browse(allres.hs, 'scoreviz')


# Dump logic
def __browse():
    print('[viz] Browsing Image')
    df2.show()


def __dump(hs, subdir):
    #print('[viz] Dumping Image')
    fpath = hs.dirs.result_dir
    if not subdir is None:
        fpath = join(fpath, subdir)
        helpers.ensurepath(fpath)
    df2.save_figure(fpath=fpath, usetitle=True)
    df2.reset()


def save_if_requested(hs, subdir):
    if not hs.args.save_figures:
        return
    #print('[viz] Dumping Image')
    fpath = hs.dirs.result_dir
    if not subdir is None:
        subdir = helpers.sanatize_fname2(subdir)
        fpath = join(fpath, subdir)
        helpers.ensurepath(fpath)
    df2.save_figure(fpath=fpath, usetitle=True)
    df2.reset()


def __dump_or_browse(hs, subdir=None):
    #fig = df2.plt.gcf()
    #fig.tight_layout()
    if BROWSE:
        __browse()
    if DUMP:
        __dump(hs, subdir)


def plot_tt_bt_tf_matches(hs, allres, qcx):
    #print('Visualizing result: ')
    #res.printme()
    res = allres.qcx2_res[qcx]
    ranks = (allres.top_true_qcx_arrays[0][qcx],
             allres.bot_true_qcx_arrays[0][qcx],
             allres.top_false_qcx_arrays[0][qcx])
    #scores = (allres.top_true_qcx_arrays[1][qcx],
             #allres.bot_true_qcx_arrays[1][qcx],
             #allres.top_false_qcx_arrays[1][qcx])
    cxs = (allres.top_true_qcx_arrays[2][qcx],
           allres.bot_true_qcx_arrays[2][qcx],
           allres.top_false_qcx_arrays[2][qcx])
    titles = ('best True rank=' + str(ranks[0]) + ' ',
              'worst True rank=' + str(ranks[1]) + ' ',
              'best False rank=' + str(ranks[2]) + ' ')
    df2.figure(fignum=1, plotnum=231)
    res.plot_matches(res, hs, cxs[0], False, fignum=1, plotnum=131, title_aug=titles[0])
    res.plot_matches(res, hs, cxs[1], False, fignum=1, plotnum=132, title_aug=titles[1])
    res.plot_matches(res, hs, cxs[2], False, fignum=1, plotnum=133, title_aug=titles[2])
    fig_title = 'fig q' + hs.cidstr(qcx) + ' TT BT TF -- ' + allres.title_suffix
    df2.set_figtitle(fig_title)
    #df2.set_figsize(_fn, 1200,675)


def dump_gt_matches(allres):
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    'Displays the matches to ground truth for all queries'
    for qcx in xrange(0, len(qcx2_res)):
        res = qcx2_res[qcx]
        res.show_gt_matches(hs, fignum=FIGNUM)
        __dump_or_browse(allres.hs, 'gt_matches' + allres.title_suffix)


def dump_orgres_matches(allres, orgres_type):
    orgres = allres.__dict__[orgres_type]
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    # loop over each query / result of interest
    for qcx, cx, score, rank in orgres.iter():
        query_gname, _  = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[qcx]])
        result_gname, _ = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[cx]])
        res = qcx2_res[qcx]
        df2.figure(fignum=FIGNUM, plotnum=121)
        df2.show_matches3(res, hs, cx, SV=False, fignum=FIGNUM, plotnum=121)
        df2.show_matches3(res, hs, cx, SV=True,  fignum=FIGNUM, plotnum=122)
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % (score, rank, query_gname,
                                                      result_gname)
        df2.set_figtitle(big_title)
        __dump_or_browse(allres.hs, orgres_type + '_matches' + allres.title_suffix)
#------------------------------
#^^^ OLD


def _annotate_image(hs, fig, ax, gx, highlight_cxs, cx_clicked_func,
                    draw_roi=True, draw_roi_lbls=True, **kwargs):
    # draw chips in the image
    cx_list = hs.gx2_cxs(gx)
    centers = []
    interact = cx_clicked_func is not None
    # Draw all chip indexes in the image
    for cx in cx_list:
        roi = hs.cx2_roi(cx)
        # Draw the ROI
        roi_lbl = hs.cidstr(cx)
        if cx in highlight_cxs:
            bbox_color = df2.ORANGE * np.array([1, 1, 1, .95])
            lbl_color  = df2.BLACK * np.array([1, 1, 1, .75])
        else:
            dark_alpha = np.array([1, 1, 1, .6])
            bbox_color = df2.DARK_ORANGE * dark_alpha
            lbl_color  = df2.BLACK       * dark_alpha
        df2.draw_roi(ax, roi, roi_lbl, bbox_color, lbl_color)
        # Index the roi centers (for interaction)
        (x, y, w, h) = roi
        xy_center = np.array([x + (w / 2), y + (h / 2)])
        centers.append(xy_center)
    # Put roi centers in numpy array
    centers = np.array(centers)

    # Create callback wrapper
    def _on_image_click(event):
        'Slot for matplotlib event'
        print('\n[viz] clicked image')
        if event.xdata is None:
            return
        if len(centers) == 0:
            return
        #print('\n'.join(['%r=%r' % tup for tup in event.__dict__.iteritems()]))
        x, y = event.xdata, event.ydata
        # Find ROI center nearest to the clicked point
        dist = (centers.T[0] - x) ** 2 + (centers.T[1] - y) ** 2
        cx = cx_list[dist.argsort()[0]]
        cx_clicked_func(cx)

    df2.disconnect_callback(fig, 'button_press_event')
    if interact:
        df2.connect_callback(fig, 'button_press_event', _on_image_click)


#def start_image_interaction(hs, gx, cx_clicked_func):


def show_image(hs, gx,
               highlight_cxs=None,
               cx_clicked_func=None,
               draw_rois=True,
               fnum=1,
               figtitle='Img',
               **kwargs):
    '''Shows an image. cx_clicked_func(cx) is a callback function'''
    gname = hs.tables.gx2_gname[gx]
    img = hs.gx2_image(gx)
    fig, ax = df2.imshow(img, title=gname, fignum=fnum, **kwargs)
    ax = df2.gca()
    if draw_rois:
        if highlight_cxs is None:
            highlight_cxs = []
        _annotate_image(hs, fig, ax, gx, highlight_cxs, cx_clicked_func,
                        draw_rois, **kwargs)
    df2.set_figtitle(figtitle)


def show_splash(fnum=1, **kwargs):
    print('[viz] show_splash()')
    splash_fpath = realpath('_frontend/splash.png')
    img = io.imread(splash_fpath)
    df2.imshow(img, fignum=fnum, **kwargs)


def show_chip_interaction(hs, cx, fnum=2, **kwargs):
    import extract_patch

    # Get chip info (make sure get_chip is called first)
    rchip = hs.get_chip(cx)
    #cidstr = hs.cidstr(cx)
    #name  = hs.cx2_name(cx)
    #gname = hs.cx2_gname(cx)

    fig = df2.figure(fignum=fnum)

    def select_ith_keypoint(fx):
        print('-------------------------------------------')
        print('[interact] viewing ith=%r keypoint' % fx)
        # Get the fx-th keypiont
        kpts = hs.get_kpts(cx)
        desc = hs.get_desc(cx)

        kp = kpts[fx]
        scale = np.sqrt(kp[2] * kp[4])
        sift = desc[fx]
        # Draw the image with keypoint fx highlighted
        fig = df2.figure(fignum=fnum)
        df2.cla()
        ell_args = {'ell_alpha': .4, 'ell_linewidth': 1.8, 'rect': False}
        # Draw chip + keypoints
        show_chip(hs, cx=cx, rchip=rchip, kpts=kpts, plotnum=(2, 1, 1),
                  fignum=fnum, ell_args=ell_args)
        # Draw highlighted point
        df2.draw_kpts2(kpts[fx:fx + 1], ell_color=df2.BLUE, **ell_args)

        # Feature strings
        xy_str = 'xy=(%.1f, %.1f)' % (kp[0], kp[1],)
        acd_str = '[(%3.1f,  0.00),\n' % (kp[2],)
        acd_str += ' (%3.1f, %3.1f)]' % (kp[3], kp[4],)

        # Draw the unwarped selected feature
        extract_patch.draw_keypoint_patch(rchip, kp, sift, plotnum=(2, 3, 4))
        ax = df2.gca()
        ax._hs_viewtype = 'unwarped'
        ax.set_title('affine feature inv(A) =')
        ax.set_xlabel(acd_str)

        # Draw the warped selected feature
        extract_patch.draw_keypoint_patch(rchip, kp, sift, warped=True, plotnum=(2, 3, 5))
        ax = df2.gca()
        ax._hs_viewtype = 'warped'
        ax.set_title('warped feature')
        ax.set_xlabel('fx=%r scale=%.1f\n%s' % (fx, scale, xy_str))

        df2.figure(fignum=fnum, plotnum=(2, 3, 6))
        ax = df2.gca()
        df2.draw_sift_signature(sift, 'sift gradient orientation histogram')
        ax._hs_viewtype = 'histogram'
        fig.canvas.draw()

    def default_chip_view():
        fig = df2.figure(fignum=fnum)
        fig.clf()
        show_chip(hs, cx=cx, draw_kpts=False)  # Toggle no keypoints view
        fig.canvas.draw()

    def _on_chip_click(event):
        #print('\n===========')
        #print('\n'.join(['%r=%r' % tup for tup in event.__dict__.iteritems()]))
        print('\n[viz] clicked chip')
        if event.xdata is None or event.inaxes is None:
            default_chip_view()
            return  # The click is not in any axis
        #print('---')
        hs_viewtype = event.inaxes.__dict__.get('_hs_viewtype', None)
        #print('hs_viewtype=%r' % hs_viewtype)
        if hs_viewtype != 'chip':
            return  # The click is not in the chip axis
        kpts = hs.get_kpts(cx)
        if len(kpts) == 0:
            print('This chip has no keypoints')
            return
        x, y = event.xdata, event.ydata
        dist = (kpts.T[0] - x) ** 2 + (kpts.T[1] - y) ** 2
        fx = dist.argmin()
        select_ith_keypoint(fx)
    #fx = 1897
    #select_ith_keypoint(fx)
    # Draw without keypoints the first time
    show_chip(hs, cx=cx, draw_kpts=False)
    df2.disconnect_callback(fig, 'button_press_event')
    df2.connect_callback(fig, 'button_press_event', _on_chip_click)


def show_chip(hs, cx=None, allres=None, res=None, info=True, draw_kpts=True,
              nRandKpts=None, kpts_alpha=None, kpts=None, rchip=None,
              ell_alpha=None, ell_color=None, prefix='', ell_args=None, fnum=2, **kwargs):
    if not res is None:
        cx = res.qcx
    if not allres is None:
        res = allres.qcx2_res[cx]
    if rchip is None:
        rchip = hs.get_chip(cx)
    title_str = prefix
    # Add info to title
    if info:
        gname = hs.cx2_gname(cx)
        name = hs.cx2_name(cx)
        ngt_str = hs.num_indexed_gt_str(cx)
        title_str += ', '.join([hs.cidstr(cx), 'name=%r' % name,
                               'gname=%r' % gname, ngt_str, ])
    fnum = kwargs.pop('fnum', fnum)
    fnum = kwargs.pop('fignum', fnum)
    fig, ax = df2.imshow(rchip, title=title_str, fignum=fnum, **kwargs)
    ax._hs_viewtype = 'chip'  # Customize axis
    #if not res is None:
    if not draw_kpts:
        return
    if kpts is None:
        kpts = hs.get_kpts(cx)
    if ell_args is None:
        ell_args = {'offset': (0, 0),
                    'ell_linewidth': 1.5,
                    'ell': True,
                    'pts': False}
    # Draw keypoints with groundtruth information
    if not res is None:
        gt_cxs = hs.get_other_indexed_cxs(cx)
        # Get keypoint indexes

        def stack_unique(fx_list):
            try:
                if len(fx_list) == 0:
                    return np.array([], dtype=int)
                stack_list = np.hstack(fx_list)
                stack_ints = np.array(stack_list, dtype=int)
                unique_ints = np.unique(stack_ints)
                return unique_ints
            except Exception as ex:
                print('Ex: %r' % ex)
                print('fx_list = %r ' % fx_list)
                print('stack_insts = %r' % stack_ints)
                print('unique_ints = %r' % unique_ints)
                print(unique_ints)
                raise
        all_fx = np.arange(len(kpts))
        cx2_fm = res.get_cx2_fm()
        fx_list1 = [fm[:, 0] for fm in cx2_fm]
        fx_list2 = [fm[:, 0] for fm in cx2_fm[gt_cxs]] if len(gt_cxs) > 0 else np.array([])
        matched_fx = stack_unique(fx_list1)
        true_matched_fx = stack_unique(fx_list2)
        noise_fx = np.setdiff1d(all_fx, matched_fx)
        # Print info
        print('[viz.show_chip()] %s has %d keypoints. %d true-matching. %d matching. %d noisy.' %
             (hs.cidstr(cx), len(all_fx), len(true_matched_fx), len(matched_fx), len(noise_fx)))
        # Get keypoints
        kpts_true  = kpts[true_matched_fx]
        kpts_match = kpts[matched_fx, :]
        kpts_noise = kpts[noise_fx, :]
        # Draw keypoints
        legend_tups = []
        ell_alpha = ell_args.pop('ell_alpha', ell_alpha)
        ell_color = ell_args.pop('ell_color', ell_color)

        # helper function taking into acount phantom labels
        def _kpts_helper(kpts_, color, alpha, label):
            df2.draw_kpts2(kpts_, ell_color=color, ell_alpha=alpha, **ell_args)
            phant_ = df2.Circle((0, 0), 1, fc=color)
            legend_tups.append((phant_, label))
        _kpts_helper(kpts_noise,  df2.RED, .1, 'Unverified')
        _kpts_helper(kpts_match, df2.BLUE, .4, 'Verified')
        _kpts_helper(kpts_true, df2.GREEN, .6, 'True Matches')
        #plt.legend(*zip(*legend_tups), framealpha=.2)
    # Just draw boring keypoints
    else:
        kpts_alpha = ell_args.pop('kpts_alpha', kpts_alpha)
        ell_alpha = ell_args.pop('ell_alpha', ell_alpha)
        ell_alpha = kpts_alpha
        if ell_alpha is None:
            ell_alpha = .4
        if not nRandKpts is None:
            nkpts1 = len(kpts)
            fxs1 = np.arange(nkpts1)
            size = nRandKpts
            replace = False
            p = np.ones(nkpts1)
            p = p / p.sum()
            fxs_randsamp = np.random.choice(fxs1, size, replace, p)
            kpts = kpts[fxs_randsamp]
            df2.set_xlabel('displaying %r/%r keypoints' % (nRandKpts, nkpts1))
            # show a random sample of kpts
        if ell_color is None:
            ell_color = df2.RED

        df2.draw_kpts2(kpts, ell_color=ell_color, ell_alpha=ell_alpha, **ell_args)


def show_keypoints(rchip, kpts, fignum=0, title=None, **kwargs):
    df2.imshow(rchip, fignum=fignum, title=title, **kwargs)
    df2.draw_kpts2(kpts)


def show_top(res, hs, figtitle='', **kwargs):
    topN_cxs = res.topN_cxs(hs)
    N = len(topN_cxs)
    figtitle += ('q%s -- TOP %r' % (hs.cidstr(res.qcx), N))
    return _show_res(hs, res, topN_cxs=topN_cxs, figtitle=figtitle,
                     all_kpts=False, **kwargs)


def res_show_analysis(res, hs, fignum=3, figtitle='', show_query=None,
                      annote=None, cx_list=None, query_cfg=None, **kwargs):
        print('[viz] show_analysis()')
        # Do we show the query image
        if show_query is None:
            show_query = not hs.args.noshow_query
        if annote is None:
            annote = hs.prefs.display_cfg.annotations

        # Compare to cx_list instead of using top ranks
        if not cx_list is None:
            print('[viz.analysis] showing a given list of cxs')
            topN_cxs = cx_list
            figtitle = 'comparing to ' + hs.cidstr(topN_cxs) + figtitle
        else:
            print('[viz.analysis] showing topN cxs')
            topN_cxs = res.topN_cxs(hs)
            if len(topN_cxs) == 0:
                warnings.warn('len(topN_cxs) == 0')
                figtitle = 'WARNING: no top scores!' + hs.cidstr(res.qcx)
            else:
                topscore = res.get_cx2_score()[topN_cxs][0]
                figtitle = ('topscore=%r -- q%s' % (topscore, hs.cidstr(res.qcx))) + figtitle

        # Do we show the ground truth?
        if hs.args.noshow_gt:
            print('[viz.analysis] not showing groundtruth')
            showgt_cxs = []
        else:
            # Show the groundtruths not returned in topN_cxs
            print('[viz.analysis] showing missed groundtruth')
            showgt_cxs = hs.get_other_indexed_cxs(res.qcx)
            showgt_cxs = np.setdiff1d(showgt_cxs, topN_cxs)

        N = len(topN_cxs)
        max_nCols = min(5, N)
        return _show_res(hs, res, gt_cxs=showgt_cxs, topN_cxs=topN_cxs,
                         figtitle=figtitle, max_nCols=max_nCols,
                         show_query=show_query, fignum=fignum,
                         annote=annote, query_cfg=query_cfg, **kwargs)


def show_matches_annote_res(res, hs, cx,
                            fignum=None,
                            plotnum=None,
                            title_aug=None,
                            **kwargs):
    '''
    Wrapper for show_matches_annote
    '''
    qcx = res.qcx
    cx2_score = res.get_cx2_score()
    cx2_fm    = res.get_cx2_fm()
    cx2_fs    = res.get_cx2_fs()
    title_suff = None
    return show_matches_annote(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx, fignum,
                               plotnum, title_aug, title_suff, **kwargs)


def show_matches_annote(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx, fignum=None,
                        plotnum=None, title_pref=None, title_suff=None,
                        show_cx=False, show_cid=True, show_gname=False,
                        show_name=True, showTF=True, showScore=True, **kwargs):
    fignum = kwargs.pop('fnum', fignum)
    ' Shows matches with annote -ations '
    #print('[viz.show_matches_annote()] Showing matches from %s' % (hs.vs_str(cx, qcx)))
    if np.isnan(cx):
        nan_img = np.zeros((100, 100), dtype=np.uint8)
        title = '(qx%r v NAN)' % (qcx)
        df2.imshow(nan_img, fignum=fignum, plotnum=plotnum, title=title)
        return
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    kpts1, kpts2   = hs.get_kpts([qcx, cx])
    score = cx2_score[cx]
    fm = cx2_fm[cx]
    fs = cx2_fs[cx]
    # Build the title string
    UNKNOWN_STR = '???'
    TRUE_STR    = 'TRUE'
    FALSE_STR   = 'FALSE'

    def is_true_match_str(qcx, cx):
        is_true, is_unknown = hs.is_true_match(qcx, cx)
        if is_unknown:
            return UNKNOWN_STR
        elif is_true:
            return TRUE_STR
        else:
            return FALSE_STR

    isgt_str  = is_true_match_str(qcx, cx)
    title = ''
    if showTF:
        title += '*' + isgt_str + '*'
    if showScore:
        score_str = (' score=' + helpers.num_fmt(score)) % (score)
        title += score_str
    if not title_pref is None:
        title = title_pref + title
    if not title_suff is None:
        title = title + title_suff
    # Draw the matches
    qcx_str = 'q' + hs.cidstr(qcx)
    cx_str = hs.cidstr(cx)
    fig, ax,  woff, hoff = df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm,
                                             fs, fignum=fignum, plotnum=plotnum,
                                             lbl1=qcx_str, lbl2=cx_str,
                                             title=title, **kwargs)
    offset = (woff, hoff)
    #df2.upperright_text(qcx_str)
    #df2.upperright_text(cx_str, offset=offset)
    #df2.lowerright_text(cx_str)
    # Finish annote -ations
    if isgt_str == UNKNOWN_STR:
        unknown_color = df2.DARK_PURP
        df2.draw_border(ax, unknown_color, 4, offset=offset)
    elif isgt_str == TRUE_STR:
        true_color = (0, 1, 0)
        df2.draw_border(ax, true_color, 4, offset=offset)
    elif isgt_str == FALSE_STR:
        false_color = (1, .2, 0)
        df2.draw_border(ax, false_color, 4, offset=offset)
    xlabel = []
    if show_gname:
        xlabel.append('gname=%r' % hs.cx2_gname(cx))
    if show_name:
        xlabel.append('name=%r' % hs.cx2_name(cx))
    if len(xlabel) > 0:
        df2.set_xlabel(', '.join(xlabel))
    return ax


def _show_res(hs, res, figtitle='', max_nCols=5, topN_cxs=None, gt_cxs=None,
              show_query=False, all_kpts=False, annote=True, query_cfg=None,
              split_plots=False, interact=True, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    #print('[viz._show_res()] %r ' % locals())
    fignum = kwargs.pop('fignum', 3)
    fignum = kwargs.pop('fnum', 3)
    #print('========================')
    #print('[viz] Show chip matches:')
    if topN_cxs is None:
        topN_cxs = []
    if gt_cxs is None:
        gt_cxs = []
    qcx = res.qcx
    all_gts = hs.get_other_indexed_cxs(qcx)
    #print('[viz._show_res()]----------------')
    print('[viz._show_res()] #topN=%r #missed_gts=%r/%r' % (len(topN_cxs),
                                                            len(gt_cxs),
                                                            len(all_gts)))
    #print('[viz._show_res()] * max_nCols=%r' % (max_nCols,))
    #print('[viz._show_res()] * show_query=%r' % (show_query,))
    ranked_cxs = res.topN_cxs(hs, N='all')
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts = nQuerySubplts + (0 if gt_cxs is None else len(gt_cxs))
    nTopNSubplts  = 0 if topN_cxs is None else len(topN_cxs)
    nTopNCols = min(max_nCols, nTopNSubplts)
    nGTCols   = min(max_nCols, nGtSubplts)
    if not split_plots:
        nGTCols = max(nGTCols, nTopNCols)
        nTopNCols = nGTCols
    nGtRows   = 0 if nGTCols == 0 else int(np.ceil(nGtSubplts / nGTCols))
    nTopNRows = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells = nGtRows * nGTCols
    if split_plots:
        nRows = nGtRows
    else:
        nRows = nTopNRows + nGtRows
    # Helper function for drawing matches to one cx

    def _show_matches_fn(cx, orank, plotnum):
        'helper for viz._show_res'
        aug = 'rank=%r\n' % orank
        #printDBG('[viz._show_res()] plotting: %r'  % (plotnum,))
        kwshow  = dict(draw_ell=annote, draw_pts=annote, draw_lines=annote,
                       ell_alpha=.5, all_kpts=all_kpts, **kwargs)
        show_matches_annote_res(res, hs, cx, title_aug=aug, fignum=fignum, plotnum=plotnum, **kwshow)

    def _show_query_fn(plotx_shift, rowcols):
        'helper for viz._show_res'
        #printDBG('[viz._show_res()] Plotting Query:')
        plotx = plotx_shift + 1
        plotnum = (rowcols[0], rowcols[1], plotx)
        #printDBG('[viz._show_res()] plotting: %r' % (plotnum,))
        show_chip(hs, res=res, plotnum=plotnum, draw_kpts=annote, prefix='q', fignum=fignum)

    # Helper to draw many cxs
    def _plot_matches_cxs(cx_list, plotx_shift, rowcols):
        'helper for viz._show_res'
        if cx_list is None:
            return
        for ox, cx in enumerate(cx_list):
            plotx = ox + plotx_shift + 1
            plotnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_cxs == cx)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            _show_matches_fn(cx, orank, plotnum)

    #query_uid = res.query_uid
    #query_uid = re.sub(r'_trainID\([0-9]*,........\)', '', query_uid)
    #query_uid = re.sub(r'_indxID\([0-9]*,........\)', '', query_uid)
    #query_uid = re.sub(r'_dcxs\(........\)', '', query_uid)
    #print('[viz._show_res()] fignum=%r' % fignum)

    fig = df2.figure(fignum=fignum)
    fig.clf()
    df2.plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query:
        _show_query_fn(0, (nRows, nGTCols))
    # Plot Ground Truth
    _plot_matches_cxs(gt_cxs, nQuerySubplts, (nRows, nGTCols))
    # Plot TopN in a new figure
    if split_plots:
        #df2.set_figtitle(figtitle + 'GT', query_uid)
        nRows = nTopNRows
        fig = df2.figure(fignum=fignum + 9000)
        fig.clf()
        df2.plt.subplot(nRows, nTopNCols, 1)
        shift_topN = 0
    else:
        shift_topN = nGtCells
    _plot_matches_cxs(topN_cxs, shift_topN, (nRows, nTopNCols))
    if split_plots:
        pass
        #df2.set_figtitle(figtitle + 'topN', query_uid)
    else:
        pass
        #df2.set_figtitle(figtitle, query_uid)
        df2.set_figtitle(figtitle)

    if interact:
        #printDBG('[viz._show_res()] starting interaction')
        # Create
        def _on_res_click(event):
            'result interaction mpl event callback slot'
            print('\n[viz] clicked result')
            if event.xdata is None:
                return
            _show_res(hs, res, figtitle=figtitle, max_nCols=max_nCols, topN_cxs=topN_cxs,
                      gt_cxs=gt_cxs, show_query=show_query, all_kpts=all_kpts,
                      annote=not annote, query_cfg=query_cfg, split_plots=split_plots,
                      interact=interact, **kwargs)
            fig.canvas.draw()

        df2.disconnect_callback(fig, 'button_press_event')
        if interact:
            df2.connect_callback(fig, 'button_press_event', _on_res_click)
    printDBG('[viz._show_res()] Finished')
    return fig

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('=================================')
    print('[viz] __main__ = vizualizations.py')
    print('=================================')
    import main
    import HotSpotter
    args = main.parse_arguments(db='MOTHERS')
    hs = HotSpotter.HotSpotter(args)
    hs.load(load_all=True)
    cx = helpers.get_arg_after('--cx', type_=int)
    qcx = hs.get_valid_cxs()[0]
    doquery = False
    dochip = True
    if doquery:
        if cx is not None:
            qcx = cx
        res = hs.query(qcx)
        N = 5
        res.show_top(hs, N)
    if dochip:
        show_chip_interaction(hs, qcx, fnum=2)
    df2.update()
    exec(df2.present())
