from __future__ import division, print_function
import __builtin__
import matplotlib
matplotlib.use('Qt4Agg')
import draw_func2 as df2
# Python
import multiprocessing
#import re
import sys
import warnings
# Scientific
import numpy as np
# Hotspotter
import fileio as io
import helpers
import extract_patch
from os.path import realpath


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
    #print(msg)
    pass


def plot_name_of_cx(hs, cx, **kwargs):
    nx = hs.tables.cx2_nx[cx]
    plot_name(hs, nx, hl_cxs=[cx], **kwargs)


def plot_name(hs, nx, nx2_cxs=None, fnum=0, hl_cxs=[], subtitle='',
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
    fig = df2.figure(fnum=fnum, pnum=pnum(0), **kwargs)
    fig.clf()
    for px, cx in enumerate(cxs):
        show_chip(hs, cx=cx, pnum=pnum(px), draw_kpts=annote, kpts_alpha=.2)
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

# Moved to _inprogress/viz.py temporarilly
#------------------------------
#^^^ OLD


def nearest_kp(x, y, kpts):
    dist = (kpts.T[0] - x) ** 2 + (kpts.T[1] - y) ** 2
    fx = dist.argmin()
    return fx, dist[fx]


def _annotate_image(hs, fig, ax, gx, highlight_cxs, cx_clicked_func,
                    draw_roi=True, draw_roi_lbls=True, **kwargs):
    # draw chips in the image
    cx_list = hs.gx2_cxs(gx)
    centers = []
    interact = cx_clicked_func is not None
    # Draw all chip indexes in the image
    for cx in cx_list:
        roi = hs.cx2_roi(cx)
        theta = hs.cx2_theta(cx)
        # Draw the ROI
        roi_lbl = hs.cidstr(cx)
        if cx in highlight_cxs:
            bbox_color = df2.ORANGE * np.array([1, 1, 1, .95])
            lbl_color  = df2.BLACK * np.array([1, 1, 1, .75])
        else:
            dark_alpha = np.array([1, 1, 1, .6])
            bbox_color = df2.DARK_ORANGE * dark_alpha
            lbl_color  = df2.BLACK       * dark_alpha
        df2.draw_roi(ax, roi, roi_lbl, bbox_color, lbl_color, theta=theta)
        # Index the roi centers (for interaction)
        (x, y, w, h) = roi
        xy_center = np.array([x + (w / 2), y + (h / 2)])
        centers.append(xy_center)
    # Put roi centers in numpy array
    centers = np.array(centers)

    # Create callback wrapper
    def _on_image_click(event):
        'Slot for matplotlib event'
        print('[viz] clicked image')
        if event.xdata is None:
            return
        if len(centers) == 0:
            return
        #printDBG('\n'.join(['%r=%r' % tup for tup in event.__dict__.iteritems()]))
        x, y = event.xdata, event.ydata
        # Find ROI center nearest to the clicked point
        dist = (centers.T[0] - x) ** 2 + (centers.T[1] - y) ** 2
        cx = cx_list[dist.argsort()[0]]
        cx_clicked_func(cx)

    if interact:
        df2.connect_callback(fig, 'button_press_event', _on_image_click)


#def start_image_interaction(hs, gx, cx_clicked_func):


def show_image(hs, gx, highlight_cxs=None, cx_clicked_func=None, draw_rois=True,
               fnum=1, figtitle='Img', **kwargs):
    '''Shows an image. cx_clicked_func(cx) is a callback function'''
    gname = hs.tables.gx2_gname[gx]
    img = hs.gx2_image(gx)
    fig, ax = df2.imshow(img, title=gname, fnum=fnum, **kwargs)
    ax = df2.gca()
    df2.disconnect_callback(fig, 'button_press_event', axes=[ax])
    if draw_rois:
        if highlight_cxs is None:
            highlight_cxs = []
        _annotate_image(hs, fig, ax, gx, highlight_cxs, cx_clicked_func,
                        draw_rois, **kwargs)
    df2.set_figtitle(figtitle)


def show_splash(fnum=1, **kwargs):
    #printDBG('[viz] show_splash()')
    splash_fpath = realpath('_frontend/splash.png')
    img = io.imread(splash_fpath)
    df2.imshow(img, fnum=fnum, **kwargs)


# CHIP INTERACTION

def show_chip_interaction(hs, cx, fnum=2, figtitle=None, **kwargs):

    # Get chip info (make sure get_chip is called first)
    rchip = hs.get_chip(cx)
    #cidstr = hs.cidstr(cx)
    #name  = hs.cx2_name(cx)
    #gname = hs.cx2_gname(cx)
    fig = df2.figure(fnum=fnum)
    df2.disconnect_callback(fig, 'button_press_event')

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
        df2.figure(fnum=fnum)
        df2.cla()
        ell_args = {'ell_alpha': .4, 'ell_linewidth': 1.8}
        # Draw chip + keypoints
        show_chip(hs, cx=cx, rchip=rchip, kpts=kpts, pnum=(2, 1, 1),
                  fnum=fnum, ell_args=ell_args)
        # Draw highlighted point
        df2.draw_kpts2(kpts[fx:fx + 1], ell_color=df2.BLUE, rect=True, **ell_args)

        # Feature strings
        xy_str   = 'xy=(%.1f, %.1f)' % (kp[0], kp[1],)
        acd_str  = '[(%3.1f,  0.00),\n' % (kp[2],)
        acd_str += ' (%3.1f, %3.1f)]' % (kp[3], kp[4],)

        # Draw the unwarped selected feature
        ax = extract_patch.draw_keypoint_patch(rchip, kp, sift, pnum=(2, 3, 4))
        ax._hs_viewtype = 'unwarped'
        ax.set_title('affine feature inv(A) =')
        ax.set_xlabel(acd_str)

        # Draw the warped selected feature
        ax = extract_patch.draw_keypoint_patch(rchip, kp, sift, warped=True, pnum=(2, 3, 5))
        ax._hs_viewtype = 'warped'
        ax.set_title('warped feature')
        ax.set_xlabel('fx=%r scale=%.1f\n%s' % (fx, scale, xy_str))

        df2.figure(fnum=fnum, pnum=(2, 3, 6))
        ax = df2.gca()
        df2.plot_sift_signature(sift, 'sift gradient orientation histogram')
        ax._hs_viewtype = 'histogram'
        #fig.canvas.draw()
        df2.draw()

    def default_chip_view():
        fig = df2.figure(fnum=fnum)
        fig.clf()
        show_chip(hs, cx=cx, draw_kpts=False)  # Toggle no keypoints view
        fig.canvas.draw()

    def _on_chip_click(event):
        #print('\n===========')
        #print('\n'.join(['%r=%r' % tup for tup in event.__dict__.iteritems()]))
        print('[viz] clicked chip')
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
        fx = nearest_kp(x, y, kpts)[0]
        select_ith_keypoint(fx)
    #fx = 1897
    #select_ith_keypoint(fx)
    # Draw without keypoints the first time
    show_chip(hs, cx=cx, draw_kpts=False)
    if figtitle is not None:
        df2.set_figtitle(figtitle)
    df2.connect_callback(fig, 'button_press_event', _on_chip_click)


def show_chip(hs, cx=None, allres=None, res=None, info=True, draw_kpts=True,
              nRandKpts=None, kpts_alpha=None, kpts=None, rchip=None,
              ell_alpha=None, ell_color=None, prefix='', ell_args=None, **kwargs):
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
    fig, ax = df2.imshow(rchip, title=title_str, **kwargs)
    ax._hs_viewtype = 'chip'  # Customize axis
    #if not res is None:
    if not draw_kpts:
        return
    kpts = kwargs.get('kpts', None)
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


def show_keypoints(rchip, kpts, fnum=0, title=None, **kwargs):
    df2.imshow(rchip, fnum=fnum, title=title, **kwargs)
    df2.draw_kpts2(kpts)


def show_top(res, hs, **kwargs):
    topN_cxs = res.topN_cxs(hs)
    N = len(topN_cxs)
    cxstr = hs.cidstr(res.qcx)
    figtitle = kwargs.pop('figtitle', 'q%s -- TOP %r' % (cxstr, N))
    max_nCols = min(5, N)
    return _show_res(hs, res, topN_cxs=topN_cxs, figtitle=figtitle,
                     max_nCols=max_nCols, draw_kpts=False,
                     draw_ell=False, draw_pts=True,
                     all_kpts=False, **kwargs)


def res_show_analysis(res, hs, **kwargs):
        print('[viz] res.show_analysis()')
        # Parse arguments
        noshow_gt  = kwargs.pop('noshow_gt', hs.args.noshow_gt)
        show_query = kwargs.pop('show_query', hs.args.noshow_query)
        cx_list    = kwargs.pop('cx_list', None)
        figtitle   = kwargs.pop('figtitle', None)

        # Debug printing
        #print('[viz.analysis] noshow_gt  = %r' % noshow_gt)
        #print('[viz.analysis] show_query = %r' % show_query)
        #print('[viz.analysis] cx_list    = %r' % cx_list)

        # Compare to cx_list instead of using top ranks
        if cx_list is None:
            print('[viz.analysis] showing topN cxs')
            topN_cxs = res.topN_cxs(hs)
            if figtitle is None:
                if len(topN_cxs) == 0:
                    warnings.warn('len(topN_cxs) == 0')
                    figtitle = 'WARNING: no top scores!' + hs.cidstr(res.qcx)
                else:
                    topscore = res.get_cx2_score()[topN_cxs][0]
                    figtitle = ('topscore=%r -- q%s' % (topscore, hs.cidstr(res.qcx)))
        else:
            print('[viz.analysis] showing a given list of cxs')
            topN_cxs = cx_list
            if figtitle is None:
                figtitle = 'comparing to ' + hs.cidstr(topN_cxs) + figtitle

        # Do we show the ground truth?
        def missed_cxs():
            showgt_cxs = hs.get_other_indexed_cxs(res.qcx)
            return np.setdiff1d(showgt_cxs, topN_cxs)
        showgt_cxs = [] if noshow_gt else missed_cxs()

        N = len(topN_cxs)
        max_nCols = min(5, N)
        return _show_res(hs, res, gt_cxs=showgt_cxs, topN_cxs=topN_cxs,
                         figtitle=figtitle, max_nCols=max_nCols,
                         show_query=show_query, **kwargs)


def res_show_chipres(res, hs, cx, **kwargs):
    'Wrapper for show_chipres(show annotated chip match result) '
    qcx = res.qcx
    cx2_score = res.get_cx2_score()
    cx2_fm    = res.get_cx2_fm()
    cx2_fs    = res.get_cx2_fs()
    cx2_fk    = res.get_cx2_fk()
    return show_chipres(hs, qcx, cx, cx2_score, cx2_fm, cx2_fs, cx2_fk,
                        **kwargs)


def show_chipres(hs, qcx, cx, cx2_score, cx2_fm, cx2_fs, cx2_fk, **kwargs):
    'shows single annotated match result.'
    fnum = kwargs.pop('fnum', None)
    pnum = kwargs.pop('pnum', None)
    #printDBG('[viz.show_chipres()] Showing matches from %s' % (vs_str))
    #printDBG('[viz.show_chipres()] fnum=%r, pnum=%r' % (fnum, pnum))
    # Test valid cx
    if np.isnan(cx):
        nan_img = np.zeros((32, 32), dtype=np.uint8)
        title = '(q%s v %r)' % (hs.cidstr(qcx), cx)
        df2.imshow(nan_img, fnum=fnum, pnum=pnum, title=title)
        return
    score = cx2_score[cx]
    fm = cx2_fm[cx]
    fs = cx2_fs[cx]
    #fk = cx2_fk[cx]
    vs_str = hs.vs_str(qcx, cx)
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    kpts1, kpts2   = hs.get_kpts([qcx, cx])
    # Build annotation strings / colors
    lbl1 = 'q' + hs.cidstr(qcx)
    lbl2 = hs.cidstr(cx)
    #(truestr, falsestr, nonamestr) = ('SameName', 'DiffName', 'NoName')
    (truestr, falsestr, nonamestr) = ('TRUE', 'FALSE', '???')
    is_true, is_unknown = hs.is_true_match(qcx, cx)
    isgt_str = nonamestr if is_unknown else (truestr if is_true else falsestr)
    match_color = {nonamestr: df2.UNKNOWN_PURP,
                   truestr:   df2.TRUE_GREEN,
                   falsestr:  df2.FALSE_RED}[isgt_str]
    # Build title
    title = '*%s*' % isgt_str if kwargs.get('showTF', True) else ''
    if kwargs.get('showScore', True):
        score_str = (' score=' + helpers.num_fmt(score)) % (score)
        title += score_str
    if 'title_pref' in kwargs:
        title = kwargs['title_pref'] + str(title)
    if 'title_suff' in kwargs:
        title = str(title) + kwargs['title_suff']
    # Build xlabel
    xlabel_ = []
    if kwargs.get('show_gname', False):
        xlabel_.append('gname=%r' % hs.cx2_gname(cx))
    if kwargs.get('show_name', True):
        xlabel_.append('name=%r' % hs.cx2_name(cx))
    xlabel = ', '.join(xlabel_)

    # Draws the chips and keypoint matches
    scm2 = df2.show_chipmatch2
    kwargs_ = dict(fs=fs, lbl1=lbl1, lbl2=lbl2, title=title, fnum=fnum,
                   pnum=pnum, vert=hs.prefs.display_cfg.vert, **kwargs)
    ax, xywh1, xywh2 = scm2(rchip1, rchip2, kpts1, kpts2, fm, **kwargs_)
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    offset2 = (x2, y2)
    df2.draw_border(ax, match_color, 4, offset=offset2)
    df2.set_xlabel(xlabel)
    ax._hs_viewtype = 'chipres %s' % vs_str
    return ax, xywh1, xywh2


def interact_chipres(hs, res, cx, fnum=4, **kwargs):
    'Interacts with a single chipres'
    print('[viz] interact_chipres()')
    # Initialize interaction by drawing matches
    qcx = res.qcx
    fig = df2.figure(fnum=fnum, doclf=True, trueclf=True)
    df2.disconnect_callback(fig, 'button_press_event')
    ax, xywh1, xywh2 = res.show_chipres(hs, cx, fnum=fnum, pnum=(1, 1, 1), **kwargs)
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    kpts1, kpts2   = hs.get_kpts([qcx, cx])
    desc1, desc2   = hs.get_desc([qcx, cx])
    fm = res.cx2_fm[cx]

    # Define interaction functions
    def _select_fm(mx):
        print('\n[viz] view feature match mx=%r' % mx)
        # Get the mx-th feature match
        fx1, fx2 = fm[mx]
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1), (rchip2, kp2, sift2)]
        nRows = len(extracted_list) + 1
        # Draw chips + feature matches
        df2.figure(fnum=fnum, pnum=(nRows, 1, 1), doclf=True, trueclf=True)
        pnum1 = (nRows, 1, 1)
        ax, xywh1, xywh2 = res.show_chipres(hs, cx, fnum=fnum, pnum=pnum1,
                                            draw_lines=False, ell_alpha=.4,
                                            ell_linewidth=1.8, colors=df2.BLUE,
                                            **kwargs)
        # Draw selected match
        sel_fm = np.array([(fx1, fx2)])
        df2.draw_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, fnum=fnum,
                        pnum=pnum1, draw_lines=False, rect=True, colors=df2.ORANGE)
        # Helper functions
        draw_patch = extract_patch.draw_keypoint_patch
        plot_siftsig = df2.plot_sift_signature
        pnum_ = lambda px: (nRows, 3, px)

        def draw_feat_row(rchip, kp, sift, px):
            #printDBG('[viz] draw_feat_row px=%r' % px)
            # Draw the unwarped selected feature
            ax = draw_patch(rchip, kp, sift, fnum=fnum, pnum=pnum_(px + 1))
            ax._hs_viewtype = 'unwarped'
            # Draw the warped selected feature
            ax = draw_patch(rchip, kp, sift, fnum=fnum, pnum=pnum_(px + 2),
                            warped=True)
            ax._hs_viewtype = 'warped'
            # Draw the SIFT representation
            sigtitle = '' if px != 3 else 'sift gradient orientation histogram'
            ax = plot_siftsig(sift, sigtitle, fnum=fnum, pnum=pnum_(px + 3))
            ax._hs_viewtype = 'histogram'
            return px + 3
        px = 3  # plot offset
        for (rchip, kp, sift) in extracted_list:
            px = draw_feat_row(rchip, kp, sift, px)
        fig.canvas.draw()

    def _svviz(cx):
        printDBG('ctrl+clicked cx=%r' % cx)
        fig = df2.figure(fnum=4, doclf=True, trueclf=True)
        df2.disconnect_callback(fig, 'button_press_event')
        viz_spatial_verification(hs, res.qcx, cx2=cx, fnum=4)
        fig.canvas.draw()

    def _on_chipres_clicked(event):
        printDBG('[viz] clicked chipres')
        (x, y) = (event.xdata, event.ydata)
        # Out of axes click
        if None in [x, y, event.inaxes]:
            return interact_chipres(hs, res, fnum)
        hs_viewtype = event.inaxes.__dict__.get('_hs_viewtype', '')
        printDBG('hs_viewtype=%r' % hs_viewtype)
        # Click in match axes
        if hs_viewtype.find('chipres') == 0:
            # Ctrl-Click
            key = '' if event.key is None else event.key
            print('key = %r' % key)
            if key.find('control') == 0:
                print('[viz] result control clicked')
                return _svviz(cx)
            # Normal Click
            # Select nearest feature match to the click
            kpts1_m = kpts1[fm[:, 0]]
            kpts2_m = kpts2[fm[:, 1]]
            x2, y2, w2, h2 = xywh2
            _mx1, _dist1 = nearest_kp(x, y, kpts1_m)
            _mx2, _dist2 = nearest_kp(x - x2, y - y2, kpts2_m)
            mx = _mx1 if _dist1 < _dist2 else _mx2
            _select_fm(mx)
        elif hs_viewtype.find('warped') == 0:
            printDBG('[viz] clicked warped')
        elif hs_viewtype.find('unwarped') == 0:
            printDBG('[viz] clicked unwarped')
        elif hs_viewtype.find('histogram') == 0:
            printDBG('[viz] clicked hist')
        else:
            printDBG('[viz] what did you click?!')
    printDBG('[viz] Drawing and starting interaction')
    fig = df2.gcf()
    df2.draw()
    df2.connect_callback(fig, 'button_press_event', _on_chipres_clicked)


def _show_res(hs, res, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    #printDBG('[viz._show_res()] %s ' % helpers.printableVal(locals()))
    #printDBG = print
    fnum       = kwargs.get('fnum', 3)
    figtitle   = kwargs.get('figtitle', '')
    topN_cxs   = kwargs.get('topN_cxs', [])
    gt_cxs     = kwargs.get('gt_cxs',   [])
    all_kpts   = kwargs.get('all_kpts', False)
    show_query = kwargs.get('show_query', False)
    max_nCols  = kwargs.get('max_nCols', 5)
    interact   = kwargs.get('interact', True)
    annote     = kwargs.pop('annote', True)  # this is toggled

    printDBG('========================')
    printDBG('[viz._show_res()]----------------')
    all_gts = hs.get_other_indexed_cxs(res.qcx)
    _tup = tuple(map(len, (topN_cxs, gt_cxs, all_gts)))
    print('[viz._show_res()] #topN=%r #missed_gts=%r/%r' % _tup)
    printDBG('[viz._show_res()] * fnum=%r' % (fnum,))
    printDBG('[viz._show_res()] * figtitle=%r' % (figtitle,))
    printDBG('[viz._show_res()] * max_nCols=%r' % (max_nCols,))
    printDBG('[viz._show_res()] * show_query=%r' % (show_query,))
    ranked_cxs = res.topN_cxs(hs, N='all')
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts    = nQuerySubplts + (0 if gt_cxs is None else len(gt_cxs))
    nTopNSubplts  = 0 if topN_cxs is None else len(topN_cxs)
    nTopNCols     = min(max_nCols, nTopNSubplts)
    nGTCols       = min(max_nCols, nGtSubplts)
    nGTCols       = max(nGTCols, nTopNCols)
    nTopNCols     = nGTCols
    nGtRows       = 0 if nGTCols == 0 else int(np.ceil(nGtSubplts / nGTCols))
    nTopNRows     = 0 if nTopNCols == 0 else int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells      = nGtRows * nGTCols
    nRows         = nTopNRows + nGtRows

    # Helpers
    def _show_query_fn(plotx_shift, rowcols):
        'helper for viz._show_res'
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #printDBG('[viz._show_res()] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annote)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['res'] = res
        _kwshow['pnum'] = pnum
        show_chip(hs, **_kwshow)

    def _show_matches_fn(cx, orank, pnum):
        'Helper function for drawing matches to one cx'
        aug = 'rank=%r\n' % orank
        #printDBG('[viz._show_res()] plotting: %r'  % (pnum,))
        _kwshow  = dict(draw_ell=annote, draw_pts=False, draw_lines=annote,
                        ell_alpha=.5, all_kpts=all_kpts)
        _kwshow.update(kwargs)
        _kwshow['fnum'] = fnum
        _kwshow['pnum'] = pnum
        _kwshow['title_aug'] = aug
        res.show_chipres(hs, cx, **_kwshow)

    def _plot_matches_cxs(cx_list, plotx_shift, rowcols):
        'helper for viz._show_res to draw many cxs'
        #printDBG('[viz._show_res()] Plotting Chips %s:' % hs.cidstr(cx_list))
        if cx_list is None:
            return
        for ox, cx in enumerate(cx_list):
            plotx = ox + plotx_shift + 1
            pnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_cxs == cx)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            _show_matches_fn(cx, orank, pnum)

    fig = df2.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), doclf=True, trueclf=True)
    df2.disconnect_callback(fig, 'button_press_event')
    df2.plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query:
        _show_query_fn(0, (nRows, nGTCols))
    # Plot Ground Truth
    _plot_matches_cxs(gt_cxs, nQuerySubplts, (nRows, nGTCols))
    shift_topN = nGtCells
    _plot_matches_cxs(topN_cxs, shift_topN, (nRows, nTopNCols))
    df2.set_figtitle(figtitle)

    # Result Interaction
    if interact:
        printDBG('[viz._show_res()] starting interaction')

        def _ctrlclicked_cx(cx):
            printDBG('ctrl+clicked cx=%r' % cx)
            fig = df2.figure(fnum=4, doclf=True, trueclf=True)
            df2.disconnect_callback(fig, 'button_press_event')
            viz_spatial_verification(hs, res.qcx, cx2=cx, fnum=4)
            fig.canvas.draw()
            df2.bring_to_front(fig)

        def _clicked_cx(cx):
            printDBG('clicked cx=%r' % cx)
            res.interact_chipres(hs, cx, fnum=4)
            fig = df2.gcf()
            fig.canvas.draw()
            df2.bring_to_front(fig)

        def _clicked_none():
            # Toggle if the click is not in any axis
            printDBG('clicked none')
            _show_res(hs, res, annote=not annote, **kwargs)
            fig.canvas.draw()

        def _on_res_click(event):
            'result interaction mpl event callback slot'
            print('[viz] clicked result')
            if event.xdata is None or event.inaxes is None:
                print('clicked outside axes')
                return _clicked_none()
            hs_viewtype = event.inaxes.__dict__.get('_hs_viewtype', '')
            printDBG(event.__dict__)
            printDBG('hs_viewtype=%r' % hs_viewtype)
            # Clicked a specific chipres
            if hs_viewtype.find('chipres') == 0:
                cid = int(hs_viewtype[hs_viewtype.find(' v ') + 3:-1])
                cx  = hs.cid2_cx(cid)
                # Ctrl-Click
                key = '' if event.key is None else event.key
                print('key = %r' % key)
                if key.find('control') == 0:
                    print('[viz] result control clicked')
                    return _ctrlclicked_cx(cx)
                # Left-Click
                else:
                    print('[viz] result clicked')
                    return _clicked_cx(cx)

        df2.connect_callback(fig, 'button_press_event', _on_res_click)
    printDBG('[viz._show_res()] Finished')
    return fig


# ---- TEST FUNCTIONS ---- #
def ensure_fm(hs, cx1, cx2, fm=None, res='db'):
    '''A feature match (fm) is a list of M 2-tuples.
    fm = [(0, 5), (3,2), (11, 12), (4,4)]
    fm[:,0] are keypoint indexes into kpts1
    fm[:,1] are keypoint indexes into kpts2
    '''
    if fm is not None:
        return fm
    print('[viz] ensure_fm()')
    import match_chips3 as mc3
    import QueryResult as qr
    if res == 'db':
        query_args = hs.prefs.query_cfg.flat_dict()
        query_args['sv_on'] = False
        query_args['use_cache'] = False
        # Query without spatial verification to get assigned matches
        print('query_args = %r' % (query_args))
        res = mc3.query_database(hs, cx1, **query_args)
    elif res == 'gt':
        # For testing purposes query_groundtruth is a bit faster than
        # query_database. But there is no reason you cant query_database
        query_args = hs.prefs.query_cfg.flat_dict()
        query_args['sv_on'] = False
        query_args['use_cache'] = False
        print('query_args = %r' % (query_args))
        res = mc3.query_groundtruth(hs, cx1, **query_args)
    assert isinstance(res, qr.QueryResult)
    # Get chip index to feature match
    fm = res.cx2_fm[cx2]
    if len(fm) == 0:
        raise Exception('No feature matches for %s' % hs.vs_str(cx1, cx2))
    print('[viz] len(fm) = %r' % len(fm))
    return fm


def ensure_cx2(hs, cx1, cx2=None):
    if cx2 is not None:
        return cx2
    print('[viz] ensure_cx2()')
    gt_cxs = hs.get_other_indexed_cxs(cx1)  # list of ground truth chip indexes
    if len(gt_cxs) == 0:
        msg = 'q%s has no groundtruth' % hs.cidstr(cx1)
        msg += 'cannot perform tests without groundtruth'
        raise Exception(msg)
    cx2 = gt_cxs[0]  # Pick a ground truth to test against
    print('[viz] cx2 = %r' % cx2)
    return cx2


def viz_spatial_verification(hs, cx1, **kwargs):
    #kwargs = {}
    import helpers
    import spatial_verification2 as sv2
    import cv2
    print('\n======================')
    cx2 = ensure_cx2(hs, cx1, kwargs.pop('cx2', None))
    print('[viz] viz_spatial_verification  %s' % hs.vs_str(cx1, cx2))
    fnum = kwargs.get('fnum', 4)
    fm  = ensure_fm(hs, cx1, cx2, kwargs.pop('fm', None), kwargs.pop('res', 'db'))
    # Get keypoints
    rchip1 = kwargs['rchip1'] if 'rchip1' in kwargs else hs.get_chip(cx1)
    rchip2 = kwargs['rchip2'] if 'rchip1' in kwargs else hs.get_chip(cx2)
    kpts1 = kwargs['kpts1'] if 'kpts1' in kwargs else hs.get_kpts(cx1)
    kpts2 = kwargs['kpts2'] if 'kpts2' in kwargs else hs.get_kpts(cx2)
    dlen_sqrd2 = rchip2.shape[0] ** 2 + rchip2.shape[1] ** 2
    # rchips are in shape = (height, width)
    (h1, w1) = rchip1.shape[0:2]
    (h2, w2) = rchip2.shape[0:2]
    #wh1 = (w1, h1)
    wh2 = (w2, h2)
    #print('[viz.sv] wh1 = %r' % (wh1,))
    #print('[viz.sv] wh2 = %r' % (wh2,))

    # Get affine and homog mapping from rchip1 to rchip2
    xy_thresh = hs.prefs.query_cfg.sv_cfg.xy_thresh
    max_scale = hs.prefs.query_cfg.sv_cfg.scale_thresh_high
    min_scale = hs.prefs.query_cfg.sv_cfg.scale_thresh_low
    homog_args = [kpts1, kpts2, fm, xy_thresh, max_scale, min_scale, dlen_sqrd2, 4]
    try:
        Aff, aff_inliers = sv2.homography_inliers(*homog_args, just_affine=True)
        H, inliers = sv2.homography_inliers(*homog_args, just_affine=False)
    except Exception as ex:
        print('[viz] homog_args = %r' % (homog_args))
        print('[viz] ex = %r' % (ex,))
        raise
    print(helpers.horiz_string(['H = ', str(H)]))
    print(helpers.horiz_string(['Aff = ', str(Aff)]))

    # Transform the chips
    print('warp homog')
    rchip1_Ht = cv2.warpPerspective(rchip1, H, wh2)
    print('warp affine')
    rchip1_At = cv2.warpAffine(rchip1, Aff[0:2, :], wh2)

    rchip2_blendA = np.zeros((h2, w2), dtype=rchip2.dtype)
    rchip2_blendH = np.zeros((h2, w2), dtype=rchip2.dtype)
    rchip2_blendA = rchip2 / 2 + rchip1_At / 2
    rchip2_blendH = rchip2 / 2 + rchip1_Ht / 2

    df2.figure(fnum=fnum, pnum=(3, 4, 1), doclf=True, trueclf=True)

    def _draw_chip(title, chip, px, *args, **kwargs):
        df2.imshow(chip, *args, title=title, fnum=fnum, pnum=(3, 4, px), **kwargs)

    # Draw original matches, affine inliers, and homography inliers
    def _draw_matches(title, fm, px):
        # Helper with common arguments to df2.show_chipmatch2
        dmkwargs = dict(fs=None, title=title, all_kpts=False, draw_lines=True,
                        doclf=True, fnum=fnum, pnum=(3, 3, px))
        df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm, show_nMatches=True, **dmkwargs)

    # Draw the Assigned -> Affine -> Homography matches
    _draw_matches('Assigned matches', fm, 1)
    _draw_matches('Affine inliers', fm[aff_inliers], 2)
    _draw_matches('Homography inliers', fm[inliers], 3)
    # Draw the Affine Transformations
    _draw_chip('Source', rchip1, 5)
    _draw_chip('Affine', rchip1_At, 6)
    _draw_chip('Destination', rchip2, 7)
    _draw_chip('Aff Blend', rchip2_blendA, 8)
    # Draw the Homography Transformation
    _draw_chip('Source', rchip1, 9)
    _draw_chip('Homog', rchip1_Ht, 10)
    _draw_chip('Destination', rchip2, 11)
    _draw_chip('Homog Blend', rchip2_blendH, 12)


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
