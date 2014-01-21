from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off, rrr, profile, printDBG) = \
    __common__.init(__name__, '[viz]', DEBUG=False)
import matplotlib
matplotlib.use('Qt4Agg')
# Python
import multiprocessing
#import re
import warnings
# Scientific
import numpy as np
# Hotspotter
import draw_func2 as df2
import fileio as io
import helpers

#from interaction import interact_keypoints, interact_chipres, interact_chip # NOQA


FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5, name=6)


@profile
def draw():
    df2.adjust_subplots_safe()
    df2.draw()


def register_FNUMS(FNUMS_):
    global FNUMS
    FNUMS = FNUMS_


@profile
def show_descriptors_match_distances(orgres2_distance, fnum=1, db_name='', **kwargs):
    disttype_list = orgres2_distance.itervalues().next().keys()
    orgtype_list = orgres2_distance.keys()
    (nRow, nCol) = len(orgtype_list), len(disttype_list)
    nColors = nRow * nCol
    color_list = df2.distinct_colors(nColors)
    df2.figure(fnum=fnum, docla=True, doclf=True)
    pnum_ = lambda px: (nRow, nCol, px + 1)
    plot_type = helpers.get_arg_after('--plot-type', default='plot')

    # Remember min and max val for each distance type (l1, emd...)
    distkey2_min = {distkey: np.uint64(-1) for distkey in disttype_list}
    distkey2_max = {distkey: 0 for distkey in disttype_list}

    def _distplot(dists, color, label, distkey, plot_type=plot_type):
        data = sorted(dists)
        ax = df2.gca()
        min_ = distkey2_min[distkey]
        max_ = distkey2_max[distkey]
        if plot_type == 'plot':
            df2.plot(data, color=color, label=label)
            #xticks = np.linspace(np.min(data), np.max(data), 3)
            #yticks = np.linspace(0, len(data), 5)
            #ax.set_xticks(xticks)
            #ax.set_yticks(yticks)
            ax.set_ylim(min_, max_)
            ax.set_xlim(0, len(dists))
            ax.set_ylabel('distance')
            ax.set_xlabel('matches indexes (sorted by distance)')
            df2.legend(loc='lower right')
        if plot_type == 'pdf':
            df2.plot_pdf(data, color=color, label=label)
            ax.set_ylabel('pr')
            ax.set_xlabel('distance')
            ax.set_xlim(min_, max_)
            df2.legend(loc='upper right')
        df2.dark_background(ax)
        df2.small_xticks(ax)
        df2.small_yticks(ax)

    px = 0
    for orgkey in orgtype_list:
        for distkey in disttype_list:
            dists = orgres2_distance[orgkey][distkey]
            if len(dists) == 0:
                continue
            min_ = dists.min()
            max_ = dists.max()
            distkey2_min[distkey] = min(distkey2_min[distkey], min_)
            distkey2_max[distkey] = max(distkey2_max[distkey], max_)

    for orgkey in orgtype_list:
        for distkey in disttype_list:
            print(((orgkey, distkey)))
            dists = orgres2_distance[orgkey][distkey]
            df2.figure(fnum=fnum, pnum=pnum_(px))
            color = color_list[px]
            title = distkey + ' ' + orgkey
            label = 'P(%s | %s)' % (distkey, orgkey)
            _distplot(dists, color, label, distkey, **kwargs)
            #ax = df2.gca()
            #ax.set_title(title)
            px += 1

    subtitle = 'the matching distances between sift descriptors'
    title = '(sift) matching distances'
    if db_name != '':
        title = db_name + ' ' + title
    df2.set_figtitle(title, subtitle)
    df2.adjust_subplots_safe()

#=============
# Splash Viz
#=============


def show_splash(fnum=1, **kwargs):
    #printDBG('[viz] show_splash()')
    splash_fpath = io.splash_img_fpath()
    img = io.imread(splash_fpath)
    df2.imshow(img, fnum=fnum, **kwargs)

#=============
# Name Viz
#=============


def show_name_of(hs, cx, **kwargs):
    nx = hs.tables.cx2_nx[cx]
    show_name(hs, nx, sel_cxs=[cx], **kwargs)


def show_name(hs, nx, nx2_cxs=None, fnum=0, sel_cxs=[], subtitle='',
              annote=False, **kwargs):
    print('[viz] show_name nx=%r' % nx)
    nx2_name = hs.tables.nx2_name
    cx2_nx   = hs.tables.cx2_nx
    name = nx2_name[nx]
    if not nx2_cxs is None:
        cxs = nx2_cxs[nx]
    else:
        cxs = np.where(cx2_nx == nx)[0]
    print('[viz] show_name %r' % hs.cidstr(cxs))
    ncxs  = len(cxs)
    #nCols = int(min(np.ceil(np.sqrt(ncxs)), 5))
    nCols = int(min(ncxs, 5))
    nRows = int(np.ceil(ncxs / nCols))
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    #gs2 = gridspec.GridSpec(nRows, nCols)
    pnum = lambda px: (nRows, nCols, px + 1)
    fig = df2.figure(fnum=fnum, pnum=pnum(0), **kwargs)
    fig.clf()
    # Trigger computation of all chips in parallel
    hs.refresh_features(cxs)
    for px, cx in enumerate(cxs):
        show_chip(hs, cx=cx, pnum=pnum(px), draw_ell=annote, kpts_alpha=.2)
        if cx in sel_cxs:
            ax = df2.gca()
            df2.draw_border(ax, df2.GREEN, 4)
        #plot_cx3(hs, cx)
    if isinstance(nx, np.ndarray):
        nx = nx[0]
    if isinstance(name, np.ndarray):
        name = name[0]

    figtitle = 'nx=%r -- name=%r' % (nx, name)
    df2.set_figtitle(figtitle)
    #if not annote:
        #title += ' noannote'
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    #df2.set_figtitle(title, subtitle)


#==========================
# Image Viz
#==========================


@profile
def _annotate_roi(hs, ax, cx, sel_cxs, draw_lbls, annote):
    # Draw an roi around a chip in the image
    roi, theta = hs.cx2_roi(cx), hs.cx2_theta(cx)
    if annote:
        is_sel =  cx in sel_cxs
        label = hs.cx2_name(cx)
        label = hs.cidstr(cx) if label == '____' else label
        label = label if draw_lbls else None
        lbl_alpha  = .75 if is_sel else .6
        bbox_alpha = .95 if is_sel else .6
        lbl_color  = df2.BLACK * lbl_alpha
        bbox_color = (df2.ORANGE if is_sel else df2.DARK_ORANGE) * bbox_alpha
        df2.draw_roi(roi, label, bbox_color, lbl_color, theta=theta, ax=ax)
    # Index the roi centers (for interaction)
    (x, y, w, h) = roi
    xy_center = np.array([x + (w / 2), y + (h / 2)])
    return xy_center


@profile
def _annotate_image(hs, ax, gx, sel_cxs, draw_lbls, annote):
    # draw chips in the image
    cx_list = hs.gx2_cxs(gx)
    centers = []
    # Draw all chip indexes in the image
    for cx in cx_list:
        xy_center = _annotate_roi(hs, ax, cx, sel_cxs, draw_lbls, annote)
        centers.append(xy_center)
    # Put roi centers in the axis
    centers = np.array(centers)
    ax._hs_centers = centers
    ax._hs_cx_list = cx_list


@profile
def show_image(hs, gx, sel_cxs=[], fnum=1, figtitle='Img', annote=True,
               draw_lbls=True, **kwargs):
    # Shows an image with annotations
    gname = hs.tables.gx2_gname[gx]
    title = 'gx=%r gname=%r' % (gx, gname)
    img = hs.gx2_image(gx)
    fig = df2.figure(fnum=fnum, docla=True)
    fig, ax = df2.imshow(img, title=title, fnum=fnum, **kwargs)
    ax._hs_viewtype = 'image'
    _annotate_image(hs, ax, gx, sel_cxs, draw_lbls, annote)
    df2.set_figtitle(figtitle)


#==========================
# Chip Viz
#==========================


@profile
def _annotate_qcx_match_results(hs, res, qcx, kpts):
    '''Draws which keypoints successfully matched'''
    def stack_unique(fx_list):
        # concatenates variable length lists
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

    def _kpts_helper(kpts_, color, alpha, label):
        # helper function taking into acount phantom labels
        df2.draw_kpts2(kpts_, ell_color=color, ell_alpha=alpha)
        df2.phantom_legend_label(label, color)

    gt_cxs = hs.get_other_indexed_cxs(qcx)
    all_fx = np.arange(len(kpts))
    cx2_fm = res.get_cx2_fm()
    fx_list1 = [fm[:, 0] for fm in cx2_fm]
    fx_list2 = [fm[:, 0] for fm in cx2_fm[gt_cxs]] if len(gt_cxs) > 0 else np.array([])
    matched_fx = stack_unique(fx_list1)
    true_matched_fx = stack_unique(fx_list2)
    noise_fx = np.setdiff1d(all_fx, matched_fx)
    # Print info
    tup = (hs.cidstr(qcx), len(all_fx), len(matched_fx), len(true_matched_fx), len(noise_fx))
    print('[viz.show_chip] %s has %d kpts. #Matches: %d, true=%d, noisy=%d.' % tup)
    # Get keypoints
    kpts_true  = kpts[true_matched_fx]
    kpts_match = kpts[matched_fx, :]
    kpts_noise = kpts[noise_fx, :]
    # Draw keypoints
    #ell_alpha = ell_args.pop('ell_alpha', ell_alpha)
    #ell_color = ell_args.pop('ell_color', ell_color)
    _kpts_helper(kpts_noise,  df2.RED, .1, 'Unverified')
    _kpts_helper(kpts_match, df2.BLUE, .4, 'Verified')
    _kpts_helper(kpts_true, df2.GREEN, .6, 'True Matches')


@profile
def _annotate_kpts(kpts, sel_fx, draw_ell, draw_pts, nRandKpts=None):
    ell_args = {
        'ell': draw_ell,
        'pts': draw_pts,
        'ell_alpha': .4,
        'ell_linewidth': 2,
        'ell_color': 'distinct',
    }
    if draw_ell and nRandKpts is not None:
        # show a random sample of kpts
        nkpts1 = len(kpts)
        fxs1 = np.arange(nkpts1)
        size = nRandKpts
        replace = False
        p = np.ones(nkpts1)
        p = p / p.sum()
        fxs_randsamp = np.random.choice(fxs1, size, replace, p)
        kpts = kpts[fxs_randsamp]
        # TODO Fix this. This should not set the xlabel
        df2.set_xlabel('displaying %r/%r keypoints' % (nRandKpts, nkpts1))
    elif draw_ell or draw_pts:
        # draw all keypoints
        if sel_fx is not None:
            ell_args['ell_color'] = df2.BLUE
        df2.draw_kpts2(kpts, **ell_args)
    if sel_fx is not None:
        # Draw selected keypoint
        sel_kpts = kpts[sel_fx:sel_fx + 1]
        df2.draw_kpts2(sel_kpts, ell_color=df2.ORANGE, arrow=True, rect=True)


@profile
def show_chip(hs, cx=None, allres=None, res=None, draw_ell=True,
              draw_pts=False, nRandKpts=None, prefix='', sel_fx=None, **kwargs):
    if allres is not None:
        res = allres.qcx2_res[cx]
    if res is not None:
        cx = res.qcx
    rchip = kwargs['rchip'] if 'rchip' in kwargs else hs.get_chip(cx)
    # Add info to title
    title_list = []
    title_list += [hs.cidstr(cx)]
    # FIXME
    #title_list += ['gname=%r' % hs.cx2_gname(cx)]
    #title_list += ['name=%r'  % hs.cx2_name(cx)]
    #title_list += [hs.num_indexed_gt_str(cx)]
    title_str = prefix + ', '.join(title_list)
    fig, ax = df2.imshow(rchip, title=title_str, **kwargs)
    # Add user data to axis
    ax._hs_viewtype = 'chip'
    ax._hs_cx = cx
    if draw_ell or draw_pts:
        # FIXME
        kpts  = kwargs['kpts']  if 'kpts'  in kwargs else hs.get_kpts(cx)
        if res is not None:
            # Draw keypoints with groundtruth information
            _annotate_qcx_match_results(hs, res, cx, kpts)
        else:
            # Just draw boring keypoints
            _annotate_kpts(kpts, sel_fx, draw_ell, draw_pts, nRandKpts)


@profile
def show_keypoints(rchip, kpts, draw_ell=True, draw_pts=False, sel_fx=None, fnum=0,
                   pnum=None, **kwargs):
    df2.imshow(rchip, fnum=fnum, pnum=pnum, **kwargs)
    _annotate_kpts(kpts, sel_fx, draw_ell, draw_pts)
    ax = df2.gca()
    ax._hs_viewtype = 'keypoints'
    ax._hs_kpts = kpts

#==========================
# ChipRes Viz
#==========================


def res_show_chipres(res, hs, cx, **kwargs):
    'Wrapper for show_chipres(show annotated chip match result) '
    qcx = res.qcx
    cx2_score = res.get_cx2_score()
    cx2_fm    = res.get_cx2_fm()
    cx2_fs    = res.get_cx2_fs()
    cx2_fk    = res.get_cx2_fk()
    return show_chipres(hs, qcx, cx, cx2_score, cx2_fm, cx2_fs, cx2_fk,
                        **kwargs)


def show_chipres(hs, qcx, cx, cx2_score, cx2_fm, cx2_fs, cx2_fk,
                 fnum=None, pnum=None, sel_fm=[], **kwargs):
    'shows single annotated match result.'
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
    #vs_str = hs.vs_str(qcx, cx)
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    kpts1, kpts2   = hs.get_kpts([qcx, cx])
    # Build annotation strings / colors
    lbl1 = 'q' + hs.cidstr(qcx)
    lbl2 = hs.cidstr(cx)
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
    if kwargs.get('time_appart', True):
        xlabel_.append(hs.get_timedelta_str(qcx, cx))
    xlabel = ', '.join(xlabel_)
    # Draws the chips and keypoint matches
    kwargs_ = dict(fs=fs, lbl1=lbl1, lbl2=lbl2, title=title, fnum=fnum,
                   pnum=pnum, vert=hs.prefs.display_cfg.vert)
    kwargs_.update(kwargs)
    ax, xywh1, xywh2 = df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm, **kwargs_)
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    if len(sel_fm) > 0:
        # Draw any selected matches
        _smargs = dict(rect=True, colors=df2.ORANGE)
        df2.draw_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **_smargs)
    offset2 = (x2, y2)
    df2.draw_border(ax, match_color, 4, offset=offset2)
    df2.set_xlabel(xlabel)
    ax._hs_viewtype = 'chipres'
    ax._hs_qcx = qcx
    ax._hs_cx = cx
    return ax, xywh1, xywh2


#==========================
# Result Viz
#==========================


@profile
def show_top(res, hs, *args, **kwargs):
    topN_cxs = res.topN_cxs(hs)
    N = len(topN_cxs)
    cxstr = hs.cidstr(res.qcx)
    figtitle = kwargs.pop('figtitle', 'q%s -- TOP %r' % (cxstr, N))
    max_nCols = min(5, N)
    return _show_res(hs, res, topN_cxs=topN_cxs, figtitle=figtitle,
                     max_nCols=max_nCols, draw_kpts=False, draw_ell=False,
                     all_kpts=False, **kwargs)


@profile
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


@profile
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
    annote     = kwargs.pop('annote', 2)  # this is toggled

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
        _kwshow['draw_ell'] = annote == 1
        _kwshow['draw_lines'] = annote >= 1
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

    fig = df2.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), docla=True, doclf=True)
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
            fnum = FNUMS['special']
            fig = df2.figure(fnum=fnum, docla=True, doclf=True)
            df2.disconnect_callback(fig, 'button_press_event')
            viz_spatial_verification(hs, res.qcx, cx2=cx, fnum=fnum)
            fig.canvas.draw()
            df2.bring_to_front(fig)

        def _clicked_cx(cx):
            printDBG('clicked cx=%r' % cx)
            fnum = FNUMS['inspect']
            res.interact_chipres(hs, cx, fnum=fnum)
            fig = df2.gcf()
            fig.canvas.draw()
            df2.bring_to_front(fig)

        def _clicked_none():
            # Toggle if the click is not in any axis
            printDBG('clicked none')
            #print(kwargs)
            _show_res(hs, res, annote=(annote + 1) % 3, **kwargs)
            fig.canvas.draw()

        def _on_res_click(event):
            'result interaction mpl event callback slot'
            print('[viz] clicked result')
            if event.xdata is None or event.inaxes is None:
                #print('clicked outside axes')
                return _clicked_none()
            ax = event.inaxes
            hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
            printDBG(event.__dict__)
            printDBG('hs_viewtype=%r' % hs_viewtype)
            # Clicked a specific chipres
            if hs_viewtype.find('chipres') == 0:
                cx = ax.__dict__.get('_hs_cx')
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

#==========================#
#  --- TESTING FUNCS ---   #
#==========================#


def ensure_fm(hs, cx1, cx2, fm=None, res='db'):
    '''A feature match (fm) is a list of M 2-tuples.
    fm = [(0, 5), (3,2), (11, 12), (4,4)]
    fm[:,0] are keypoint indexes into kpts1
    fm[:,1] are keypoint indexes into kpts2
    '''
    if fm is not None:
        return fm
    print('[viz] ensure_fm()')
    import QueryResult as qr
    if res == 'db':
        query_args = hs.prefs.query_cfg.flat_dict()
        query_args['sv_on'] = False
        query_args['use_cache'] = False
        # Query without spatial verification to get assigned matches
        print('query_args = %r' % (query_args))
        res = hs.query(cx1, **query_args)
    elif res == 'gt':
        # For testing purposes query_groundtruth is a bit faster than
        # query_database. But there is no reason you cant query_database
        query_args = hs.prefs.query_cfg.flat_dict()
        query_args['sv_on'] = False
        query_args['use_cache'] = False
        print('query_args = %r' % (query_args))
        res = hs.query_groundtruth(cx1, **query_args)
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


def viz_spatial_verification(hs, cx1, figtitle='Spatial Verification View', **kwargs):
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

    df2.figure(fnum=fnum, pnum=(3, 4, 1), docla=True, doclf=True)

    def _draw_chip(title, chip, px, *args, **kwargs):
        df2.imshow(chip, *args, title=title, fnum=fnum, pnum=(3, 4, px), **kwargs)

    # Draw original matches, affine inliers, and homography inliers
    def _draw_matches(title, fm, px):
        # Helper with common arguments to df2.show_chipmatch2
        dmkwargs = dict(fs=None, title=title, all_kpts=False, draw_lines=True,
                        docla=True, fnum=fnum, pnum=(3, 3, px))
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
    df2.set_figtitle(figtitle)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('=================================')
    print('[viz] __main__ = vizualizations.py')
    print('=================================')
    import main
    hs = main.main()
    cx = helpers.get_arg_after('--cx', type_=int)
    qcx = hs.get_valid_cxs()[0]
    if cx is not None:
        qcx = cx
    res = hs.query(qcx)
    res.show_top(hs)
    exec(df2.present())
