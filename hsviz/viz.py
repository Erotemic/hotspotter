from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr, profile, printDBG) = \
    __common__.init(__name__, '[viz]', DEBUG=False)
import matplotlib
matplotlib.use('Qt4Agg')
#import re
import warnings
# Scientific
import numpy as np
# Hotspotter
import draw_func2 as df2
import extract_patch
from hscom import fileio as io
from hscom import helpers

#from interaction import interact_keypoints, interact_chipres, interact_chip # NOQA


FNUMS = dict(image=1, chip=2, res=3, inspect=4, special=5, name=6)

IN_IMAGE_OVERRIDE = helpers.get_arg('--in-image-override', type_=bool, default=None)
SHOW_QUERY_OVERRIDE = helpers.get_arg('--show-query-override', type_=bool, default=None)
NO_LABEL_OVERRIDE = helpers.get_arg('--no-label-override', type_=bool, default=None)


@profile
def draw():
    df2.adjust_subplots_safe()
    df2.draw()


def register_FNUMS(FNUMS_):
    global FNUMS
    FNUMS = FNUMS_


def get_square_row_cols(nSubplots, max_cols=5):
    nCols = int(min(nSubplots, max_cols))
    #nCols = int(min(np.ceil(np.sqrt(ncxs)), 5))
    nRows = int(np.ceil(nSubplots / nCols))
    return nRows, nCols


@profile
def show_descriptors_match_distances(orgres2_distance, fnum=1, db_name='', **kwargs):
    disttype_list = orgres2_distance.itervalues().next().keys()
    orgtype_list = orgres2_distance.keys()
    (nRow, nCol) = len(orgtype_list), len(disttype_list)
    nColors = nRow * nCol
    color_list = df2.distinct_colors(nColors)
    df2.figure(fnum=fnum, docla=True, doclf=True)
    pnum_ = lambda px: (nRow, nCol, px + 1)
    plot_type = helpers.get_arg('--plot-type', default='plot')

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
    nRows, nCols = get_square_row_cols(len(cxs))
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

    figtitle = 'Name View nx=%r name=%r' % (nx, name)
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
def _annotate_qcx_match_results(hs, res, qcx, kpts, cx2_color):
    '''Draws which keypoints successfully matched'''
    #print('[viz] !!! ANNOTATE QCX MATCH RESULTS !!!')

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

    if cx2_color is not None:
        # Show which keypoints match chosen chips with color
        for cx, color in cx2_color.iteritems():
            try:
                qfxs = res.cx2_fm[cx][:, 0]
                kpts_ = np.empty((0, 5)) if len(qfxs) == 0 else kpts[qfxs]
                _kpts_helper(kpts_, color, .4, hs.cidstr(cx))
            except Exception as ex:
                print('qfxs=%r' % qfxs)
                print('kpts.shape=%r' % (kpts.shape,))
                print(ex)
                raise
    else:
        # Show which keypoints match groundtruth, etc...
        gt_cxs = hs.get_other_indexed_cxs(qcx)
        all_fx = np.arange(len(kpts))
        cx2_fm = res.get_cx2_fm()
        fx_list1 = [fm[:, 0] for fm in cx2_fm]
        fx_list2 = [fm[:, 0] for fm in cx2_fm[gt_cxs]] if len(gt_cxs) > 0 else np.array([])
        matched_fx = stack_unique(fx_list1)
        true_matched_fx = stack_unique(fx_list2)
        noise_fx = np.setdiff1d(all_fx, matched_fx)
        # Print info
        #tup = (hs.cidstr(qcx), len(all_fx), len(matched_fx), len(true_matched_fx), len(noise_fx))
        #print('[viz] %s has %d kpts. #Matches: %d, true=%d, noisy=%d.' % tup)
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
def _annotate_kpts(kpts, sel_fx, draw_ell, draw_pts, color=None, nRandKpts=None, rect=False):
    #print('[viz] _annotate_kpts()')
    if color is None:
        color = 'distinct' if sel_fx is None else df2.ORANGE
    ell_args = {
        'ell': draw_ell,
        'pts': draw_pts,
        'rect': rect,
        'ell_alpha': .4,
        'ell_linewidth': 2,
        'ell_color': color,
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
        df2.draw_kpts2(kpts, **ell_args)
    if sel_fx is not None:
        # Draw selected keypoint
        sel_kpts = kpts[sel_fx:sel_fx + 1]
        df2.draw_kpts2(sel_kpts, ell_color=df2.BLUE, arrow=True, rect=True)


@profile
def show_chip(hs, cx=None, allres=None, res=None, draw_ell=True,
              draw_pts=False, nRandKpts=None, prefix='', sel_fx=None,
              color=None, in_image=False, sel_fx2=None, **kwargs):
    printDBG('[viz] show_chip()')
    if allres is not None:
        res = allres.qcx2_res[cx]
    if res is not None:
        cx = res.qcx
    if in_image:
        rchip = hs.cx2_image(cx)
    else:
        rchip = kwargs['rchip'] if 'rchip' in kwargs else hs.get_chip(cx)
    # Add info to title
    title_list = []
    title_list += [hs.cidstr(cx)]
    # FIXME
    #title_list += ['gname=%r' % hs.cx2_gname(cx)]
    title_list += ['name=%r'  % hs.cx2_name(cx)]
    #title_list += [hs.num_indexed_gt_str(cx)]
    if NO_LABEL_OVERRIDE:
        title_str = ''
    else:
        title_str = prefix + ', '.join(title_list)
    fig, ax = df2.imshow(rchip, title=title_str, **kwargs)
    # Add user data to axis
    ax._hs_viewtype = 'chip'
    ax._hs_cx = cx
    if draw_ell or draw_pts:
        # FIXME
        if in_image:
            kpts = cx2_imgkpts(hs, [cx])[0]
        else:
            kpts = kwargs['kpts'] if 'kpts' in kwargs else hs.get_kpts(cx)
        if sel_fx2 is not None:
            sel_fx2 = np.array(sel_fx2)
            kpts = kpts[sel_fx2]
        if res is not None:
            # Draw keypoints with groundtruth information
            cx2_color = kwargs.get('cx2_color', None)
            _annotate_qcx_match_results(hs, res, cx, kpts, cx2_color)
        else:
            # Just draw boring keypoints
            _annotate_kpts(kpts, sel_fx, draw_ell, draw_pts, color, nRandKpts)


@profile
def show_keypoints(rchip, kpts, draw_ell=True, draw_pts=False, sel_fx=None, fnum=0,
                   pnum=None, color=None, rect=False, **kwargs):
    df2.imshow(rchip, fnum=fnum, pnum=pnum, **kwargs)
    _annotate_kpts(kpts, sel_fx, draw_ell, draw_pts, color=color, rect=rect)
    ax = df2.gca()
    ax._hs_viewtype = 'keypoints'
    ax._hs_kpts = kpts

#==========================
# ChipRes Viz
#==========================


# HACK!
def build_transform2(roi, chipsz, theta):
    (x, y, w, h) = roi
    (w_, h_) = chipsz
    sx = (w_ / w)  # ** 2
    sy = (h_ / h)  # ** 2
    cos_ = np.cos(-theta)
    sin_ = np.sin(-theta)
    tx = -(x + (w / 2))
    ty = -(y + (h / 2))

    T1 = np.array([[1, 0, tx],
                   [0, 1, ty],
                   [0, 0, 1]], np.float64)

    S = np.array([[sx, 0,  0],
                  [0, sy,  0],
                  [0,  0,  1]], np.float64)

    R = np.array([[cos_, -sin_, 0],
                  [sin_,  cos_, 0],
                  [   0,     0, 1]], np.float64)

    T2 = np.array([[1, 0, (w_ / 2)],
                   [0, 1, (h_ / 2)],
                   [0, 0, 1]], np.float64)

    M = T2.dot(R.dot(S.dot(T1)))
    return M


# HACK!
def cx2_imgkpts(hs, cx_list):
    roi_list = hs.cx2_roi(cx_list)
    theta_list = hs.cx2_theta(cx_list)
    chipsz_list = hs.cx2_rchip_size(cx_list)
    kpts_list = hs.get_kpts(cx_list)

    imgkpts_list = []
    flatten_xs = np.array([[0, 2], [1, 2], [0, 0], [1, 0], [1, 1]])
    for roi, theta, chipsz, kpts in zip(roi_list, theta_list, chipsz_list, kpts_list):
        # HOLY SHIT THIS IS JANKY
        M = build_transform2(roi, chipsz, theta)
        invA_list = [np.array([[a, 0, x], [c, d, y], [0, 0, 1]]) for (x, y, a, c, d) in kpts]
        invM = np.linalg.inv(M)
        invMinvA_list = [invM.dot(invA) for invA in invA_list]
        flatten_xs = np.array([[0, 2], [1, 2], [0, 0], [1, 0], [1, 1]])
        imgkpts = [[invMinvA[index[0], index[1]] for index in flatten_xs] for invMinvA in invMinvA_list]
        imgkpts_list.append(np.array(imgkpts))
    return imgkpts_list


def res_show_chipres(res, hs, cx, **kwargs):
    'Wrapper for show_chipres(show annotated chip match result) '
    return show_chipres(hs, res, cx, **kwargs)


def show_chipres(hs, res, cx, fnum=None, pnum=None, sel_fm=[], in_image=False, **kwargs):
    'shows single annotated match result.'
    qcx = res.qcx
    #cx2_score = res.get_cx2_score()
    cx2_fm    = res.get_cx2_fm()
    cx2_fs    = res.get_cx2_fs()
    #cx2_fk    = res.get_cx2_fk()
    #printDBG('[viz.show_chipres()] Showing matches from %s' % (vs_str))
    #printDBG('[viz.show_chipres()] fnum=%r, pnum=%r' % (fnum, pnum))
    # Test valid cx
    printDBG('[viz] show_chipres()')
    if np.isnan(cx):
        nan_img = np.zeros((32, 32), dtype=np.uint8)
        title = '(q%s v %r)' % (hs.cidstr(qcx), cx)
        df2.imshow(nan_img, fnum=fnum, pnum=pnum, title=title)
        return
    fm = cx2_fm[cx]
    fs = cx2_fs[cx]
    #fk = cx2_fk[cx]
    #vs_str = hs.vs_str(qcx, cx)
    # Read query and result info (chips, names, ...)
    if in_image:
        # TODO: rectify build_transform2 with cc2
        # clean up so its not abysmal
        rchip1, rchip2 = [hs.cx2_image(_) for _ in [qcx, cx]]
        kpts1, kpts2   = cx2_imgkpts(hs, [qcx, cx])
    else:
        rchip1, rchip2 = hs.get_chip([qcx, cx])
        kpts1, kpts2   = hs.get_kpts([qcx, cx])

    # Build annotation strings / colors
    lbl1 = 'q' + hs.cidstr(qcx)
    lbl2 = hs.cidstr(cx)
    if in_image:
        # HACK!
        lbl1 = None
        lbl2 = None
    # Draws the chips and keypoint matches
    kwargs_ = dict(fs=fs, lbl1=lbl1, lbl2=lbl2, fnum=fnum,
                   pnum=pnum, vert=hs.prefs.display_cfg.vert)
    kwargs_.update(kwargs)
    ax, xywh1, xywh2 = df2.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm, **kwargs_)
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    if len(sel_fm) > 0:
        # Draw any selected matches
        _smargs = dict(rect=True, colors=df2.BLUE)
        df2.draw_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **_smargs)
    offset1 = (x1, y1)
    offset2 = (x2, y2)
    annotate_chipres(hs, res, cx, xywh2=xywh2, in_image=in_image, offset1=offset1, offset2=offset2, **kwargs)
    return ax, xywh1, xywh2


def annotate_chipres(hs, res, cx, showTF=True, showScore=True, title_pref='',
                     title_suff='', show_gname=False, show_name=True,
                     time_appart=True, in_image=False, offset1=(0, 0),
                     offset2=(0, 0), show_query=True, xywh2=None, **kwargs):
    printDBG('[viz] annotate_chipres()')
    #print('Did not expect args: %r' % (kwargs.keys(),))
    qcx = res.qcx
    score = res.cx2_score[cx]
    # TODO Use this function when you clean show_chipres
    (truestr, falsestr, nonamestr) = ('TRUE', 'FALSE', '???')
    is_true, is_unknown = hs.is_true_match(qcx, cx)
    isgt_str = nonamestr if is_unknown else (truestr if is_true else falsestr)
    match_color = {nonamestr: df2.UNKNOWN_PURP,
                   truestr:   df2.TRUE_GREEN,
                   falsestr:  df2.FALSE_RED}[isgt_str]
    # Build title
    title = '*%s*' % isgt_str if showTF else ''
    if showScore:
        score_str = (' score=' + helpers.num_fmt(score)) % (score)
        title += score_str
    title = title_pref + str(title) + title_suff
    # Build xlabel
    xlabel_ = []
    if 'show_gname':
        xlabel_.append('gname=%r' % hs.cx2_gname(cx))
    if 'show_name':
        xlabel_.append('name=%r' % hs.cx2_name(cx))
    if 'time_appart':
        xlabel_.append('\n' + hs.get_timedelta_str(qcx, cx))
    xlabel = ', '.join(xlabel_)
    ax = df2.gca()
    ax._hs_viewtype = 'chipres'
    ax._hs_qcx = qcx
    ax._hs_cx = cx
    if NO_LABEL_OVERRIDE:
        title = ''
        xlabel = ''
    df2.set_title(title, ax)
    df2.set_xlabel(xlabel, ax)
    if in_image:
        roi1 = hs.cx2_roi(qcx) + np.array(list(offset1) + [0, 0])
        roi2 = hs.cx2_roi(cx) + np.array(list(offset2) + [0, 0])
        theta1 = hs.cx2_theta(qcx)
        theta2 = hs.cx2_theta(cx)
        # HACK!
        lbl1 = 'q' + hs.cidstr(qcx)
        lbl2 = hs.cidstr(cx)
        if show_query:
            df2.draw_roi(roi1, bbox_color=df2.ORANGE, label=lbl1, theta=theta1)
        df2.draw_roi(roi2, bbox_color=match_color, label=lbl2, theta=theta2)
        # No matches draw a red box
        if len(res.cx2_fm[cx]) == 0:
            df2.draw_boxedX(roi2, theta=theta2)
    else:
        if xywh2 is None:
            xy, w, h = df2._axis_xy_width_height(ax)
            xywh2 = (xy[0], xy[1], w, h)
        df2.draw_border(ax, match_color, 4, offset=offset2)
        # No matches draw a red box
        if len(res.cx2_fm[cx]) == 0:
            df2.draw_boxedX(xywh2)


#==========================
# Result Viz
#==========================


@profile
def show_top(res, hs, *args, **kwargs):
    topN_cxs = res.topN_cxs(hs)
    N = len(topN_cxs)
    cxstr = hs.cidstr(res.qcx)
    figtitle = kwargs.pop('figtitle', 'q%s -- TOP %r' % (cxstr, N))
    return _show_res(hs, res, topN_cxs=topN_cxs, figtitle=figtitle,
                     draw_kpts=False, draw_ell=False,
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

        return _show_res(hs, res, gt_cxs=showgt_cxs, topN_cxs=topN_cxs,
                         figtitle=figtitle, show_query=show_query, **kwargs)


@profile
def _show_res(hs, res, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    #printDBG('[viz._show_res()] %s ' % helpers.printableVal(locals()))
    #printDBG = print
    in_image = hs.prefs.display_cfg.show_results_in_image
    annote     = kwargs.pop('annote', 2)  # this is toggled
    fnum       = kwargs.get('fnum', 3)
    figtitle   = kwargs.get('figtitle', '')
    topN_cxs   = kwargs.get('topN_cxs', [])
    gt_cxs     = kwargs.get('gt_cxs',   [])
    all_kpts   = kwargs.get('all_kpts', False)
    interact   = kwargs.get('interact', True)
    show_query = kwargs.get('show_query', False)
    dosquare   = kwargs.get('dosquare', False)
    if SHOW_QUERY_OVERRIDE is not None:
        show_query = SHOW_QUERY_OVERRIDE

    max_nCols = 5
    if len(topN_cxs) in [6, 7]:
        max_nCols = 3
    if len(topN_cxs) in [8]:
        max_nCols = 4

    printDBG('========================')
    printDBG('[viz._show_res()]----------------')
    all_gts = hs.get_other_indexed_cxs(res.qcx)
    _tup = tuple(map(len, (topN_cxs, gt_cxs, all_gts)))
    printDBG('[viz._show_res()] #topN=%r #missed_gts=%r/%r' % _tup)
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

    # HACK:
    _color_list = df2.distinct_colors(len(topN_cxs))
    cx2_color = {cx: _color_list[ox] for ox, cx in enumerate(topN_cxs)}

    if IN_IMAGE_OVERRIDE is not None:
        in_image = IN_IMAGE_OVERRIDE

    # Helpers
    def _show_query_fn(plotx_shift, rowcols):
        'helper for viz._show_res'
        plotx = plotx_shift + 1
        pnum = (rowcols[0], rowcols[1], plotx)
        #print('[viz] Plotting Query: pnum=%r' % (pnum,))
        _kwshow = dict(draw_kpts=annote)
        _kwshow.update(kwargs)
        _kwshow['prefix'] = 'q'
        _kwshow['res'] = res
        _kwshow['pnum'] = pnum
        _kwshow['cx2_color'] = cx2_color
        _kwshow['draw_ell'] = annote >= 1
        #_kwshow['in_image'] = in_image
        show_chip(hs, **_kwshow)
        #if in_image:
            #roi1 = hs.cx2_roi(res.qcx)
            #df2.draw_roi(roi1, bbox_color=df2.ORANGE, label='q' + hs.cidstr(res.qcx))

    def _plot_matches_cxs(cx_list, plotx_shift, rowcols):

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
            # If we already are showing the query dont show it here
            if not show_query:
                _kwshow['draw_ell'] = annote == 1
                _kwshow['draw_lines'] = annote >= 1
                res_show_chipres(res, hs, cx, in_image=in_image, **_kwshow)
            else:
                _kwshow['draw_ell'] = annote >= 1
                if annote == 2:
                    # TODO Find a better name
                    _kwshow['color'] = cx2_color[cx]
                    _kwshow['sel_fx2'] = res.cx2_fm[cx][:, 1]
                show_chip(hs, cx, in_image=in_image, **_kwshow)
                annotate_chipres(hs, res, cx, in_image=in_image, show_query=not show_query)

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

    if dosquare:
        # HACK
        nSubplots = nGtSubplts + nTopNSubplts
        nRows, nCols = get_square_row_cols(nSubplots, 3)
        nTopNCols = nGTCols = nCols
        shift_topN = 1
        printDBG('nRows, nCols = (%r, %r)' % (nRows, nCols))
    else:
        shift_topN = nGtCells

    if nGtSubplts == 1:
        nGTCols = 1

    fig = df2.figure(fnum=fnum, pnum=(nRows, nGTCols, 1), docla=True, doclf=True)
    df2.disconnect_callback(fig, 'button_press_event')
    df2.plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query:
        _show_query_fn(0, (nRows, nGTCols))
    # Plot Ground Truth
    _plot_matches_cxs(gt_cxs, nQuerySubplts, (nRows, nGTCols))
    _plot_matches_cxs(topN_cxs, shift_topN, (nRows, nTopNCols))
    figtitle += ' q%s name=%s' % (hs.cidstr(res.qcx), hs.cx2_name(res.qcx))
    df2.set_figtitle(figtitle, incanvas=not NO_LABEL_OVERRIDE)

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
    df2.adjust_subplots_safe()
    printDBG('[viz._show_res()] Finished')
    return fig

#==========================#
#  --- TESTING FUNCS ---   #
#==========================#


def kp_info(kp):
    xy_str   = 'xy=(%.1f, %.1f)' % (kp[0], kp[1],)
    acd_str  = '[(%3.1f,  0.00),\n' % (kp[2],)
    acd_str += ' (%3.1f, %3.1f)]' % (kp[3], kp[4],)
    scale = np.sqrt(kp[2] * kp[4])
    return xy_str, acd_str, scale


def draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, prevsift=None,
                  cx=None, info='', type_=None):
    pnum_ = lambda px: (nRows, nCols, px)

    def _draw_patch(**kwargs):
        return extract_patch.draw_keypoint_patch(rchip, kp, sift, **kwargs)

    # Feature strings
    xy_str, acd_str, scale = kp_info(kp)

    # Draw the unwarped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 1))
    ax._hs_viewtype = 'unwarped'
    ax._hs_cx = cx
    ax._hs_fx = fx
    unwarped_lbl = 'affine feature inv(A) =\n' + acd_str
    df2.set_xlabel(unwarped_lbl, ax)

    # Draw the warped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 2), warped=True)
    ax._hs_viewtype = 'warped'
    ax._hs_cx = cx
    ax._hs_fx = fx
    warped_lbl = ('warped feature\n' +
                  'fx=%r scale=%.1f\n' +
                  '%s' + info) % (fx, scale, xy_str)
    df2.set_xlabel(warped_lbl, ax)

    border_color = {None: None,
                    'query': None,
                    'match': df2.BLUE,
                    'norm': df2.ORANGE}[type_]
    if border_color is not None:
        df2.draw_border(ax, color=border_color)

    # Draw the SIFT representation
    sigtitle = '' if px != 3 else 'sift histogram'
    ax = df2.plot_sift_signature(sift, sigtitle, fnum=fnum, pnum=pnum_(px + 3))
    ax._hs_viewtype = 'histogram'
    if prevsift is not None:
        from hotspotter import algos
        dist_list = ['L1', 'L2', 'hist_isect', 'emd']
        distmap = algos.compute_distances(sift, prevsift, dist_list)
        dist_str = ', '.join(['(%s, %.1E)' % (key, val) for key, val in distmap.iteritems()])
        df2.set_xlabel(dist_str)
    return px + nCols

#----


def show_nearest_descriptors(hs, qcx, qfx, fnum=None):
    if fnum is None:
        fnum = df2.next_fnum()
    # Inspect the nearest neighbors of a descriptor
    dx2_cx = hs.qdat._data_index.ax2_cx
    dx2_fx = hs.qdat._data_index.ax2_fx
    K      = hs.qdat.cfg.nn_cfg.K
    Knorm  = hs.qdat.cfg.nn_cfg.Knorm
    checks = hs.qdat.cfg.nn_cfg.checks
    flann  = hs.qdat._data_index.flann
    qfx2_desc = hs.get_desc(qcx)[qfx:qfx + 1]

    try:
        (qfx2_dx, qfx2_dist) = flann.nn_index(qfx2_desc, K + Knorm, checks=checks)
        qfx2_cx = dx2_cx[qfx2_dx]
        qfx2_fx = dx2_fx[qfx2_dx]

        def get_extract_tuple(cx, fx, k=-1):
            rchip = hs.get_chip(cx)
            kp    = hs.get_kpts(cx)[fx]
            sift  = hs.get_desc(cx)[fx]
            if k == -1:
                info = '\nquery %s, fx=%r' % (hs.cidstr(cx), fx)
                type_ = 'query'
            elif k < K:
                type_ = 'match'
                info = '\nmatch %s, fx=%r k=%r, dist=%r' % (hs.cidstr(cx), fx, k, qfx2_dist[0, k])
            elif k < Knorm + K:
                type_ = 'norm'
                info = '\nnorm  %s, fx=%r k=%r, dist=%r' % (hs.cidstr(cx), fx, k, qfx2_dist[0, k])
            else:
                raise Exception('[viz] problem k=%r')
            return (rchip, kp, sift, fx, cx, info, type_)

        extracted_list = []
        extracted_list.append(get_extract_tuple(qcx, qfx, -1))
        for k in xrange(K + Knorm):
            tup = get_extract_tuple(qfx2_cx[0, k], qfx2_fx[0, k], k)
            extracted_list.append(tup)
        #print('[viz] K + Knorm = %r' % (K + Knorm))

        # Draw the _select_ith_match plot
        nRows, nCols = len(extracted_list), 3
        # Draw selected feature matches
        prevsift = None
        df2.figure(fnum=fnum, docla=True, doclf=True)
        px = 0  # plot offset
        for (rchip, kp, sift, fx, cx, info, type_) in extracted_list:
            print('[viz] ' + info.replace('\n', ''))
            px = draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px,
                               prevsift=prevsift, cx=cx, info=info, type_=type_)
            prevsift = sift

        df2.adjust_subplots_safe(hspace=1)

    except Exception as ex:
        print('[viz] Error in show nearest descriptors')
        print(ex)
        raise


#----

def ensure_fm(hs, cx1, cx2, fm=None, res='db'):
    '''A feature match (fm) is a list of M 2-tuples.
    fm = [(0, 5), (3,2), (11, 12), (4,4)]
    fm[:,0] are keypoint indexes into kpts1
    fm[:,1] are keypoint indexes into kpts2
    '''
    if fm is not None:
        return fm
    print('[viz] ensure_fm()')
    from hotspotter import QueryResult as qr
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
    from hscom import helpers
    from hotspotter import spatial_verification2 as sv2
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

    rchip2_blendA = np.zeros(rchip2.shape, dtype=rchip2.dtype)
    rchip2_blendH = np.zeros(rchip2.shape, dtype=rchip2.dtype)
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
