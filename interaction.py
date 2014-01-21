from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off, rrr, profile, printDBG)\
    = __common__.init(__name__, '[inter]', DEBUG=False)
# Python
import multiprocessing
# Scientific
import numpy as np
# Hotspotter
import draw_func2 as df2
import extract_patch
import vizualizations as viz
import helpers
from _tpl import mask_creator


# RCOS TODO: We should change the fnum, pnum figure layout into one managed by
# gridspec.

#==========================
# HELPERS
#==========================


def nearest_point(x, y, pts):
    dists = (pts.T[0] - x) ** 2 + (pts.T[1] - y) ** 2
    fx = dists.argmin()
    mindist = dists[fx]
    other_fx = np.where(mindist == dists)[0]
    if len(other_fx > 0):
        np.random.shuffle(other_fx)
        fx = other_fx[0]
    return fx, mindist


def kp_info(kp):
    xy_str   = 'xy=(%.1f, %.1f)' % (kp[0], kp[1],)
    acd_str  = '[(%3.1f,  0.00),\n' % (kp[2],)
    acd_str += ' (%3.1f, %3.1f)]' % (kp[3], kp[4],)
    scale = np.sqrt(kp[2] * kp[4])
    return xy_str, acd_str, scale


def detect_keypress(fig):
    def on_key_press(event):
        if event.key == 'shift':
            shift_is_held = True  # NOQA

    def on_key_release(event):
        if event.key == 'shift':
            shift_is_held = False  # NOQA
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)


def draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, prevsift=None):
    pnum_ = lambda px: (nRows, nCols, px)

    def _draw_patch(**kwargs):
        return extract_patch.draw_keypoint_patch(rchip, kp, sift, **kwargs)

    # Feature strings
    xy_str, acd_str, scale = kp_info(kp)

    # Draw the unwarped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 1))
    ax._hs_viewtype = 'unwarped'
    unwarped_lbl = 'affine feature inv(A) =\n' + acd_str
    df2.set_xlabel(unwarped_lbl, ax)

    # Draw the warped selected feature
    ax = _draw_patch(fnum=fnum, pnum=pnum_(px + 2), warped=True)
    ax._hs_viewtype = 'warped'
    warped_lbl = ('warped feature\n' + 'fx=%r scale=%.1f\n' + '%s') % (fx, scale, xy_str)
    df2.set_xlabel(warped_lbl, ax)

    # Draw the SIFT representation
    sigtitle = '' if px != 3 else 'sift histogram'
    ax = df2.plot_sift_signature(sift, sigtitle, fnum=fnum, pnum=pnum_(px + 3))
    ax._hs_viewtype = 'histogram'
    if prevsift is not None:
        import algos
        dist_list = ['L1', 'L2', 'hist_isect', 'emd']
        distmap = algos.compute_distances(sift, prevsift, dist_list)
        dist_str = ', '.join(['%s:%.1e' % (key, val) for key, val in distmap.iteritems()])
        df2.set_xlabel(dist_str)
    return px + nCols


def begin_interaction(type_, fnum):
    print('[inter] starting %s interaction' % type_)
    fig = df2.figure(fnum=fnum, docla=True, doclf=True)
    ax = df2.gca()
    df2.disconnect_callback(fig, 'button_press_event', axes=[ax])
    return fig


#==========================
# Image Interaction
#==========================

@profile
def interact_image(hs, gx, sel_cxs=[], select_cx_func=None, fnum=1, **kwargs):
    fig = begin_interaction('image', fnum)

    # Create callback wrapper
    @profile
    def _on_image_click(event):
        print_('[inter] clicked image')
        if event is None or event.inaxes is None or event.xdata is None:
            # Toggle draw lbls
            print(' ...out of axis')
            kwargs['draw_lbls'] = not kwargs.pop('draw_lbls', True)
            interact_image(hs, gx, sel_cxs=sel_cxs,
                           select_cx_func=select_cx_func, **kwargs)
        else:
            ax = event.inaxes
            hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
            print_(' hs_viewtype=%r' % hs_viewtype)
            centers = ax.__dict__.get('_hs_centers')
            if len(centers) == 0:
                print(' ...no chips to click')
                return
            x, y = event.xdata, event.ydata
            # Find ROI center nearest to the clicked point
            cx_list = ax._hs_cx_list
            centers = ax._hs_centers
            centx = nearest_point(x, y, centers)[0]
            cx = cx_list[centx]
            print(' ...clicked cx=%r' % cx)
            if select_cx_func is not None:
                select_cx_func(cx)
        viz.draw()

    viz.show_image(hs, gx, sel_cxs, **kwargs)
    viz.draw()
    df2.connect_callback(fig, 'button_press_event', _on_image_click)


#==========================
# Name Interaction
#==========================

@profile
def interact_name(hs, nx, sel_cxs=[], select_cx_func=None, fnum=5, **kwargs):
    fig = begin_interaction('name', fnum)

    def _on_name_click(event):
        print_('[inter] clicked name')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ax is None or x is None:
            # The click is not in any axis
            print('... out of axis')
        hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
        print_(' hs_viewtype=%r' % hs_viewtype)
        if hs_viewtype == 'chip':
            cx = ax.__dict__.get('_hs_cx')
            print('... cx=%r' % cx)
            viz.show_name(hs, nx, fnum=fnum, sel_cxs=[cx])
            select_cx_func(cx)
        viz.draw()

    viz.show_name(hs, nx, fnum=fnum, sel_cxs=sel_cxs)
    viz.draw()
    df2.connect_callback(fig, 'button_press_event', _on_name_click)
    pass


#==========================
# Chip Interaction
#==========================


# CHIP INTERACTION 2
@profile
def interact_chip(hs, cx, fnum=2, figtitle=None, **kwargs):
    fig = begin_interaction('chip', fnum)
    # Get chip info (make sure get_chip is called first)
    rchip = hs.get_chip(cx)
    annote_ptr = [False]

    def _select_ith_kpt(fx):
        # Get the fx-th keypiont
        kpts = hs.get_kpts(cx)
        desc = hs.get_desc(cx)
        kp, sift = kpts[fx], desc[fx]
        # Draw chip + keypoints + highlighted plots
        _chip_view(pnum=(2, 1, 1), sel_fx=fx)
        # Draw the selected feature plots
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _chip_view(pnum=(1, 1, 1), **kwargs):
        df2.figure(fnum=fnum, pnum=pnum, docla=True, doclf=True)
        # Toggle no keypoints view
        viz.show_chip(hs, cx=cx, rchip=rchip, fnum=fnum, pnum=pnum, **kwargs)
        df2.set_figtitle(figtitle)

    def _on_chip_click(event):
        print_('[inter] clicked chip')
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ax is None or x is None:
            # The click is not in any axis
            print('... out of axis')
            annote_ptr[0] = (annote_ptr[0] + 1) % 3
            mode = annote_ptr[0]
            draw_ell = mode == 1
            draw_pts = mode == 2
            print('... default kpts view mode=%r' % mode)
            _chip_view(draw_ell=draw_ell, draw_pts=draw_pts)
        else:
            hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
            print_(' hs_viewtype=%r' % hs_viewtype)
            if hs_viewtype == 'chip' and event.key == 'shift':
                print('... masking')
                # TODO: Do better integration of masking
                _chip_view()
                df2.disconnect_callback(fig, 'button_press_event')
                mc = mask_creator.MaskCreator(df2.gca())  # NOQA
            elif hs_viewtype == 'chip':
                kpts = hs.get_kpts(cx)
                if len(kpts) > 0:
                    fx = nearest_point(x, y, kpts)[0]
                    print('... clicked fx=%r' % fx)
                    _select_ith_kpt(fx)
                else:
                    print('... len(kpts) == 0')
        viz.draw()

    # Draw without keypoints the first time
    _chip_view(draw_ell=False, draw_pts=False)
    viz.draw()
    df2.connect_callback(fig, 'button_press_event', _on_chip_click)


@profile
def interact_keypoints(rchip, kpts, desc, fnum=0, figtitle=None, nodraw=False, **kwargs):
    fig = begin_interaction('keypoint', fnum)
    annote_ptr = [1]

    def _select_ith_kpt(fx):
        print_('[interact] viewing ith=%r keypoint' % fx)
        # Get the fx-th keypiont
        kp, sift = kpts[fx], desc[fx]
        # Draw the image with keypoint fx highlighted
        _viz_keypoints(fnum, (2, 1, 1), sel_fx=fx)
        # Draw the selected feature
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _viz_keypoints(fnum, pnum=(1, 1, 1), **kwargs):
        df2.figure(fnum=fnum, docla=True, doclf=True)
        viz.show_keypoints(rchip, kpts, fnum=fnum, pnum=pnum, **kwargs)
        if figtitle is not None:
            df2.set_figtitle(figtitle)

    def _on_keypoints_click(event):
        print_('[viz] clicked keypoint view')
        if event is None  or event.xdata is None or event.inaxes is None:
            annote_ptr[0] = (annote_ptr[0] + 1) % 3
            mode = annote_ptr[0]
            draw_ell = mode == 1
            draw_pts = mode == 2
            print('... default kpts view mode=%r' % mode)
            _viz_keypoints(fnum, draw_ell=draw_ell, draw_pts=draw_pts)
        else:
            ax = event.inaxes
            hs_viewtype = ax.__dict__.get('_hs_viewtype', None)
            print_(' viewtype=%r' % hs_viewtype)
            if hs_viewtype == 'keypoints':
                kpts = ax.__dict__.get('_hs_kpts', [])
                if len(kpts) == 0:
                    print('...nokpts')
                else:
                    print('...nearest')
                    x, y = event.xdata, event.ydata
                    fx = nearest_point(x, y, kpts)[0]
                    _select_ith_kpt(fx)
            else:
                print('...unhandled')
        viz.draw()

    # Draw without keypoints the first time
    _viz_keypoints(fnum)
    df2.connect_callback(fig, 'button_press_event', _on_keypoints_click)
    if not nodraw:
        viz.draw()

#==========================
# Chipres Interaction
#==========================


@profile
def interact_chipres(hs, res, cx=None, fnum=4, figtitle='Inspect Query Result', **kwargs):
    'Interacts with a single chipres, '
    fig = begin_interaction('chipres', fnum)
    qcx = res.qcx
    if cx is None:
        cx = res.topN_cxs(hs, 1)[0]
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    fm = res.cx2_fm[cx]
    mx = kwargs.pop('mx', None)
    xywh2_ptr = [None]
    annote_ptr = [0]

    # Draw default
    @profile
    def _chipmatch_view(pnum=(1, 1, 1), **kwargs):
        mode = annote_ptr[0]
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        df2.figure(fnum=fnum, docla=True, doclf=True)
        tup = viz.res_show_chipres(res, hs, cx, fnum=fnum, pnum=pnum,
                                   draw_lines=draw_lines, draw_ell=draw_ell, **kwargs)
        ax, xywh1, xywh2 = tup
        xywh2_ptr[0] = xywh2
        df2.set_figtitle(figtitle)

    # Draw clicked selection
    @profile
    def _select_ith_match(mx):
        annote_ptr[0] = 1
        # Get the mx-th feature match
        fx1, fx2 = fm[mx]
        kpts1, kpts2 = hs.get_kpts([qcx, cx])
        desc1, desc2 = hs.get_desc([qcx, cx])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1), (rchip2, kp2, sift2, fx2)]
        nRows, nCols = len(extracted_list) + 1, 3
        # Draw matching chips and features
        pnum1 = (nRows, 1, 1)
        sel_fm = np.array([(fx1, fx2)])
        _crargs = dict(ell_alpha=.4, ell_linewidth=1.8, colors=df2.BLUE,
                       sel_fm=sel_fm, **kwargs)
        _chipmatch_view(pnum1, vert=False, **_crargs)
        # Draw selected feature matches
        px = 1 * nCols  # plot offset
        prevsift = None
        for (rchip, kp, sift, fx) in extracted_list:
            px = draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, prevsift)
            prevsift = sift

    # Draw ctrl clicked selection
    def _sv_view(cx):
        fnum = viz.FNUMS['special']
        fig = df2.figure(fnum=fnum, docla=True, doclf=True)
        df2.disconnect_callback(fig, 'button_press_event')
        viz.viz_spatial_verification(hs, res.qcx, cx2=cx, fnum=fnum)
        viz.draw()

    # Callback
    @profile
    def _click_chipres_click(event):
        print_('[inter] clicked chipres')
        (x, y, ax) = (event.xdata, event.ydata, event.inaxes)
        # Out of axes click
        if None in [x, y, ax]:
            print('... out of axis')
            _chipmatch_view()
            viz.draw()
            return
        hs_viewtype = ax.__dict__.get('_hs_viewtype', '')
        print_(' hs_viewtype=%r ' % hs_viewtype)
        key = '' if event.key is None else event.key
        print_('key=%r ' % key)
        ctrl_down = key.find('control') == 0
        # Click in match axes
        if hs_viewtype == 'chipres' and ctrl_down:
            # Ctrl-Click
            print('.. control click')
            return _sv_view(cx)
        elif hs_viewtype == 'chipres':
            if len(fm) == 0:
                print('[inter] no feature matches to click')
            else:
                # Normal Click
                # Select nearest feature match to the click
                kpts1, kpts2 = hs.get_kpts([qcx, cx])
                kpts1_m = kpts1[fm[:, 0]]
                kpts2_m = kpts2[fm[:, 1]]
                x2, y2, w2, h2 = xywh2_ptr[0]
                _mx1, _dist1 = nearest_point(x, y, kpts1_m)
                _mx2, _dist2 = nearest_point(x - x2, y - y2, kpts2_m)
                mx = _mx1 if _dist1 < _dist2 else _mx2
                print('... clicked mx=%r' % mx)
                _select_ith_match(mx)
        elif hs_viewtype == 'warped':
            print('... clicked warped')
        elif hs_viewtype == 'unwarped':
            print('... clicked unwarped')
        elif hs_viewtype == 'histogram':
            print('... clicked hist')
        else:
            print('... what did you click?!')
        viz.draw()

    if mx is None:
        _chipmatch_view()
    else:
        _select_ith_match(mx)
    df2.connect_callback(fig, 'button_press_event', _click_chipres_click)
    viz.draw()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    import main
    hs = main.main()
    cx = helpers.get_arg_after('--cx', type_=int)
    qcx = hs.get_valid_cxs()[0]
    if cx is not None:
        qcx = cx

    res = hs.query(qcx)
    interact_chip(hs, qcx, fnum=1)
    interact_chipres(hs, res, fnum=2)
    df2.update()
    exec(df2.present())
