from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr, profile, printDBG)\
    = __common__.init(__name__, '[inter]', DEBUG=False)
# Scientific
import numpy as np
# Hotspotter
import draw_func2 as df2
import viz
from hstpl import mask_creator


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


def detect_keypress(fig):
    def on_key_press(event):
        if event.key == 'shift':
            shift_is_held = True  # NOQA

    def on_key_release(event):
        if event.key == 'shift':
            shift_is_held = False  # NOQA
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)


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
            if centers is None or len(centers) == 0:
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
        else:
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
def interact_chip(hs, cx, fnum=2, figtitle=None, fx=None, **kwargs):
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
        viz.draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)

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
            else:
                print('...Unknown viewtype')
        viz.draw()

    # Draw without keypoints the first time
    if fx is not None:
        _select_ith_kpt(fx)
    else:
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
        viz.draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)

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
def interact_chipres(hs, res, cx=None, fnum=4, figtitle='Inspect Query Result',
                     same_fig=False, **kwargs):
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
    from hscom.Printable import DynStruct
    last_state = DynStruct()
    last_state.same_fig = same_fig
    last_state.last_fx = 0

    # Draw default
    @profile
    def _chipmatch_view(pnum=(1, 1, 1), **kwargs):
        mode = annote_ptr[0]
        draw_ell = mode >= 1
        draw_lines = mode == 2
        annote_ptr[0] = (annote_ptr[0] + 1) % 3
        df2.figure(fnum=fnum, docla=True, doclf=True)
        # TODO RENAME This to remove res and rectify with show_chipres
        tup = viz.res_show_chipres(res, hs, cx, fnum=fnum, pnum=pnum,
                                   draw_lines=draw_lines, draw_ell=draw_ell,
                                   colorbar_=True, **kwargs)
        ax, xywh1, xywh2 = tup
        xywh2_ptr[0] = xywh2

        df2.set_figtitle(figtitle + hs.vs_str(qcx, cx))

    # Draw clicked selection
    @profile
    def _select_ith_match(mx, qcx, cx):
        #----------------------
        # Get info for the _select_ith_match plot
        annote_ptr[0] = 1
        # Get the mx-th feature match
        cx1, cx2 = qcx, cx
        fx1, fx2 = fm[mx]
        fscore2  = res.cx2_fs[cx2][mx]
        fk2      = res.cx2_fk[cx2][mx]
        kpts1, kpts2 = hs.get_kpts([cx1, cx2])
        desc1, desc2 = hs.get_desc([cx1, cx2])
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        info1 = '\nquery'
        info2 = '\nk=%r fscore=%r' % (fk2, fscore2)
        last_state.last_fx = fx1

        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1, cx1, info1),
                          (rchip2, kp2, sift2, fx2, cx2, info2)]
        # Normalizng Keypoint
        if hasattr(res, 'filt2_meta') and 'lnbnn' in res.filt2_meta:
            qfx2_norm = res.filt2_meta['lnbnn']
            # Normalizing chip and feature
            (cx3, fx3, normk) = qfx2_norm[fx1]
            rchip3 = hs.get_chip(cx3)
            kp3 = hs.get_kpts(cx3)[fx3]
            sift3 = hs.get_desc(cx3)[fx3]
            info3 = '\nnorm %s k=%r' % (hs.cidstr(cx3), normk)
            extracted_list.append((rchip3, kp3, sift3, fx3, cx3, info3))
        else:
            print('WARNING: meta doesnt exist')

        #----------------------
        # Draw the _select_ith_match plot
        nRows, nCols = len(extracted_list) + same_fig, 3
        # Draw matching chips and features
        sel_fm = np.array([(fx1, fx2)])
        pnum1 = (nRows, 1, 1) if same_fig else (1, 1, 1)
        _chipmatch_view(pnum1, vert=False, ell_alpha=.4, ell_linewidth=1.8,
                        colors=df2.BLUE, sel_fm=sel_fm, **kwargs)
        # Draw selected feature matches
        px = nCols * same_fig  # plot offset
        prevsift = None
        if not same_fig:
            fnum2 = fnum + len(viz.FNUMS)
            fig2 = df2.figure(fnum=fnum2, docla=True, doclf=True)
        else:
            fnum2 = fnum
        for (rchip, kp, sift, fx, cx, info) in extracted_list:
            px = viz.draw_feat_row(rchip, fx, kp, sift, fnum2, nRows, nCols, px,
                                   prevsift=prevsift, cx=cx, info=info)
            prevsift = sift
        if not same_fig:
            df2.connect_callback(fig2, 'button_press_event', _click_chipres_click)
            df2.set_figtitle(figtitle + hs.vs_str(qcx, cx))

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
        if event is None:
            return
        button = event.button
        is_right_click = button == 3
        if is_right_click:
            return
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
                _select_ith_match(mx, qcx, cx)
        elif hs_viewtype in ['warped', 'unwarped']:
            hs_cx = ax.__dict__.get('_hs_cx', None)
            hs_fx = ax.__dict__.get('_hs_fx', None)
            if hs_cx is not None:
                interact_chip(hs, hs_cx, fx=hs_fx, fnum=df2.next_fnum())
        else:
            print('...Unknown viewtype')
        viz.draw()

    if mx is None:
        _chipmatch_view()
    else:
        _select_ith_match(mx, qcx, cx)

    from hsgui import guitools

    def toggle_samefig():
        interact_chipres(hs, res, cx=cx, fnum=fnum, figtitle=figtitle, same_fig=not same_fig, **kwargs)

    def query_last_feature():
        viz.show_nearest_descriptors(hs, qcx, last_state.last_fx, df2.next_fnum())
        fig3 = df2.gca()
        df2.connect_callback(fig3, 'button_press_event', _click_chipres_click)
        df2.update()

    toggle_samefig_key = 'Toggle same_fig (currently %r)' % same_fig

    opt2_callback = [
        (toggle_samefig_key, toggle_samefig),
        ('query last feature', query_last_feature),
        ('cancel', lambda: print('cancel')), ]
    guitools.popup_menu(fig.canvas, opt2_callback, fig.canvas)
    df2.connect_callback(fig, 'button_press_event', _click_chipres_click)
    viz.draw()
