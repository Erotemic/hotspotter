from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[inter]')
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


def printDBG(msg):
    pass


def nearest_kp(x, y, kpts):
    dist = (kpts.T[0] - x) ** 2 + (kpts.T[1] - y) ** 2
    fx = dist.argmin()
    return fx, dist[fx]


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
    #printDBG('[inter] draw_feat_row px=%r' % px)
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


# CHIP INTERACTION
def interact_keypoints(rchip, kpts, desc, fnum, figtitle=None, nodraw=False, **kwargs):
    fig = df2.figure(fnum=fnum)
    df2.disconnect_callback(fig, 'button_press_event')
    draw_kpts_ptr = [False]

    def _select_ith_kpt(fx):
        print('-------------------------------------------')
        print('[interact] viewing ith=%r keypoint' % fx)
        # Get the fx-th keypiont
        kp = kpts[fx]
        sift = desc[fx]
        # Draw the image with keypoint fx highlighted
        df2.figure(fnum=fnum)
        df2.cla()
        ell_args = {'ell_alpha': 1, 'ell_linewidth': 2}
        _viz_keypoints(fnum, (2, 1, 1), ell_color=df2.BLUE, ell_args=ell_args)
        # Draw highlighted point
        df2.draw_kpts2(kpts[fx:fx + 1], ell_color=df2.ORANGE, arrow=True, rect=True, **ell_args)

        # Draw the selected feature
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)

        df2.adjust_subplots_safe()
        #fig.canvas.draw()

    def _viz_keypoints(fnum, pnum, draw_kpts=True, **kwargs):
        fig = df2.figure(fnum=fnum)
        fig.clf()
        # Draw chip
        df2.imshow(rchip, pnum=pnum, fnum=fnum)
        # Draw all keypoints
        if draw_kpts:
            df2.draw_kpts2(kpts, **kwargs)
        ax = df2.gca()
        ax._hs_viewtype = 'keypoints'

    def _on_keypoints_click(event):
        import sys
        print_ = sys.stdout.write
        print_('[viz] clicked keypoint view')
        if event is None  or event.xdata is None or event.inaxes is None:
            print('...default')
            draw_kpts_ptr[0] = not draw_kpts_ptr[0]
            _viz_keypoints(fnum, (1, 1, 1), draw_kpts=draw_kpts_ptr[0])
        else:
            hs_viewtype = event.inaxes.__dict__.get('_hs_viewtype', None)
            print_(' %r' % hs_viewtype)
            if hs_viewtype != 'keypoints':
                print('...unhandled')
            elif len(kpts) == 0:
                print('...nokpts')
            else:
                print('...nearest')
                x, y = event.xdata, event.ydata
                fx = nearest_kp(x, y, kpts)[0]
                _select_ith_kpt(fx)
        if event is not None:
            df2.draw()
    # Draw without keypoints the first time
    _on_keypoints_click(None)
    if figtitle is not None:
        df2.set_figtitle(figtitle)
    df2.connect_callback(fig, 'button_press_event', _on_keypoints_click)
    if not nodraw:
        df2.draw()


def interact_chip(hs, cx, fnum=2, figtitle=None, **kwargs):
    # Get chip info (make sure get_chip is called first)
    rchip = hs.get_chip(cx)
    #cidstr = hs.cidstr(cx)
    #name  = hs.cx2_name(cx)
    #gname = hs.cx2_gname(cx)
    fig = df2.figure(fnum=fnum)
    df2.disconnect_callback(fig, 'button_press_event')

    def _select_ith_kpt(fx):
        print('-------------------------------------------')
        print('[interact] viewing ith=%r keypoint' % fx)
        # Get the fx-th keypiont
        kpts = hs.get_kpts(cx)
        desc = hs.get_desc(cx)

        kp = kpts[fx]
        sift = desc[fx]
        # Draw the image with keypoint fx highlighted
        df2.figure(fnum=fnum)
        df2.cla()
        ell_args = {'ell_alpha': .4, 'ell_linewidth': 1.8}
        # Draw chip + keypoints
        viz.show_chip(hs, cx=cx, rchip=rchip, kpts=kpts, pnum=(2, 1, 1),
                      fnum=fnum, ell_args=ell_args)
        # Draw highlighted point
        df2.draw_kpts2(kpts[fx:fx + 1], ell_color=df2.BLUE, rect=True, **ell_args)

        # Draw the selected feature
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, None)
        df2.adjust_subplots_safe()
        df2.draw()

    def default_chip_view():
        fig = df2.figure(fnum=fnum)
        fig.clf()
        viz.show_chip(hs, cx=cx, draw_kpts=False)  # Toggle no keypoints view
        df2.adjust_subplots_safe()
        fig.canvas.draw()

    def _on_chip_click(event):
        #print('\n===========')
        print('\n'.join(['%r=%r' % tup for tup in event.__dict__.iteritems()]))
        print('[inter] clicked chip')
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
        if event.key == 'shift':
            print('masking')
            # TODO: Do better integration of masking
            default_chip_view()
            df2.disconnect_callback(fig, 'button_press_event')
            ax = df2.gca()
            mc = mask_creator.MaskCreator(ax)  # NOQA
            df2.adjust_subplots_safe()
            fig.canvas.draw()
        else:
            x, y = event.xdata, event.ydata
            fx = nearest_kp(x, y, kpts)[0]
            _select_ith_kpt(fx)
    #fx = 1897
    #_select_ith_kpt(fx)
    # Draw without keypoints the first time
    viz.show_chip(hs, cx=cx, draw_kpts=False)
    if figtitle is not None:
        df2.set_figtitle(figtitle)
    df2.connect_callback(fig, 'button_press_event', _on_chip_click)


def interact_chipres(hs, res, cx=None, fnum=4, figtitle='Inspect Query Result', **kwargs):
    'res = back.current_res'
    'Interacts with a single chipres, '
    # Get data
    qcx = res.qcx
    if cx is None:
        cx = res.topN_cxs(hs, 1)[0]
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    kpts1, kpts2   = hs.get_kpts([qcx, cx])
    desc1, desc2   = hs.get_desc([qcx, cx])
    fm = res.cx2_fm[cx]
    mx = kwargs.pop('mx', None)
    xywh2_ptr = [None]
    annote_ptr = [True]

    # Draw default
    def _chipmatch_view():
        print('[inter] interact_chipres(qcx=%r, cx=%r)' % (qcx, cx))
        fig = df2.figure(fnum=fnum, doclf=True, trueclf=True)
        annote = annote_ptr[0]
        ax, xywh1, xywh2 = res.show_chipres(hs, cx, fnum=fnum, pnum=(1, 1, 1),
                                            draw_lines=annote, draw_ell=annote, **kwargs)
        df2.set_figtitle(figtitle)
        xywh2_ptr[0] = xywh2
        # Toggle annote
        annote_ptr[0] = not annote
        df2.adjust_subplots_safe()
        fig.canvas.draw()

    # Draw clicked selection
    def _select_ith_match(mx):
        annote_ptr[0] = True
        print('\n[inter] view feature match mx=%r' % mx)
        # Helper functions and args
        # Get the mx-th feature match
        fx1, fx2 = fm[mx]
        kp1, kp2     = kpts1[fx1], kpts2[fx2]
        sift1, sift2 = desc1[fx1], desc2[fx2]
        # Extracted keypoints to draw
        extracted_list = [(rchip1, kp1, sift1, fx1), (rchip2, kp2, sift2, fx2)]
        chipres_rows = 1  # Number of rows for showing the chip result
        nRows = len(extracted_list) + chipres_rows
        nCols = 3
        #-----------------
        # Draw chips + feature matches
        pnum1 = (nRows, 1, 1)
        _crargs = dict(fnum=fnum, pnum=pnum1, draw_lines=False, **kwargs)
        _bmargs = dict(ell_alpha=.4, ell_linewidth=1.8, colors=df2.BLUE, **_crargs)
        _smargs = dict(rect=True, colors=df2.ORANGE, **_crargs)
        fig = df2.figure(fnum=fnum, pnum=pnum1, doclf=True, trueclf=True)
        # Draw background matches
        ax, xywh1, xywh2 = res.show_chipres(hs, cx, vert=False, **_bmargs)
        xywh2_ptr[0] = xywh2
        # Draw selected match
        sel_fm = np.array([(fx1, fx2)])
        df2.draw_fmatch(xywh1, xywh2, kpts1, kpts2, sel_fm, **_smargs)
        #-----------------
        px = chipres_rows * nCols  # plot offset
        prevsift = None
        for (rchip, kp, sift, fx) in extracted_list:
            px = draw_feat_row(rchip, fx, kp, sift, fnum, nRows, nCols, px, prevsift)
            prevsift = sift
        df2.set_figtitle(figtitle)
        df2.adjust_subplots_safe()
        fig.canvas.draw()

    # Draw ctrl clicked selection
    def _sv_view(cx):
        printDBG('ctrl+clicked cx=%r' % cx)
        fnum = viz.FNUMS['special']
        fig = df2.figure(fnum=fnum, doclf=True, trueclf=True)
        df2.disconnect_callback(fig, 'button_press_event')
        viz.viz_spatial_verification(hs, res.qcx, cx2=cx, fnum=fnum)
        df2.adjust_subplots_safe()
        fig.canvas.draw()

    # Callback
    def _click_chipres_callback(event):
        printDBG('[inter] clicked chipres')
        (x, y) = (event.xdata, event.ydata)
        # Out of axes click
        if None in [x, y, event.inaxes]:
            return _chipmatch_view()
        hs_viewtype = event.inaxes.__dict__.get('_hs_viewtype', '')
        printDBG('hs_viewtype=%r' % hs_viewtype)
        # Click in match axes
        if hs_viewtype.find('chipres') == 0:
            # Ctrl-Click
            key = '' if event.key is None else event.key
            print('[inter] key = %r' % key)
            if key.find('control') == 0:
                print('[inter] result control clicked')
                return _sv_view(cx)
            # Normal Click
            # Select nearest feature match to the click
            if len(fm) == 0:
                print('[inter] no feature matches to click')
                return
            kpts1_m = kpts1[fm[:, 0]]
            kpts2_m = kpts2[fm[:, 1]]
            x2, y2, w2, h2 = xywh2_ptr[0]
            _mx1, _dist1 = nearest_kp(x, y, kpts1_m)
            _mx2, _dist2 = nearest_kp(x - x2, y - y2, kpts2_m)
            mx = _mx1 if _dist1 < _dist2 else _mx2
            _select_ith_match(mx)
        elif hs_viewtype.find('warped') == 0:
            printDBG('[inter] clicked warped')
        elif hs_viewtype.find('unwarped') == 0:
            printDBG('[inter] clicked unwarped')
        elif hs_viewtype.find('histogram') == 0:
            printDBG('[inter] clicked hist')
        else:
            printDBG('[inter] what did you click?!')

    # Disconnect other callbacks and initialize interaction.
    fig_ = df2.figure(fnum=fnum, doclf=True, trueclf=True, **kwargs)
    df2.disconnect_callback(fig_, 'button_press_event')
    if mx is None:
        _chipmatch_view()
    else:
        _select_ith_match(mx)
    df2.connect_callback(fig_, 'button_press_event', _click_chipres_callback)
    printDBG('[inter] Drawing and starting interaction')
    df2.adjust_subplots_safe()
    df2.draw()

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
