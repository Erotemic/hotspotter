#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import params
import load_data2 as ld2
import draw_func2 as df2
import match_chips2 as mc2
import vizualizations as viz
import helpers
import cv2
import spatial_verification2 as sv2
import sys

def reload_module():
    import imp, sys
    print('[extract] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def __cv2_warp_kwargs():
    flags = (cv2.INTER_LINEAR, cv2.INTER_NEAREST)[0]
    borderMode = cv2.BORDER_CONSTANT
    warp_kwargs = dict(flags=flags, borderMode=borderMode)
    return warp_kwargs

def quantize_to_pixel_with_offset(z, radius, low, high):
    ''' Quantizes a small area into an indexable pixel location 
    Returns: pixel_range=(iz1, iz2), subpxl_offset
    Pixels:
    +   ___+___+___          +
    ^        ^ ^             ^
    z1       z iz           z2              
            _______________ < radius
                _____________ < quantized radius '''      
    (z1, z2) = (z-radius, z+radius)
    iz1 = max(np.floor(z1), low)
    iz2 = min(np.ceil(z2), high)
    z_radius1 = np.ceil(z - iz1)
    z_radius2 = np.ceil(iz2 - z)
    z_radius = min(z_radius1, z_radius2)
    (iz1, iz2) = (z-z_radius, z+z_radius)
    z_radius = np.ceil(z - iz1)
    return iz1, iz2, z_radius

def kp2_sf(kp):
    'computes scale factor of keypoint'
    (x, y, a, c, d) = kp
    sf = np.sqrt(1/(a*d))
    return sf

def get_patch(rchip, kp):
    'Returns cropped unwarped patch around a keypoint'
    (x, y, a, c, d) = kp
    sf = 1.5 * np.sqrt(kp2_sf(kp))
    (chip_h, chip_w) = rchip.shape[0:2]
    ix1, ix2, xm = quantize_to_pixel_with_offset(x, sf, 0, chip_w)
    iy1, iy2, ym = quantize_to_pixel_with_offset(y, sf, 0, chip_h)
    patch = rchip[iy1:iy2, ix1:ix2]
    subkp = kp.copy() # subkeypoint in patch coordinates
    subkp[0:2] = (xm, ym)
    return patch, subkp

def get_warped_patch(rchip, kp):
    'Returns warped patch around a keypoint'
    (x, y, a, c, d) = kp
    sf = kp2_sf(kp)
    s = 3*np.sqrt(sf)
    (h, w) = rchip.shape[0:2]
    T = np.array([[1, 0, -x+1],
                  [0, 1, -y+1],
                  [0, 0,    1]])
    A = np.array([[a,  0, 0],
                  [c,  d, 0],
                  [0,  0, 1]])
    S = np.array([[sf, 0, 0],
                  [0, sf, 0],
                  [0, 0,  1]])
    X = np.array([[1, 0, s/2],
                  [0, 1, s/2],
                  [0, 0,   1]])
    rchip_h, rchip_w = rchip.shape[0:2]
    dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
    def mat_mult(*args):
        # args[0] is the first transform
        M = np.eye(3)
        for Z in args:
            M = Z.dot(M)
        return M
    # Do I need to scale before A and then after A?  I think so. 
    M = mat_mult(T, S, A, X)
    #print('-------')
    #print('Warping')
    #print('kp = %r ' % (kp,))
    #print('sf = %r ' % (sf,))
    #print('* T = \n%r' % (T,))
    #print('* A = \n%r' % (A,))
    #print('* S = \n%r' % (S,))
    #print('* X = \n%r' % (X,))
    #print('* M = \n%r' % (M,))
    #print('* dsize=%r' % (dsize,))
    #print('-------')
    warped_patch = cv2.warpAffine(rchip, M[0:2], tuple(dsize), **__cv2_warp_kwargs())
    #warped_patch = cv2.warpPerspective(rchip, M, dsize, **__cv2_warp_kwargs())
    wkp = np.array([(s/2, s/2, 1/sf, 0., 1/sf)])
    return warped_patch, wkp

def get_top_scoring_feats(cx2_fs):
    # Gets the top scoring features match indexes from a query
    # Returns [(cx, fm, feat_score)]
    top_scoring_feats = []
    for cx in xrange(len(cx2_fs)):
        fs = cx2_fs[cx]
        for mx in xrange(len(fs)):
            feat_score = fs[mx]
            top_scoring_feats.append((cx, mx, feat_score))
    top_scoring_feats = sorted(top_scoring_feats, key=lambda x: x[2])[::-1]
    return top_scoring_feats

def printDBG(msg):
    pass 

def get_normalizer_cx_rchip_cx(hs, desc):
    printDBG('query desc.shape = %r ' % (desc.shape,))
    (qfx2_cx, qfx2_fx, qfx2_dists) = mc2.desc_nearest_neighbors(desc, hs.matcher.vsmany_args)
    cxN = qfx2_cx[:,-1][0]
    fxN = qfx2_fx[:,-1][0]
    rchipN = hs.get_chip(cxN)
    print('---')
    print('qfx2_fx = %r' % (qfx2_fx,))
    print('qfx2_cx = %r' % (qfx2_cx,))
    print('qfx2_nx = %r' % (hs.tables.cx2_nx[qfx2_cx],))
    print('qfx2_dists = %r' % (qfx2_dists,))
    print('cx Normalizer: %r' % cxN)
    print('fx Normalizer: %r' % fxN)
    print('---')
    return cxN, fxN, rchipN
    
def get_top_scoring_patches(hs, res, low, high):
    qcx = res.qcx
    cx2_fs = res.cx2_fs_V
    cx2_fm = res.cx2_fm_V
    top_scoring_feats = get_top_scoring_feats(cx2_fs)
    rchip1 = hs.get_chip(qcx)
    def get_patches(hs, cx, rchip, fx):
        kp   = hs.feats.cx2_kpts[cx][fx]
        desc = hs.feats.cx2_desc[cx][fx]
        patch, subkp = get_patch(rchip, kp)
        wpatch, wkp = get_warped_patch(rchip, kp)
        return (kp, subkp, wkp, patch, wpatch, cx, desc)
    top_patches_list = []
    #print('Top Scoring Features: cx, mx, feat_score')
    for cx, mx, feat_score in top_scoring_feats[low:high]:
        rchip2 = hs.get_chip(cx)
        fx1, fx2 = cx2_fm[cx][mx]
        # Get query patches
        patches1 = get_patches(hs, qcx, rchip1, fx1)
        # Get result patches
        patches2 = get_patches(hs,  cx, rchip2, fx2)
        # Get k+1th patches
        if params.__MATCH_TYPE__ == 'vsmany':
            desc1 = patches1[-1]
            cxN, fxN, rchipN = get_normalizer_cx_rchip_cx(hs, desc1)
            patchesN = get_patches(hs, cxN, rchipN, fxN)
        else:
            patchesN = None
        top_patches_list.append((patches1, patches2, patchesN, cx, feat_score))
        #print('cx=%r, mx=%r, feat_score=%r' %(cx, mx, feat_score))
    return top_patches_list

def viz_top_features(hs, res, low, high, fignum=0, draw_chips=True):
    from collections import defaultdict
    qcx = res.qcx
    cx2_nx = hs.tables.cx2_nx
    top_patches_list = get_top_scoring_patches(hs, res, low, high)
    num_rows = high-low
    num_cols = 4
    if params.__MATCH_TYPE__ == 'vsmany':
        num_cols = 6
    # Initialize Figure
    fig = df2.figure(fignum+1, plotnum=(num_rows, num_cols,1))
    cx2_rchip = defaultdict(int)
    cx2_kplist = defaultdict(list)
    def draw_one_kp(patch, kp, plotx, color, desc=None):
        fig, ax = df2.imshow(patch, plotnum=(num_rows, num_cols, plotx))
        df2.draw_kpts2([kp], ell_color=color, pts=True)
        if not desc is None and not '--nodesc' in sys.argv:
            df2.draw_sift(desc, [kp])
        df2.draw_border(ax, color, 1)

    for tx, (patches1, patches2, patchesN, cx, feat_score) in enumerate(top_patches_list):
        (kp1, subkp1, wkp1, patch1, wpatch1, cx1, desc1) = patches1
        (kp2, subkp2, wkp2, patch2, wpatch2, cx2, desc2) = patches2
        # draw on table
        # Draw Query Keypoint
        plotx = (tx*num_cols)
        draw_one_kp(patch1,  subkp1, plotx+1, df2.GREEN, desc1)
        qnx = cx2_nx[qcx]
        df2.plt.gca().set_xlabel('qcx=%r; qnx=%r' % (qcx, qnx))
        draw_one_kp(wpatch1, wkp1,   plotx+2, df2.GREEN, desc1)

        # Draw ith < k match
        draw_one_kp(patch2,  subkp2, plotx+3, df2.BLUE, desc2)
        nx = cx2_nx[cx]
        df2.plt.gca().set_xlabel('cx=%r; nx=%r' % (cx, nx))
        draw_one_kp(wpatch2, wkp2,   plotx+4, df2.BLUE, desc2)
        df2.plt.gca().set_xlabel('score=%r' % (feat_score))

        # Draw k+1th match
        if params.__MATCH_TYPE__ == 'vsmany':
            (kpN, subkpN, wkpN, patchN, wpatchN, cxN, descN) = patchesN
            draw_one_kp(patchN,  subkpN, plotx+5, df2.ORANGE, descN)
            nxN = cx2_nx[qcx]
            df2.plt.gca().set_xlabel('cxN=%r; nxN=%r' % (cxN, nxN))
            draw_one_kp(wpatchN, wkpN,   plotx+6, df2.ORANGE, descN)
        # Get other info
        cx2_rchip[cx] = hs.get_chip(cx)
        cx2_kplist[qcx].append(kp1)
        cx2_kplist[cx].append(kp2)
    # Draw annotations on
    df2.figure(plotnum=(num_rows, num_cols,1), title='Query Patch')
    df2.figure(plotnum=(num_rows, num_cols,2), title='Query Patch')
    df2.figure(plotnum=(num_rows, num_cols,3), title='Result Patch')
    df2.figure(plotnum=(num_rows, num_cols,4), title='Result Warped')

    if params.__MATCH_TYPE__ == 'vsmany':
        df2.figure(plotnum=(num_rows, num_cols,5), title='Normalizer Patch')
        df2.figure(plotnum=(num_rows, num_cols,6), title='Normalizer Warped')

    df2.set_figtitle('Top '+str(low)+' to '+str(high)+' scoring matches')
    if not draw_chips:
        return
    #
    # Draw on full images
    cx2_rchip[qcx] = hs.get_chip(qcx)
    cx_keys  = cx2_kplist.keys()
    cx_vals = map(len, cx2_kplist.values())
    cx_list = [x for (y,x) in sorted(zip(cx_vals, cx_keys))][::-1]
    num_chips = len(cx_list)
    pltnum_fn = lambda ix: (int(np.ceil(num_chips/2)), 2, ix)
    plotnum = pltnum_fn(1)
    fig2 = df2.figure(fignum-1, plotnum=plotnum)
    for ix, cx in enumerate(cx_list):
        plotnum = pltnum_fn(ix+1)
        fig2 = df2.figure(fignum-1, plotnum=plotnum)
        rchip = cx2_rchip[cx]
        fig, ax = df2.imshow(rchip)
        title_pref = ''
        color = df2.BLUE
        if res.qcx == cx:
            title_pref = 'q'
            color = df2.GREEN
        df2.draw_kpts2(cx2_kplist[cx], ell_color=color, ell_alpha=1)
        df2.draw_border(ax, color, 2)
        nx = cx2_nx[cx]
        ax.set_title(title_pref+'cx = %r; nx=%r' % (cx, nx))
        gname = hs.get_gname(cx)
        ax.set_xlabel(gname)
    #df2.draw()

def test(hs, qcx):
    fx2_scale = sv2.keypoint_scale(hs.feats.cx2_kpts[qcx])
    fx = fx2_scale.argsort()[::-1][40]
    rchip = hs.get_chip(qcx)
    kp = hs.feats.cx2_kpts[qcx][fx]
    # Show full image and keypoint
    df2.figure(fignum=9000, doclf=True)
    df2.imshow(rchip, plotnum=(1,3,1))
    df2.draw_kpts2([kp], ell_color=(1,0,0), pts=True)
    # Show cropped image and keypoint
    patch, subkp = get_patch(rchip, kp)
    df2.imshow(patch, plotnum=(1,3,2))
    df2.draw_kpts2([subkp], ell_color=(1,0,0), pts=True)
    # Show warped image and keypoint
    wpatch, wkp = get_warped_patch(rchip, kp)
    df2.imshow(wpatch, plotnum=(1,3,3))
    df2.draw_kpts2([wkp], ell_color=(1,0,0), pts=True)
    #
    df2.set_figtitle('warp test')

def test2(hs, qcx, low, high):
    res = mc2.build_result_qcx(hs, qcx)
    viz.BROWSE = False
    viz.DUMP = False
    viz.FIGNUM = 232
    viz.plot_cx2(hs, res, 'analysis')
    viz_top_features(hs, res, low=low, high=high, draw_chips=True)

def test3():
    desc = np.random.rand(128)
    desc = desc / np.sqrt((desc**2).sum())
    desc = np.round(desc * 255)
    print desc
    fig = df2.figure(fignum=43)
    ax = df2.plt.gca()
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    df2.draw_sift(desc)
    

if __name__ == '__main__':
    print("[patch] __name__ == 'extract_patch.py'")
    if not 'hs' in vars():
        if len(sys.argv) == 1:
            db_dir = params.GZ
            qcx = 111
            low = 0
            high=6
        else:
            db_dir = params.DEFAULT
            qcx = helpers.get_arg_after('--qcx', type_=int)
            low = helpers.get_arg_after('--low', type_=int)
            high = helpers.get_arg_after('--high', type_=int)
            if qcx is None:
                raise Exception('fds')
                qcx = 1
            if low is None:
                low = 0
            if high is None:
                high = 6

        hs = ld2.HotSpotter()
        hs.load_all(db_dir)
    #test(hs, qcx)
    #test3()
    test2(hs, qcx, low=low, high=high)
    df2.all_figures_tight_layout()
    exec(df2.present(no_tile=True))
