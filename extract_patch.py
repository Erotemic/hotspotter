#exec(open('__init__.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import match_chips2 as mc2
import cv2
import spatial_verification2 as sv2
import sys
import params

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
    sf = np.sqrt(kp2_sf(kp))
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
                  [0, 0, 1]])
    rchip_h, rchip_w = rchip.shape[0:2]
    dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
    def mat_mult(*args):
        # args[0] is the first transform
        M = np.eye(3)
        for Z in args:
            M = Z.dot(M)
        return M
    I = np.eye(3)
    Z = I.copy()
    M = mat_mult(T, A, S, X)
    print('-------')
    print('Warping')
    print('kp = %r ' % (kp,))
    print('sf = %r ' % (sf,))
    print('* T = \n%r' % (T,))
    print('* A = \n%r' % (A,))
    print('* S = \n%r' % (S,))
    print('* X = \n%r' % (X,))
    print('* M = \n%r' % (M,))
    print('* dsize=%r' % (dsize,))
    print('-------')
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

def get_top_scoring_patches(hs, res, N):
    qcx = res.qcx
    cx2_fs = res.cx2_fs_V
    cx2_fm = res.cx2_fm_V
    top_scoring_feats = get_top_scoring_feats(cx2_fs)
    rchip1 = hs.get_chip(qcx)
    def get_patches(hs, cx, rchip, fx):
        kp = hs.feats.cx2_kpts[cx][fx]
        desc = hs.feats.cx2_desc[cx][fx]
        patch, subkp = get_patch(rchip, kp)
        wpatch, wkp = get_warped_patch(rchip, kp)
        return (kp, subkp, wkp, patch, wpatch, desc)
    top_patches_list = []
    print('Top Scoring Features: cx, mx, feat_score')
    for cx, mx, feat_score in top_scoring_feats[0:N]:
        rchip2 = hs.get_chip(cx)
        fx1, fx2 = cx2_fm[cx][mx]
        # Get query patches
        patches1 = get_patches(hs, qcx, rchip1, fx1)
        patches2 = get_patches(hs,  cx, rchip2, fx2)
        top_patches_list.append((patches1, patches2, cx, feat_score))
        print('cx=%r, mx=%r, feat_score=%r' %(cx, mx, feat_score))
    return top_patches_list

def viz_top_features(hs, res, N=10, fignum=0):
    from collections import defaultdict
    qcx = res.qcx
    top_patches_list = get_top_scoring_patches(hs, res, N)
    fig = df2.figure(fignum, plotnum=(N,4,1))
    cx2_rchip = defaultdict(int)
    cx2_kplist = defaultdict(list)
    def draw_one_kp(patch, kp, index):
        df2.imshow(patch, plotnum=(N,4,index))
        df2.draw_kpts2([kp], ell_color=(1,0,0), pts=True)
    for tx, (patches1, patches2, cx, feat_score) in enumerate(top_patches_list):
        (kp1, subkp1, wkp1, patch1, wpatch1, desc1) = patches1
        (kp2, subkp2, wkp2, patch2, wpatch2, desc2) = patches2
        # draw on table
        draw_one_kp(patch1, subkp1, (tx*4)+1)
        draw_one_kp(wpatch1, wkp1, (tx*4)+2)
        draw_one_kp(patch2, subkp2, (tx*4)+3)
        draw_one_kp(wpatch2, wkp2, (tx*4)+4)
        df2.plt.gca().set_xlabel('cx=%r, score=%r' % (cx, feat_score))
        # Get other info
        cx2_rchip[cx] = hs.get_chip(cx)
        cx2_kplist[qcx].append(kp1)
        cx2_kplist[cx].append(kp2)
    df2.set_figtitle('chosen keypoint')
    cx2_rchip[qcx] = hs.get_chip(qcx)
    #
    # Draw on full images
    cx_keys  = cx2_kplist.keys()
    cx_vals = map(len, cx2_kplist.values())
    cx_list = [x for (y,x) in sorted(zip(cx_vals, cx_keys))][::-1]
    num_chips = len(cx_list)
    pltnum_fn = lambda ix: (int(np.ceil(num_chips/2)), 2, ix)
    plotnum = pltnum_fn(1)
    fig2 = df2.figure(fignum+1, plotnum=plotnum)
    for ix, cx in enumerate(cx_list):
        plotnum = pltnum_fn(ix+1)
        fig2 = df2.figure(fignum+1, plotnum=plotnum)
        rchip = cx2_rchip[cx]
        df2.imshow(rchip)
        df2.draw_kpts2(cx2_kplist[cx])
    #df2.draw()

def test(hs, qcx, fx):
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
    df2.set_figtitle('chosen keypoint')

def test2(hs, qcx, N=10):
    res = mc2.build_result_qcx(hs, qcx)
    viz_top_features(hs, res)

if __name__ == '__main__':
    print("[patch] __name__ == 'extract_patch.py'")
    if not 'hs' in vars():
        hs = ld2.HotSpotter()
        hs.load_all(params.GZ)
    qcx = 111
    fx2_scale = sv2.keypoint_scale(hs.feats.cx2_kpts[qcx])
    fx = fx2_scale.argsort()[::-1][40]
    #fx = 300
    #test(hs, qcx, fx)
    test2(hs, qcx, N=10)
    exec(df2.present())
