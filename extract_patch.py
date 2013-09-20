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

def get_subchip(rchip, kp, radius):
    x, y, a, c, d = kp[0]
    (chip_h, chip_w) = rchip.shape[0:2]
    ix1, ix2, xm = quantize_to_pixel_with_offset(x, radius, 0, chip_w)
    iy1, iy2, ym = quantize_to_pixel_with_offset(y, radius, 0, chip_h)
    subchip = rchip[iy1:iy2, ix1:ix2]
    subkp = kp.copy() # subkeypoint in subchip coordinates
    subkp[0,0:2] = (xm, ym)
    return subchip, subkp

def show_feature(rchip, kp, subkp, **kwargs):
    df2.figure(**kwargs)
    df2.imshow(rchip, plotnum=(1,2,1), **kwargs)
    df2.draw_kpts2(kp, ell_color=(1,0,0), pts=True)
    df2.imshow(subchip, plotnum=(2,2,2), **kwargs)
    df2.draw_kpts2(subkp, ell_color=(1,0,0), pts=True)

def target_dsize(img, M):
    # Get img bounds under transformation
    (minx, maxx, miny, maxy) = sv2.transformed_bounds(img, M)
    Mw, Mh = (maxx-minx, maxy-miny)
    # If any border forced below, return a translation to append to M
    tx = -min(0, minx)
    ty = -min(0, miny)
    # Round to integer size
    dsize = tuple(map(int, np.ceil((Mw, Mh))))
    return dsize, tx, ty

def warp_image(img, M):
    img_size = img.shape[0:2]
    # Find the target warped img extent, add any tranlations
    dsize, tx, ty = target_dsize(img, M)
    M = M.copy()
    M[0,2] += tx
    M[1,2] += ty
    print('warp %r -> %r' % (img_size, dsize))
    #warp_img = cv2.warpAffine(img, M[0:2], dsize, **__cv2_warp_kwargs())
    warp_img = cv2.warpPerspective(img, M, dsize, **__cv2_warp_kwargs())
    return warp_img, tx, ty, M


if not 'hs' in vars():
    hs = ld2.HotSpotter()
    hs.load_all(params.GZ)
    qcx = 111
    cx = 305
    # Database descriptor + keypoints
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    def get_features(cx):
        rchip = hs.get_chip(cx)
        rchip_size = cx2_rchip_size[cx]
        fx2_kp   = cx2_kpts[cx]
        fx2_scale = sv2.keypoint_scale(fx2_kp)
        fx2_desc = cx2_desc[cx]
        return rchip, rchip_size, fx2_kp, fx2_scale, fx2_desc
    # Query features
    rchip1, rchip_size1, fx2_kp1, fx2_scale1, fx2_desc1 = get_features(qcx)
    # Result features
    rchip2, rchip_size2, fx2_kp2, fx2_scale2, fx2_desc2 = get_features(cx)
    # Vsmany index
    vsmany_index = hs.matcher._Matcher__vsmany_index
    #c2.precompute_index_vsmany(hs)
    #qcx2_res = mc2.run_matching(hs)

    #params.__MATCH_TYPE__ = 'bagofwords'
    #hs.load_matcher()
    #resBOW = mc2.build_result_qcx(hs, qcx)
    #df2.show_match_analysis(hs, resBOW, N=5, fignum=1)
    
    params.__MATCH_TYPE__ = 'vsmany'
    hs.load_matcher()
    params.__VSMANY_SCORE_FN__ = 'LNRAT'
    resLNRAT = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resLNRAT, N=5, fignum=1)

    params.__VSMANY_SCORE_FN__ = 'LNBNN'
    resLNBNN = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resLNBNN, N=5, fignum=2)

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    resRATIO = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resRATIO, N=5, fignum=3)

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    
    params.__MATCH_TYPE__ = 'vsone'
    hs.load_matcher()
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=4)
    df2.present()

    #allres = init_allres(hs, qcx2_res, SV, oxford=oxford)

    def get_vsmany_all_data():
        vsmany_all_assign = mc2.assign_matches_vsmany(qcx, cx2_desc, vsmany_index)
        cx2_fm, cx2_fs, cx2_score = vsmany_all_assign
        vsmany_all_svout = mc2.spatially_verify_matches(qcx, cx2_kpts, cx2_rchip_size, cx2_fm, cx2_fs)
        return vsmany_all_assign, vsmany_all_svout

    def get_vsmany_data(vsmany_all_data, cx):
        ' Assigned matches (vsmany)'
        vsmany_all_assign, vsmany_all_svout = vsmany_all_data
        vsmany_cx_assign = map(lambda _: _[cx],  vsmany_all_assign)
        vsmany_cx_svout  = map(lambda _: _[cx],  vsmany_all_svout)
        return vsmany_cx_assign, vsmany_cx_svout

    def get_vsone_data(cx):
        ' Assigned matches (vsone)'
        vsone_flann, checks = mc2.get_vsone_flann(fx2_desc1)
        fm, fs = mc2.match_vsone(fx2_desc2, vsone_flann, checks)
        fm_V, fs_V, H = mc2.spatially_verify(fx2_kp1, fx2_kp2, rchip_size2, fm, fs, qcx, cx)
        score = fs.sum(); score_V = fs_V.sum()
        vsone_cx_assign = fm, fs, score
        vsone_cx_svout = fm_V, fs_V, score_V
        return vsone_cx_assign, vsone_cx_svout

    # Assign + Verify
    params.__USE_CHIP_EXTENT__ = False
    vsmany_all_data = get_vsmany_all_data()
    vsmany_data = get_vsmany_data(vsmany_all_data, cx)
    vsone_data = get_vsone_data(cx)

    params.__USE_CHIP_EXTENT__ = True
    vsmany_all_data2 = get_vsmany_all_data()
    vsmany_data2 = get_vsmany_data(vsmany_all_data2, cx)
    vsone_data2 = get_vsone_data(cx)

    def show_matchers_compare(vsmany_data, vsone_data, fignum=0, figtitle=''):
        vsmany_cx_assign, vsmany_cx_svout = vsmany_data
        vsone_cx_assign, vsone_cx_svout = vsone_data
        # Show vsmany
        fm, fs, score = vsmany_cx_assign
        fm_V, fs_V, score_V = vsmany_cx_svout
        plot_kwargs = dict(all_kpts=False, fignum=fignum)
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm, fs,
                          plotnum=(2,2,1), title='vsmany assign', **plot_kwargs)
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm_V, fs_V,
                          plotnum=(2,2,2),  title='vsmany verified', **plot_kwargs)
        # Show vsone
        fm, fs, score      = vsone_cx_assign
        fm_V, fs_V, score_V = vsone_cx_svout
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm, fs,
                          plotnum=(2,2,3), title='vsone assign', **plot_kwargs)
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm_V, fs_V,
                          plotnum=(2,2,4), title='vsone verified', **plot_kwargs)
        df2.set_figtitle(figtitle)

    show_matchers_compare(vsmany_data, vsone_data, fignum=1, figtitle='kpt extent')
    show_matchers_compare(vsmany_data2, vsone_data2, fignum=2, figtitle='chip extent')
    df2.update()

    fx2_feature = hs.get_feature_fn(qcx)

def show_feature_fx(fx):
    rchip = rchip1
    kp, scale, radius, desc = fx2_feature(fx)
    subchip, subkp = get_subchip(rchip, kp, radius)
    show_feature(rchip, kp, subkp, fignum=fx)
    df2.update()

import sys
sys.exit(1)

rchip = rchip1
fx2_kp = fx2_kp1
fx = 2294
(kp, scale, radius, desc) = fx2_feature(fx)
subchip, subkp = get_subchip(rchip, kp, radius)

show_feature(rchip, kp, subkp, fignum=fx)

(x, y, a, c, d) = kp[0]
# Transformation from ellipse to a unit circle
A = np.array([(a, 0, 0),
              (c, d, 0),
              (0, 0, 1)])
# Scale up so we can see the keypoint
sf = 1000
S = np.array([(sf*3,  0, 0), 
              ( 0, sf*3, 0),
              ( 0,  0, 1)])

print('A')
print A
A = A.dot(S)
print A

fx2_kpIS = sv2.sqrt_inv(fx2_kp)
kpIS = fx2_kpIS[fx]
print('warp subchip')
#subchip2 = np.swapaxes(subchip, 0, 1)
warp_subchip, tx, ty, M = warp_image(subchip, A)
#warp_subchip = np.swapaxes(warp_subchip, 0, 1)

print('warp_subchip.shape = %r ' % (warp_subchip.shape,))
#circle_a = 3/(np.sqrt(3*sf))
xm = subkp[0,0]
ym = subkp[0,1]
[Axm, Aym,_]  = M.dot(np.array([[xm],[ym],[1]])).flatten()

circle_a = 1/sf
#(sf*3*np.sqrt(3))
unit_circle = np.array([(Axm, Aym, circle_a, 0., circle_a)])

# Plot full, cropped, warp


df2.imshow(warp_subchip, plotnum=(2,2,4), fignum=fx)
df2.draw_kpts2(unit_circle, ell_color=(1,0,0), pts=True)

#----

#fig = df2.figure(fignum=10)
#ax = df2.plt.gca()
#ax.set_xlim(-1,1)
#ax.set_ylim(-1,1)
#ax.set_aspect('equal')
##---
#df2.draw_sift(desc)

exec(df2.present())
#import sys
#sys.exit(1)
