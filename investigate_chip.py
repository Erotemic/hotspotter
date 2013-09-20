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
    hs.load_all(params.GZ, matcher=False)
    hs.set_samples()
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
    #c2.precompute_index_vsmany(hs)
    #qcx2_res = mc2.run_matching(hs)

    params.__MATCH_TYPE__ = 'vsmany'
    hs.load_matcher()
    vsmany_index = hs.matcher._Matcher__vsmany_index
    params.__VSMANY_SCORE_FN__ = 'LNRAT'
    resLNRAT = mc2.build_result_qcx(hs, qcx)
    fig1 = df2.show_match_analysis(hs, resLNRAT, N=5, fignum=1, figtitle=' LNRAT')
    #fig.tight_layout()

    params.__VSMANY_SCORE_FN__ = 'LNBNN'
    resLNBNN = mc2.build_result_qcx(hs, qcx)
    fig2 = df2.show_match_analysis(hs, resLNBNN, N=5, fignum=2, figtitle=' LNBNN')

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    resRATIO = mc2.build_result_qcx(hs, qcx)
    fig3 = df2.show_match_analysis(hs, resRATIO, N=5, fignum=3, figtitle=' RATIO')

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    
    params.__MATCH_TYPE__ = 'bagofwords'
    hs.load_matcher()
    resBOW = mc2.build_result_qcx(hs, qcx)
    fig4 = df2.show_match_analysis(hs, resBOW, N=5, fignum=4, figtitle=' bagofwords')
    
    params.__MATCH_TYPE__ = 'vsone'
    hs.load_matcher()
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    fig5 = df2.show_match_analysis(hs, res_vsone, N=5, fignum=5, figtitle=' vsone')


    fig6 = df2.show_match_analysis(hs, resLNBNN, N=20, fignum=6,
                                   figtitle=' LNBNN More', show_query=False)

    df2.update()

    

    res = resLNBNN


def top_matching_features(res, axnum=None, match_type=''):
    cx2_fs = res.cx2_fs_V
    cx_fx_fs_list = []
    for cx in xrange(len(cx2_fs)):
        fx2_fs = cx2_fs[cx]
        for fx in xrange(len(fx2_fs)):
            fs = fx2_fs[fx]
            cx_fx_fs_list.append((cx, fx, fs))

    cx_fx_fs_sorted = np.array(sorted(cx_fx_fs_list, key=lambda x: x[2])[::-1])

    sorted_score = cx_fx_fs_sorted[:,2]
    fig = df2.figure(0)
    df2.plot(sorted_score)

