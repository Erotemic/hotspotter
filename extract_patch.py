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


def show_feature(rchip, kp, subkp, **kwargs):
    df2.figure(**kwargs)

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

def warp_image(img, kp, sf):
    (x, y, a, c, d) = kp[0]
    A = np.array([[a,  0, 0],
                  [c,  d, 0],
                  [0,  0, 1]])
    T = np.array([[1, 0, -x+1],
                  [0, 1, -y+1],
                  [0, 0,    1]])
    S = np.array([[sf, 0, sf/2],
                  [0, sf, sf/2],
                  [0,  0,    1]])
    img_h, img_w = img.shape[0:2]
    dsize = np.array(np.ceil(np.array([sf, sf])), dtype=int)
    print(dsize)
    Z = S.dot(A).dot(T)
    print("Z=\n"+repr(A))
    warp_img = cv2.warpAffine(img, Z[0:2], tuple(dsize), **__cv2_warp_kwargs())
    #warp_img = cv2.warpPerspective(img, M, dsize, **__cv2_warp_kwargs())
    return warp_img

if __name__ == '__main__':
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
        res = mc2.build_result_qcx(hs, qcx)
    fig = df2.show_match_analysis(hs, res, N=5, fignum=1)
    df2.update()

    rchip = rchip1
    fx2_kp = fx2_kp1
    fx = 200#2294
    kp = fx2_kp1[fx:fx+1]
    scale = fx2_scale1[fx]
    desc = fx2_desc1[fx]
    #(kp, scale, radius, desc) = fx2_feature(fx)
    def get_subchip(rchip, kp, radius):
        x, y, a, c, d = kp[0]
        (chip_h, chip_w) = rchip.shape[0:2]
        ix1, ix2, xm = quantize_to_pixel_with_offset(x, radius, 0, chip_w)
        iy1, iy2, ym = quantize_to_pixel_with_offset(y, radius, 0, chip_h)
        subchip = rchip[iy1:iy2, ix1:ix2]
        subkp = kp.copy() # subkeypoint in subchip coordinates
        subkp[0,0:2] = (xm, ym)
        return subchip, subkp
    sf = np.sqrt(scale)*3*np.sqrt(3)
    subchip, subkp = get_subchip(rchip, kp, sf)
    warp_subchip   = warp_image(rchip, kp, sf)
    unit_circle = np.array([(sf/2, sf/2, 1/sf, 0., 1/sf)])

    # Plot chosen keypoint
    fig = df2.figure(9001, plotnum=(2,2,1))

    df2.imshow(rchip, plotnum=(1,3,1))
    df2.draw_kpts2(kp, ell_color=(1,0,0), pts=True)

    df2.imshow(subchip, plotnum=(1,3,2))
    df2.draw_kpts2(subkp, ell_color=(1,0,0), pts=True)

    df2.imshow(warp_subchip, plotnum=(1,3,3))
    df2.draw_kpts2(unit_circle, ell_color=(1,0,0), pts=True)

    df2.set_figtitle('chosen keypoint')

    # Plot full, cropped, warp
    df2.draw()
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
