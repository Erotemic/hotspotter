#exec(open('__init__.py').read())
from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[extract]')
import numpy as np
import draw_func2 as df2
import cv2
from numpy import sqrt


def rrr():
    import imp
    import sys
    print('[extract] Reloading: ' + __name__)
    imp.reload(sys.modules[__name__])


def svd(M):
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U, S, V


def draw_warped_keypoint_patch(rchip, kp, **kwargs):
    return draw_keypoint_patch(rchip, kp, warped=True, **kwargs)


def draw_keypoint_patch(rchip, kp, sift=None, warped=False, **kwargs):
    #print('--------------------')
    #print('[extract] Draw Patch')
    if warped:
        wpatch, wkp = get_warped_patch(rchip, kp)
        patch = wpatch
        subkp = wkp
    else:
        patch, subkp = get_patch(rchip, kp)
    #print('[extract] kp    = '+str(kp))
    #print('[extract] subkp = '+str(subkp))
    #print('[extract] patch.shape = %r' % (patch.shape,))
    color = (0, 0, 1)
    # HACK: convert to gray
    from PIL import Image
    patch = np.asarray(Image.fromarray(patch).convert('L'))
    fig, ax = df2.imshow(patch, **kwargs)
    df2.draw_kpts2([subkp], ell_color=color, pts=True)
    if not sift is None:
        df2.draw_sift(sift, [subkp])
    return ax
    #df2.draw_border(df2.gca(), color, 1)


def get_warped_patch(rchip, kp):
    'Returns warped patch around a keypoint'
    (x, y, a, c, d) = kp
    sfx, sfy = kp2_sf(kp)
    s = 41  # sf
    ss = sqrt(s) * 3
    (h, w) = rchip.shape[0:2]
    # Translate to origin(0,0) = (x,y)
    T = np.array([[1, 0, -x],
                  [0, 1, -y],
                  [0, 0,  1]])
    A = np.linalg.inv(
        np.array([[a, 0, 0],
                  [c, d, 0],
                  [0, 0, 1]]))
    S2 = np.array([[ss, 0, 0],
                   [0, ss, 0],
                   [0,  0, 1]])
    X = np.array([[1, 0, s / 2],
                  [0, 1, s / 2],
                  [0, 0,     1]])
    rchip_h, rchip_w = rchip.shape[0:2]
    dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
    M = X.dot(S2).dot(A).dot(T)
    cv2_flags = (cv2.INTER_LINEAR, cv2.INTER_NEAREST)[0]
    cv2_borderMode = cv2.BORDER_CONSTANT
    cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}
    warped_patch = cv2.warpAffine(rchip, M[0:2], tuple(dsize), **cv2_warp_kwargs)
    #warped_patch = cv2.warpPerspective(rchip, M, dsize, **__cv2_warp_kwargs())
    wkp = np.array([(s / 2, s / 2, ss, 0., ss)])
    return warped_patch, wkp


def get_patch(rchip, kp):
    'Returns cropped unwarped patch around a keypoint'
    (x, y, a, c, d) = kp
    sfx, sfy = kp2_sf(kp)
    ratio = max(sfx, sfy) / min(sfx, sfy)
    radx = sfx * ratio
    rady = sfy * ratio
    #print('[get_patch] sfy=%r' % sfy)
    #print('[get_patch] sfx=%r' % sfx)
    #print('[get_patch] ratio=%r' % ratio)
    (chip_h, chip_w) = rchip.shape[0:2]
    #print('[get_patch()] chip wh = (%r, %r)' % (chip_w, chip_h))
    #print('[get_patch()] kp = %r ' % kp)
    quantx = quantize_to_pixel_with_offset(x, radx, 0, chip_w)
    quanty = quantize_to_pixel_with_offset(y, rady, 0, chip_h)
    ix1, ix2, xm = quantx
    iy1, iy2, ym = quanty
    patch = rchip[iy1:iy2, ix1:ix2]
    subkp = kp.copy()  # subkeypoint in patch coordinates
    subkp[0:2] = (xm, ym)
    return patch, subkp


def quantize_to_pixel_with_offset(z, radius, low, high):
    ''' Quantizes a small area into an indexable pixel location
    Returns: pixel_range=(iz1, iz2), subpxl_offset
    Pixels:
    +___+___+___+___+___+___+___+___+
      ^     ^ ^                    ^
      z1    z iz                   z2
            ________________________ < radius
                _____________________ < quantized radius
    ========|
                '''
    #print('quan pxl: z=%r, radius=%r, low=%r, high=%r' % (z, radius, low, high))
    #print('-----------')
    #print('z = %r' % z)
    #print('radius = %r' % radius)
    # Non quantized bounds
    z1 = z - radius
    z2 = z + radius
    #print('bounds = %r, %r' % (z1, z2))
    # Quantized bounds
    iz1 = int(max(np.floor(z1), low))
    iz2 = int(min(np.ceil(z2), high))
    #print('ibounds = %r, %r' % (iz1, iz2))
    # Quantized min radius
    z_radius1 = int(np.ceil(z - iz1))
    z_radius2 = int(np.ceil(iz2 - z))
    z_radius = min(z_radius1, z_radius2)
    #print('z_radius=%r' % z_radius)
    return iz1, iz2, z_radius


def kp2_sf(kp):
    'computes scale factor of keypoint'
    (x, y, a, c, d) = kp
    A = np.array(((a, 0), (c, d)))
    U, S, V = svd(A)
    # sf = np.sqrt(1 / (a * d))
    sfx = S[1]
    sfy = S[0]
    return sfx, sfy
