#exec(open('__init__.py').read())
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off,
 rrr, profile, printDBG) = __common__.init(__name__, '[extract]', DEBUG=False)
# Science
from itertools import izip
import cv2
import numpy as np
from numpy import sqrt
# Hotspotter
import draw_func2 as df2
from hscom import util
import vtool.keypoint as ktool
import vtool.linalg as ltool


def rrr():
    import imp
    import sys
    print('[extract] Reloading: ' + __name__)
    imp.reload(sys.modules[__name__])


def draw_warped_keypoint_patch(rchip, kp, **kwargs):
    return draw_keypoint_patch(rchip, kp, warped=True, **kwargs)


@util.indent_decor('[extract_patch.dkp]')
def draw_keypoint_patch(rchip, kp, sift=None, warped=False, **kwargs):
    #print('--------------------')
    #print('[extract] Draw Patch')
    kpts = np.array([kp])
    if warped:
        wpatches, wkpts = get_warped_patches(rchip, kpts)
        patches = wpatches
        subkpts = wkpts
    else:
        patches, subkpts = get_unwarped_patches(rchip, kpts)
    print('[extract] kpts[0]    = %r' % (kpts[0]))
    print('[extract] subkpts[0] = %r' % (subkpts[0]))
    print('[extract] patches[0].shape = %r' % (patches[0].shape,))
    color = (0, 0, 1)
    patch = patches[0]
    fig, ax = df2.imshow(patch, **kwargs)
    sifts = np.array([sift])
    subkpts_ = np.array(subkpts)
    df2.draw_kpts2(subkpts_, ell_color=color, pts=True, sifts=sifts)
    #if not sift is None:
        #df2.draw_sift(sift, subkpts)
    return ax
    #df2.draw_border(df2.gca(), color, 1)


def get_warped_patches(rchip, kpts):
    'Returns warped patch around a keypoint'
    warped_patches = []
    warped_subkpts = []
    xs, ys = ktool.get_xys(kpts)
    oris = ktool.get_oris(kpts)
    invV_mats = ktool.get_invV_mats(kpts, with_trans=False, ashomog=True)
    V_mats = ktool.get_V_mats(invV_mats)
    S_list = ktool.orthogonal_scales(invV_mats)
    kpts_iter = izip(xs, ys, V_mats, oris, S_list)
    s = 41  # sf
    try:
        for x, y, V, ori, (sfx, sfy) in kpts_iter:
            ss = sqrt(s) * 3
            (h, w) = rchip.shape[0:2]
            # Translate to origin(0,0) = (x,y)
            T = ltool.translation_mat(-x, -y)
            R = ltool.rotation_mat(ori)
            S = ltool.scale_mat(ss)
            X = ltool.translation_mat(s / 2, s / 2)
            M = X.dot(S).dot(R).dot(V).dot(T)
            # Prepare to warp
            rchip_h, rchip_w = rchip.shape[0:2]
            dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
            cv2_flags = cv2.INTER_LANCZOS4
            cv2_borderMode = cv2.BORDER_CONSTANT
            cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}
            # Warp
            warped_patch = cv2.warpAffine(rchip, M[0:2], tuple(dsize), **cv2_warp_kwargs)
            # Build warped keypoints
            wkp = np.array((s / 2, s / 2, ss, 0., ss, ori))
            warped_patches.append(warped_patch)
            warped_subkpts.append(wkp)
    except Exception as ex:
        print(ex)
        #util.embed()
        raise
    return warped_patches, warped_subkpts


def get_unwarped_patches(rchip, kpts):
    'Returns cropped unwarped patch around a keypoint'
    _xs, _ys = ktool.get_xys(kpts)
    S_list = ktool.orthogonal_scales(kpts=kpts)
    patches = []
    subkpts = []

    for (kp, x, y, (sfx, sfy)) in izip(kpts, _xs, _ys, S_list):
        sfx, sfy = S_list[0, :]
        ratio = max(sfx, sfy) / min(sfx, sfy)
        radius_x = sfx * ratio
        radius_y = sfy * ratio
        (chip_h, chip_w) = rchip.shape[0:2]
        # Get integer grid coordinates to crop at
        quantx = quantize_to_pixel_with_offset(x, radius_x, 0, chip_w)
        quanty = quantize_to_pixel_with_offset(y, radius_y, 0, chip_h)
        ix1, ix2, xm = quantx
        iy1, iy2, ym = quanty
        # Crop the keypoint out of the image
        patch = rchip[iy1:iy2, ix1:ix2]
        subkp = kp.copy()  # subkeypoint in patch coordinates
        subkp[0:2] = (xm, ym)
        patches.append(patch)
        subkpts.append(subkp)
    return patches, subkpts


def quantize_to_pixel_with_offset(z, radius, low, high):
    '''
    Quantizes a small area into an indexable pixel location
    Useful for extracting the range of a keypoint from an image.
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
