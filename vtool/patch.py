# LICENCE
from __future__ import print_function, division
# Python
from itertools import izip
# Science
import cv2
import numpy as np
from numpy import array, sqrt
# VTool
import vtool.histogram as htool
import vtool.keypoint as ktool
import vtool.linalg as ltool


def patch_gradient(image, ksize=1):
    image_ = array(image, dtype=np.float64)
    gradx = cv2.Sobel(image_, cv2.CV_64F, 1, 0, ksize=ksize)
    grady = cv2.Sobel(image_, cv2.CV_64F, 0, 1, ksize=ksize)
    return gradx, grady


def patch_mag(gradx, grady):
    return np.sqrt((gradx ** 2) + (grady ** 2))


def patch_ori(gradx, grady):
    np.tau = 2 * np.pi
    gori = np.arctan2(grady, gradx)  # outputs from -pi to pi
    gori[gori < 0] = gori[gori < 0] + np.tau  # map to 0 to tau (keep coords)
    return gori


def get_unwarped_patches(rchip, kpts):
    # TODO: CLEAN ME (FIX CROP EXTENT PROBLEMS. It is an issue with svd or no skew?)
    'Returns cropped unwarped patch around a keypoint'
    _xs, _ys = ktool.get_xys(kpts)
    S_list = ktool.orthogonal_scales(kpts=kpts)
    patches = []
    subkpts = []

    for (kp, x, y, (sfy, sfx)) in izip(kpts, _xs, _ys, S_list):
        ratio = np.sqrt(max(sfx, sfy) / min(sfx, sfy))
        radius_x = sfx * ratio
        radius_y = sfy * ratio
        (chip_h, chip_w) = rchip.shape[0:2]
        # Get integer grid coordinates to crop at
        ix1, ix2, xm = htool.subbin_bounds(x, radius_x, 0, chip_w)
        iy1, iy2, ym = htool.subbin_bounds(y, radius_y, 0, chip_h)
        # Crop the keypoint out of the image
        patch = rchip[iy1:iy2, ix1:ix2]
        subkp = kp.copy()  # subkeypoint in patch coordinates
        subkp[0:2] = (xm, ym)
        patches.append(patch)
        subkpts.append(subkp)
    return patches, subkpts


def get_warped_patches(rchip, kpts):
    # TODO: CLEAN ME
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
    return warped_patches, warped_subkpts


def get_patch(imgBGR, kp):
    kpts = np.array([kp])
    wpatches, wkps = get_warped_patches(imgBGR, kpts)
    wpatch = wpatches[0]
    wpatchLAB = cv2.cvtColor(wpatch, cv2.COLOR_BGR2LAB)
    wpatchL = wpatchLAB[:, :, 0]
    return wpatchL


def get_orientation_histogram(gori):
    # Get wrapped histogram (because we are finding a direction)
    hist_, edges_ = np.histogram(gori.flatten(), bins=8)
    hist, edges = htool.wrap_histogram(hist_, edges_)
    centers = htool.hist_edges_to_centers(edges)
    return hist, centers


def find_kpts_direction(imgBGR, kpts):
    theta_list = []
    for kp in kpts:
        patch = get_patch(imgBGR, kp)
        gradx, grady = patch_gradient(patch)
        gori = patch_ori(gradx, grady)
        hist, centers = get_orientation_histogram(gori)
        # Find submaxima
        maxima_x, maxima_y, argmaxima = htool.hist_argmaxima(hist, centers)
        submaxima_x, submaxima_y = htool.interpolate_submaxima(argmaxima, hist, centers)
        theta = submaxima_x[submaxima_y.argmax()]
        theta_list.append(theta)
    print(kpts.shape)
    print(len(theta_list))
    kpts2 = np.vstack([kpts.T, theta_list]).T
    return kpts2
