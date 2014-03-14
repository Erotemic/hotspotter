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
import vtool.image as gtool


np.tau = 2 * np.pi  # tauday.com


def patch_gradient(image, ksize=1):
    image_ = array(image, dtype=np.float64)
    gradx = cv2.Sobel(image_, cv2.CV_64F, 1, 0, ksize=ksize)
    grady = cv2.Sobel(image_, cv2.CV_64F, 0, 1, ksize=ksize)
    return gradx, grady


def patch_mag(gradx, grady):
    return np.sqrt((gradx ** 2) + (grady ** 2))


def patch_ori(gradx, grady):
    'returns patch orientation relative to the gravity vector'
    gori = np.arctan2(grady, gradx)  # outputs from -pi to pi
    gori[gori < 0] = gori[gori < 0] + np.tau  # map to 0 to tau (keep coords)
    gori = (gori + ktool.GRAVITY_THETA) % np.tau  # normalize relative to gravity
    return gori


def get_unwarped_patches(rchip, kpts):
    'Returns cropped unwarped patch around a keypoint'
    _xs, _ys = ktool.get_xys(kpts)
    xyexnts = ktool.get_xy_axis_extents(kpts=kpts)
    patches = []
    subkpts = []

    for (kp, x, y, (sfx, sfy)) in izip(kpts, _xs, _ys, xyexnts):
        radius_x = sfx * 1.5
        radius_y = sfy * 1.5
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
    'Returns warped patch around a keypoint'
    # TODO: CLEAN ME
    warped_patches = []
    warped_subkpts = []
    xs, ys = ktool.get_xys(kpts)
    # rotate relative to the gravity vector
    oris = ktool.get_oris(kpts)
    invV_mats = ktool.get_invV_mats(kpts, with_trans=False, ashomog=True)
    V_mats = ktool.get_V_mats(invV_mats)
    kpts_iter = izip(xs, ys, V_mats, oris)
    s = 41  # sf
    for x, y, V, ori in kpts_iter:
        ss = sqrt(s) * 3
        (h, w) = rchip.shape[0:2]
        # Translate to origin(0,0) = (x,y)
        T = ltool.translation_mat(-x, -y)
        R = ltool.rotation_mat(-ori)
        S = ltool.scale_mat(ss)
        X = ltool.translation_mat(s / 2, s / 2)
        M = X.dot(S).dot(R).dot(V).dot(T)
        # Prepare to warp
        dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
        # Warp
        warped_patch = gtool.warpAffine(rchip, M, dsize)
        # Build warped keypoints
        wkp = np.array((s / 2, s / 2, ss, 0., ss, 0))
        warped_patches.append(warped_patch)
        warped_subkpts.append(wkp)
    return warped_patches, warped_subkpts


def get_warped_patch(imgBGR, kp, gray=False):
    kpts = np.array([kp])
    wpatches, wkpts = get_warped_patches(imgBGR, kpts)
    wpatch = wpatches[0]
    wkp = wkpts[0]
    if gray:
        wpatch = gtool.cvt_BGR2L(wpatch)
    return wpatch, wkp


def get_orientation_histogram(gori):
    # Get wrapped histogram (because we are finding a direction)
    hist_, edges_ = np.histogram(gori.flatten(), bins=8)
    hist, edges = htool.wrap_histogram(hist_, edges_)
    centers = htool.hist_edges_to_centers(edges)
    return hist, centers


def find_kpts_direction(imgBGR, kpts):
    ori_list = []
    for kp in kpts:
        patch, wkp = get_warped_patch(imgBGR, kp, gray=True)
        gradx, grady = patch_gradient(patch)
        gori = patch_ori(gradx, grady)
        hist, centers = get_orientation_histogram(gori)
        # Find submaxima
        submaxima_x, submaxima_y = htool.hist_interpolated_submaxima(hist, centers)
        ori = submaxima_x[submaxima_y.argmax()] % np.tau
        ori_list.append(ori)
    _oris = np.array(ori_list, dtype=kpts.dtype)
    # discard old orientatiosn if they exist
    kpts2 = np.vstack([kpts[:, 0:5].T, _oris]).T
    return kpts2
