from __future__ import division, print_function
from hscom import __common__
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, '[cov]', DEBUG=False)
# Standard
from itertools import product as iprod
import math
# Science
import cv2
import numpy as np
# HotSpotter
from hscom import helpers


def get_coverage(kpts, chip_size):
    area_matrix = get_coverage_map(kpts, chip_size)
    return 0


def pdf_norm2d(x_, y_):
    x = np.array([x_, y_])
    sigma = np.eye(2)
    mu = np.array([0, 0])
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
    np.tau = 2 * np.pi
    norm_const = 1.0 / ( math.pow(np.tau, float(size) / 2) * math.pow(det, 1.0 / 2))
    x_mu = np.matrix(x - mu)
    inv = np.linalg.inv(sigma)
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return norm_const * result


def get_gaussimg(width, resolution):
    half_width = width / 2
    gauss_xs = np.linspace(-half_width, half_width, resolution)
    gauss_ys = np.linspace(-half_width, half_width, resolution)

    gaussspace_xys = np.array(list(iprod(gauss_xs, gauss_ys)))
    gausspace_score = np.array([pdf_norm2d(x, y) for (x, y) in gaussspace_xys])
    gausspace_score -= gausspace_score.min()
    gausspace_score /= gausspace_score.max()

    size = (resolution, resolution)
    gaussimg = gausspace_score.reshape(size).T
    gaussimg = np.array(gaussimg, dtype=np.float32)
    return gaussimg


def build_transforms(kpts, chip_size, src_size):
    (h, w) = chip_size
    (h_, w_) = src_size
    T1 = np.array(((1, 0, -w_ / 2),
                   (0, 1, -h_ / 2),
                   (0, 0,       1),))
    S1 = np.array(((1 / w_,      0,  0),
                   (0,      1 / h_,  0),
                   (0,           0,  1),))
    aff_list = [np.array(((a, 0, x),
                          (c, d, y),
                          (0, 0, 1),)) for (x, y, a, c, d) in kpts]
    perspective_list = [A.dot(S1).dot(T1) for A in aff_list]
    transform_list = [M[0:2] for M in perspective_list]
    return transform_list


def warp_srcimg_to_kpts(kpts, srcimg, chip_size):
    (h, w) = chip_size
    dst_img = np.zeros((h, w), dtype=np.float32)
    dst_copy = dst_img.copy()
    src_size = srcimg.shape
    transform_list = build_transforms(kpts, (h, w), src_size)
    flags = cv2.INTER_LINEAR  # cv2.INTER_LANCZOS4
    dsize = (w, h)
    boderMode = cv2.BORDER_CONSTANT
    mark_progress, end_progress = helpers.progress_func(len(transform_list))
    for count, M in enumerate(transform_list):
        if count % 10 == 0:
            mark_progress(count)
        warped = cv2.warpAffine(srcimg, M, dsize,
                                dst=dst_copy,
                                flags=flags, borderMode=boderMode,
                                borderValue=0).T
        catmat = np.dstack((warped.T, dst_img))
        dst_img = catmat.max(axis=2)
    mark_progress(count)
    end_progress()
    return dst_img


def get_coverage_map(kpts, chip_size):
    # Create gaussian image to warp
    np.tau = 2 * np.pi
    srcimg = get_gaussimg(np.tau, 55)
    dst_img = warp_srcimg_to_kpts(kpts, srcimg, chip_size)
    return dst_img
