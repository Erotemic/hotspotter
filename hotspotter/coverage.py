from __future__ import division, print_function
from hscom import __common__
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, '[cov]', DEBUG=False)
# Standard
from itertools import izip
from itertools import product as iprod
import math
# Science
import cv2
import numpy as np
# HotSpotter
from hscom import helpers
import matching_functions as mf


def score_chipmatch_coverage(hs, qcx, chipmatch, qdat):
    prescore_method = 'csum'
    nShortlist = 100
    dcxs_ = set(qdat._dcxs)
    (cx2_fm, cx2_fs, cx2_fk) = chipmatch
    cx2_prescore = mf.score_chipmatch(hs, qcx, chipmatch, prescore_method, qdat)
    topx2_cx = cx2_prescore.argsort()[::-1]  # Only allow indexed cxs to be in the top results
    topx2_cx = [cx for cx in iter(topx2_cx) if cx in dcxs_]
    nRerank = min(len(topx2_cx), nShortlist)
    cx2_score = [0 for _ in xrange(len(cx2_fm))]
    mark_progress, end_progress = helpers.progress_func(nRerank,
                                                        flush_after=10,
                                                        lbl='[cov] Compute coverage ')
    for topx in xrange(nRerank):
        mark_progress(topx)
        cx2 = topx2_cx[topx]
        fm = cx2_fm[cx2]
        fs = cx2_fs[cx2]
        covscore = get_match_coverage_score(hs, qcx, cx2, fm, fs)
        cx2_score[cx2] = covscore
    end_progress()
    return cx2_score


def get_match_coverage_score(hs, cx1, cx2, fm, fs):
    if len(fm) == 0:
        return 0
    dstimg1, dstimg2 = get_match_coverage_images(hs, cx1, cx2, fm, fs, scale_factor=.1)
    score1 = dstimg1.sum() / (dstimg1.shape[0] * dstimg1.shape[1])
    score2 = dstimg2.sum() / (dstimg2.shape[0] * dstimg2.shape[1])
    covscore = (score1 + score2) / 2
    return covscore


def get_match_coverage_images(hs, cx1, cx2, fm, mx2_score, scale_factor):
    chip1 = hs.get_chip(cx1)
    chip2 = hs.get_chip(cx2)
    kpts1_m = hs.get_kpts(cx1)[fm[:, 0]]
    kpts2_m = hs.get_kpts(cx2)[fm[:, 1]]

    srcimg = get_gaussimg()
    dstimg1 = warp_srcimg_to_kpts(kpts1_m, srcimg, chip1.shape[0:2],
                                  fx2_score=mx2_score, scale_factor=scale_factor)

    dstimg2 = warp_srcimg_to_kpts(kpts2_m, srcimg, chip2.shape[0:2],
                                  fx2_score=mx2_score, scale_factor=scale_factor)
    return dstimg1, dstimg2


def get_keypoint_coverage(kpts, chip_shape, dstimg=None, scale_factor=.2):
    if dstimg is None:
        dstimg = get_coverage_map(kpts, chip_shape, scale_factor=scale_factor)
    percent = dstimg.sum() / dstimg.size
    return percent


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


def get_gaussimg(width=3, resolution=7):
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


def build_transforms(kpts, chip_shape, src_shape, scale_factor):
    (h, w) = chip_shape
    (h_, w_) = src_shape
    T1 = np.array(((1, 0, -w_ / 2),
                   (0, 1, -h_ / 2),
                   (0, 0,       1),))
    S1 = np.array(((1 / w_,      0,  0),
                   (0,      1 / h_,  0),
                   (0,           0,  1),))
    aff_list = [np.array(((a, 0, x),
                          (c, d, y),
                          (0, 0, 1),)) for (x, y, a, c, d) in kpts]
    S2 = np.array(((scale_factor,      0,  0),
                   (0,      scale_factor,  0),
                   (0,           0,  1),))
    perspective_list = [S2.dot(A).dot(S1).dot(T1) for A in aff_list]
    transform_list = [M[0:2] for M in perspective_list]
    return transform_list


def warp_srcimg_to_kpts(fx2_kp, srcimg, chip_shape, fx2_score=None, scale_factor=.2):
    if len(fx2_kp) == 0:
        return None
    if fx2_score is None:
        fx2_score = np.ones(len(fx2_kp))
    # Build destination image
    (h, w) = map(int, (chip_shape[0] * scale_factor, chip_shape[1] * scale_factor))
    dstimg = np.zeros((h, w), dtype=np.float32)
    dst_copy = dstimg.copy()
    src_shape = srcimg.shape
    # Build keypoint transforms
    fx2_M = build_transforms(fx2_kp, (h, w), src_shape, scale_factor)
    # cv2 warp flags
    dsize = (w, h)
    flags = cv2.INTER_LINEAR  # cv2.INTER_LANCZOS4
    boderMode = cv2.BORDER_CONSTANT
    # mark prooress
    mark_progress, end_progress = helpers.progress_func(len(fx2_M),
                                                        flush_after=20,
                                                        mark_after=1000,
                                                        lbl='coverage warp ')
    # For each keypoint warp a gaussian scaled by the feature score
    # into the image
    count = 0
    for count, (M, score) in enumerate(izip(fx2_M, fx2_score)):
        mark_progress(count)
        warped = cv2.warpAffine(srcimg * score, M, dsize,
                                dst=dst_copy,
                                flags=flags, borderMode=boderMode,
                                borderValue=0).T
        catmat = np.dstack((warped.T, dstimg))
        dstimg = catmat.max(axis=2)
    mark_progress(count)
    end_progress()
    return dstimg


def get_coverage_map(kpts, chip_shape, scale_factor=.2):
    # Create gaussian image to warp
    np.tau = 2 * np.pi
    srcimg = get_gaussimg()
    dstimg = warp_srcimg_to_kpts(kpts, srcimg, chip_shape, scale_factor=scale_factor)
    return dstimg
