
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
from hscom import helpers as util
from . import matching_functions as mf

SCALE_FACTOR_DEFAULT = .05
METHOD_DEFAULT = 0


def score_chipmatch_coverage(hs, qcx, chipmatch, qreq, method=0):
    prescore_method = 'csum'
    nShortlist = 100
    dcxs_ = set(qreq._dcxs)
    (cx2_fm, cx2_fs, cx2_fk) = chipmatch
    cx2_prescore = mf.score_chipmatch(hs, qcx, chipmatch, prescore_method, qreq)
    topx2_cx = cx2_prescore.argsort()[::-1]  # Only allow indexed cxs to be in the top results
    topx2_cx = [cx for cx in iter(topx2_cx) if cx in dcxs_]
    nRerank = min(len(topx2_cx), nShortlist)
    cx2_score = [0 for _ in range(len(cx2_fm))]
    mark_progress, end_progress = util.progress_func(nRerank, flush_after=10,
                                                     lbl='[cov] Compute coverage')
    for topx in range(nRerank):
        mark_progress(topx)
        cx2 = topx2_cx[topx]
        fm = cx2_fm[cx2]
        fs = cx2_fs[cx2]
        covscore = get_match_coverage_score(hs, qcx, cx2, fm, fs, method=method)
        cx2_score[cx2] = covscore
    end_progress()
    return cx2_score


def get_match_coverage_score(hs, cx1, cx2, fm, fs, **kwargs):
    if len(fm) == 0:
        return 0
    if not 'scale_factor' in kwargs:
        kwargs['scale_factor'] = SCALE_FACTOR_DEFAULT
    if not 'method' in kwargs:
        kwargs['method'] = METHOD_DEFAULT
    sel_fx1, sel_fx2 = fm.T
    method = kwargs.get('method', 0)
    score1 = get_cx_match_covscore(hs, cx1, sel_fx1, fs, **kwargs)
    if method in [0, 2]:
        # 0 and 2 use both score
        score2 = get_cx_match_covscore(hs, cx2, sel_fx2, fs, **kwargs)
        covscore = (score1 + score2) / 2
    elif method in [1, 3]:
        # 1 and 3 use just score 1
        covscore = score1
    else:
        raise NotImplemented('[cov] method=%r' % method)
    return covscore


def get_cx_match_covscore(hs, cx, sel_fx, mx2_score, **kwargs):
    dstimg = get_cx_match_covimg(hs, cx, sel_fx, mx2_score, **kwargs)
    score = dstimg.sum() / (dstimg.shape[0] * dstimg.shape[1])
    return score


def get_cx_match_covimg(hs, cx, sel_fx, mx2_score, **kwargs):
    chip = hs.get_chip(cx)
    kpts = hs.get_kpts(cx)
    mx2_kp = kpts[sel_fx]
    srcimg = get_gaussimg()
    # 2 and 3 are scale modes
    if kwargs.get('method', 0) in [2, 3]:
        # Bigger keypoints should get smaller weights
        mx2_scale = np.sqrt([a * d for (x, y, a, c, d) in mx2_kp])
        mx2_score = mx2_score / mx2_scale
    dstimg = warp_srcimg_to_kpts(mx2_kp, srcimg, chip.shape[0:2],
                                 fx2_score=mx2_score, **kwargs)
    return dstimg


def get_match_coverage_images(hs, cx1, cx2, fm, mx2_score, **kwargs):
    sel_fx1, sel_fx2 = fm.T
    dstimg1 = get_cx_match_covimg(hs, cx1, sel_fx1, mx2_score, **kwargs)
    dstimg2 = get_cx_match_covimg(hs, cx1, sel_fx1, mx2_score, **kwargs)
    return dstimg1, dstimg2


def warp_srcimg_to_kpts(fx2_kp, srcimg, chip_shape, fx2_score=None, **kwargs):
    if len(fx2_kp) == 0:
        return None
    if fx2_score is None:
        fx2_score = np.ones(len(fx2_kp))
    scale_factor = kwargs.get('scale_Factor', SCALE_FACTOR_DEFAULT)
    # Build destination image
    (h, w) = list(map(int, (chip_shape[0] * scale_factor, chip_shape[1] * scale_factor)))
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
    mark_progress, end_progress = util.progress_func(len(fx2_M),
                                                     flush_after=20,
                                                     mark_after=1000,
                                                     lbl='coverage warp ')
    # For each keypoint warp a gaussian scaled by the feature score
    # into the image
    count = 0
    for count, (M, score) in enumerate(zip(fx2_M, fx2_score)):
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


def get_coverage_map(kpts, chip_shape, **kwargs):
    # Create gaussian image to warp
    np.tau = 2 * np.pi
    srcimg = get_gaussimg()
    dstimg = warp_srcimg_to_kpts(kpts, srcimg, chip_shape, **kwargs)
    return dstimg
