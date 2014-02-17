from __future__ import division, print_function
from hscom import __common__
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, '[cov]', DEBUG=False)
import numpy as np
from numpy import array
import math
import scipy.stats as stats
from hscom import helpers
from itertools import izip, cycle
from itertools import product as iprod


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


def get_coverage_map(kpts, chip_size):
    np.set_printoptions(threshold=2)
    #kpts = kpts[::10]
    #mean = [0, 0]
    #cov = [[1, 0], [0, 1]]
    #var = stats.norm(loc=0, scale=1)
    #gauss_pxls = np.ones([2, 2])
    (h, w) = chip_size
    print('chip_size=%r' % ((w, h),))
    gauss_xs = np.linspace(-3.14, 3.14, 55)
    gauss_ys = np.linspace(-3.14, 3.14, 55)
    #gauss_xs = np.ones(11)
    #gauss_ys = np.ones(11)

    print('len(kpts)) = %r' % len(kpts))
    print('len(gauss_xs) = %r' % len(gauss_xs))

    try:
        raise KeyError
        #biglist = helpers.load_testdata('biglist')
        #gausspace_score = helpers.load_testdata('gausspace_score')
        #chipspace_xys = helpers.load_testdata('chipspace_xys')
        #gaussimg = helpers.load('gaussimg')
    except KeyError:
        gaussspace_xys = np.array(list(iprod(gauss_xs, gauss_ys)))
        #gausspace_score = np.ones(len(gaussspace_xys))
        gausspace_score = np.array([pdf_norm2d(x, y) for (x, y) in gaussspace_xys])
        #gausspace_score = np.sqrt(gausspace_score)
        #gausspace_score = gausspace_score / gausspace_score.sum()
        gausspace_score -= gausspace_score.min()
        gausspace_score /= gausspace_score.max()
        #gausspace_score = gausspace_score / (gausspace_score.max() - gausspace_score.min())
        #gausspace_score[gausspace_score < 0] = 0
        #gausspace_score = gausspace_score - gausspace_score.min()
        #gausspace_score /= gausspace_score.sum()
        print('gausspace_score')
        print(gausspace_score)

        #gausspace_score = np.sqrt(var.pdf(gaussspace_xys ** 2).sum(1))
        #gausspace_score = 1 - var.pdf(gaussspace_xys)[:, 0]
        w_, h_ = (len(gauss_xs), len(gauss_ys))
        gaussimg = gausspace_score.reshape((w_, h_)).T
        gaussimg = np.array(gaussimg, dtype=np.float32)
        np.set_printoptions(precision=3, threshold=1000000, linewidth=180)
        #from hsviz import draw_func2 as df2
        #df2.imshow(gaussimg, fnum=432)

        #biglist = np.array([(x1, y1, x, y, a, c, d)
                            #for x1, y1 in gaussspace_xys
                            #for x, y, a, c, d in kpts])
        #x1s, y1s, xs, ys, as_, cs, ds = biglist.T

        #chipspace_xys = np.array(((as_ * x1s) + xs,
                                  #(cs  * x1s) + (ds * y1s) + ys)).T
        #chipspace_xys = np.vstack([gausspace_xys] * len(kpts))
        #helpers.save_testdata('biglist')
        #helpers.save_testdata('chipspace_xys')
        #helpers.save_testdata('gausspace_score')
        #helpers.save_testdata('gaussimg')

    import cv2
    area_matrix = np.zeros((h, w), dtype=np.float32)
    area_matrix_copy = area_matrix.copy().T
    (h_, w_) = gaussimg.shape
    for count, (x, y, a, c, d) in enumerate(kpts):
        T1 = np.array(((1, 0, -(w_ / 2)),
                       (0, 1, -(h_ / 2)),
                       (0, 0, 1)))

        S1 = np.array([[1 / (w_), 0,  0],
                       [0,  1 / (h_),  0],
                       [0,  0,  1]], np.float64)

        aff = np.array(((a, 0, x),
                        (c, d, y),
                        (0, 0, 1),))

        M = aff.dot(S1).dot(T1)
        M = M[0:2]
        flags = cv2.INTER_NEAREST  # cv2.INTER_LANCZOS4
        #dsize = area_matrix_copy.shape[::-1]
        dsize = area_matrix_copy.shape[::1]  # shape[::-1]
        boderMode = cv2.BORDER_CONSTANT
        warped = cv2.warpAffine(gaussimg, M, dsize,
                                flags=flags, borderMode=boderMode,
                                borderValue=0).T
        catmat = np.dstack((warped.T, area_matrix))
        newmat = catmat.max(-1)
        area_matrix = newmat
        from hsviz import draw_func2 as df2
        if count % 80 == 79:
            #print(area_matrix)
            area_img = np.array(np.round(area_matrix * 255), dtype=np.uint8)
            df2.imshow(warped, fnum=count + 10)
            df2.draw()
    print('area_matrix.max()')

    print(area_matrix.max())
    print(area_matrix.min())
    area_matrix = area_img

    return area_img
