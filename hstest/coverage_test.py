#!/usr/bin/env python
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr, profile,
 printDBG) = __common__.init(__name__, '[covt]', DEBUG=False)

from hsviz import viz
import multiprocessing
import numpy as np  # NOQA
from hsdev import test_api
from hotspotter import coverage
from hsviz import draw_func2 as df2


def test_find_coverage_score(hs, res):
    qreq = hs.qreq
    chipmatch = (res.cx2_fm, res.cx2_fs, res.cx2_fk)
    qcx = res.qcx
    #qcx2_chipmatch = {qcx: chipmatch}
    cx2_score = coverage.score_chipmatch_coverage(hs, qcx, chipmatch, qreq)
    res.cx2_score
    #print('=========')
    #print('')
    #print('new score')
    #print(cx2_score)
    #print('')
    #print('old score')
    #print(res.cx2_score)
    #print('=========')
    return cx2_score


def test_result_coverage(hs, res, cx2, scale_factor):
    cx1 = res.qcx
    fm = res.cx2_fm[cx2]
    fs = res.cx2_fs[cx2]

    chip1 = hs.get_chip(cx1)
    chip2 = hs.get_chip(cx2)
    kpts1_m = hs.get_kpts(cx1)[fm[:, 0]]
    kpts2_m = hs.get_kpts(cx2)[fm[:, 1]]

    dstimg1 = coverage.warp_srcimg_to_kpts(kpts1_m, srcimg, chip1.shape[0:2],
                                           fx2_score=fs, scale_factor=scale_factor)

    dstimg2 = coverage.warp_srcimg_to_kpts(kpts2_m, srcimg, chip2.shape[0:2],
                                           fx2_score=fs, scale_factor=scale_factor)

    args_ = (kpts1_m, kpts2_m)
    kwargs_ = dict(fs=fs, scale_factor=scale_factor, draw_ell=False,
                   draw_lines=False, heatmap=True)
    return dstimg1, dstimg2, args_, kwargs_


if __name__ == '__main__':
    exec_str = 'exec(open("_tests/test_coverage.py").read())'
    multiprocessing.freeze_support()
    np.set_printoptions(precision=2, threshold=1000000, linewidth=180)
    # --- LOAD TABLES --- #
    hs = test_api.main(preload=True)
    # Test variables
    cx = test_api.get_test_cxs(hs, 1)[0]
    if hs.get_db_name() == 'HSDB_zebra_with_mothers':
        cx = hs.cid2_cx(13)
    kpts = hs.get_kpts(cx)
    chip = hs.get_chip(cx)
    chip_shape = chip.shape[0:2]
    np.tau = 2 * np.pi
    fnum = 2
    # Reload
    test_api.reload_all()
    coverage.rrr()

    scale_factor = .1

    # Get source gaussian
    srcimg = coverage.get_gaussimg()

    # Get Chip Coverage
    dstimg = coverage.warp_srcimg_to_kpts(kpts, srcimg, chip_shape,
                                          scale_factor=scale_factor)
    dstimg_thresh = dstimg.copy()
    dstimg_thresh[dstimg_thresh > 0] = 1

    # Get matching coverage
    hs.prefs.query_cfg.agg_cfg.score_method = 'coverage'
    print(hs.get_cache_uid())
    res = hs.query(cx)
    nTop = 2
    for tx in xrange(nTop):
        cx2 = res.topN_cxs(hs)[tx]
        dstimg1, dstimg2, args_, kwargs_ = test_result_coverage(hs, res, cx2, scale_factor)
        test_find_coverage_score(hs, res)
        res.show_chipres(hs, cx2, fnum=fnum)
        df2.set_figtitle('matching viz' + str(tx), incanvas=False)
        fnum += 1

        df2.show_chipmatch2(dstimg1, dstimg2, *args_, fnum=fnum, **kwargs_)
        df2.set_figtitle('matching coverage' + str(tx))
        fnum += 1

    df2.imshow(srcimg, fnum=fnum, heatmap=True)
    df2.set_figtitle('gaussian weights')
    fnum += 1

    df2.imshow(dstimg, fnum=fnum, heatmap=True)
    df2.set_figtitle('chip coverage map')
    fnum += 1

    df2.imshow(dstimg_thresh, fnum=fnum, heatmap=True)
    df2.set_figtitle('thresholded chip coverage map')
    fnum += 1

    viz.show_chip(hs, cx, fnum=fnum)
    df2.set_figtitle('chip', incanvas=False)
    fnum += 1
    exec(viz.df2.present())
