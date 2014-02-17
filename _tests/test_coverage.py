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


if __name__ == '__main__':
    exec_str = 'exec(open("_tests/test_coverage.py").read())'
    multiprocessing.freeze_support()
    np.set_printoptions(precision=2, threshold=1000000, linewidth=180)
    #from hsdev import dev_api
    #dev_api.rrr()
    #dev_api._reload()
    # --- LOAD TABLES --- #
    hs = test_api.main(preload=True)
    # Test variables
    valid_cxs = hs.get_valid_cxs()
    cx = valid_cxs[0]
    kpts = hs.get_kpts(cx)
    chip = hs.get_chip(cx)
    chip_size = chip.shape[0:2]

    coverage.rrr()

    np.tau = 2 * np.pi
    fnum = 2

    srcimg = coverage.get_gaussimg(2, 5)
    print(srcimg)
    dstimg = coverage.warp_srcimg_to_kpts(kpts, srcimg, chip_size, 0.5)
    df2.imshow(srcimg * 255, fnum=fnum)
    fnum += 1
    df2.imshow(dstimg * 255, fnum=fnum)
    fnum += 1
    dstimg2 = dstimg.copy()
    dstimg2[dstimg2 > 0] = 1
    df2.imshow(dstimg2 * 255, fnum=fnum)
    fnum += 1
    df2.update()

    percent = coverage.get_keypoint_coverage(kpts, chip_size, dstimg=dstimg)

    # Show Result
    print('coverage = %r%%' % percent)
    viz.show_chip(hs, cx, fnum=fnum)
    fnum += 1
    exec(viz.df2.present())
