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
    # Run Test
    area_img = coverage.get_coverage_map(kpts, chip_size)
    #area_img = np.round(area_matrix * 255)
    fnum = 2
    df2.imshow(area_img, fnum=fnum)
    fnum += 1
    df2.update()

    #percent = coverage.get_coverage(kpts, chip_size)

    # Show Result
    #print('coverage = %r' % percent)
    viz.show_chip(hs, cx, fnum=fnum)
    fnum += 1
    exec(viz.df2.present())
