from hotspotter import HotSpotterAPI as api
from hotspotter import chip_compute2 as cc2
from hscom import argparse2
from hscom import helpers
from hscom import helpers as util
from hsviz import viz
import multiprocessing
import numpy as np  # NOQA

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Debugging vars
    chip_cfg = None
#l')=103.7900s
    cx_list = None
    kwargs = {}
    # --- LOAD TABLES --- #
    args = argparse2.parse_arguments(defaultdb='NAUTS')
    hs = api.HotSpotter(args)
    hs.load_tables()
    hs.update_samples()
    # --- LOAD CHIPS --- #
    force_compute = helpers.get_flag('--force', default=False)
    cc2.load_chips(hs, force_compute=force_compute)
    cx = helpers.get_arg('--cx', type_=int)
    if not cx is None:
        #tau = np.pi * 2
        #hs.change_theta(cx, tau / 8)
        viz.show_chip(hs, cx, draw_kpts=False, fnum=1)
        viz.show_image(hs, hs.cx2_gx(cx), fnum=2)
    else:
        print('usage: feature_compute.py --cx [cx]')
    exec(viz.df2.present())
