#!/usr/env python
from __future__ import division, print_function
from hotspotter import HotSpotterAPI as api
from hotspotter import feature_compute2 as fc2
from hscom import helpers
from hscom import helpers as util
from hsviz import viz
from hscom import argparse2
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('[fc2] __main__ = feature_compute2.py')
    # Read Args
    cx = helpers.get_arg('--cx', type_=int)
    delete_features = helpers.get_flag('--delete-features', default=False)
    nRandKpts = helpers.get_arg('--nRandKpts', type_=int)
    # Debugging vars
    feat_cfg = None
    cx_list = None
    kwargs = {}
    # --- LOAD TABLES --- #
    args = argparse2.parse_arguments(db='NAUTS')
    hs = api.HotSpotter(args)
    hs.load_tables()
    # --- LOAD CHIPS --- #
    hs.update_samples()
    hs.load_chips()
    # Delete features if needed
    if delete_features:
        fc2.clear_feature_cache(hs)
    # --- LOAD FEATURES --- #
    fc2.load_features(hs)
    if not cx is None:
        viz.show_chip(hs, cx, nRandKpts=nRandKpts)
    else:
        print('usage: feature_compute.py --cx [cx] --nRandKpts [num] [--delete-features]')

    exec(viz.df2.present())
