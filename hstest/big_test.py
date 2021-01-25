#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG

import builtins
from hsdev import test_api
import multiprocessing
import sys
import numpy as np
from hsdev import dbgimport  # NOQA

INTERACTIVE = '--interactive' in sys.argv or '-i' in sys.argv


def print(msg):
    builtins.print('\n=============================')
    builtins.print(msg)
    if INTERACTIVE:
        input('press enter to continue')


if __name__ == '__main__':
    #dbgimport.hsdev_printoff()
    dbgimport.hsgui_printoff()
    dbgimport.hsviz_printoff()
    dbgimport.hscom_printoff()

    print('[TEST] BIGTEST')
    multiprocessing.freeze_support()

    print('[TEST] CLONEDB')
    dbname = 'NAUT_Dan'
    clonename = test_api.clone_database(dbname)

    print('[TEST] TESTMAIN.MAIN')
    # WARNING! DO NOT USE --DB or --DBDIR on this yet! IT WILL DELETE IMAGES ON
    # REAL (IE NON COPIED DATABASES)
    hs, back, app, is_root = test_api.main_init(defaultdb=clonename, preload=True)
    hs.default_preferences()

    #print('[TEST] GET VALID CID')
    cid = test_api.get_valid_cid(hs)

    print('[TEST] ADDCHIP')
    cid1 = back.add_chip(gx=0, roi=[800, 350, 500, 500])

    print('[TEST] SELECTCID')
    back.select_cid(cid1)

    print('[TEST] QUERY')
    back.query(cid=cid1)

    print('[TEST] RESELECT ROI')
    back.reselect_roi(cid=cid1, roi=[700, 400, 700, 700])

    print('[TEST] RESELECT ORI')
    np.tau = 2 * np.pi  # tauday.com
    theta = np.tau / 8
    back.reselect_ori(cid=cid1, theta=theta)

    print('[TEST] DELETE IMAGE')
    valid_gxs = hs.get_valid_gxs()
    gx = valid_gxs[np.where(valid_gxs > 0)][0]
    back.delete_image(gx=gx)

    print('[TEST] SELECT CID')
    cid2 = test_api.get_valid_cid(hs)
    back.select_cid(cid2)

    print('[TEST] QUERY')
    res = back.query(cid=cid2)

    #print('[TEST] CHANGE CHIP PROPERTY')
    #back.change_chip_property(cid)

    print('[TEST] END TEST')
    test_api.main_loop(app, is_root, back, runqtmain=INTERACTIVE)
