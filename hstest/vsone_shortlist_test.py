#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import __builtin__
from hsdev import test_api
import multiprocessing
import sys
from hsdev import dbgimport  # NOQA

INTERACTIVE = '--interactive' in sys.argv or '-i' in sys.argv


def print(msg):
    __builtin__.print('\n=============================')
    __builtin__.print(msg)
    if INTERACTIVE:
        raw_input('press enter to continue')


if __name__ == '__main__':
    print('[TEST] VSONE SHORTLIST TEST')
    multiprocessing.freeze_support()

    print('[TEST] CLONEDB')
    dbname = 'GZ'

    print('[TEST] TESTMAIN.MAIN')
    hs, back, app, is_root = test_api.main_init(defaultdb=dbname, preload=False)
    hs.default_preferences()

    print('[TEST] LOAD')
    hs.load()

    print('[TEST] = QUERY')
    hs.prefs.display_cfg.showanalysis = True
    test_cids = hs.cx2_cid(test_api.get_test_cxs(hs))
    cid = 306
    back.select_cid(cid)
    res = back.query(cid=cid)

    cx = hs.cid2_cx(cid)
    res2 = hs.query_groundtruth(cx, query_type='vsone')

    print('[TEST] END TEST')
    test_api.main_loop(app, is_root, back, runqtmain=INTERACTIVE)
