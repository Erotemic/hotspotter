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
    multiprocessing.freeze_support()
    print('[TEST] MULTIQUERY TEST')
    dbgimport.hsgui_printoff()
    dbgimport.hsviz_printoff()
    dbgimport.mf.print_off()

    print('[TEST] CLONEDB')

    print('[TEST] TESTMAIN.MAIN')
    dbname = 'GZ'
    hs, back, app, is_root = test_api.main_init(defaultdb=dbname, preload=True)

    #print('[TEST] GET VALID CID')
    cid1 = test_api.get_valid_cid(hs, 0)
    cid2 = test_api.get_valid_cid(hs, 1)
    cid3 = test_api.get_valid_cid(hs, 2)

    print('[TEST] QUERY')
    res1 = back.query(cid=cid1)

    print('[TEST] QUERY')
    res2 = back.query(cid=cid2)

    print('[TEST] QUERY')
    res3 = back.query(cid=cid3)

    print('[TEST] END TEST')
    test_api.main_loop(app, is_root, back, runqtmain=INTERACTIVE)
