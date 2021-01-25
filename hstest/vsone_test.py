#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG

import builtins
from hsdev import test_api
import multiprocessing
import sys
from hsdev import dbgimport  # NOQA

INTERACTIVE = '--interactive' in sys.argv or '-i' in sys.argv


def print(msg):
    builtins.print('\n=============================')
    builtins.print(msg)
    if INTERACTIVE:
        input('press enter to continue')


if __name__ == '__main__':
    print('[TEST] VSONETEST')
    multiprocessing.freeze_support()

    print('[TEST] CLONEDB')
    #dbname = 'MOTHERS'
    dbname = 'NAUTS'

    print('[TEST] TESTMAIN.MAIN')
    hs, back, app, is_root = test_api.main_init(defaultdb=dbname, preload=False)
    hs.default_preferences()

    print('[TEST] LOAD')
    hs.load()

    print('[TEST] GET VALID CID')
    cid = test_api.get_valid_cid(hs)

    print('[TEST] HS FEATS')
    print(hs.feats)

    print('[TEST] HS TABLES')
    print(hs.tables)

    print('[TEST] QUERY')
    res1 = back.query(cid=cid, query_type='vsmany', K=6, Knorm=4, ratio_thresh=0)
    res2 = back.query(cid=cid, query_type='vsone', K=1, Knorm=1, ratio_thresh=1.6)

    print('[TEST] END TEST')
    test_api.main_loop(app, is_root, back, runqtmain=INTERACTIVE)
