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
    print('[TEST] SMALLTEST')
    multiprocessing.freeze_support()

    print('[TEST] TESTMAIN.MAIN')
    # WARNING! DO NOT USE --DB or --DBDIR on this yet! IT WILL DELETE IMAGES ON
    # REAL (IE NON COPIED DATABASES)
    hs, back, app, is_root = test_api.main_init(defaultdb='NAUTS', preload=True)

    print('[TEST] END TEST')
    test_api.main_loop(app, is_root, back, runqtmain=INTERACTIVE)
