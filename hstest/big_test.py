#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
from hsdev import test_api
from hsgui import guitools
from hscom import helpers as util
from hscom import params
import multiprocessing
import sys
from os.path import join
import numpy as np
from hsdev import dbgimport


def clone_database(dbname):
    work_dir = params.get_workdir()
    dir1 = join(work_dir, dbname)
    dir2 = join(work_dir, dbname + 'Clone')
    util.delete(dir2)
    util.copy_all(dir1, dir2, '*')
    util.copy_all(join(dir1, 'images'), join(dir2, 'images'), '*')
    util.copy_all(join(dir1, '_hsdb'), join(dir2, '_hsdb'), '*')
    return dbname + 'Clone'


if __name__ == '__main__':
    dbgimport.all_printoff()
    # INITIALIZATION CODE
    # For windows
    print('[TEST] BIGTEST')
    multiprocessing.freeze_support()
    print('[TEST] INITQtAPP')
    app, is_root = guitools.init_qtapp()
    dbname = 'NAUT_Dan'
    print('[TEST] CLONEDB')
    clonename = clone_database(dbname)
    print('[TEST] TESTMAIN.MAIN')
    hs, back = test_api.main(defaultdb=clonename, preload=True, app=app)
    def get_valid_cid(hs):
        try:
            test_cxs = test_api.get_test_cxs(hs)
            test_cids = hs.cx2_cid(test_cxs)
            cid = test_cids[0]
        except IndexError as ex:
            print('Index Error: %s' % str(ex))
            raise
            print(hs.tables)
            print('cx2_cid: %r' % hs.tables.cx2_cid)
            print(ex)
            print(test_cxs)
            print(test_cids)
            print(cid)
        return cid
    print('[TEST] SELECTGX')
    #back.select_gx(0)
    cid = get_valid_cid(hs)
    #print('[TEST] ADDCHIP')
    #cid = back.add_chip(gx=0, roi=[800, 350, 500, 500])
    #print('[TEST] SELECTCID')
    #back.select_cid(cid)
    print('[TEST] QUERY')
    back.query(cid=cid)
    #print('[TEST] RESELECT ROI')
    #back.reselect_roi(cid=cid, roi=[700, 400, 700, 700])
    #print('[TEST] RESELECT ORI')
    #back.reselect_ori(cid=cid, theta=1.57)
    print('[TEST] DELETE IMAGE')
    valid_gxs = hs.get_valid_gxs()
    gx = valid_gxs[np.where(valid_gxs > 0)][0]
    back.delete_image(gx=gx)
    print('[TEST] DELETE IMAGE')
    cid = get_valid_cid(hs)
    back.select_cid(cid)
    print('[TEST] QUERY')
    res = back.query()
    print('[TEST] CHANGE CHIP PROPERTY')
    #back.change_chip_property(cid)

    if '--cmd' in sys.argv:
        util.embed()
