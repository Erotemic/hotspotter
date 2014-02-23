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
    # INITIALIZATION CODE
    # For windows
    multiprocessing.freeze_support()
    # Initialize a qt app (or get parent's)
    app, is_root = guitools.init_qtapp()
    dbname = 'NAUT_Dan'
    clonename = clone_database(dbname)
    # Create a HotSpotter API (hs) and GUI backend (back)
    hs, back = test_api.main(defaultdb=clonename, preload=True, app=app)

    back.select_gx(0)
    cid = back.add_chip(gx=0, roi=[800, 350, 500, 500])
    back.select_cid(cid)
    back.query(cid=cid)
    back.reselect_roi(cid=cid, roi=[700, 400, 700, 700])
    back.reselect_ori(cid=cid, theta=1.57)

    back.delete_image()

    back.select_cid(4)
    back.query()

    back.change_chip_property(cid)

    #hs.add_chip()
    #hs.add_image()
    #hs.add_delete_chip()
    #hs.add_delete_image()
    #hs.change_name()
    #hs.change_roi()
    #hs.change_ori()
    #hs.change_prop()

    cid = hs.cx2_cid(cx)

    # TESTING CODE
    # Query the chip-id
    res = back.query(cid)
    # LAAEFTR: use res to do stuff
    # Run Qt Loop to use the GUI
    if '--cmd' in sys.argv:
        util.embed()
