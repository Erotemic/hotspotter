#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
from hsdev import test_api
from hsgui import guitools
from hscom import helpers as util
import multiprocessing
import sys
from os.path import join


def get_workdir():
    return 'D:\\data\work'


def get_dbdir(dbname):
    workdir = get_workdir()
    return join(workdir, dbname)


def clone_database(dbname):
    dir1 = get_dbdir(dbname)
    dir2 = get_dbdir(dbname + 'Clone')
    util.delete(dir2)
    util.copy(dir1, dir2)
    return dir2


if __name__ == '__main__':
    # INITIALIZATION CODE
    # For windows
    multiprocessing.freeze_support()
    # Initialize a qt app (or get parent's)
    app, is_root = guitools.init_qtapp()
    dbname = 'NAUTS_Dan'
    dir2 = clone_database(dbname)
    clonename = dir2 + 'Clone'
    # Create a HotSpotter API (hs) and GUI backend (back)
    hs, back = test_api.main(defaultdb=clonename, preload=True, app=app)
    # The test api returns a list of interesting chip indexes
    cx = test_api.get_test_cxs(hs, 1)[0]
    # Convert chip-index in to chip-id
    cid = hs.cx2_cid(cx)

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
