#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG

from hscom import helpers as util
from hsdev import test_api
from hsgui import guitools
from os.path import join
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()
    app, is_root = guitools.init_qtapp()
    hs, back = test_api.main(defaultdb=None, preload=False, app=app)

    # Build the test db name
    work_dir = back.get_work_directory()
    new_dbname = 'scripted_test_db'
    new_dbdir = join(work_dir, new_dbname)

    # Remove it if it exists
    util.delete(new_dbdir)

    back.new_database(new_dbdir)

    back.import_images_from_file()

    guitools.run_main_loop(app, is_root, back, frequency=100)
