#!/usr/bin/python2.7
from __init__ import *

import sys
import gui
import fileio as io
import convert_db as convert_db

def config_matplotlib():
    # configure matplotlib 
    import matplotlib
    if matplotlib.get_backend() != 'Qt4Agg':
        print('main> Configuring matplotlib for Qt4')
        matplotlib.use('Qt4Agg')
    
def catch_ctrl_c(signal, frame):
    print('Caught ctrl+c')

def signal_reset():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # reset ctrl+c behavior

def signal_set():
    signal.signal(signal.SIGINT, catch_ctrl_c)


def tmp_get_database_dir():
    db_dir = None
    if not '-nc' in sys.argv and not '--nocache' in sys.argv: 
        db_dir = io.global_cache_read('db_dir')
        if db_dir == '.': 
            db_dir = None

    if db_dir is None or not exists(db_dir):
        db_dir = gui.select_directory('Select a directory to be used as the database')
        io.global_cache_write('db_dir', db_dir)

    do_convert = not exists(join(db_dir, '.hs_internals'))
    if do_convert:
        img_dpath = join(db_dir,'images')
        if not exists(img_dpath):
            img_dpath = gui.select_directory('Select directory with images in it')
        convert_db.convert_named_chips(db_dir, img_dpath)
    return db_dir


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = main.py')

    app, is_root = gui.init_qtapp()
    signal_set()

    #gui.show_open_db_dlg()

    db_dir = tmp_get_database_dir()
    hs = ld2.HotSpotter()
    hs.load_all(db_dir)
    qcx2_res = mc2.run_matching(hs)
    hs.vrd()
    allres = rr2.report_all(hs, qcx2_res, SV=True, matrix=True, allqueries=True)
    print('Exiting HotSpotter')
    sys.exit(0)

    #mainwin = PyQt4.Qt.QMainWindow()
    #mainwin.setWindowTitle('Dummy Main Window')
    #mainwin.show()
    #print('Running the application event loop')
    #helpers.flush()
    run_qtapp.run_qtapp()
