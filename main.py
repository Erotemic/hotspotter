from __init__ import *
import gui
import fileio as io
import convert_to_hotspotterdb as convert_hsdb

def config_matplotlib():
    # configure matplotlib 
    import matplotlib
    print('Configuring matplotlib for Qt4')
    matplotlib.use('Qt4Agg')
    
def catch_ctrl_c(signal, frame):
    print('Caught ctrl+c')

def signal_reset():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # reset ctrl+c behavior

def signal_set():
    signal.signal(signal.SIGINT, catch_ctrl_c)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = main.py')

    app, is_root = gui.init_qtapp()
    signal_set()

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
        convert_hsdb.convert_named_chips(db_dir, img_dpath)

    hs = load_data2.HotSpotter(db_dir)

    if '--vrd' in sys.argv:
        helpers.vd(hs.dirs.result_dir)
        sys.exit(1)

    qcx2_res = mc2.run_matching(hs)
    allres = rr2.init_allres(hs, qcx2_res, SV=True)
    rr2.dump_all(allres)

    print('Exiting HotSpotter')
    sys.exit(0)

    #mainwin = PyQt4.Qt.QMainWindow()
    #mainwin.setWindowTitle('Dummy Main Window')
    #mainwin.show()
    #print('Running the application event loop')
    #helpers.flush()
    #sys.exit(app.exec_())
