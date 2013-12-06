#!/usr/bin/python2.7
from __future__ import division, print_function

def on_ctrl_c(signal, frame):
    print('Caught ctrl+c')

def signal_reset():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # reset ctrl+c behavior

def signal_set():
    signal.signal(signal.SIGINT, on_ctrl_c)

def main():
    import sys
    import signal
    import helpers
    import gui
    import fileio as io
    import HotSpotter
    helpers.print_off()
    print('[main] Running: __main__ = main.py')
    app, is_root = gui.init_qtapp()
    #gui.show_open_db_dlg()
    hs = HotSpotter.HotSpotter()
    hs.load2()
    main_win = gui.make_main_window(hs)
    #mainwin = gui.make_dummy_main_window()
    #opendb_dlg = gui.show_open_db_dlg()
    gui.run_main_loop(app, is_root)
    return locals()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main_locals = main()
