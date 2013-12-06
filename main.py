#!/usr/bin/python2.7
from __future__ import division, print_function

def on_ctrl_c(signal, frame):
    print('Caught ctrl+c')

def signal_reset():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # reset ctrl+c behavior

def signal_set():
    signal.signal(signal.SIGINT, on_ctrl_c)

def main(use_gui=False):
    import sys
    import signal
    import helpers
    import gui
    import fileio as io
    import HotSpotter
    if use_gui:
        helpers.print_off()
        print('[main] Running: __main__ = main.py')
        app, is_root = gui.init_qtapp()
    hs = HotSpotter.HotSpotter()
    hs.load2()
    if use_gui:
        #gui.show_open_db_dlg()
        main_win = gui.make_main_window(hs)
        gui.run_main_loop(app, is_root)
    return locals()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main_locals = main(True)
