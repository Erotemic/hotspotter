#!/usr/bin/python2.7
'''Hotspotter main script
Runs hotspotter gui

!!! IMPORTANT DEVELOPER NOTICE !!!
Import as few things as possible at the global level in this module. Import at
the function level instead. The reason is multiprocesing will fork this module
many times. Less imports means less parallel overhead.
'''
from __future__ import division, print_function
import matplotlib
matplotlib.use('Qt4Agg')
import multiprocessing

def on_ctrl_c(signal, frame):
    import sys
    print('Caught ctrl+c')
    print('Hotspotter parent process killed by ctrl+c')
    sys.exit(0)


def signal_reset():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior


def signal_set():
    import signal
    signal.signal(signal.SIGINT, on_ctrl_c)


if __name__ == '__main__':
    # Necessary for windows parallelization
    multiprocessing.freeze_support()
    import argparse2
    args = argparse2.parse_arguments()
    import HotSpotter
    import guitools
    import helpers
    print('main.py')
    signal_set()
    # Run qt app
    app, is_root = guitools.init_qtapp()
    # Parse arguments
    args = argparse2.fix_args_with_cache(args)
    if args.vdd:
        helpers.vd(args.dbdir)
        args.vdd = False
    # Build hotspotter database
    hs = HotSpotter.HotSpotter(args)
    backend = guitools.make_main_window(hs, app)

    cids = hs.args.query
    # Preload data if you do any of these flags
    if hs.args.autoquery or len(cids) > 0:
        hs.load(load_all=True)
    # Autocompute all queries
    if hs.args.autoquery:
        backend.precompute_queries()
    if len(cids) > 0:
        try:
            qcid = cids[0]
            res = backend.query(qcid)
        except ValueError as ex:
            print('[main] ValueError = %r' % (ex,))
            if hs.args.strict:
                raise
        else:
            hs.load(load_all=False)
    # Connect database to the backend gui
    app.setActiveWindow(backend.win)

    # Allow for a IPython connection by passing the --cmd flag
    embedded = False
    exec(helpers.ipython_execstr())
    if not embedded:
        # If not in IPython run the QT main loop
        guitools.run_main_loop(app, is_root, backend)
    signal_reset()
