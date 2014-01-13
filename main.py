#!/usr/bin/env python
'''Hotspotter main script
Runs hotspotter gui

!!! IMPORTANT DEVELOPER NOTICE !!!
Import as few things as possible at the global level in this module. Import at
the function level instead. The reason is multiprocesing will fork this module
many times. Less imports means less parallel overhead.
'''
from __future__ import division, print_function
import multiprocessing


def dependencies_for_myprogram():
    # Let pyintaller find these modules
    from scipy.sparse.csgraph import _validation  # NOQA
    from scipy.special import _ufuncs_cxx  # NOQA


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


def preload_args_process(args):
    import helpers
    import sys
    # Process relevant args
    cids = args.query
    if args.vdd:
        helpers.vd(args.dbdir)
        args.vdd = False
    load_all = args.autoquery or len(cids) > 0
    if helpers.inIPython() or '--cmd' in sys.argv:
        args.nosteal = True
    return load_all, cids


def postload_args_process(hs):
    # --- Run Startup Commands ---
    # Autocompute all queries
    if hs.args.autoquery:
        back.precompute_queries()
    # Run a query
    qcid_list = args.query
    tx_list = args.txs
    res = None
    if len(qcid_list) > 0:
        qcid = qcid_list[0]
        tx = tx_list[0] if len(tx_list) > 0 else None
        res = back.query(qcid, tx)
    return res


if __name__ == '__main__':
    # Necessary for windows parallelization
    multiprocessing.freeze_support()
    import matplotlib
    matplotlib.use('Qt4Agg')
    import argparse2
    args = argparse2.parse_arguments()
    import HotSpotter
    import guitools
    import guiback
    import helpers
    print('main.py')
    # Listen for ctrl+c
    signal_set()
    # Run qt app
    app, is_root = guitools.init_qtapp()
    # Parse arguments
    args = argparse2.fix_args_with_cache(args)
    load_all, cids = preload_args_process(args)
    argparse2.execute_initial(args)

    # --- Build HotSpotter API ---
    hs = HotSpotter.HotSpotter(args)
    # Load all data if needed now, otherwise be lazy
    try:
        hs.load(load_all=load_all)
    except ValueError as ex:
        print('[main] ValueError = %r' % (ex,))
        if hs.args.strict:
            raise
    # Create main window only after data is loaded
    back = guiback.make_main_window(hs, app)
    # --- Run Startup Commands ---
    res = postload_args_process(hs)
    # Connect database to the back gui
    #app.setActiveWindow(back.front)

    # Allow for a IPython connection by passing the --cmd flag
    embedded = False
    exec(helpers.ipython_execstr())
    if not embedded:
        # If not in IPython run the QT main loop
        guitools.run_main_loop(app, is_root, back, frequency=100)
    signal_reset()
