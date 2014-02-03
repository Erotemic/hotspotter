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
    from hscom import helpers
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


def postload_args_process(hs, back):
    # --- Run Startup Commands ---
    # Autocompute all queries
    if hs.args.autoquery:
        back.precompute_queries()
    # Run a query
    qcid_list = hs.args.query
    tx_list = hs.args.txs
    res = None
    if len(qcid_list) > 0:
        qcid = qcid_list[0]
        tx = tx_list[0] if len(tx_list) > 0 else None
        res = back.query(qcid, tx)
    selgxs = hs.args.selgxs
    if len(selgxs) > 0:
        back.select_gx(selgxs[0])
    selnxs = hs.args.selnxs
    if len(selnxs) > 0:
        name = hs.nx2_name(selnxs[0])
        back.select_name(name)
    selcxs = hs.args.selcxs
    if len(selcxs) > 0:
        back.select_cx(selcxs[0])
    cids = hs.args.select_cid
    if len(cids) > 0:
        cxs = hs.cid2_cx(cids)
        back.select_cx(cxs[0])

    return res


def imports():
    pass
    # TODO: Rename this to something better
    #from hotspotter import load_data2 as ld2
    #from hsgui import guiback
    #from hsgui import guifront
    #from hsviz import draw_func2 as df2
    #ld2.print_off()
    #guiback.print_off()
    #guifront.print_off()
    #df2.print_off()


def main(defaultdb='NAUTS', usedbcache=False, default_load_all=True):
    import matplotlib
    matplotlib.use('Qt4Agg')
    imports()
    from hscom import argparse2
    from hscom import helpers
    from hotspotter import HotSpotterAPI as api
    args = argparse2.parse_arguments(defaultdb=defaultdb)
    # Parse arguments
    args = argparse2.fix_args_with_cache(args)
    if usedbcache:
        load_all, cids = preload_args_process(args)
    else:
        args = argparse2.fix_args_shortnames(args)
        load_all = helpers.get_flag('--load-all', default_load_all)

    # Preload process args
    if args.delete_global:
        from hscom import fileio as io
        io.delete_global_cache()

    # --- Build HotSpotter API ---
    hs = api.HotSpotter(args)
    setcfg = args.setcfg
    if setcfg is not None:
        import experiment_harness
        print('[main] setting cfg to %r' % setcfg)
        varried_list = experiment_harness.get_varried_params_list([setcfg])
        cfg_dict = varried_list[0]
        #print(cfg_dict)
        hs.prefs.query_cfg.update_cfg(**cfg_dict)
        hs.prefs.save()
        #hs.prefs.printme()
        #hs.default_preferences()

    # Load all data if needed now, otherwise be lazy
    try:
        hs.load(load_all=load_all)
        from hotspotter import fileio as io
        db_dir = hs.dirs.db_dir
        io.global_cache_write('db_dir', db_dir)
    except ValueError as ex:
        print('[main] ValueError = %r' % (ex,))
        if hs.args.strict:
            raise
    return hs

#==================
# MAIN ENTRY POINT
#==================

if __name__ == '__main__':
    # Necessary for windows parallelization
    multiprocessing.freeze_support()
    hs = main(defaultdb=None, usedbcache=True)
    from hsgui import guitools
    from hsgui import guiback
    from hscom import helpers
    print('main.py')
    # Listen for ctrl+c
    signal_set()
    # Run qt app
    app, is_root = guitools.init_qtapp()
    # Create main window only after data is loaded
    back = guiback.make_main_window(hs, app)
    # --- Run Startup Commands ---
    res = postload_args_process(hs, back)
    # Connect database to the back gui
    #app.setActiveWindow(back.front)

    # Allow for a IPython connection by passing the --cmd flag
    embedded = False
    if helpers.get_flag('--cmd'):
        import scripts
        import generate_training
        import sys

        def do_encounters(seconds=None):
            if not 'seconds' in vars() or seconds is None:
                seconds = 5
            scripts.rrr()
            do_enc_loc = scripts.compute_encounters(hs, back, seconds)
            return do_enc_loc

        def do_extract_encounter(eid=None):
            #if not 'eid' in vars() or eid is None:
            #eid = 'ex=269_nGxs=21'
            eid = 'ex=61_nGxs=18'
            scripts.rrr()
            extr_enc_loc = scripts.extract_encounter(hs, eid)
            export_subdb_locals = extr_enc_loc['export_subdb_locals']
            return extr_enc_loc, export_subdb_locals

        def do_generate_training():
            generate_training.rrr()
            return generate_training.generate_detector_training_data(hs, (256, 448))

        def do_import_database():
            scripts.rrr()
            #from os.path import expanduser, join
            #workdir = expanduser('~/data/work')
            #other_dbdir = join(workdir, 'hsdb_exported_138_185_encounter_eid=1 nGxs=43')

        def vgd():
            return generate_training.vgd(hs)

        #from PyQt4.QtCore import pyqtRemoveInputHook
        #from IPython.lib.inputhook import enable_qt4
        #pyqtRemoveInputHook()
        #enable_qt4()
        exec(helpers.ipython_execstr())
        sys.exit(1)
    if not embedded:
        # If not in IPython run the QT main loop
        guitools.run_main_loop(app, is_root, back, frequency=100)

    signal_reset()
