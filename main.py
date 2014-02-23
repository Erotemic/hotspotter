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


def postload_args_process(hs, back):
    from hscom import params
    # --- Run Startup Commands ---
    # Autocompute all queries
    if params.args.autoquery:
        back.precompute_queries()
    # Run a query
    qcid_list = params.args.query
    tx_list = params.args.txs
    qfx_list = params.args.qfxs
    cid_list = params.args.cids
    res = None
    if len(qcid_list) > 0:
        qcid = qcid_list[0]
        tx = tx_list[0] if len(tx_list) > 0 else None
        # Run a query
        try:
            res = back.query(qcid, tx)
            back.select_cid(qcid, show=False)
            if len(cid_list) > 0:
                # Interact with the query
                cx = hs.cid2_cx(cid_list[0])
                if len(qfx_list) > 0:
                    qfx = qfx_list[0]
                    mx = res.get_match_index(hs, cx, qfx)
                    res.interact_chipres(hs, cx, fnum=4, mx=mx)
                    res.show_nearest_descriptors(hs, qfx)
                else:
                    res.interact_chipres(hs, cx, fnum=4)
        except AssertionError as ex:
            print(ex)
    # Select on startup commands
    selgxs = params.args.selgxs
    if len(selgxs) > 0:
        back.select_gx(selgxs[0])
    selnxs = params.args.selnxs
    if len(selnxs) > 0:
        name = hs.nx2_name(selnxs[0])
        back.select_name(name)
    selcids = params.args.selcids
    if len(selcids) > 0:
        selcxs = hs.cid2_cx(selcids)
        back.select_cx(selcxs[0])
    return locals()


#==================
# MAIN ENTRY POINT
#==================

if __name__ == '__main__':
    # Necessary for windows parallelization
    multiprocessing.freeze_support()
    # Run Main Function
    from hsgui import guitools
    from hscom import helpers as util
    from hsdev import test_api
    print('main.py')
    # Listen for ctrl+c
    test_api.signal_set()
    # Run qt app
    app, is_root = guitools.init_qtapp()
    # Run main script with backend
    hs, back = test_api.main(defaultdb=None, preload=False, app=app)
    # --- Run Startup Commands ---
    postload_locals = postload_args_process(hs, back)
    res = postload_locals['res']
    # Connect database to the back gui
    #app.setActiveWindow(back.front)

    # Allow for a IPython connection by passing the --cmd flag
    embedded = False
    if util.get_flag('--cmd'):
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
        exec(util.ipython_execstr())
        sys.exit(1)
    if not embedded:
        # If not in IPython run the QT main loop
        guitools.run_main_loop(app, is_root, back, frequency=100)

    test_api.signal_reset()
