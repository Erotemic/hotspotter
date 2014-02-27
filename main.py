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
    from hsdev import test_api
    from hsdev import dbgimport
    print('main.py')
    dbgimport.hsgui_printoff()
    dbgimport.hsviz_printoff()
    dbgimport.mf.print_off()
    # Run main script with backend
    hs, back, app, is_root = test_api.main_init()
    # --- Run Startup Commands ---
    postload_locals = postload_args_process(hs, back)
    res = postload_locals['res']
    # Connect database to the back gui
    #app.setActiveWindow(back.front)
    test_api.main_loop(app, is_root, back)
