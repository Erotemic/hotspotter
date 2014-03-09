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
    # This processes command line arguments and runs corresponding commands on
    # startup.
    from hsdev import params

    # Run a query of...
    # currently each list is assumed to be of length 1 or 0
    # anything else will cause errors
    qcid_list = params.args.query  # Query qcid list
    # Inspect results against...
    tx_list = params.args.txs    # the chips with top ranked index OR
    cid_list = params.args.cids  # the chip-ids
    # Perform further inspection of...
    qfx_list = params.args.qfxs  # the query feature index
    res = None
    if len(qcid_list) > 0:
        qcid = qcid_list[0]
        tx = tx_list[0] if len(tx_list) > 0 else None
        try:
            res = back.query(qcid, tx)  # Run query with optional tx
            back.select_cid(qcid, show=False)  # Select query
            if len(cid_list) > 0:
                cx = hs.cid2_cx(cid_list[0])
                if len(qfx_list) == 0:
                    # Just interact with the query
                    res.interact_chipres(hs, cx, fnum=4, mode=1)
                else:
                    # Interact with query and features
                    qfx = qfx_list[0]
                    mx = res.get_match_index(hs, cx, qfx)
                    res.interact_chipres(hs, cx, fnum=4, mx=mx)
                    res.show_nearest_descriptors(hs, qfx)
        except AssertionError as ex:
            print(ex)

    # Select image indexes
    selgxs = params.args.selgxs
    if len(selgxs) > 0:
        back.select_gx(selgxs[0])

    # Select name indexes
    selnxs = params.args.selnxs
    if len(selnxs) > 0:
        name = hs.nx2_name(selnxs[0])
        back.select_name(name)

    # Select chip ids
    selcids = params.args.selcids
    if len(selcids) > 0:
        selcxs = hs.cid2_cx(selcids)
        back.select_cx(selcxs[0])

    # Autocompute all queries
    if params.args.autoquery:
        back.precompute_queries()

    return locals()


#==================
# MAIN ENTRY POINT
#==================

if __name__ == '__main__':
    # Necessary for windows parallelization
    multiprocessing.freeze_support()
    # Run Main Function
    #from hsviz import draw_func2 as df2  # NOQA
    from hsdev import test_api
    #from hsdev import dbgimport
    print('main.py')
    #dbgimport.hsgui_printoff()
    #dbgimport.hsviz_printoff()
    #dbgimport.mf.print_off()
    # Run main script with backend
    hs, back, app, is_root = test_api.main_init()
    # --- Run Startup Commands ---
    postload_locals = postload_args_process(hs, back)
    res = postload_locals['res']
    # Connect database to the back gui
    #app.setActiveWindow(back.front)
    test_api.main_loop(app, is_root, back)
