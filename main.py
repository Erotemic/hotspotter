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


def postload_interpret_cmdline(hs, back):
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
    fx_list = params.args.fxs   # the query feature index
    selgxs = params.args.selgxs
    selnxs = params.args.selnxs
    selcids = params.args.selcids
    cfgstrdict_list = params.args.update_cfg
    res = None

    if len(cfgstrdict_list) > 0:
        cfgdict = {}
        for strdict in cfgstrdict_list:
            print(strdict)
            key, val = strdict.split(':')
            if val.lower() in ['none']:
                val = None
            elif val.lower() in ['true']:
                val = True
            elif val.lower() in ['false']:
                val = False
            elif '.' in val:
                val = float(val)
            cfgdict[key] = val
        hs.prefs.query_cfg.update_cfg(**cfgdict)

    # Autocompute all queries
    if params.args.batchfeats:
        back.precompute_feats()

    if len(qcid_list) > 0:
        qcid = qcid_list[0]
        tx = tx_list[0] if len(tx_list) > 0 else None
        try:
            res = back.query(qcid, tx)  # Run query with optional tx
            back.select_cid(qcid, show=False)  # Select query
            if len(cid_list) > 0:
                cx = hs.cid2_cx(cid_list[0])
                if len(fx_list) == 0:
                    # Just interact with the query
                    res.interact_chipres(hs, cx, fnum=4, mode=1)
                else:
                    # Interact with query and features
                    qfx = fx_list[0]
                    mx = res.get_match_index(hs, cx, qfx)
                    res.interact_chipres(hs, cx, fnum=4, mx=mx)
                    res.show_nearest_descriptors(hs, qfx)
        except AssertionError as ex:
            print(ex)

    # Select image indexes
    if len(selgxs) > 0:
        back.select_gx(selgxs[0])

    # Select name indexes
    if len(selnxs) > 0:
        name = hs.nx2_name(selnxs[0])
        back.select_name(name)

    # Select chip ids
    if len(selcids) > 0:
        fx = None if len(fx_list) == 0 else fx_list[0]
        selcxs = hs.cid2_cx(selcids)
        back.select_cx(selcxs[0], noimage=True, fx=fx)

    if params.args.batchquery:
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
    from hsdev import main_api
    main_api.inject_colored_exception_hook()
    #from hsdev import dbgimport
    print('[main] main.py')
    #dbgimport.hsgui_printoff()
    #dbgimport.hsviz_printoff()
    #dbgimport.mf.print_off()
    # Run main script with backend
    hs, back, app, is_root = main_api.main_init()
    # --- Run Startup Commands ---
    postload_locals = postload_interpret_cmdline(hs, back)
    res = postload_locals['res']
    # Connect database to the back gui
    #app.setActiveWindow(back.front)
    main_api.main_loop(app, is_root, back)
