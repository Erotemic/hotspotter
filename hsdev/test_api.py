from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr,
 profile, printDBG) = __common__.init(__name__, '[tapi]', DEBUG=False)


def signal_reset():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # reset ctrl+c behavior


def signal_set():
    import signal
    signal.signal(signal.SIGINT, on_ctrl_c)


def on_ctrl_c(signal, frame):
    import sys
    print('Caught ctrl+c')
    print('Hotspotter parent process killed by ctrl+c')
    sys.exit(0)


def parse_arguments(defaultdb, usedbcache):
    from hscom import argparse2
    from hscom import params
    from hscom import helpers as util
    from hscom import fileio as io
    import sys
    args = argparse2.parse_arguments(defaultdb=defaultdb)
    # Parse arguments
    args = argparse2.fix_args_with_cache(args)
    if usedbcache:
        if args.vdd:
            util.vd(args.dbdir)
            args.vdd = False
        if util.inIPython() or '--cmd' in sys.argv:
            args.nosteal = True
    params.args = args
    # Preload process args
    if args.delete_global:
        io.delete_global_cache()
    return args


def main(defaultdb='cache', preload=False, app=None):
    from hscom import fileio as io
    from hscom import params
    from hotspotter import HotSpotterAPI as api
    from hsgui import guitools
    from hsgui import guiback
    if app is True:
        app, is_root = guitools.init_qtapp()
    args = parse_arguments(defaultdb, defaultdb == 'cache')
    # --- Build HotSpotter API ---
    if app is None:
        hs = api.HotSpotter(args)
    else:
        back = guiback.make_main_window(app)
        hs = back.open_database(args.dbdir)
    setcfg = args.setcfg
    if setcfg is not None:
        # FIXME move experiment harness to hsdev
        import experiment_harness
        print('[tapi.main] setting cfg to %r' % setcfg)
        varied_list = experiment_harness.get_varied_params_list([setcfg])
        cfg_dict = varied_list[0]
        hs.prefs.query_cfg.update_cfg(**cfg_dict)
        hs.prefs.save()
        hs.prefs.printme()
    # Load all data if needed now, otherwise be lazy
    try:
        load_all = preload
        hs.load(load_all=load_all)
        db_dir = hs.dirs.db_dir
        io.global_cache_write('db_dir', db_dir)
    except ValueError as ex:
        print('[tapi.main] ValueError = %r' % (ex,))
        if params.args.strict:
            raise
    if app is not None:
        return hs, back
    else:
        from hsgui import guitools
        app, is_root = guitools.init_qtapp()
        hs.app = app
    return hs


def clone_database(dbname):
    from hscom import params
    from hscom import helpers as util
    from os.path import join
    work_dir = params.get_workdir()
    dir1 = join(work_dir, dbname)
    dir2 = join(work_dir, dbname + 'Clone')
    util.delete(dir2)
    util.copy_all(dir1, dir2, '*')
    util.copy_all(join(dir1, 'images'), join(dir2, 'images'), '*')
    util.copy_all(join(dir1, '_hsdb'), join(dir2, '_hsdb'), '*')
    return dbname + 'Clone'


def get_valid_cid(hs, num=0):
    try:
        test_cxs = get_test_cxs(hs)
        test_cids = hs.cx2_cid(test_cxs)
        if len(test_cids) == 0:
            raise IndexError('THERE ARE NO TEST_CIDS IN THIS DATABASE')
        cid = test_cids[num % len(test_cids)]
    except IndexError as ex:
        print('Index Error: %s' % str(ex))
        raise
        print(hs.tables)
        print('cx2_cid: %r' % hs.tables.cx2_cid)
        print(ex)
        print(test_cxs)
        print(test_cids)
        print(cid)
    return cid


def get_test_cxs(hs, max_testcases=None):
    valid_cxs = get_qcx_list(hs)
    if max_testcases is not None:
        #maxcx = max(valid_cxs)
        #max_ = max(len(valid_cxs) - 1, cxs)
        #if max_ == 0:
            #raise ValueError('[test_api] Database does not have test cxs')
        valid_cxs = valid_cxs[0:max_testcases]
    return valid_cxs


def get_qcx_list(hs):
    ''' Function for getting the list of queries to test '''
    import numpy as np
    from hscom import params
    from hscom import helpers as util
    print('[tapi!] get_qcx_list()')

    valid_cxs = hs.get_valid_cxs()
    def get_cases(hs, with_hard=True, with_gt=True, with_nogt=True, with_notes=False):
        qcx_list = []
        if with_hard:
            if 'hard' in hs.tables.prop_dict:
                for cx in iter(valid_cxs):
                    if hs.cx2_property(cx, 'hard') == 'True':
                        qcx_list += [cx]
        if with_hard:
            if 'Notes' in hs.tables.prop_dict:
                for cx in iter(valid_cxs):
                    if hs.cx2_property(cx, 'Notes') != '':
                        qcx_list += [cx]
        if with_gt and not with_nogt:
            for cx in iter(valid_cxs):
                gt_cxs = hs.get_other_indexed_cxs(cx)
                if len(gt_cxs) > 0:
                    qcx_list += [cx]
        if with_gt and with_nogt:
            qcx_list = valid_cxs
        return qcx_list

    # Sample a large pool of query indexes
    histids = None if params.args.histid is None else np.array(params.args.histid)
    if params.args.all_cases:
        print('[tapi] all cases')
        qcx_all = get_cases(hs, with_gt=True, with_nogt=True)
    elif params.args.all_gt_cases:
        print('[tapi] all gt cases')
        qcx_all = get_cases(hs, with_hard=True, with_gt=True, with_nogt=False)
    elif params.args.qcid is None:
        # FIXEME: BUG
        print('[tapi] did not select cases')
        qcx_all = get_cases(hs, with_hard=True, with_gt=False, with_nogt=False)
    else:
        print('[tapi] Chosen qcid=%r' % params.args.qcid)
        qcx_all =  util.ensure_iterable(hs.cid2_cx(params.args.qcid))
    # Filter only the ones you want from the large pool
    if histids is None:
        qcx_list = qcx_all
    else:
        histids = util.ensure_iterable(histids)
        print('[tapi] Chosen histids=%r' % histids)
        qcx_list = [qcx_list[id_] for id_ in histids]

    if len(qcx_list) == 0:
        msg = '[tapi.get_qcxs] no qcx_list history'
        print(msg)
        import sys
        if '--vstrict' in sys.argv:  # if params.args.vstrict:
            raise Exception(msg)
        print(valid_cxs)
        qcx_list = valid_cxs[0:1]
    print('[tapi] len(qcx_list) = %d' % len(qcx_list))
    qcx_list = util.unique_keep_order(qcx_list)
    print('[tapi] qcx_list = %r' % qcx_list)
    return qcx_list


def reload_all():
    import dev_reload
    dev_reload.rrr()
    dev_reload.reload_all_modules()


def main_init(defaultdb=None, preload=False, app=None):
    from hsgui import guitools
    # Listen for ctrl+c
    signal_set()
    # Run Qt App
    app, is_root = guitools.init_qtapp()
    hs, back = main(defaultdb=defaultdb, preload=preload, app=app)
    return hs, back, app, is_root


def main_loop(app, is_root, back, runqtmain=True):
    from hscom import helpers as util
    from hsgui import guitools
    import sys
    hs = back.hs  # NOQA
    # Allow for a IPython connection by passing the --cmd flag
    embedded = util.inIPython()
    if not embedded and util.get_flag('--cmd'):
        print('Embedding')
        util.embed()
        sys.exit(1)
    if not embedded and runqtmain:
        print('Running main loop')
        # If not in IPython run the QT main loop
        guitools.run_main_loop(app, is_root, back, frequency=100)
    signal_reset()
    print('hotspotter will exit')
