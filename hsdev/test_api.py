from __future__ import division, print_function


def parse_arguments(defaultdb, usedbcache):
    from hscom import argparse2
    from hscom import params
    from hscom import helpers
    from hscom import fileio as io
    import sys
    args = argparse2.parse_arguments(defaultdb=defaultdb)
    # Parse arguments
    args = argparse2.fix_args_with_cache(args)
    # fix args shortnam
    if (args.dbdir is None) and (args.db is not None):
        try:
            args.dbdir = params.dev_databases[args.db]
        except KeyError:
            pass
    # Lookup shortname
    try:
        inverse_dev_databases = params.inverse_dev_databases()
        args.db = inverse_dev_databases[args.dbdir]
    except KeyError:
        pass
    if usedbcache:
        if args.vdd:
            helpers.vd(args.dbdir)
            args.vdd = False
        if helpers.inIPython() or '--cmd' in sys.argv:
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
    if app is True:
        from hsgui import guitools
        app, is_root = guitools.init_qtapp()
    args = parse_arguments(defaultdb, defaultdb == 'cache')
    # --- Build HotSpotter API ---
    if app is None:
        hs = api.HotSpotter(args)
    else:
        from hsgui import guiback
        back = guiback.make_main_window(app)
        hs = back.open_database(args.dbdir)
    setcfg = args.setcfg
    if setcfg is not None:
        import experiment_harness
        print('[main] setting cfg to %r' % setcfg)
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
        print('[main] ValueError = %r' % (ex,))
        if params.args.strict:
            raise
    if app is not None:
        return hs, back
    else:
        from hsgui import guitools
        app, is_root = guitools.init_qtapp()
        hs.app = app
    return hs


def get_test_cxs(hs, max_testcases=None):
    valid_cxs = hs.get_valid_cxs()
    if max_testcases is not None:
        max_ = max(max_testcases, len(valid_cxs) - 1)
        valid_cxs = valid_cxs[0:max_]
    return valid_cxs


def reload_all():
    import dev_reload
    dev_reload.rrr()
    dev_reload.reload_all_modules()
