#!/usr/env python
from __future__ import division, print_function
import multiprocessing
from hotspotter import encounter
from hscom import helpers
from hsviz import draw_func2 as df2

USE_TESTCACHE = True


def test_encounter(hs):
    exec(open('hotspotter/encounter.py').read())
    encounter.rrr()
    try:
        if USE_TESTCACHE:
            raise KeyError('use_testcache=False')
        ex2_cxs = helpers.load_testdata('ex2_cxs', uid=hs.get_db_name())
    except KeyError:
        ex2_cxs = encounter.get_chip_encounters(hs)
        helpers.stash_testdata('ex2_cxs', uid=hs.get_db_name())
    cxs = ex2_cxs[-1]
    assert len(cxs) > 1
    qcx2_res = encounter.intra_query_cxs(hs, cxs)
    # Make a graph between the chips
    graph = encounter.make_chip_graph(qcx2_res)
    encounter.viz_chipgraph(hs, graph, fnum=20, with_images=False)
    #encounter.viz_chipgraph(hs, graph, fnum=20, with_images=True)
    df2.update()


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


def main(defaultdb='NAUTS', usedbcache=True, app=None):
    from hscom import fileio as io
    from hscom import params
    from hotspotter import HotSpotterAPI as api
    args = parse_arguments(defaultdb, usedbcache)
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
        load_all = False
        hs.load(load_all=load_all)
        db_dir = hs.dirs.db_dir
        io.global_cache_write('db_dir', db_dir)
    except ValueError as ex:
        print('[main] ValueError = %r' % (ex,))
        if params.args.strict:
            raise
    if app is not None:
        return hs, back
    return hs


if __name__ == '__main__':
    multiprocessing.freeze_support()
    hs = main()
    test_encounter(hs)
    exec(df2.present())
'''
python _tests/test_encounter.py --dbdir ~/data/work/MISC_Jan12
python _tests/test_encounter.py --dbdir ~/data/work/NAUTS_Dan
'''
