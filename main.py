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


def parse_arguments(**kwargs):
    '''Defines the arguments for hotspotter'''
    import argparse
    parser = argparse.ArgumentParser(description='HotSpotter - Investigate Chip', prefix_chars='+-')
    addarg = parser.add_argument

    def add_meta(switch, type, default, help, step=None, **kwargs):
        dest = switch.strip('-').replace('-', '_')
        addarg(switch, metavar=dest, type=type, default=default, help=help, **kwargs)
        if not step is None:
            add_meta(switch + '-step', type, step, help='', **kwargs)

    def add_bool(switch, default=True, help=''):
        action = 'store_false' if default else 'store_true'
        dest = switch.strip('-').replace('-', '_')
        addarg(switch, dest=dest, action=action, default=default, help=help)

    def add_var(switch, default, help, **kwargs):
        add_meta(switch, None, default, help, **kwargs)

    def add_int(switch, default, help, **kwargs):
        add_meta(switch, int, default, help, **kwargs)

    def add_float(switch, default, help, **kwargs):
        add_meta(switch, float, default, help, **kwargs)

    def add_str(switch, default, help, **kwargs):
        add_meta(switch, str, default, help, **kwargs)

    def test_bool(switch):
        add_bool(switch, False, '')
    add_int('--qcid',  None, 'query chip-id to investigate', nargs='*')
    add_int('--ocid',  [], 'query chip-id to investigate', nargs='*')
    add_int('--histid', None, 'history id (hard cases)', nargs='*')
    add_int('--r', [], 'view row', nargs='*')
    add_int('--c', [], 'view col', nargs='*')
    add_int('--nRows', 1, 'number of rows')
    add_int('--nCols', 1, 'number of cols')
    add_float('--xy-thresh', .001, '', step=.005)
    add_float('--ratio-thresh', 1.2, '', step=.1)
    add_int('--K', 2, 'for K-nearest-neighbors', step=20)
    add_int('--sthresh', (10, 80), 'scale threshold', nargs='*')
    add_str('--score-method', 'csum', 'aggregation method')
    add_int('--query',  None, 'query chip-id to investigate', nargs='*')
    add_bool('--nopresent', default=False)
    add_bool('--save-figures', default=False)
    add_bool('--noannote', default=False)
    # Program behavior
    num_cpus = multiprocessing.cpu_count()
    num_proc_help = 'default to number of cpus = %d' % (num_cpus)
    add_int('--num-procs', num_cpus, num_proc_help)
    add_bool('--serial', default=False, help='Forces num_procs=1')
    add_bool('--strict', default=False, help='Force failure in iffy areas')

    # Database selections
    add_str('--db', 'DEFAULT', 'specifies the short name of the database to load')
    add_str('--dbdir', None, 'specifies the full path of the database to load')
    add_bool('--dbM', default=False)
    add_bool('--dbG', default=False)
    # View Directories
    add_bool('--vrd', default=False)
    add_bool('--vcd', default=False)
    add_bool('--vdd', default=False)
    add_bool('--vrdq', default=False)
    add_bool('--vcdq', default=False)
    add_bool('--show-res', default=False)
    add_bool('--noprinthist', default=True)
    # Testing flags
    add_bool('--test-vsmany', default=False)
    add_bool('--test-vsone', default=False)
    add_bool('--all-cases', default=False)
    add_bool('--all-gt-cases', default=False)
    # Cache flags
    add_bool('--nocache-db', default=False,
             help='forces user to specify database directory')
    add_bool('--nocache-chips', default=False)
    add_bool('--nocache-query', default=False)
    add_bool('--nocache-feats', default=False)
    # Plotting Args
    add_bool('--noshow-query', default=False)
    add_bool('--noshow-gt', default=False)
    add_bool('--printoff', default=False)
    add_bool('--horiz', default=True)
    add_bool('--darken', default=False)
    add_str('--tests', [], 'integer or test name', nargs='*')
    add_str('--show-best', [], 'integer or test name', nargs='*')
    add_str('--show-worst', [], 'integer or test name', nargs='*')
    args, unknown = parser.parse_known_args()
    args.__dict__.update(**kwargs)
    args = args_postprocess(args)
    args = fix_args_shortnames(args)
    return args


def args_postprocess(args):
    from os.path import realpath, exists
    # Postprocess args
    if args.serial:
        args.num_procs = 1
    if args.darken:
        import draw_func2 as df2
        df2.DARKEN = .5
    if args.dbM:
        args.db = 'MOTHERS'
    if args.dbG:
        args.db = 'GZ'
    if args.dbdir is not None:
        # The full path is specified
        args.dbdir = realpath(args.dbdir)
    if args.dbdir is None or args.dbdir in ['', ' ', '.'] or not exists(args.dbdir):
        args.dbdir = None
    return args


def fix_args_shortnames(args):
    import params
    if args.dbdir is None and args.db is not None:
        # The shortname is specified
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
    return args


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


def fix_args_with_cache(args):
    'Returns the database directory based on cache'
    import fileio as io
    if args.dbdir is None and not args.nocache_db:
        # Read from cache
        args.dbdir = io.global_cache_read('db_dir')
        if args.dbdir in ['.', '', ' ']:
            args.dbdir = None
        print('[main] trying to read db_dir from cache: %r' % args.dbdir)
    args = fix_args_shortnames(args)
    return args


if __name__ == '__main__':
    # Necessary for windows parallelization
    multiprocessing.freeze_support()
    import HotSpotter
    import guitools
    import helpers
    print('main.py')
    signal_set()
    # Run qt app
    app, is_root = guitools.init_qtapp()
    args = parse_arguments()
    # Parse arguments
    args = fix_args_with_cache(args)
    if args.vdd:
        helpers.vd(args.dbdir)
        args.vdd = False
    # Build hotspotter database
    hs = HotSpotter.HotSpotter(args)
    try:
        hs.load(load_all=False)
    except ValueError as ex:
        print('[main] ValueError = %r' % (ex,))
        if hs.args.strict:
            raise
    # Connect database to the backend gui
    backend = guitools.make_main_window(hs, app)
    app.setActiveWindow(backend.win)
    # Allow for a IPython connection by passing the --cmd flag
    embedded = False
    exec(helpers.ipython_execstr())
    if not embedded:
        # If not in IPython run the QT main loop
        guitools.run_main_loop(app, is_root, backend)
    signal_reset()
