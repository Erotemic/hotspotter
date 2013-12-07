#!/usr/bin/python2.7
from __future__ import division, print_function

# Moved this up for faster help responce time
def parse_arguments():
    import argparse
    '''
    Defines the arguments for investigate_chip.py
    '''
    parser = argparse.ArgumentParser(description='HotSpotter - Investigate Chip', prefix_chars='+-')
    def_on  = {'action':'store_false', 'default':True}
    def_off = {'action':'store_true', 'default':False}
    addarg = parser.add_argument
    def add_meta(switch, type, default, help, step=None, **kwargs):
        dest = switch.strip('-').replace('-','_')
        addarg(switch, metavar=dest, type=type, default=default, help=help, **kwargs)
        if not step is None:
            add_meta(switch+'-step', type, step, help='', **kwargs)
    def add_bool(switch, default=True, help=''):
        action = 'store_false' if default else 'store_true' 
        dest = switch.strip('-').replace('-','_')
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
    add_int('--K', 10, 'for K-nearest-neighbors', step=20)
    add_int('--sthresh', (10,80), 'scale threshold', nargs='*')
    add_bool('--nopresent', default=False)
    add_bool('--save-figures', default=False)
    add_bool('--noannote', default=False)
    # Database selections
    add_str('--db', 'DEFAULT', 'specifies the short name of the database to load')
    add_str('--dbdir', None, 'specifies the full path of the database to load')
    add_bool('--dbM', default=False)
    add_bool('--dbG', default=False)
    # View Directories
    add_bool('--vrd', default=False)
    add_bool('--vcd', default=False)
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
    add_bool('--printoff', default=False)
    add_bool('--horiz', default=True)
    add_str('--tests', [], 'integer or test name', nargs='*')
    add_str('--show-best', [], 'integer or test name', nargs='*')
    add_str('--show-worst', [], 'integer or test name', nargs='*')
    args, unknown = parser.parse_known_args()
    return args

def fix_args(args):
    import params
    from os.path import exists
    if args.dbM: args.db = 'MOTHERS'
    if args.dbG: args.db = 'GZ'
    if args.dbdir is not None:
        # The full path is specified
        args.dbdir = realpath(args.dbdir)
    if args.dbdir is None and args.db is not None:
        # The shortname is specified
        try:
            args.dbdir = params.dev_databases[args.db]
        except KeyError:
            pass
    try:
        # Lookup shortname
        try:
            inverse_dev_databases = params.inverse_dev_databases()
            args.db = inverse_dev_databases[args.db_dir]
        except KeyError:
            pass
    except Exception as ex:
        pass
    if args.dbdir is None or args.dbdir in ['', ' ', '.'] or not exists(args.dbdir):
        args.dbdir = None
    return args

def on_ctrl_c(signal, frame):
    import sys
    print('Caught ctrl+c')
    print('Hotspotter parent process killed by ctrl+c')
    sys.exit(0)

def signal_reset():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL) # reset ctrl+c behavior

def signal_set():
    import signal
    signal.signal(signal.SIGINT, on_ctrl_c)

def user_selection(db_dir):
    'Returns the database directory based on args, cache, or gui selection'
    import fileio as io
    import gui
    if db_dir is None and not args.nocache_db:
        # Read from cache
        db_dir = io.global_cache_read('db_dir')
        if db_dir in ['.','']: db_dir = None
        print('[main] trying to read db_dir from cache: '+repr(db_dir))
    if db_dir is None and gui.IS_INIT:
        # All has failed. Ask the user and write to global cache.
        db_dir = gui.select_directory('Select (or create) a database directory.')
        print('[main] user selects database: '+db_dir)
        io.global_cache_write('db_dir', db_dir)
    return db_dir

if __name__ == '__main__':
    from multiprocessing import freeze_support
    import HotSpotter
    import gui
    freeze_support()
    print('main.py')
    app, is_root = gui.init_qtapp()
    args = fix_args(parse_arguments())
    args.dbdir = user_selection(args.dbdir)
    hs = HotSpotter.HotSpotter(args).load()
    main_win = gui.make_main_window(hs)
    gui.run_main_loop(app, is_root)
