
import multiprocessing
import argparse
from . import cross_platform
# seemlessly fix any path issues
cross_platform.ensure_pythonpath()

#======================
# Globals
#======================

DEBUG = False  # True
ARGS_ = None

#======================
# Helpers
#======================


def switch_sanataize(switch):
    if isinstance(switch, str):
        dest = switch.strip('-').replace('-', '_')
    else:
        if isinstance(switch, tuple):
            switch = switch
        elif isinstance(switch, list):
            switch = tuple(switch)
        dest = switch[0].strip('-').replace('-', '_')
    return dest, switch

#======================
# Wrapper Class
#======================


class ArgumentParser2(object):
    'Wrapper around argparse.ArgumentParser with convinence functions'
    def __init__(self, parser):
        self.parser = parser
        self._add_arg = parser.add_argument

    def add_arg(self, switch, *args, **kwargs):
        #print('[argparse2] add_arg(%r) ' % (switch,))
        if isinstance(switch, tuple):
            args = tuple(list(switch) + list(args))
            return self._add_arg(*args, **kwargs)
        else:
            return self._add_arg(switch, *args, **kwargs)

    def add_meta(self, switch, type, default=None, help='', **kwargs):
        #print('[argparse2] add_meta()')
        dest, switch = switch_sanataize(switch)
        self.add_arg(switch, metavar=dest, type=type, default=default, help=help, **kwargs)

    def add_flag(self, switch, default=False, **kwargs):
        #print('[argparse2] add_flag()')
        action = 'store_false' if default else 'store_true'
        dest, switch = switch_sanataize(switch)
        self.add_arg(switch, dest=dest, action=action, default=default, **kwargs)

    def add_int(self, switch, *args, **kwargs):
        self.add_meta(switch, int,  *args, **kwargs)

    def add_intlist(self, switch, *args, **kwargs):
        self.add_meta(switch, int,  *args, nargs='*', **kwargs)

    def add_float(self, switch, *args, **kwargs):
        self.add_meta(switch, float, *args, **kwargs)

    def add_str(self, switch, *args, **kwargs):
        self.add_meta(switch, str, *args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        return ArgumentParser2(self.parser.add_argument_group(*args, **kwargs))


def make_argparse2(description, *args, **kwargs):
    formatter_classes = [
        argparse.RawDescriptionHelpFormatter,
        argparse.RawTextHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter]
    return ArgumentParser2(
        argparse.ArgumentParser(prog='HotSpotter',
                                description=description,
                                prefix_chars='+-',
                                formatter_class=formatter_classes[2], *args,
                                **kwargs))


#======================
# Argument Definitions
#======================


def commands_argparse(parser2):
    parser2 = parser2.add_argument_group('Commands')
    parser2.add_str('--setcfg', help='standard config name')
    parser2.add_flag('--autoquery')
    parser2.add_intlist('--query', default=[], help='query chip-id to investigate')
    parser2.add_intlist('--select-cids', default=[], help='chip indexes to view')
    parser2.add_intlist('--selgxs', default=[], help='image indexes to view')
    parser2.add_intlist('--selcids', default=[], help='chip indexes to view')
    parser2.add_intlist('--selnxs', default=[], help='name indexes to view')
    parser2.add_intlist('--txs', default=[], help='investigate match to top x of querys')
    parser2.add_intlist('--cids', default=[], help='investigate match cx')
    parser2.add_intlist('--qfxs', default=[], help='investigate match to cx via qfxs')
    parser2.add_int('--qcid', help='query chip-id to investigate', nargs='*')
    parser2.add_intlist('--ocid', help='query chip-id to investigate')
    parser2.add_int('--histid', help='history id (hard cases)')
    parser2.add_intlist(('--sel-rows', '-r'), help='view row')
    parser2.add_intlist(('--sel-cols', '-c'), help='view col')
    parser2.add_flag('--view-all', help='view all')
    parser2.add_flag('--nopresent')
    parser2.add_flag(('--save-figures', '--dump'))
    parser2.add_flag('--noannote')
    parser2.add_flag('--quiet')


def dev_argparse(parser2):
    parser2 = parser2.add_argument_group('Dev')
    # Testing flags
    parser2.add_flag('--all-cases')
    parser2.add_flag('--all-gt-cases')
    # Plotting Args
    parser2.add_flag('--noshow-query')
    parser2.add_flag('--noshow-gt')
    parser2.add_flag('--horiz', True)
    parser2.add_flag('--darken')
    parser2.add_str(('--tests', '--test', '-t'),  [], 'integer or test name', nargs='*')
    # View Directories
    parser2.add_flag('--vrd')
    parser2.add_flag('--vcd')
    parser2.add_flag('--vdd')
    parser2.add_flag('--vrdq')
    parser2.add_flag('--vcdq')
    parser2.add_flag('--show-res')
    parser2.add_flag('--noprinthist', True)


def database_argparse(parser2):
    # Database selections
    parser2 = parser2.add_argument_group('Database')
    parser2.add_str('--db', 'DEFAULT', 'specifies the short name of the database to load')
    parser2.add_str('--dbdir', None, 'specifies the full path of the database to load')


def behavior_argparse(parser2):
    # Program behavior
    parser2 = parser2.add_argument_group('Behavior')
    # TODO UNFILTER THIS HERE AND CHANGE PARALLELIZE TO KNOW ABOUT
    # MEMORY USAGE
    num_cpus = max(min(6, multiprocessing.cpu_count()), 1)
    num_proc_help = 'default to number of cpus = %d' % (num_cpus)
    parser2.add_int('--num-procs', num_cpus, num_proc_help)
    parser2.add_flag('--serial', help='Forces num_procs=1')
    parser2.add_flag('--strict', help='Force failure in iffy areas')
    parser2.add_flag('--nosteal', help='GUI will not steal stdout')
    parser2.add_flag('--noshare', help='GUI will not share stdout')
    parser2.add_flag('--nogui', help='Will not start the gui')
    parser2.add_flag('--withexif', help='Reads EXIF data')
    parser2.add_flag('--verbose-cache')
    parser2.add_flag('--verbose-load')
    parser2.add_flag('--aggroflush', help='Agressively flushes')
    parser2.add_flag(('--nomemory', '--nomem'), help='runs tests without' +
                     'keeping results in memory')


def cfg_argparse(parser2):
    parser2 = parser2.add_argument_group('Config')
    # TODO: This line alone makes this module not belong in hscom
    # I dont know where it should go but fix it
    from hotspotter import Config
    _qcfg = Config.default_vsmany_cfg(None)
    _fcfg = Config.default_feat_cfg(None)
    _ccfg = Config.default_chip_cfg()
    _dcfg = Config.default_display_cfg()
    _filtcfg = _qcfg.filt_cfg
    _svcfg = _qcfg.sv_cfg
    _nncfg = _qcfg.nn_cfg
    _aggcfg = _qcfg.agg_cfg
    defcfg_list = [_fcfg, _ccfg, _filtcfg, _svcfg, _nncfg, _aggcfg, _dcfg]
    for cfg in defcfg_list:
        for key, val in cfg.items():
            if key.find('_') == 0:
                continue
            elif isinstance(val, int):
                parser2.add_int('--' + key, default=val)
            elif isinstance(val, float):
                parser2.add_float('--' + key, default=val)
            elif val is None:
                parser2.add_float('--' + key, default=val)
            elif isinstance(val, str):
                parser2.add_str('--' + key, default=val)
    # TODO: pass in default hs.prefs to auto populate this


def cache_argparse(parser2):
    # Cache flags
    parser2 = parser2.add_argument_group('Cache')
    parser2.add_flag(('--delete-cache', '--dc'))
    parser2.add_flag('--quit')
    parser2.add_flag(('--delete-global', '--delete-global-cache'))
    parser2.add_flag('--nocache-db', help='forces user to specify database directory')
    parser2.add_flag('--nocache-chips')
    parser2.add_flag('--nocache-query')
    parser2.add_flag('--nocache-feats')
    parser2.add_flag('--nocache-flann')
    parser2.add_flag('--nocache-prefs')


#======================
# Argument Postprocessing
#======================


def args_postprocess(args):
    from os.path import realpath, exists
    global ARGS_
    # Postprocess args
    if args.serial:
        args.num_procs = 1
    if args.darken:
        import draw_func2 as df2
        df2.DARKEN = .5
    if args.dbdir is not None:
        # The full path is specified
        args.dbdir = realpath(args.dbdir)
    if args.dbdir is None or args.dbdir in ['', ' ', '.'] or not exists(args.dbdir):
        args.dbdir = None
    ARGS_ = args
    return args


def fix_args_shortnames(args):
    from . import params
    global ARGS_
    #print('[argparse2] fix_args_shortnames(): %r' % args.db)
    #print('[argparse2] mapping %r to %r' % (args.db, args.dbdir))
    # The shortname is specified
    if (args.dbdir is None) and (args.db is not None):
        args.dbdir = params.db_to_dbdir(args.db)
    #print('[argparse2] mapped %r to %r' % (args.db, args.dbdir))
    ARGS_ = args
    return args


def fix_args_with_cache(args):
    'Returns the database directory based on cache'
    from . import fileio as io
    global ARGS_
    if args.dbdir is None and not args.nocache_db:
        # Read from cache
        args.dbdir = io.global_cache_read('db_dir')
        if args.dbdir in ['.', '', ' ']:
            args.dbdir = None
        print('[main] trying to read db_dir from cache: %r' % args.dbdir)
    # --db has priority over --dbdir
    args = fix_args_shortnames(args)
    ARGS_ = args
    return args


#======================
# Parser Driver
#======================


def parse_arguments(defaultdb=None, **kwargs):
    '''Defines the arguments for hotspotter'''
    global ARGS_
    parser2 = make_argparse2('HotSpotter - Individual Animal Recognition')
                             # version='???')
    commands_argparse(parser2)
    database_argparse(parser2)
    dev_argparse(parser2)
    behavior_argparse(parser2)
    #cfg_argparse(parser2)
    cache_argparse(parser2)
    args, unknown = parser2.parser.parse_known_args()
    #args, unknown = parser.parse_args()
    if DEBUG:
        print('[argparse2] args=%r' % args)
        print('[argparse2] unknown=%r' % unknown)
    if args.db == 'DEFAULT' and args.dbdir is None:
        args.db = defaultdb
    args.__dict__.update(**kwargs)
    args = args_postprocess(args)
    args = fix_args_shortnames(args)
    ARGS_ = args
    return args


#======================
# Test Script
#======================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('\n\n====================')
    print('__main__ == argparse2.py')
    args = parse_arguments()
    print(args)
