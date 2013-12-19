from __future__ import division, print_function
import multiprocessing
import argparse

DEBUG = False


class ArgumentParser2(object):
    'Wrapper around argparse.ArgumentParser with convinence functions'
    def __init__(self, parser):
        self.parser = parser
        self.add_arg = parser.add_argument

    def add_meta(self, switch, type, default=None, help='', **kwargs):
        dest = switch.strip('-').replace('-', '_')
        self.add_arg(switch, metavar=dest, type=type, default=default, help=help, **kwargs)

    def add_flag(self, switch, default=False, *args, **kwargs):
        action = 'store_false' if default else 'store_true'
        dest = switch.strip('-').replace('-', '_')
        self.add_arg(switch, dest=dest, action=action, default=default, *args, **kwargs)

    def add_var(self, switch, **kwargs):
        self.add_meta(switch, None, *args, **kwargs)

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


def main_argparse(parser2):
    parser2 = parser2.add_argument_group('Main')
    parser2.add_flag('--autoquery')
    parser2.add_intlist('--query', default=[], help='query chip-id to investigate')
    parser2.add_int('--qcid', help='query chip-id to investigate', nargs='*')
    parser2.add_intlist('--ocid', help='query chip-id to investigate')
    parser2.add_int('--histid', help='history id (hard cases)')
    parser2.add_intlist('--r', help='view row')
    parser2.add_intlist('--c', help='view col')
    parser2.add_flag('--nopresent')
    parser2.add_flag('--save-figures')
    parser2.add_flag('--noannote')
    parser2.add_flag('--quiet')


def dev_argparse(parser2):
    parser2 = parser2.add_argument_group('Dev')
    # Misc
    parser2.add_flag('--export-qon-list')
    # Testing flags
    parser2.add_flag('--test-vsmany')
    parser2.add_flag('--test-vsone')
    parser2.add_flag('--all-cases')
    parser2.add_flag('--all-gt-cases')
    # Plotting Args
    parser2.add_flag('--noshow-query')
    parser2.add_flag('--noshow-gt')
    parser2.add_flag('--printoff')
    parser2.add_flag('--horiz', True)
    parser2.add_flag('--darken')
    parser2.add_str('--tests',  [], 'integer or test name', nargs='*')
    # View Directories
    parser2.add_flag('--vrd')
    parser2.add_flag('--vcd')
    parser2.add_flag('--vdd')
    parser2.add_flag('--vrdq')
    parser2.add_flag('--vcdq')
    parser2.add_flag('--show-res')
    parser2.add_flag('--noprinthist', True)


def database_argparase(parser2):
    # Database selections
    parser2 = parser2.add_argument_group('Database')
    parser2.add_str('--db', 'DEFAULT', 'specifies the short name of the database to load')
    parser2.add_str('--dbdir', None, 'specifies the full path of the database to load')
    parser2.add_flag('--dbM')
    parser2.add_flag('--dbG')


def behavior_argparse(parser2):
    # Program behavior
    parser2 = parser2.add_argument_group('Behavior')
    num_cpus = multiprocessing.cpu_count()
    num_proc_help = 'default to number of cpus = %d' % (num_cpus)
    parser2.add_int('--num-procs', num_cpus, num_proc_help)
    parser2.add_flag('--serial', help='Forces num_procs=1')
    parser2.add_flag('--strict', help='Force failure in iffy areas')


def cfg_argparse(parser2):
    parser2 = parser2.add_argument_group('Config')
    import DataStructures as ds
    _qcfg = ds.default_vsmany_cfg(None)
    _fcfg = ds.default_feat_cfg(None)
    _ccfg = ds.default_chip_cfg()
    _filtcfg = _qcfg.filt_cfg
    _svcfg = _qcfg.sv_cfg
    _nncfg = _qcfg.nn_cfg
    _aggcfg = _qcfg.agg_cfg
    defcfg_list = [_fcfg, _ccfg, _filtcfg, _svcfg, _nncfg, _aggcfg]
    for cfg in defcfg_list:
        for key, val in cfg.iteritems():
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
    #parser.add_float('--xy-thresh', None, '')
    #parser.add_float('--ratio-thresh', None, '')
    #parser.add_int('--K', help='for K-nearest-neighbors')
    #parser.add_int('--N', help='num neighbors to show')
    #parser.add_int('--feat_min_scale')
    #parser.add_int('--feat_max_scale')
    #parser.add_str('--score-method', help='aggregation method')


def cache_argparse(parser2):
    # Cache flags
    parser2 = parser2.add_argument_group('Cache')
    parser2.add_flag('--nocache-db', help='forces user to specify database directory')
    parser2.add_flag('--nocache-chips')
    parser2.add_flag('--nocache-query')
    parser2.add_flag('--nocache-feats')
    parser2.add_flag('--nocache-flann')
    parser2.add_flag('--nocache-prefs')


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


def parse_arguments(**kwargs):
    '''Defines the arguments for hotspotter'''

    parser2 = make_argparse2('HotSpotter - Individual Animal Recognition',
                             version='???')
    main_argparse(parser2)
    database_argparase(parser2)
    dev_argparse(parser2)
    behavior_argparse(parser2)
    cfg_argparse(parser2)
    cache_argparse(parser2)
    args, unknown = parser2.parser.parse_known_args()
    #args, unknown = parser.parse_args()
    if DEBUG:
        print('[argparse2] args=%r' % args)
        print('[argparse2] unknown=%r' % unknown)
    args.__dict__.update(**kwargs)
    args = args_postprocess(args)
    args = fix_args_shortnames(args)
    return args


if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('\n\n====================')
    print('__main__ == argparse2.py')
    args = parse_arguments()
    print(args)
