from __future__ import division, print_function
import argparse
import multiprocessing

def printDBG(msg, *args):
    pass
# Moved this up for faster help responce time
def parse_arguments():
    '''
    Defines the arguments for investigate_chip.py
    '''
    printDBG('==================')
    printDBG('[invest] ---------')
    printDBG('[invest] ARGPARSE')
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
    add_str('--db', None, 'specifies the short name of the database to load')
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
    printDBG('[invest] args    = %r' % (args,))
    printDBG('[invest] unknown = %r' % (unknown,))
    printDBG('[invest] ---------')
    printDBG('==================')
    return args

#args = None
#if multiprocessing.current_process().name == 'MainProcess':
    #args = parse_arguments()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
