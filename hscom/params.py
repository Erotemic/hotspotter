'''Parameters module: DEPRICATE THIS
    stores a bunch of global variables used by the other modules
    It also reads from sys.argv'''
from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[params]')
# Python
from os.path import exists, join
import fileio as io
import sys

'''
print(' * __name__ = %s' % __name__)
print(' * sys.argv = %r' % sys.argv)
print(' * sys.checkinterval   = %r' % sys.getcheckinterval())
print(' * sys.defaultencoding = %r' % sys.getdefaultencoding())
print(' * sys.filesystemencoding = %r' % sys.getfilesystemencoding())
print(' * sys.prefix = %r' % sys.prefix)
'''


#=====================================================
# DEV DATABASE GLOBALS
#=====================================================
#def find_workdir():
    #workdir_list = [
        #'D:/data/work',
        #'/media/Store/data/work', ]
    #for workdir in workdir_list:
        #if exists(workdir):
            #return workdir
    #return join(expanduser('~'),  'data/work')


WORKDIR_CACHEID = 'work_directory_cache_id'


# TODO: workdir doesnt belong in params
def get_workdir(allow_gui=True):
    work_dir = io.global_cache_read(WORKDIR_CACHEID, default='.')
    if work_dir is not '.' and exists(work_dir):
        return work_dir
    if allow_gui:
        work_dir = set_workdir()
    return None


def set_workdir(work_dir=None, allow_gui=True):
    if work_dir is None and allow_gui:
        work_dir = guiselect_workdir()
    if work_dir is None or not exists(work_dir):
        raise AssertionError('invalid workdir=%r' % work_dir)
    io.global_cache_write(WORKDIR_CACHEID, work_dir)
    return work_dir


def guiselect_workdir():
    from hsgui import guitools
    # Gui selection
    work_dir = guitools.select_directory('Work dir not currently set.' +
                                         'Select a work directory')
    # Make sure selection is ok
    if not exists(work_dir):
        msg_try = 'Directory %r does not exist.' % work_dir
        opt_try = ['Try Again']
        try_again = guitools._user_option(None, msg_try, 'get work dir failed', opt_try, False)
        if try_again == 'Try Again':
            return guiselect_workdir()
    return work_dir

# Common databases I use
dbalias_dict = {
    'NAUTS':            'NAUT_Dan',
    'WD':               'WD_Siva',
    'LF':               'LF_all',
    'GZ':               'GZ_ALL',
    'MOTHERS':          'HSDB_zebra_with_mothers',
    'FROGS':            'Frogs',
    'TOADS':            'WY_Toads',
    'SEALS':            'Seals',

    'OXFORD':           'Oxford_Buildings',
    'PARIS':            'Paris_Buildings',

    'JAG_KELLY':        'JAG_Kelly',
    'JAG_KIERYN':       'JAG_Kieryn',
    'WILDEBEAST':       'Wildebeast',
    'WDOGS':            'WD_Siva',

    'PZ':               'PZ_FlankHack',
    'PZ2':              'PZ-Sweatwater',
    'PZ_MARIANNE':      'PZ_Marianne',
    'PZ_DANEXT_TEST':   'PZ_DanExt_Test',
    'PZ_DANEXT_ALL':    'PZ_DanExt_All',

    'LF_ALL':           'LF_all',
    'WS_HARD':          'WS_hard',
    'SONOGRAMS':        'sonograms',

}
dbalias_dict['JAG'] = dbalias_dict['JAG_KELLY']
#dbalias_dict['DEFAULT'] = dbalias_dict['NAUTS']
dbalias_dict['DEFAULT'] = None
# Add values from the database dict as global vars
#for key, val in dbalias_dict.iteritems():
    #exec('%s = %r' % (key, val))
#def inverse_dev_databases():
    #return {val: key for (key, val) in dbalias_dict.iteritems()}


def db_to_dbdir(db):
    work_dir = get_workdir()
    dbdir = join(work_dir, db)
    if not exists(dbdir) and db in dbalias_dict:
        dbdir = join(work_dir, dbalias_dict[db.upper()])
    if not exists(dbdir):
        import os
        import helpers as util
        print('!!!!!!!!!!!!!!!!!!!!!')
        print('[params] WARNING: db=%r not found in work_dir=%r' %
              (db, work_dir))
        fname_list = os.listdir(work_dir)
        lower_list = [fname.lower() for fname in fname_list]
        index = util.listfind(lower_list, db.lower())
        if index is not None:
            print('[params] WARNING: db capitalization seems to be off')
            if not '--strict' in sys.argv:
                print('[params] attempting to fix it')
                db = fname_list[index]
                dbdir = join(work_dir, db)
                print('[params] dbdir=%r' % dbdir)
                print('[params] db=%r' % db)
        if not exists(dbdir):
            print('[params] Valid DBs:')
            print('\n'.join(fname_list))
            print('[params] dbdir=%r' % dbdir)
            print('[params] db=%r' % db)
            print('[params] work_dir=%r' % work_dir)

            raise AssertionError('[params] FATAL ERROR. Cannot load database')
        print('!!!!!!!!!!!!!!!!!!!!!')
    return dbdir


#=====================================================
# Flann Configurations
#=====================================================

hs1_params = {'algorithm': 'kdtree',
              'trees':     4,
              'checks':    128}

quick_and_dirty_params = {'algorithm': 'kdtree',
                          'trees': 8,
                          'checks': 8}

mother_hesaff_tuned_params = {'algorithm':           'kmeans',
                              'branching':           16,
                              'build_weight':        0.009999999776482582,
                              'cb_index':            0.20000000298023224,
                              'centers_init':        'default',
                              'checks':              154,
                              'cores':               0,
                              'eps':                 0.0,
                              'iterations':          5,
                              'key_size_':           20L,
                              'leaf_max_size':       4,
                              'log_level':           'warning',
                              'max_neighbors':       -1,
                              'memory_weight':       0.0,
                              'multi_probe_level_':  2L,
                              'random_seed':         94222758,
                              'sample_fraction':     0.10000000149011612,
                              'sorted':              1,
                              'speedup':             23.30769157409668,
                              'table_number_':       12L,
                              'target_precision':    0.8999999761581421,
                              'trees':               1}

BOW_AKMEANS_FLANN_PARAMS = {'algorithm': 'kdtree',
                            'trees':    8,
                            'checks':   64}

BOW_WORDS_FLANN_PARAMS   = hs1_params
VSMANY_FLANN_PARAMS      = hs1_params
VSONE_FLANN_PARAMS       = hs1_params
FLANN_PARAMS       = hs1_params

AKMEANS_MAX_ITERS = 100

CACHE_QUERY    = True
REVERIFY_QUERY = False
RESAVE_QUERY   = False

WHITEN_FEATS   = False

#BOW_NUM_WORDS      = long(5e4)  # Vocab size for bag of words
#BOW_NDESC_PER_WORD = 14


#def OXFORD_defaults():
    ## best scale params
    #import params
    #params.XY_THRESH         = 0.01
    #params.SCALE_THRESH_HIGH = 8
    #params.SCALE_THRESH_LOW  = 0.5
    #params.CHIP_SQRT_AREA = None
    #params.BOW_NUM_WORDS  = long(1e6)


#def GZ_defaults():
    #import params
    #params.BOW_NUM_WORDS  = 87638


#def PZ_defaults():
    #import params
    #params.BOW_NUM_WORDS  = 139454


#def MOTHERS_defaults():
    #import params
    #params.BOW_NUM_WORDS  = 16225
