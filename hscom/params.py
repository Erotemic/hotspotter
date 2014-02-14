'''Parameters module: DEPRICATE THIS
    stores a bunch of global variables used by the other modules
    It also reads from sys.argv'''
from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[params]')
# Python
from os.path import exists, expanduser, join


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
def find_workdir():
    workdir_list = [
        'D:/data/work',
        '/media/Store/data/work', ]
    for workdir in workdir_list:
        if exists(workdir):
            return workdir
    return join(expanduser('~'),  'data/work')

WORK_DIR = find_workdir()

#WORK_DIR2 = 'D:/data/work'
#WORK_DIR  = '/media/SSD_Extra/work'

# Common databases I use
dev_databases = {
    'SONOGRAMS':       WORK_DIR + '/sonograms',
    'NAUTS':           WORK_DIR + '/NAUT_Dan',

    'OXFORD':          WORK_DIR + '/Oxford_Buildings',
    'PARIS':           WORK_DIR + '/Paris_Buildings',

    'JAG_KELLY':       WORK_DIR + '/JAG_Kelly',
    'JAG_KIERYN':      WORK_DIR + '/JAG_Kieryn',
    'WILDEBEAST':      WORK_DIR + '/Wildebeast',
    'WDOGS':           WORK_DIR + '/WD_Siva',

    'GZ':              WORK_DIR + '/GZ_ALL',
    'MOTHERS':         WORK_DIR + '/HSDB_zebra_with_mothers',

    'PZ':              WORK_DIR + '/PZ_FlankHack',
    'PZ2':             WORK_DIR + '/PZ-Sweatwater',
    'PZ_MARIANNE':     WORK_DIR + '/PZ_Marianne',
    'PZ_DANEXT_TEST':  WORK_DIR + '/PZ_DanExt_Test',
    'PZ_DANEXT_ALL':   WORK_DIR + '/PZ_DanExt_All',

    'LF_ALL':          WORK_DIR + '/LF_all',
    'WS_HARD':         WORK_DIR + '/WS_hard',

    'FROGS':           WORK_DIR + '/Frogs',
    'TOADS':           WORK_DIR + '/WY_Toads',
    'SEALS':           WORK_DIR + '/Seals',
}
dev_databases['JAG'] = dev_databases['JAG_KELLY']
#dev_databases['DEFAULT'] = dev_databases['NAUTS']
dev_databases['DEFAULT'] = None
# Add values from the database dict as global vars
for key, val in dev_databases.iteritems():
    exec('%s = %r' % (key, val))


def inverse_dev_databases():
    return {val: key for (key, val) in dev_databases.iteritems()}


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
