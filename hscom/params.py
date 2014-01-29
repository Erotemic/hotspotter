'''Parameters module: DEPRICATE THIS
    stores a bunch of global variables used by the other modules
    It also reads from sys.argv'''
from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[params]')
# Python
import sys
import helpers
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

#=====================================================
# ALGO GLOBALS
#=====================================================
# Double __ means It is an algorithm varaible
#---  CHIP COMPUTE ---
__CHIP_SQRT_AREA__ = 750
__GRABCUT__        = False
__HISTEQ__         = False
__REGION_NORM__    = False
__RANK_EQ__        = False
__LOCAL_EQ__       = False
__MAXCONTRAST__    = False
#--- FEATURE COMPUTE ---
__FEAT_TYPE__    = 'HESAFF'    # Feature type to use
#--- MATCH CHIPS ---
__MATCH_TYPE__         = 'vsmany'  # Matching type
__VSMANY_K__           = 5         # Number of matches for one-vs-many
__USE_RECIPROCAL_NN__  = True      # Number of matches for one-vs-many
__USE_SPATIAL_NN__     = True      # Number of matches for one-vs-many
__VSMANY_SCORE_FN__    = 'LNBNN'  # LNRAT, LNBNN, RATIO, PL, BORDA
__BOW_NUM_WORDS__      = long(5e4)  # Vocab size for bag of words
__BOW_NDESC_PER_WORD__ = 14
__VSONE_RATIO_THRESH__ = 1.5       # Thresholds for one-vs-one
#---------------------
# SPATIAL VERIFICATION
__NUM_RERANK__   = 1000  # Number of top matches to spatially re-rank
__XY_THRESH__    = .002  # % diaglen of keypoint extent
__USE_CHIP_EXTENT__ = False  # use roi as threshold instead of kpts extent
__SCALE_THRESH_HIGH__ = 2.0
__SCALE_THRESH_LOW__  = 0.5
#=====================================================
# FUNCTIONS
#=====================================================


# -------------------


if '--histeq' in sys.argv:
    #print('[params] with histogram equalization')
    __HISTEQ__ = True
if '--grabcut' in sys.argv:
    __GRABCUT__ = True
    #print('[params] with grabcut')
if '--regnorm' in sys.argv:
    __REGION_NORM__ = True
if '--rankeq' in sys.argv:
    __RANK_EQ__ = True
if '--norankeq' in sys.argv:
    __RANK_EQ__ = False
if '--localeq' in sys.argv:
    __LOCAL_EQ__ = True
if '--maxcont' in sys.argv:
    __MAXCONTRAST__ = True


if '--whiten' in sys.argv or '--white' in sys.argv:
    #print('[params] with whitening')
    WHITEN_FEATS = True

if '--vsone' in sys.argv:
    __MATCH_TYPE__ = 'vsone'
if '--vsmany' in sys.argv:
    __MATCH_TYPE__ = 'vsmany'
if '--bagofwords' in sys.argv:
    __MATCH_TYPE__ = 'bagofwords'

if '--lnbnn' in sys.argv:
    __VSMANY_SCORE_FN__ = 'LNBNN'
if '--ratio' in sys.argv:
    __VSMANY_SCORE_FN__ = 'RATIO'
if '--lnrat' in sys.argv:
    __VSMANY_SCORE_FN__ = 'LNRAT'

if '--cache-query' in sys.argv:
    CACHE_QUERY = True
if '--nocache-query' in sys.argv:
    CACHE_QUERY = False


if '--reverify' in sys.argv:
    REVERIFY_QUERY = True

if '--resave-query' in sys.argv:
    RESAVE_QUERY = True  # 4H4X5

if '--print-checks' in sys.argv:
    helpers.PRINT_CHECKS = True


def OXFORD_defaults():
    # best scale params
    import params
    params.__XY_THRESH__         = 0.01
    params.__SCALE_THRESH_HIGH__ = 8
    params.__SCALE_THRESH_LOW__  = 0.5
    params.__CHIP_SQRT_AREA__ = None
    params.__BOW_NUM_WORDS__  = long(1e6)


def GZ_defaults():
    import params
    params.__BOW_NUM_WORDS__  = 87638


def PZ_defaults():
    import params
    params.__BOW_NUM_WORDS__  = 139454


def MOTHERS_defaults():
    import params
    params.__BOW_NUM_WORDS__  = 16225

if '--dbG' in sys.argv:
    sys.argv += ['--db GZ']
if '--dbM' in sys.argv:
    sys.argv += ['--db MOTHERS']

for argv in iter(sys.argv):
    argv_u =  argv.upper()
    """
    if multiprocessing.current_process().name == 'MainProcess':
        print('[params] argv: %r' % argv_u)
    """
    if argv_u in dev_databases.keys():
        """
        if multiprocessing.current_process().name == 'MainProcess':
            print('\n'.join(['[params] Default Database set to:'+argv.upper(),
                            '[params] Previously: '+str(DEFAULT)]))
        """
        DEFAULT = dev_databases[argv_u]
        if argv_u == 'OXFORD' or argv_u == 'PHILBIN':
            """
            if multiprocessing.current_process().name == 'MainProcess':
                print('[params] Overloading OXFORD parameters')
            """
            # dont resize oxford photos
            OXFORD_defaults()
        if argv_u == 'GZ':
            GZ_defaults()
        if argv_u == 'PZ':
            PZ_defaults()
        if argv_u == 'MOTHERS':
            MOTHERS_defaults()
