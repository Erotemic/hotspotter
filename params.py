'''Parameters module:
    stores a bunch of global variables used by the other modules
    It also reads from sys.argv'''
from __future__ import division, print_function
import __builtin__
import sys
import os.path
import numpy as np
import helpers
import multiprocessing

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[params] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module


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

WORK_DIR = 'D:/data/work'
WORK_DIR2 = 'D:/data/work'

if sys.platform == 'linux2':
    WORK_DIR  = '/media/SSD_Extra/work'
    WORK_DIR2 = '/media/Store/data/work'

# Common databases I use
dev_databases = {
    'SONOGRAMS'      : WORK_DIR  + '/sonograms',
    'NAUTS'          : WORK_DIR  + '/NAUT_Dan',

    'OXFORD'         : WORK_DIR  + '/Oxford_Buildings',
    'PARIS'          : WORK_DIR  + '/Paris_Buildings',

    'JAG_KELLY'      : WORK_DIR  + '/JAG_Kelly',
    'JAG_KIERYN'     : WORK_DIR2 + '/JAG_Kieryn',
    'WILDEBEAST'     : WORK_DIR2 + '/Wildebeast',
    'WDOGS'          : WORK_DIR  + '/WD_Siva',

    'GZ'             : WORK_DIR  + '/GZ_ALL',
    'MOTHERS'        : WORK_DIR  + '/HSDB_zebra_with_mothers',

    'PZ'             : WORK_DIR  + '/PZ_FlankHack',
    'PZ2'            : WORK_DIR  + '/PZ-Sweatwater',
    'PZ_MARIANNE'    : WORK_DIR  + '/PZ_Marianne',
    'PZ_DANEXT_TEST' : WORK_DIR  + '/PZ_DanExt_Test',
    'PZ_DANEXT_ALL'  : WORK_DIR2 + '/PZ_DanExt_All',

    'LF_ALL'         : WORK_DIR  + '/LF_all',
    'WS_HARD'        : WORK_DIR  + '/WS_hard',

    'FROGS'          : WORK_DIR2 + '/Frogs',
    'TOADS'          : WORK_DIR2 + '/WY_Toads'
}
#dev_databases['DEFAULT'] = dev_databases['NAUTS']
dev_databases['DEFAULT'] = None
# Add values from the database dict as global vars
for key, val in dev_databases.iteritems():
    exec('%s = %r' % (key, val))

def inverse_dev_databases():
    return {val:key for (key, val) in dev_databases.iteritems()}


#=====================================================
# Flann Configurations
#=====================================================

hs1_params = {'algorithm':'kdtree',
              'trees'    :4,
              'checks'   :128}

quick_and_dirty_params = {'algorithm':'kdtree',
              'trees'    :8,
              'checks'   :8}

mother_hesaff_tuned_params = {'algorithm'          : 'kmeans',
                              'branching'          : 16,
                              'build_weight'       : 0.009999999776482582,
                              'cb_index'           : 0.20000000298023224,
                              'centers_init'       : 'default',
                              'checks'             : 154,
                              'cores'              : 0,
                              'eps'                : 0.0,
                              'iterations'         : 5,
                              'key_size_'          : 20L,
                              'leaf_max_size'      : 4,
                              'log_level'          : 'warning',
                              'max_neighbors'      : -1,
                              'memory_weight'      : 0.0,
                              'multi_probe_level_' : 2L,
                              'random_seed'        : 94222758,
                              'sample_fraction'    : 0.10000000149011612,
                              'sorted'             : 1,
                              'speedup'            : 23.30769157409668,
                              'table_number_'      : 12L,
                              'target_precision'   : 0.8999999761581421,
                              'trees'              : 1}

BOW_AKMEANS_FLANN_PARAMS = {'algorithm':'kdtree',
                                'trees'    :8,
                                'checks'   :64}

BOW_WORDS_FLANN_PARAMS   = hs1_params 
VSMANY_FLANN_PARAMS      = hs1_params
VSONE_FLANN_PARAMS       = hs1_params
FLANN_PARAMS       = hs1_params

AKMEANS_MAX_ITERS = 100

VERBOSE_CACHE = False
VERBOSE_LOAD_DATA = True
VERBOSE_MATCHING = True
VERBOSE_IO = 2


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
__VSMANY_SCORE_FN__    = 'LNBNN' # LNRAT, LNBNN, RATIO, PL, BORDA
__BOW_NUM_WORDS__      = long(5e4) # Vocab size for bag of words
__BOW_NDESC_PER_WORD__ = 14
__VSONE_RATIO_THRESH__ = 1.5       # Thresholds for one-vs-one
#---------------------
# SPATIAL VERIFICATION
__NUM_RERANK__   = 1000 # Number of top matches to spatially re-rank
__XY_THRESH__    = .002 # % diaglen of keypoint extent
__USE_CHIP_EXTENT__ = False # use roi as threshold instead of kpts extent
__SCALE_THRESH_HIGH__ = 2.0
__SCALE_THRESH_LOW__  = 0.5
#=====================================================
# FUNCTIONS
#=====================================================

def param_string():
    import re
    param_list = []
    param_list_append = param_list.append
    def has_bad_dependency(key):
        bad_depends1 = (key.find('BOW') in [0,2]    and __MATCH_TYPE__ != 'bagofwords')
        bad_depends2 = (key.find('VSMANY') in [0,2] and __MATCH_TYPE__ != 'vsmany')
        bad_depends3 = (key.find('VSONE') in [0,2]  and __MATCH_TYPE__ != 'vsone')
        return any([bad_depends1, bad_depends2, bad_depends3])
    for key in globals().iterkeys():
        if re.match('__[A-Z_]*__', key):
            if has_bad_dependency(key): continue
            param_list_append(key + ' = ' + repr(globals()[key]))
    param_str = '\n'.join(param_list)
    return param_str

#---------------------
# Strings corresponding to unique ids used by different files

def get_chip_uid():
    isorig = (__CHIP_SQRT_AREA__ is None or __CHIP_SQRT_AREA__ <= 0)
    histeq = ['','_histeq'][__HISTEQ__]
    grabcut = ['','_grabcut'][__GRABCUT__]
    myeq = ['','_regnorm'][__REGION_NORM__]
    rankeq = ['','_rankeq'][__RANK_EQ__]
    localeq = ['','_localeq'][__LOCAL_EQ__]
    maxcontrast = ['','_maxcont'][__MAXCONTRAST__]
    resize = ['_szorig', ('_sz%r' % __CHIP_SQRT_AREA__)][not isorig] 
    chip_uid = resize + grabcut + histeq + myeq + rankeq + localeq + maxcontrast
    return chip_uid

def get_feat_uid():
    feat_type = '_'+__FEAT_TYPE__
    whiten = ['','_white'][WHITEN_FEATS]
    # depends on chip
    feat_uid = feat_type + whiten + get_chip_uid()
    return feat_uid

TRAIN_INDX_SAMPLE_ID = ''
TRAIN_SAMPLE_ID = ''
INDX_SAMPLE_ID = ''
TEST_SAMPLE_ID = ''

def get_matcher_uid(with_train=True, with_indx=True):
    matcher_uid = '_'+__MATCH_TYPE__
    if __MATCH_TYPE__ == 'bagofwords':
        matcher_uid += '_W%d' % __BOW_NUM_WORDS__
    if with_train:
        matcher_uid += '_trainID('+TRAIN_SAMPLE_ID+')'
    if with_indx:
        matcher_uid += '_indxID('+INDX_SAMPLE_ID+')'
    # depends on feat
    matcher_uid += get_feat_uid()
    return matcher_uid

def get_indexed_uid(with_train=True, with_indx=True):
    indexed_uid = ''
    if with_train:
        indexed_uid += '_trainID('+TRAIN_SAMPLE_ID+')'
    if with_indx:
        indexed_uid += '_indxID('+INDX_SAMPLE_ID+')'
    # depends on feat
    indexed_uid += get_feat_uid()
    return indexed_uid

def get_query_uid():
    query_uid = ''
    query_uid += '_sv(' 
    query_uid += str(__NUM_RERANK__)
    query_uid += ',' + str(__XY_THRESH__)
    query_uid += ',' + str(__SCALE_THRESH_HIGH__)
    query_uid += ',' + str(__SCALE_THRESH_LOW__)
    query_uid += ',cdl' * __USE_CHIP_EXTENT__ # chip diag len
    query_uid += ')'
    if __MATCH_TYPE__ == 'vsmany':
        query_uid += '_k%d' % __VSMANY_K__
    if __MATCH_TYPE__ == 'vsone':
        query_uid += '_rat'+str(__VSONE_RATIO_THRESH__)
    query_uid += '_rnn' * __USE_RECIPROCAL_NN__
    query_uid += '_snn' * __USE_SPATIAL_NN__
    if __MATCH_TYPE__ == 'vsmany':
        query_uid += '_%s' % __VSMANY_SCORE_FN__
    query_uid += get_matcher_uid()
    return query_uid

def get_algo_uid():
    'Unique id for the entire algorithm'
    algo_uid = get_query_uid()
    return algo_uid

# -------------------

def mothers_problem_pairs():
    '''MOTHERS Dataset: difficult (qcx, cx) query/result pairs'''
    #-
    viewpoint \
            = [
        ( 16,  17),
        ( 19,  20),
        ( 73,  71),
        ( 75,  78),
        (108, 112), # query is very dark
        (110, 108),
                ]
    #-
    quality   \
            = [
        (27, 26),   #minor viewpoint
        (52, 53),
        (67, 68),   #stupid hard case (query from 68 to 67 direction is better (start with foal)
        (73, 71),
    ]
    #-
    lighting  \
            = [
        (105, 104),
        ( 49,  50), #brush occlusion on legs
        ( 93,  94),
    ]
    #-
    confused  \
            = [
    ]
    #-
    occluded  \
            = [
        (64,65),
    ]
    #-
    return locals()


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
    RESAVE_QUERY = True # 4H4X5

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
    sys.argv+=['--db GZ']
if '--dbM' in sys.argv:
    sys.argv+=['--db MOTHERS']

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


def make_pref_object():
    'Not finished yet, but its a start'
    import Pref
    Pref.rrr()
    # --- chip compute ---
    chip = Pref.Pref()
    chip.chip_sqrt_area = 750
    chip.histeq         = False
    chip.region_norm    = False
    chip.rank_eq        = False
    chip.local_eq       = False
    chip.maxcontrast    = False
    #-----------------------
    # --- feature compute ---
    feat = Pref.Pref()
    feat.feat_type       = 'HESAFF'  # Feature type to use
    #-----------------------
    # -- vsone --
    vsone  = Pref.Pref()
    vsone.ratio_thresh   = 1.5       # Thresholds for one-vs-one
    # -- vsmany --
    vsmany = Pref.Pref()
    vsmany.K             = 5         # Number of matches for one-vs-many
    vsmany.score_type    = 'LNBNN'
    # -- bow --
    bow = Pref.Pref()
    bow.num_words    = long(5e4) # Vocab size for bag of words
    bow.nDesc_per_word = 14 # Vocab size for bag of words
    # --- match chips ---
    match = Pref.Pref()
    match.match_type = 'vsmany'  # Matching type
    match.use_krnn = False
    match.use_snn  = False
    #-----------------------
    # --- spatial verification ---
    verify  = Pref.Pref()
    verify.num_rerank        = 1000      # Number of top matches to spatially re-rank
    verify.xy_thresh         = .1        # % diaglen of keypoint extent
    verify.scale_thresh_low  = .5
    verify.scale_thresh_high = 2
    #-----------------------
    # --- all preferences ---
    prefs = Pref.Pref()
    prefs.chip = chip
    prefs.match = match
    prefs.vsmany = vsmany
    prefs.vsone = vsone
    prefs.bow = bow
    prefs.verify = verify
    #-----------------------
    epw = prefs.createQWidget()

if __name__ == '__main__':
    print('[params] __main__ = params.py')
    print('[params] Param string:')
    print(helpers.indent(param_string()))
