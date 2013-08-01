import os.path
import sys
import helpers


# Number of processessors
__NUM_PROCS__ = 9
# Feature type to use
__FEAT_TYPE__    = 'HESAFF'
# Matching type
__MATCH_TYPE__   = '1vM'
# Number of matches for one-vs-many
__K__            = 2
# Thresholds for one-vs-one
__RATIO_THRESH__ = 1.5
# Number of top matches to spatially re-rank
__NUM_RERANK__   = 50
# Percentage of the diagonal length of keypoint extent
__XY_THRESH__    = .05
hs1_params = {'algorithm':'kdtree',
              'trees'    :4,
              'checks'   :128}

# Unwhitened
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

__FLANN_ONCE_PARAMS__ = hs1_params
__FLANN_PARAMS__      = hs1_params

print(' * Setting default parameters')
print(' * Reading command line parameters')

__CACHE_QUERY__ = True
__REVERIFY_QUERY__ = False
__RESAVE_QUERY__ = False

__WHITEN_FEATS__  = False
__HISTEQ__        = False

if '--histeq' in sys.argv:
    print(' * with histogram equalization')
    __HISTEQ__ = True
if '--whiten' in sys.argv or '--white' in sys.argv:
    print(' * with whitening')
    __WHITEN_FEATS__ = True
if '--1v1' in sys.argv:
    __MATCH_TYPE__ = '1v1'
if '--1vM' in sys.argv:
    __MATCH_TYPE__ = '1vM'
if '--BOW' in sys.argv:
    __MATCH_TYPE__ = 'BOW'

if '--cache-query' in sys.argv:
    __CACHE_QUERY__ = True
if '--nocache-query' in sys.argv:
    __CACHE_QUERY__ = False


if '--reverify' in sys.argv:
    __REVERIFY_QUERY__ = True

if '--resave-query' in sys.argv:
    __RESAVE_QUERY__ = False # 4H4X5

if '--print-checks' in sys.argv:
    helpers.__PRINT_CHECKS__ = True

# MAJOR HACKS 
#__FORCE_REQUERY_CX__ = set([0,1])
__FORCE_REQUERY_CX__ = set([])
