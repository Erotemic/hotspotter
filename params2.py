import os.path


# Number of processessors
__NUM_PROCS__ = 9

# Feature type to use
__FEAT_TYPE__    = 'HESAFF'
# Number of matches for one-vs-many
__K__            = 2
# Thresholds for one-vs-one
__RATIO_THRESH__ = 1.5
# Number of top matches to spatially re-rank
__NUM_RERANK__   = 50
# Percentage of the diagonal length of keypoint extent
__XY_THRESH__    = .05
hs1_params = {'algorithm' :'kdtree',
              'trees'     :4,
              'checks'    :128}

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
__FLANN_ONCE_PARAMS__ = mother_hesaff_tuned_params
__FLANN_PARAMS__      = hs1_params

__LAZY_MATCHING__ = True

__WHITEN_FEATS__ = False

__HISTEQ__ = False
