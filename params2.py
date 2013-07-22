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
__FLANN_PARAMS__ = {'algorithm' :'kdtree',
                      'trees'     :4,
                      'checks'    :128}

