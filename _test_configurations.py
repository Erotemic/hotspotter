vsmany_2 = {
    'query_type'     : ['vsmany'],
    'checks'         : [1024],#, 8192],
    'K'              : [5], #5, 10],
    'Knorm'          : [1], #2, 3],
    'Krecip'         : [0], #, 5, 10],
    'roidist_weight' : [0], # 1,]
    'recip_weight'   : [0], # 1,] 
    'bursty_weight'  : [0], # 1,]
    'ratio_weight'   : [0,1], # 1,]
    'lnbnn_weight'   : [0,1], # 1,]
    'lnrat_weight'   : [0,1], # 1,]
    'roidist_thresh' : [None], # .5,] 
    'recip_thresh'   : [0], # 0
    'bursty_thresh'  : [None], #
    'ratio_thresh'   : [None], # 1.2, 1.6
    'lnbnn_thresh'   : [None], # 
    'lnrat_thresh'   : [None], #
    'nShortlist'   : [1000],
    'sv_on'        : [True], #True, False],
    'score_method' : ['csum'],#, 'pl'], #, 'nsum', 'borda', 'topk', 'nunique']
    'max_alts'     : [1000],
}

vsmany_3456 = {
    'query_type'     : ['vsmany'],
    'checks'         : [256, 1024],#, 8192],
    'K'              : [5, 10, 30], #5, 10],
    'Knorm'          : [1, 3, 5], #2, 3],
    'Krecip'         : [0, 1, 5, 10], #, 5, 10],
    'roidist_weight' : [0], # 1,]
    'recip_weight'   : [0], # 1,] 
    'bursty_weight'  : [0], # 1,]
    'ratio_weight'   : [0], # 1,]
    'lnbnn_weight'   : [0,1], # 1,]
    'lnrat_weight'   : [0,1], # 1,]
    'roidist_thresh' : [None, .5], # .5,] 
    'recip_thresh'   : [0], # 0
    'bursty_thresh'  : [None], #
    'ratio_thresh'   : [None], # 1.2, 1.6
    'lnbnn_thresh'   : [None], # 
    'lnrat_thresh'   : [None], #
    'nShortlist'   : [500],
    'sv_on'        : [True], #True, False],
    'score_method' : ['pl', 'plw', 'csum'],#, 'pl'], #, 'nsum', 'borda', 'topk', 'nunique']
    'max_alts'     : [200, 600],
}

vsone_1 = {
    'query_type'     : ['vsone'],
    'checks'         : [256],#, 8192],
    'K'              : [1], #5, 10],
    'Knorm'          : [1], #2, 3],
    'Krecip'         : [0], #, 5, 10],
    'roidist_weight' : [0], # 1,]
    'recip_weight'   : [0], # 1,] 
    'bursty_weight'  : [0], # 1,]
    'ratio_weight'   : [1], # 1,]
    'lnbnn_weight'   : [0], # 1,]
    'lnrat_weight'   : [0], # 1,]
    'roidist_thresh' : [None], # .5,] 
    'recip_thresh'   : [0], # 0
    'bursty_thresh'  : [None], #
    'ratio_thresh'   : [1.5], # 1.2, 1.6
    'lnbnn_thresh'   : [None], # 
    'lnrat_thresh'   : [None], #
    'nShortlist'   : [1000],
    'sv_on'        : [True], #True, False],
    'score_method' : ['csum'],#, 'pl'], #, 'nsum', 'borda', 'topk', 'nunique']
    'max_alts'     : [500],
}

vsone_std = {
    'query_type'     : 'vsone',
    'checks'         : 256,
    'K'              : 1, 
    'Knorm'          : 1, 
    'Krecip'         : 0, 
    'ratio_weight'   : 1, 
    'lnbnn_weight'   : 0, 
    'ratio_thresh'   : 1.5, 
}
