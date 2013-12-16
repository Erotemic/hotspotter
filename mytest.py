import spatial_verification2 as sv2
sv2.rrr()
query_cfg.update_cfg(sv_on=True,
                 K=5,
                 use_chip_extent=True, 
                 xy_thresh=.002,
                 just_affine=False, 
                 scale_thresh=(.5, 2))
