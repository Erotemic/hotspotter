import spatial_verification2 as sv2
sv2.rrr()
q_cfg.update_cfg(sv_on=True, K=5, use_chip_extent=False, xy_thresh=.02,
                 just_affine=True, scale_thresh=(.5, 1.5))
