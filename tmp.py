sv1.reload_module()
sv2.reload_module()
kpts1_m = kpts1[fm[:, 0], :].T
kpts2_m = kpts2[fm[:, 1], :].T
with helpers.Timer('sv1') as t: 
    hinlier_tup1 = sv1.H_homog_from_DELSAC(kpts1_m, kpts2_m,
                                            xy_thresh, 
                                            scale_thresh_high,
                                            scale_thresh_low)
with helpers.Timer('sv2') as t: 
    hinlier_tup2 = sv2.homography_inliers(kpts1, kpts2, fm,
                                            xy_thresh, 
                                            scale_thresh_high,
                                            scale_thresh_low)

print('==========')
print(sv1.__DBG1__[0])
print('----------')
print(sv2.__DBG2__[0])
print('==========')
print(np.array(sv2.__DBG2__) == np.array(sv1.__DBG1__))
print('All=%r ' % all(np.array(sv2.__DBG2__) == np.array(sv1.__DBG1__)))

