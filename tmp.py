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

xyz = np.vstack([xy1_ma, np.ones(len(xy1_ma[0]))])

xy1_ma = sv1.DBG
x1_ma, y1_ma = sv2.DBG

x_m, y_m = (x1_ma, y1_ma)

xy1_ma[1] == y1_ma
print('==========')
print(sv1.DBG)
print('----------')
print(sv2.DBG)
print('==========')
print(np.array(sv1.DBG) == np.array(sv2.DBG))
print('All=%r ' % all(np.array(sv1.DBG) == np.array(sv2.DBG)))


