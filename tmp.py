x_extent_sqrd = (x_m.max() - x_m.min()) ** 2
y_extent_sqrd = (y_m.max() - y_m.min()) ** 2
diaglen_sqrd_sv2 = x_extent_sqrd + y_extent_sqrd
xy_thresh_sqrd_sv2 = diaglen_sqrd_sv2 * xy_thresh


    
img2_extent = (kpts2_m[0:2, :].max(1) - kpts2_m[0:2, :].min(1))[0:2]
diaglen_sqrd_sv1 = np.sum(np.array(img2_extent, dtype=np.float64)**2)
xy_thresh_sqrd_sv1 = diaglen_sqrd_sv1 * xy_thresh

print np.array([x2_extent, y2_extent]) == img2_extent
print diaglen_sqrd_sv2 == diaglen_sqrd_sv1
print xy_thresh_sqrd_sv1 == xy_thresh_sqrd_sv2

