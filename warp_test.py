#import drawing_functions2 as df2
#import matplotlib.pyplot as plt


# INPUT from match_chips2:
# cx2_fm, cx2_fs, qcx, cx2_nx

import cvransac2
import drawing_functions2
import hotspotter.helpers
import imp
imp.reload(cvransac2)
imp.reload(drawing_functions2)
imp.reload(hotspotter.helpers)
from hotspotter.helpers import *
from cvransac2 import H_homog_from_RANSAC, H_homog_from_DELSAC, H_homog_from_PCVSAC, H_homog_from_CV2SAC

# TEST PARAMETERS
__VIEW_TOP__ = 10

__SHOW_PLAIN_CHIPS__ = True
__SHOW_KPTS_CHIPS__ = True
__SHOW_ASSIGNED_FEATURE_MATCHES__ = True
__SHOW_INLIER_MATCHES__ = True
__OTHER_X__ = 0
__xy_thresh_percent__ = .10

def cx2_other_cx(cx):
    nx = cx2_nx[cx]
    other_cx_, = np.where(cx2_nx == nx)
    other_cx  = other_cx_[other_cx_ != cx]
    return other_cx

# INITIALIZATION DEPENDS ON RUNNING MATCH_CHIPS2 FIRST
df2.reset()

# VIEW TOP SCORES AND GROUND TRUTH SCORES
print('\n\n=============================')
print(' INSPECTING QCX MATCHES      ')
print('=============================')
qnx = cx2_nx[qcx]
other_cx = cx2_other_cx(qcx)
print('Inspecting matches of qcx=%d name=%s' % (qcx, nx2_name[qnx]))
print(' * Matched against %d other chips' % len(cx2_fm))
print(' * Ground truth chip indexes:\n   other_cx=%r' % other_cx)
cx2_score = np.array([np.sum(fs) for fs in cx2_fs])
top_cx = cx2_score.argsort()[::-1]
top_scores = cx2_score[top_cx] 
top_nx = cx2_nx[top_cx]
view_top = min(len(top_scores), __VIEW_TOP__)
print('---------------------------------------')
print('The ground truth scores are: ')
for cx in iter(other_cx):
    score = cx2_score[cx]
    print('--> cx=%4d, score=%6.2f' % (cx, score))
print('---------------------------------------')
print('The top %d chips and scores are: ' % view_top)
for topx in xrange(view_top):
    tscore = top_scores[topx]
    tcx = top_cx[topx]
    tnx = cx2_nx[tcx]
    _mark = '-->' if tnx == qnx else '  -'
    print(_mark+' cx=%4d, score=%6.2f' % (tcx, tscore))
print('---------------------------------------')
print('---------------------------------------')

# INPUT CHIPS
cx = other_cx[__OTHER_X__]
kpts1  = cx2_kpts[qcx]
kpts2  = cx2_kpts[cx]
rchip1 = cv2.imread(cx2_rchip_path[qcx])
rchip2 = cv2.imread(cx2_rchip_path[cx])
rchipkpts1 = draw_kpts(rchip1, kpts1)
rchipkpts2 = draw_kpts(rchip2, kpts2)
print('---------------------------------------')
print(' * Inspecting match to: cx = %d' % cx)
print('Drawing chips we are inspecting')
print('  * rchip1.shape = %r ' % (rchip1.shape,)) 
print('  * rchip2.shape = %r ' % (rchip2.shape,)) 
if __SHOW_PLAIN_CHIPS__:
    df2.imshow(rchip1, fignum=9001, title='querychip qcx=%d' % qcx)
    df2.imshow(rchip2, fignum=9002, title='reschip  cx=%d' %  cx)
if __SHOW_KPTS_CHIPS__:
    df2.imshow(rchipkpts1, fignum=2, title='With Keypoints qcx=%d' % qcx)
    df2.imshow(rchipkpts2, fignum=3, title='With Keypoints  cx=%d' %  cx)

# ASSIGNED MATCHES
print('---------------------------------------')
print(' Getting the assigned feature matches')
fm12  = cx2_fm[cx]
# ugg transpose, I like row first, but ransac seems not to
kpts1_m = kpts1[fm12[:,0]].T
kpts2_m = kpts2[fm12[:,1]].T
assert kpts1_m.shape[0] == 5 and kpts2_m.shape[0] == 5, 'needs ellipses'
print(' * feature matches (assigned) fm12.shape = %r' % (fm12.shape,))
print(' * kpts1.shape %r' % (kpts1.shape,))
print(' * kpts2.shape %r' % (kpts2.shape,))
print('---------------------------------------')
print('Drawing the assigned matches')
if __SHOW_ASSIGNED_FEATURE_MATCHES__:
    df2.show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12, fignum=4,
                 title='Assigned feature matches qcx=%d cx=%d ' % (qcx, cx))

# SPATIAL VERIFICATION PARAMS SETUP
img1_extent = (kpts1_m[0:2,:].max(1) - kpts1_m[0:2,:].min(1))[0:2]
img2_extent = (kpts2_m[0:2,:].max(1) - kpts2_m[0:2,:].min(1))[0:2]
xy_thresh12_sqrd = np.sum(img1_extent**2) * (__xy_thresh_percent__**2)
xy_thresh21_sqrd = np.sum(img2_extent**2) * (__xy_thresh_percent__**2)

__PRINT_THRESH_INFO__ = False
if __PRINT_THRESH_INFO__:
    print('---------------------------------------')
    print('Computing the xy_threshold:')
    print(' * Threshold is %.1f%% of diagonal length' % (__xy_thresh_percent__*100))
    print(' * img1_extent = %r ' % img1_extent)
    print(' * img2_extent = %r ' % img2_extent)
    print(' * img1_diag_len = %.2f ' % np.sqrt(np.sum(img1_extent**2)))
    print(' * img2_diag_len = %.2f ' % np.sqrt(np.sum(img2_extent**2)))
    print(' * xy_thresh12_sqrd=%.2f' % np.sqrt(xy_thresh12_sqrd))
    print(' * xy_thresh21_sqrd=%.2f' % np.sqrt(xy_thresh21_sqrd))
    print('---------------------------------------')


# HOMOGRAPHY ESTIMATIONS
print('---------------------------------------')
def __test_homog(func_homog, testname, fignum):
    print('--------------------------')
    print('Testing homog: '+testname)
    with Timer(msg=testname+' SV12'):
        H12, inliers12 = func_homog(kpts1_m, kpts2_m, xy_thresh12_sqrd) 
    print(' * num inliers12 = %d' % inliers12.sum())
    fm1_SV1 = fm12[inliers12,:]
    if __SHOW_INLIER_MATCHES__:
        df2.show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm1_SV1, fignum=fignum, title=testname+' SV1')
    return (H12, inliers12)
print('Estimating homography')
pcvsac_homog = __test_homog(H_homog_from_PCVSAC, 'pcv RanSaC',  5)
ransac_homog = __test_homog(H_homog_from_RANSAC, ' my RanSaC',  7)
delsac_homog = __test_homog(H_homog_from_DELSAC, ' my DElSaC',  9)
cv2sac_homog = __test_homog(H_homog_from_CV2SAC, 'cv2 RanSaC', 11)
print('---------------------------------------')


# WARP IMAGES
print('---------------------------------------')
print('Warping Images')
from numpy.linalg import inv
def __test_warp(homog_tup, testname, fignum):
    H1 = homog_tup[0]
    rchip1_H1 = cv2.warpPerspective(rchip1,      H1, rchip2.shape[0:2][::-1])
    rchip2_H1 = cv2.warpPerspective(rchip2, inv(H1), rchip1.shape[0:2][::-1])

    df2.imshow(rchip1_H1, fignum=fignum+0, title=testname+' warped querychip1')
    df2.imshow(rchip2_H1, fignum=fignum+1, title=testname+' warped reschip2')

__test_warp(cv2sac_homog, 'cv2 RanSaC', fignum=12)
__test_warp(delsac_homog, ' my DElSaC', fignum=16)
__test_warp(ransac_homog, ' my RanSaC', fignum=20)
__test_warp(pcvsac_homog, 'pcv RanSaC', fignum=24)


#rchip1_H_12 = cv2.warpPerspective(rchip1, H, rchip2.shape[0:2][::-1])
#rchip1_H_21 = cv2.warpPerspective(rchip1, inv(H), rchip2.shape[0:2][::-1])

#df2.imshow(rchip1_H_12, fignum=5, title='chip1_H_12')
#df2.imshow(rchip1_H_21, fignum=6, title='chip1_H_21')
#df2.imshow(rchip2_H_12, fignum=7, title='chip2_H_12')
#df2.imshow(rchip2_H_21, fignum=8, title='chip2_H_21')

df2.present()
