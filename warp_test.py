#import matplotlib.pyplot as plt

# INPUT from match_chips2:
# cx2_fm, cx2_fs, qcx, cx2_nx

import drawing_functions2 as df2
import feature_compute2   as fc2
import match_chips2       as mc2
import hotspotter.tpl.cv2 as cv2
import load_data2
import cvransac2
import hotspotter.helpers
import imp
imp.reload(cvransac2)
imp.reload(df2)
imp.reload(fc2)
imp.reload(mc2)
imp.reload(hotspotter.helpers)
from hotspotter.helpers import *
from cvransac2 import H_homog_from_RANSAC, H_homog_from_DELSAC, H_homog_from_PCVSAC, H_homog_from_CV2SAC
from hotspotter.tpl.pyflann import FLANN
from numpy.linalg import inv

# TEST PARAMETERS

__VIEW_TOP__ = 10

__SHOW_PLAIN_CHIPS__ = False
__SHOW_KPTS_CHIPS__ = True
__SHOW_ASSIGNED_FEATURE_MATCHES__ = False
__SHOW_INLIER_MATCHES__ = False
__SHOW_WARP__ = False
__OTHER_X__ = 0
__xy_thresh_percent__ = mc2.__xy_thresh_percent__
__FEAT_TYPE__ = 'HESAFF'
__WARP_FEATURE_TYPE__ = 'SIFT'
__oldfeattype = None


# INITIALIZE 

qcx = 1

# reload data if hs was deleted
if not 'hs' in vars():
    print('hs is not in vars... reloading')
    hs = mc2.load_hotspotter(load_data2.MOTHERS)

cx2_cid  = hs.tables.cx2_cid
cx2_nx   = hs.tables.cx2_nx
nx2_name = hs.tables.nx2_name
cx2_rchip_path = hs.cpaths.cx2_rchip_path

# rerun query if feature type has changed
if __oldfeattype != __FEAT_TYPE__:
    print('The feature type is new or has changed')
    hs.feats.set_feat_type(__FEAT_TYPE__)
    cx2_kpts = hs.feats.cx2_kpts
    cx2_desc = hs.feats.cx2_desc
    flann_1vM = mc2.precompute_index_1vM(hs)
    cx2_fm, cx2_fs = mc2.assign_matches_1vM(qcx, cx2_cid, cx2_desc, flann_1vM)
    __oldfeattype = __FEAT_TYPE__

cx2_fm_SV, cx2_fs_SV = mc2.spatially_verify_1vX(qcx, cx2_kpts, cx2_fm, cx2_fs)

def cx2_other_cx(cx):
    nx = cx2_nx[cx]
    other_cx_, = np.where(cx2_nx == nx)
    other_cx  = other_cx_[other_cx_ != cx]
    return other_cx

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
# Get spatially verified initial 1vM scores
cx2_score  = np.array([np.sum(fs) for fs in cx2_fs_SV])
top_cx     = cx2_score.argsort()[::-1]
top_scores = cx2_score[top_cx] 
top_nx     = cx2_nx[top_cx]
view_top   = min(len(top_scores), __VIEW_TOP__)
print('---------------------------------------')
print('The ground truth scores are: ')
for cx in iter(other_cx):
    score = cx2_score[cx]
    print('--> cx=%4d, score=%6.2f' % (cx, score))
print('---------------------------------------')
print('The top %d chips and scores are: ' % view_top)
for topx in xrange(view_top):
    tscore = top_scores[topx]
    tcx    = top_cx[topx]
    tnx    = cx2_nx[tcx]
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
print('---------------------------------------')
print(' * Inspecting match to: cx = %d' % cx)
print('Drawing chips we are inspecting')
print('  * rchip1.shape = %r ' % (rchip1.shape,)) 
print('  * rchip2.shape = %r ' % (rchip2.shape,)) 
if __SHOW_PLAIN_CHIPS__:
    df2.imshow(rchip1, fignum=9001, title='querychip qcx=%d' % qcx)
    df2.imshow(rchip2, fignum=9002, title='reschip  cx=%d' %  cx)
if __SHOW_KPTS_CHIPS__:
    df2.show_keypoints(rchip1, kpts1, fignum=2, title='qcx=%d nkpts=%d' % (qcx,len(kpts1)))
    df2.show_keypoints(rchip2, kpts2, fignum=3, title='cx=%d nkpts=%d' % (cx,len(kpts1)))

# ASSIGNED MATCHES
print('---------------------------------------')
print(' Getting the assigned feature matches')
fm  = cx2_fm[cx]
fs  = cx2_fs[cx]
# ugg transpose, I like row first, but ransac seems not to
kpts1_m = kpts1[fm[:,0]].T
kpts2_m = kpts2[fm[:,1]].T
assert kpts1_m.shape[0] == 5 and kpts2_m.shape[0] == 5, 'needs ellipses'
print(' * feature matches (assigned) fm.shape = %r' % (fm.shape,))
print(' * kpts1.shape %r' % (kpts1.shape,))
print(' * kpts2.shape %r' % (kpts2.shape,))
print('---------------------------------------')
print('Drawing the assigned matches')
if __SHOW_ASSIGNED_FEATURE_MATCHES__:
    df2.show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm, fignum=4,
                 title='Assigned feature matches qcx=%d cx=%d ' % (qcx, cx))

# SPATIAL VERIFICATION PARAMS SETUP
img1_extent = (kpts1_m[0:2,:].max(1) - kpts1_m[0:2,:].min(1))[0:2]
img2_extent = (kpts2_m[0:2,:].max(1) - kpts2_m[0:2,:].min(1))[0:2]
xy_thresh12_sqrd = np.sum(img1_extent**2) * (__xy_thresh_percent__**2)
xy_thresh21_sqrd = np.sum(img2_extent**2) * (__xy_thresh_percent__**2)

__PRINT_THRESH_INFO__ = False
if __PRINT_THRESH_INFO__:
    print('---------------------------------------')
    print(' * Threshold is %.1f%% of diagonal length' % (__xy_thresh_percent__*100))
    print('Computing the xy_threshold:')
    print(' * img1_extent = %r ' % img1_extent)
    print(' * img2_extent = %r ' % img2_extent)
    print(' * img1_diag_len = %.2f ' % np.sqrt(np.sum(img1_extent**2)))
    print(' * img2_diag_len = %.2f ' % np.sqrt(np.sum(img2_extent**2)))
    print(' * xy_thresh12_sqrd=%.2f' % np.sqrt(xy_thresh12_sqrd))
    print(' * xy_thresh21_sqrd=%.2f' % np.sqrt(xy_thresh21_sqrd))
    print('---------------------------------------')


# SPATIAL VERIFICATION AND HOMOGRAPHY ESTIMATIONS
print('---------------------------------------')
print(' * Threshold is %.1f%% of diagonal length' % (__xy_thresh_percent__*100))
def __test_homog(func_homog, testname, fignum):
    with Timer(msg=testname+' SV12'):
        H12, inliers12 = func_homog(kpts1_m, kpts2_m, xy_thresh12_sqrd) 
    print((' * num inliers12 = %d '+testname) % inliers12.sum())
    fm_SV1 = fm[inliers12,:]
    if __SHOW_INLIER_MATCHES__:
        df2.show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm_SV1, fignum=fignum, title=testname+' SV1')
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
def __test_warp(homog_tup, testname, fignum):
    H1 = homog_tup[0]
    #rchip1_H1 = cv2.warpPerspective(rchip1,      H1, rchip2.shape[0:2][::-1])
    with Timer(msg='Testing warp: '+testname):
        rchip2_H1 = cv2.warpPerspective(rchip2, inv(H1), rchip1.shape[0:2][::-1])
    if __SHOW_WARP__:
        #df2.imshow(rchip1_H1, fignum=fignum+0, title=testname+' warped querychip1')
        df2.imshow(rchip2_H1, fignum=fignum+1, title=testname+' warped reschip2')
    return rchip2_H1

wrchip2_cv2sac = __test_warp(cv2sac_homog, 'cv2 RanSaC', fignum=12)
wrchip2_delsac = __test_warp(delsac_homog, ' my DElSaC', fignum=16)
wrchip2_ransac = __test_warp(ransac_homog, ' my RanSaC', fignum=20)
wrchip2_pcvsac = __test_warp(pcvsac_homog, 'pcv RanSaC', fignum=24)

print('---------------------------------------')
print('Redetecting features in warped chips')
qkpts1 = cx2_kpts[qcx]
qdesc1 = cx2_desc[qcx]

# CHOOSE output from one of previous algorithms
(H, inliers) = delsac_homog
fm_SV = fm[inliers]
fs_SV = fs[inliers]

wchip2 = wrchip2_delsac
#
with Timer(msg='detect features in warped chip'):
    wkpts2, wdesc2 = fc2.compute_features(wchip2, __WARP_FEATURE_TYPE__)
    # If we are using a different feature type compute one for the query image
    # too
    #if not hs.feats.feat_type == __WARP_FEATURE_TYPE__:
    _qkpts1, _qdesc1 = fc2.compute_features(rchip1, __WARP_FEATURE_TYPE__)
    #else: 
        #_qkpts1, _qdesc1 = qkpts1, qdesc1

# match keypoints 1v1 style
if __WARP_FEATURE_TYPE__ != 'FREAK':
    flann_1v1 = FLANN()
    flann_1v1.build_index(_qdesc1, **mc2.__FLANN_PARAMS__)
    wfm, wfs       = mc2.match_1v1(_qdesc1, wdesc2, flann_1v1)
    wfm_SV, wfs_SV = mc2.spatially_verify(_qkpts1, wkpts2, wfm, wfs)
else:
    raise NotImplemented('!!!')

# show warped with keypoints
print('-----------------')
print('Show warped with keypoints')
df2.show_keypoints(rchip1, _qkpts1, fignum=300, title='nWarpedKpts=%r ' % len(_qkpts1))
df2.show_keypoints(wchip2, wkpts2, fignum=30, title='nWarpedKpts=%r ' % len(wkpts2))

# show warped matches
print('-----------------')
print('Show warped matches')
df2.show_matches2(rchip1, rchip2, qkpts1,  kpts2,  fm,    fignum=26, title='nMatch=%d' % len(fm))
df2.show_matches2(rchip1, rchip2, qkpts1,  kpts2,  fm_SV, fignum=27, title='nMatch_SV=%d' % len(fm_SV))
df2.show_matches2(rchip1, wchip2, _qkpts1,  kpts2,  wfm, fignum=28, title='nMatchWarped=%d' % len(wfm))
df2.show_matches2(rchip1, wchip2, _qkpts1,  kpts2,  wfm_SV, fignum=29, title='nMatchWarped_SV=%d' % len(wfm_SV))

# inspect scores
print('-----------------')
print('Inspect scores')
print('Assigned matching score  : '+str(fs.sum()))
print('Verified matching score  : '+str(fs_SV.sum()))
print('Warped matching score    : '+str(wfs.sum()))
print('Warped and Verified score: '+str(wfs_SV.sum()))

# FREAK 
df2.present()
