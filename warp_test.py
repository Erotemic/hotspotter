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

#TODO: 
'''
CURRENTLY YOU ARE CLEANING THE MESS WHICH IS THE BOTTOM OF THIS SCRIPT
--- Looking to see if cv2.warpPerspective has a replicate boundary option
--- Looking to see if cv2.getAffine works 

* Fix warping issue (warp entire image)
* Local area descriptor search ( from warped image ) [CHEAT]
* Distinctiveness measure - Ratio test restricted to region
* Descript matching to coverage measure

Proposal for similarity / distinctiveness measure: 
    Algorithms + Analysis 
    ways to integrate them across the entire image in order to make a final same-zebra-liklihood 
    ways to integrate them across the entire image in order to make a final similarity score 

    Given inital transform
    Use learning techniques. 
    Make as non-huristic as possible

    We are familiar enough with the problem to do thought experiments
'''


# TEST PARAMETERS
qcx = 0

__VIEW_TOP__ = 10
__SHOW_PLAIN_CHIPS__ = True
__SHOW_KPTS_CHIPS__ = True
__SHOW_ASSIGNED_FEATURE_MATCHES__ = False
__SHOW_INLIER_MATCHES__ = False
__SHOW_WARP__ = False
__OTHER_X__ = 2
__WARP_FEATURE_TYPE__ = 'HESAFF'
__oldfeattype = None
mc2.__xy_thresh_percent__ = mc2.__xy_thresh_percent__
mc2.__FEAT_TYPE__ = mc2.__FEAT_TYPE__



# INITIALIZE 
print('----------------------')
print('Initializing warp test')
# reload data if hs was deleted
if not 'hs' in vars():
    print('hs is not in vars... reloading')
    hs = mc2.load_hotspotter(load_data2.MOTHERS)
cx2_cid  = hs.tables.cx2_cid
cx2_nx   = hs.tables.cx2_nx
nx2_name = hs.tables.nx2_name
cx2_rchip_path = hs.cpaths.cx2_rchip_path
# rerun query if feature type has changed
if __oldfeattype != mc2.__FEAT_TYPE__:
    print('The feature type is new or has changed')
    hs.feats.set_feat_type(mc2.__FEAT_TYPE__)
    cx2_kpts = hs.feats.cx2_kpts
    cx2_desc = hs.feats.cx2_desc
    flann_1vM = mc2.precompute_index_1vM(hs)
    cx2_fm, cx2_fs = mc2.assign_matches_1vM(qcx, cx2_cid, cx2_desc, flann_1vM)
    __oldfeattype = mc2.__FEAT_TYPE__
cx2_fm_V, cx2_fs_V = mc2.spatially_verify_1vX(qcx, cx2_kpts, cx2_fm, cx2_fs)
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
def print_top_scores(_cx2_fs, lbl):
    cx2_score  = np.array([np.sum(fs) for fs in _cx2_fs])
    top_cx     = cx2_score.argsort()[::-1]
    top_scores = cx2_score[top_cx] 
    top_nx     = cx2_nx[top_cx]
    view_top   = min(len(top_scores), __VIEW_TOP__)
    print('---------------------------------------')
    print('The ground truth scores '+lbl+' are: ')
    for cx in iter(other_cx):
        score = cx2_score[cx]
        print('--> cx=%4d, score=%6.2f' % (cx, score))
    print('---------------------------------------')
    print(('The top %d chips and scores '+lbl+' are: ') % view_top)
    for topx in xrange(view_top):
        tscore = top_scores[topx]
        tcx    = top_cx[topx]
        tnx    = cx2_nx[tcx]
        _mark = '-->' if tnx == qnx else '  -'
        print(_mark+' cx=%4d, score=%6.2f' % (tcx, tscore))
    print('---------------------------------------')
    print('---------------------------------------')
print_top_scores(cx2_fs,   '(assigned)')
print_top_scores(cx2_fs_V, '(assigned+V)')

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
    df2.show_keypoints(rchip1, kpts1, fignum=202, title='qcx=%d nkpts=%d' % (qcx,len(kpts1)))
    df2.show_keypoints(rchip2, kpts2, fignum=203, title='cx=%d nkpts=%d' % (cx,len(kpts1)))



# ASSIGNED MATCHES
print('---------------------------------------')
print(' INFO: Assigned feature matches        ')
fm      = cx2_fm[cx]
fs      = cx2_fs[cx]
print(' * feature matches (assigned) fm.shape = %r' % (fm.shape,))
print(' * kpts1.shape %r' % (kpts1.shape,))
print(' * kpts2.shape %r' % (kpts2.shape,))
print('---------------------------------------')
if __SHOW_ASSIGNED_FEATURE_MATCHES__:
    print('Drawing the assigned matches')
    title_str = 'qcx=%d cx=%d nMatches=%d' % (qcx, cx, len(fm))
    print(title_str)
    df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm, fs, 4, title_str)
print('---------------------------------------')



# SPATIAL VERIFICATION AND HOMOGRAPHY ESTIMATIONS
print('---------------------------------------')
def __test_homog(func_homog, testname, fignum):
    with Timer(msg=testname+' V'):
        fm_V, fs_V, H = mc2.__spatially_verify(func_homog, kpts1, kpts2, fm, fs, DBG=None)
    title_str = ('nInliers=%d '+testname+' V__') % len(fm_V)
    print('L____'+title_str+'\n')
    if __SHOW_INLIER_MATCHES__:
        df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm_V, fs_V, fignum, title_str)
    return (fm_V, fs_V, H)
print('Testing different ways to calculate homography')
pcvsac_Vtup = __test_homog(H_homog_from_PCVSAC, 'PCVSaC',  5)
ransac_Vtup = __test_homog(H_homog_from_RANSAC, 'RanSaC',  7)
delsac_Vtup = __test_homog(H_homog_from_DELSAC, 'DElSaC',  9)
cv2sac_Vtup = __test_homog(H_homog_from_CV2SAC, 'CV2SaC', 11)
print('---------------------------------------')


# WARP IMAGES
print('---------------------------------------')
print('Warping Images')
def __test_warp(Vtup, testname, fignum):
    H_21 = Vtup[2]
    with Timer(msg='Warped with H from '+testname):
        rchip2W = mc2.warp_chip(rchip2, H_21, rchip1) 
    if __SHOW_WARP__:
        title_str = testname+' Result rchip2W'
        print(' * Showing: '+str(title_str))
        df2.imshow(rchip2W, fignum=fignum+1, title=title_str)
    return rchip2W
#cv2sac_rchip2W = __test_warp(cv2sac_Vtup, 'PCVSaC', 12)
#ransac_rchip2W = __test_warp(ransac_Vtup, 'RanSaC', 20)
delsac_rchip2W = __test_warp(delsac_Vtup, 'DElSaC', 16)
#pcvsac_rchip2W = __test_warp(pcvsac_Vtup, 'CV2SaC', 24)


# PARAM CHOICE: output from one of previous algorithms
(fm_V, fs_V, H) = delsac_Vtup
rchip2W         = delsac_rchip2W


# REDETECT in warped chips
print('---------------------------------------')
print('Redetecting features in warped chips')
kpts1_1vM = cx2_kpts[qcx]
desc1_1vM = cx2_desc[qcx]

kpts2_1vM = cx2_kpts[cx]
desc2_1vM = cx2_desc[cx]
with Timer(msg='detect features in warped rchip2W'):
    kpts2W, desc2W = fc2.compute_features(rchip2W,  __WARP_FEATURE_TYPE__)

# Recomputed but not actually warped (sanity check)
with Timer(msg='(sanity check) detect in non-warped rchip1'):
    kpts1, desc1 = fc2.compute_features(rchip1, __WARP_FEATURE_TYPE__)
with Timer(msg='(sanity check) detect in non-warped rchip2'):
    kpts2, desc2 = fc2.compute_features(rchip2, __WARP_FEATURE_TYPE__)

# REMATCH warped keypoints 1v1 style
print('---------------------------------------')
print('Rematching warped keypoints 1v1 style  ')
if __WARP_FEATURE_TYPE__ == 'FREAK':
    raise NotImplemented('!!!')
flann_1v1 = FLANN()
flann_1v1.build_index(desc1, **mc2.__FLANN_PARAMS__)
#desc2 =  desc2W
fm_W,  fs_W  = mc2.match_1v1(desc2W, flann_1v1)
fm_WV, fs_WV, _ = mc2.spatially_verify(kpts1, kpts2W, fm_W, fs_W)


# INVESTIGATE: See what 1v1 scores look like next to 1vM
print('---------------------------------------')
print('Investigating 1v1 compared to 1v1 warping')
# 1v1: Assign, Verify, Warp, Redetect, Reassign, Reverify


def one_vs_one_pipeline(rchip1, rchip2, kpts1, kpts2):
    _flann = FLANN()
    _flann.build_index(desc1_1vM, **mc2.__FLANN_PARAMS__)
    # Match 1v1 + Spatial Verifiacation
    fm,   fs       = mc2.match_1v1(desc2, _flann)
    fm_V, fs_V,  H = mc2.spatially_verify(kpts1, kpts2, fm, fs)
    # Warp using H from Matching
    rchip2_W          = mc2.warp_chip(rchip2, H, rchip1)
    # 
    kpts2_W, desc2_W  = fc2.compute_features( rchip21v1W, __WARP_FEATURE_TYPE__)
    fm_W,  fs_W       = mc2.match_1v1(desc2_W,  _flann)
    fm_WV, fs_WV, H_W = mc2.spatially_verify(kpts1, kpts2_W, fm_W, fs_W)

with Timer(msg='1v1 assign, verify, warp, redetect, reassign, reverify'):

print('---------------------------------------')

# Arguments to show matches
_chip_kpts1vM  = (rchip1, rchip2,  kpts1_1vM, kpts2_1vM)
_chip_kptsW    = (rchip1, rchip2W, kpts1,     kpts2W)
_chip_kpts1v1  = (rchip1, rchip2,  kpts1,     kpts2)
_chip_kpts1v1W = (rchip1, rchip21v1W, kpts1,  kpts21v1W)

# SHOW warped with keypoints
print('-----------------')
print('1vM - show matches')
df2.show_matches2(*( _chip_kpts1vM  + (fm,   fs,    200, '1vM nMatch=%d'   % len(fm))    ))
df2.show_matches2(*( _chip_kpts1vM  + (fm_V, fs_V,  201, '1vM nMatchV=%d'  % len(fm_V))  ))
df2.show_keypoints(rchip1, kpts1_1vM, 202, 'Query  nkpts=%d' % (len(kpts1_1vM)))
df2.show_keypoints(rchip2, kpts2_1vM, 203, 'Reults nkpts=%d' % (len(kpts2_1vM)))

print('-----------------')
print('1vM + Warped - showing keypoints and matches')
df2.show_matches2(*( _chip_kptsW + (fm_W,  fs_W,  300, 'H-from-1vM nMatchW=%d'  % len(fm_W))  ))
df2.show_matches2(*( _chip_kptsW + (fm_WV, fs_WV, 301, 'H-from-1vM nMatchWV=%d' % len(fm_WV)) ))
df2.show_keypoints(rchip1 , kpts1,  302, 'Query  H-from-1vM nKptsW=%r '   % len(kpts1))
df2.show_keypoints(rchip2W, kpts2W, 303, 'Result H-from-1vM nKptsW=%r '   % len( kpts2W))


print('-----------------')
print('1v1 - showing keypoints and matches')
df2.show_matches2(*( _chip_kpts1v1 + (fm_1v1,  fs_1v1,  400, '1v1 nMatch=%d'   % len(fm_1v1))    ))
df2.show_matches2(*( _chip_kpts1v1 + (fm_1v1V, fs_1v1V, 401, '1v1 nMatchV=%d'  % len(fm_1v1V))  ))
df2.show_keypoints(rchip1, kpts1, 402, 'Query  1v1 nKpts=%r ' % len( kpts1))
df2.show_keypoints(rchip2, kpts2, 403, 'Result 1v1 nKpts=%r ' % len( kpts2))

print('-----------------')
print('1v1 + Warped - showing keypoints and matches')
df2.show_matches2(*( _chip_kpts1v1W + (fm_1v1W,  fs_1v1W,  500, 'H-from-1v1 nMatch=%d'   % len(fm_1v1W))    ))
df2.show_matches2(*( _chip_kpts1v1W + (fm_1v1WV, fs_1v1WV, 501, 'H-from-1v1 nMatchV=%d'  % len(fm_1v1WV))  ))
df2.show_keypoints(rchip1,      kpts1, 502,  'Query  H-from-1v1 nKptsW=%r ' % len( kpts1))
df2.show_keypoints(rchip21v1W, kpts21v1W, 503, 'Result H-from-1v1 nKptsW=%r ' % len( kpts21v1W))




# INFO warped / verified scores
print('-----------------')
print('INFO: warped / verified scores')
print('--')
print('Assigned 1vM matching score  : '+str(fs.sum()))
print('Verified 1vM matching score  : '+str(fs_V.sum()))
print('--')
print('Assigned 1v1 matching score  : '+str(fs_1v1.sum()))
print('Verified 1v1 matching score  : '+str(fs_1v1V.sum()))
print('--')
print('Warped matching score    : '+str(fs_W.sum()))
print('Warped and Verified score: '+str(fs_WV.sum()))



# FREAK 
df2.present()
