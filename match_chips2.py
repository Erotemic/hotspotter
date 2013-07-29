import drawing_functions2 as df2
import chip_compute2, feature_compute2, load_data2
import report_results2
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from hotspotter.Parallelize import parallel_compute
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.helpers import Timer, get_exec_src, check_path, tic, toc, myprint
from drawing_functions2 import draw_matches, draw_kpts, tile_all_figures
from hotspotter.tpl.pyflann import FLANN
from itertools import chain
from numpy import linalg
from cvransac2 import H_homog_from_RANSAC, H_homog_from_DELSAC, H_homog_from_PCVSAC, H_homog_from_CV2SAC
import params2
import algos

import imp
imp.reload(cvransac2)
imp.reload(algos)

__K__ = params2.__K__
__NUM_RERANK__   = params2.__NUM_RERANK__
__RATIO_THRESH__ = params2.__RATIO_THRESH__
__FLANN_PARAMS__ = params2.__FLANN_PARAMS__
__FEAT_TYPE__    = params2.__FEAT_TYPE__ 
__XY_THRESH__    = params2.__XY_THRESH__ = .05
__FEAT_TYPE__    = 'FREAK'


def printDBG(msg):
    pass

def runall_match(hs):
    #functools.partial
    hs.printme2()
    #cx2_res_1vM = __run_matching(hs, Matcher(hs, '1vM'))
    cx2_res_1v1 = __run_matching(hs, Matcher(hs, '1v1'))


class Matcher(object):
    '''Wrapper class: assigns matches based on matching and feature prefs'''
    def __init__(self, hs, match_type='1vM'):
        feat_type  = hs.feats.feat_type
        self.feat_type  = feat_type
        self.match_type = match_type
        self.__flann_1v1      = None
        self.__assign_matches = None
        # Curry the correct functions
        if   match_type == '1vM' and feat_type == 'FREAK':
            self.__assign_matches = assign_matches_1vM_FREAK
        elif match_type == '1v1' and feat_type == 'FREAK':
            self.__assign_matches = assign_matches_1v1_FREAK
        elif match_type == '1vM':
            self.__flann_1vM = precompute_index_1vM(hs)
            self.__assign_matches = self.__assign_matches_1vM
        elif match_type == '1v1':
            self.__assign_matches = assign_matches_1v1
        else:
           raise Exception('Unknown match_type: '+repr(match_type))
    # Function which calls the correct matcher
    def assign_matches(self, qcx, cx2_cid, cx2_desc):
        return self.__assign_matches(qcx, cx2_cid, cx2_desc)
    # Helper functions
    def __assign_matches_1vM(self, qcx, cx2_cid, cx2_desc):
        return assign_matches_1vM(qcx, cx2_cid, cx2_desc, self.__flann_1vM)

def assign_matches_1vM_FREAK(qcx, cx2_cid, cx2_desc):
    return None

cv2_matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
def match_1v1_FREAK(desc1, desc2):
    raw_matches = cv2_matcher.knnMatch(desc1, desc2, 1)
    fm = np.array([(m1[0].queryIdx-1, m1[0].trainIdx-1) for m1 in iter(raw_matches)])
    #for (qx, tx) in iter(fm):
        #if tx >= len(desc2) or qx >= len(desc1):
            #print(repr(tx))
            #print(repr(qx))
            #print(repr(len(desc1)))
            #print(repr(len(desc2)))
            #raise Exception('Error')

    fs = np.array([1/m1[0].distance for m1 in iter(raw_matches)])
    return fm, fs

def assign_matches_1v1_FREAK(qcx, cx2_cid, cx2_desc):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    desc1 = cx2_desc[qcx]
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, desc2 in enumerate(cx2_desc):
        sys.stdout.write('.'); sys.stdout.flush()
        if cx == qcx: continue
        fm, fs = match_1v1_FREAK(desc1, desc2)
        #print('fm.shape = %r ' % (repr(fm.shape),))
        #print('fm.max = %r ' % (repr(fm.max(0)),))
        #print('fm.min = %r ' % (repr(fm.min(0)),))
        #print('fs.shape = %r ' % (repr(fs.shape),))
        #print('desc1.shape = %r ' % (repr(desc1.shape),))
        #print('desc2.shape = %r ' % (repr(desc2.shape),))
        cx2_fm[cx] = fm
        cx2_fs[cx] = fs
    sys.stdout.write('DONE')
    return cx2_fm, cx2_fs

def assign_matches_1v1(qcx, cx2_cid, cx2_desc):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    desc1 = cx2_desc[qcx]
    flann_1v1 = FLANN()
    flann_1v1.build_index(desc1, **__FLANN_PARAMS__)
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, desc2 in enumerate(cx2_desc):
        sys.stdout.write('.')
        sys.stdout.flush()
        if cx == qcx: continue
        (fm, fs) = match_1v1(desc2, flann_1v1)
        cx2_fm[cx] = fm
        cx2_fs[cx] = fs
    sys.stdout.write('DONE')
    flann_1v1.delete_index()
    return cx2_fm, cx2_fs

def cv2_match(desc1, desc2):
    K = 1
    cv2_matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    raw_matches = cv2_matcher.knnMatch(desc1, desc2, K)
    matches = [(m1.trainIdx, m1.queryIdx) for m1 in raw_matches]

def aggregate_descriptors_1vM(hs):
    cx2_cid   = hs.tables.cx2_cid
    cx2_desc  = hs.feats.cx2_desc
    feat_type = hs.feats.feat_type
    print('Aggregating descriptors for one vs many')
    cx2_nFeats = [len(k) for k in cx2_desc]
    _ax2_cx = [[cx_]*nFeats for (cx_, nFeats) in iter(zip(range(len(cx2_cid)), cx2_nFeats))]
    _ax2_fx = [range(nFeats) for nFeats in iter(cx2_nFeats)]
    ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    ax2_desc   = np.vstack(cx2_desc)
    return ax2_cx, ax2_fx, ax2_desc

#@profile
def precompute_index_1vM(hs):
    # Build (or reload) one vs many flann index
    feat_dir  = hs.dirs.feat_dir
    ax2_cx, ax2_fx, ax2_desc = aggregate_descriptors_1vM(hs)
    flann_1vM = FLANN()
    flann_1vM_path = feat_dir + '/flann_1vM_'+feat_type+'.index'
    load_success = False
    if check_path(flann_1vM_path):
        try:
            print('Attempting to load flann index')
            flann_1vM.load_index(flann_1vM_path, ax2_desc)
            print('...successfully loaded flann index')
            load_success = True
        except Exception as ex:
            print('Cannot load FLANN index'+repr(ex))
    if not load_success:
        with Timer(msg='rebuilding FLANN index'):
            flann_1vM.build_index(ax2_desc, **__FLANN_PARAMS__)
            flann_1vM.save_index(flann_1vM_path)
    # Keep relevant data in the flann object. 
    # as to prevent them from loosing scope
    flann_1vM.ax2_desc = ax2_desc 
    flann_1vM.ax2_cx   = ax2_cx 
    flann_1vM.ax2_fx   = ax2_fx 
    return flann_1vM

class HotspotterQueryResult(DynStruct):
    def __init__(self):
        super(HotspotterQueryResult, self).__init__()
        self.qcx    = -1
        self.cx2_fm = []
        self.cx2_fs = []
        self.cx2_fm_V = []
        self.cx2_fs_V = []

# Work function that is the basic matching pipeline. 
# specific functions need to be bound: 
# 
# fn_assign - assign feature matches
def __run_matching(hs, matcher):
    cx2_cid  = hs.tables.cx2_cid
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts 
    cx2_res = [HotspotterQueryResult() for _ in xrange(len(cx2_cid))]
    tt_ALL = tic('all queries')
    assign_times = []
    verify_times = []
    skip_list = []
    for qcx, qcid in enumerate(cx2_cid):
        if qcid == 0: 
            skip_list.append(qcx)
            continue
        tt_A = tic('query(qcx=%d)' % qcx)
        # Assign matches with the chosen function (1v1) or (1vM)
        (cx2_fm, cx2_fs) = matcher.assign_matches(qcx, cx2_cid, cx2_desc)
        assign_times.append(toc(tt_A))
        # Spatially verify the assigned matches
        tt_V = tic('verify(qcx=%d)' % qcx)
        (cx2_fm_V, cx2_fs_V) = spatially_verify_matches(qcx, cx2_kpts, cx2_fm, cx2_fs)
        verify_times.append(toc(tt_V))
        # Assign output to a query result
        res = cx2_res[qcx]
        res.qcx = qcx
        res.cx2_fm    = cx2_fm
        res.cx2_fs    = cx2_fs
        res.cx2_fm_V = cx2_fm_V
        res.cx2_fs_V = cx2_fs_V
    if len(skip_list) > 0:
        print('Skipped more queries than you should have: %r ' % skip_list)
    total_time = toc(tt_ALL)
    # Write results out to disk
    report_results2.write_rank_results(cx2_res, hs, matcher)
    return cx2_res

#@profile
def __spatially_verify(func_homog, kpts1, kpts2, fm, fs, DBG=None):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers '''
    # ugg transpose, I like row first, but ransac seems not to
    if len(fm) == 0: 
        return (np.empty((0,2)),np.empty((0,1)), np.eye(3))
    kpts1_m = kpts1[fm[:,0],:].T
    kpts2_m = kpts2[fm[:,1],:].T
    # -----------------------------------------------
    # TODO: SHOULD THIS HAPPEN HERE? (ISSUE XY_THRESH)
    # Get match threshold 10% of matching keypoint extent diagonal
    img1_extent = (kpts1_m[0:2,:].max(1) - kpts1_m[0:2,:].min(1))[0:2]
    xy_thresh1_sqrd = np.sum(img1_extent**2) * (__XY_THRESH__**2)
    if not DBG is None:
        print('---------------------------------------')
        print('INFO: spatially_verify xy threshold:')
        print(' * Threshold is %.1f%% of diagonal length' % (__XY_THRESH__*100))
        print(' * img1_extent = %r '     % img1_extent)
        print(' * img1_diag_len = %.2f ' % np.sqrt(np.sum(img1_extent**2)))
        print(' * xy_thresh1_sqrd=%.2f'  % np.sqrt(xy_thresh1_sqrd))
        print('---------------------------------------')
    # -----------------------------------------------
    H, inliers = func_homog(kpts2_m, kpts1_m, xy_thresh1_sqrd) 
    fm_V = fm[inliers,:]
    fs_V = fs[inliers,:]
    return fm_V, fs_V, H

def spatially_verify(kpts1, kpts2, fm, fs, DBG=None):
    ''' Concrete implementation of spatial verification
        using the deterministic ellipse based sample conensus'''
    return __spatially_verify(H_homog_from_DELSAC, kpts1, kpts2, fm, fs, DBG)
spatially_verify.__doc__ += '\n'+__spatially_verify.__doc__

#@profile
def spatially_verify_matches(qcx, cx2_kpts, cx2_fm, cx2_fs):
    kpts1     = cx2_kpts[qcx]
    cx2_cscore = np.array([np.sum(fs) for fs in cx2_fs])
    top_cx     = cx2_cscore.argsort()[::-1]
    num_rerank = min(len(top_cx), __NUM_RERANK__)
    # -----------------------------------------------
    # TODO: SHOULD THIS HAPPEN HERE? (ISSUE XY_THRESH)
    #img1_extent = (kpts1[:,0:2].max(0) - qkpts1[:,0:2].min(0))[0:2]
    #xy_thresh1_sqrd = np.sum(img1_extent**2) * __xy_thresh_percent__
    # -----------------------------------------------
    # Precompute output container
    cx2_fm_V = [[] for _ in xrange(len(cx2_fm))]
    cx2_fs_V = [[] for _ in xrange(len(cx2_fs))]
    # spatially verify the top __NUM_RERANK__ results
    for topx in xrange(num_rerank):
        cx    = top_cx[topx]
        kpts2 = cx2_kpts[cx]
        fm    = cx2_fm[cx]
        fs    = cx2_fs[cx]
        fm_V, fs_V, H = spatially_verify(kpts1, kpts2, fm, fs)
        cx2_fm_V[cx] = fm_V
        cx2_fs_V[cx] = fs_V
    return cx2_fm_V, cx2_fs_V


# Feature scoring functions
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist, vdist+1E-8)+1) 
def RATIO_fn(vdist, ndist): return np.divide(ndist, vdist+1E-8)
def LNBNN_fn(vdist, ndist): return ndist - vdist 
score_fn = RATIO_fn

#@profile
def assign_matches_1vM(qcx, cx2_cid, cx2_desc, flann_1vM):
    '''
    Matches desc1 vs all database descriptors using 
    Input:
        qcx       - query chip index
        cx2_cid   - chip ID lookup table (for removing self matches)
        cx2_desc  - chip descriptor lookup table
        flann_1vM - prebuild FLANN index of aggregated database descriptors
    Output: 
        cx2_fm - C x Mx2 array of matching feature indexes
        cx2_fs - C x Mx1 array of matching feature scores
    '''
    print('Assigning 1vM feature matches from qcx=%d to %d chips' % (qcx, len(cx2_cid)))
    isQueryIndexed = True
    desc1 = cx2_desc[qcx]
    K = __K__+1 if isQueryIndexed else __K__
    # Find each query descriptor's K+1 nearest neighbors
    (qfx2_ax, qfx2_dists) = flann_1vM.nn_index(desc1, K+1, **__FLANN_PARAMS__)
    vote_dists = qfx2_dists[:, 0:K]
    norm_dists = qfx2_dists[:, K] # K+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([score_fn(_vdist.T, norm_dists) for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    qfx2_cx = flann_1vM.ax2_cx[qfx2_ax[:,0:K]]
    qfx2_fx = flann_1vM.ax2_fx[qfx2_ax[:,0:K]]
    # Build feature matches
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    num_qf = len(desc1)
    qfx2_qfx = np.tile(np.arange(num_qf).reshape(num_qf,1), (1,K)) 
    iter_matches = iter(zip(qfx2_qfx.flat, qfx2_cx.flat, qfx2_fx.flat, qfx2_score.flat))
    for qfx, cx,fx,score in iter_matches:
        if qcx == cx: continue # dont vote for yourself
        cx2_fm[cx].append((qfx,fx))
        cx2_fs[cx].append(score)
    # Convert to numpy
    for cx in xrange(len(cx2_cid)): cx2_fm[cx] = np.array(cx2_fm[cx])
    for cx in xrange(len(cx2_cid)): cx2_fs[cx] = np.array(cx2_fs[cx])
    return cx2_fm, cx2_fs

def match_1v1(desc2, flann_1v1, ratio_thresh=1.2, burst_thresh=None, DBG=False):
    '''
    Matches desc2 vs desc1 using Lowe's ratio test
    Input:
        desc2         - other descriptors (N2xD)
        flann_1v1     - prebuild FLANN index of desc1 (query descriptors (N1xD)) 
    Thresholds: 
        ratio_thresh = 1.2 - keep match if dist(2)/dist(1) > ratio_thresh
        burst_thresh = 1   - keep match if 0 < matching_freq(desc1) <= burst_thresh 
    Output: 
        fm - Mx2 array of matching feature indexes
        fs - Mx1 array of matching feature scores
    '''
    # features to their matching query features
    (fx2_qfx, fx2_dist) = flann_1v1.nn_index(desc2, 2, **__FLANN_PARAMS__)
    # RATIO TEST
    fx2_ratio  = np.divide(fx2_dist[:,1], fx2_dist[:,0]+1E-8)
    fx_passratio, = np.where(fx2_ratio > ratio_thresh)
    fx = fx_passratio
    # BURSTINESS TEST
    # Find frequency of descriptor matches
    # Select the query features which only matched < burst_thresh
    # Convert qfx to fx
    # FIXME: there is probably a better way of doing this.
    if not burst_thresh is None:
        qfx2_frequency = np.bincount(fx2_qfx[:,0])
        qfx_occuring   = qfx2_frequency > 0
        qfx_nonbursty  = qfx2_frequency <= burst_thresh
        qfx_nonbursty_unique,= np.where(np.bitwise_and(qfx_occuring, qfx_nonbursty))
        _qfx_set      = set(qfx_nonbursty_unique.tolist())
        fx2_nonbursty = [_qfx in _qfx_set for _qfx in iter(fx2_qfx[:,0])]
        fx_nonbursty, = np.where(fx2_nonbursty)
        fx  = np.intersect1d(fx, fx_nonbursty, assume_unique=True)
    # RETURN 1v1 matches and scores
    qfx = fx2_qfx[fx,0]
    fm  = np.array(zip(qfx, fx))
    fs  = fx2_ratio[fx]
    # DEBUG
    if DBG:
        print('-------------')
        print('Matching 1v1:')
        print(' * Ratio threshold: %r ' % ratio_thresh)
        print(' * Burst threshold: %r ' % burst_thresh)
        print(' * fx_passratio.shape   = %r ' % (  fx_passratio.shape,))
        if not burst_thresh is None:
            print(' * fx_nonbursty.shape   = %r ' % (  fx_nonbursty.shape,))
        print(' * fx.shape   = %r ' % (  fx.shape,))
        print(' * qfx.shape  = %r ' % (  qfx.shape,))
    return (fm, fs)

def warp_chip(rchip2, H, rchip1):
    rchip2W = cv2.warpPerspective(rchip2, H, rchip1.shape[0:2][::-1])
    return rchip2W

class HotSpotter(DynStruct):
    def __init__(self):
        super(HotSpotter, self).__init__()
        self.tables = None
        self.feats  = None
        self.cpaths = None
        self.dirs   = None

# TODO, this should go in a more abstracted module
def load_hotspotter(db_dir):
    # --- LOAD DATA --- #
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
    # --- LOAD CHIPS --- #
    hs_cpaths = chip_compute2.load_chip_paths(hs_dirs, hs_tables)
    # --- LOAD FEATURES --- #
    hs_feats  = feature_compute2.load_chip_features(hs_dirs, hs_tables, hs_cpaths)
    hs_feats.set_feat_type(__FEAT_TYPE__)
    # --- BUILD HOTSPOTTER --- #
    hs = HotSpotter()
    hs.tables = hs_tables
    hs.feats  = hs_feats
    hs.cpaths = hs_cpaths
    hs.dirs   = hs_dirs
    return hs

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # --- CHOOSE DATABASE --- #
    db_dir = load_data2.MOTHERS
    hs = load_hotspotter(db_dir)

    __TEST_MODE__ = True
    if __TEST_MODE__:
        runall_match(hs)
        pass

    ## DEV ONLY CODE ##
    __DEV_MODE__ = False
    if __DEV_MODE__: 
        print(hs_cpaths)
        print(hs_dirs)
        print(hs_tables)
        print(hs_feats)
        # Convinent but bad # 
        exec(hs_cpaths.execstr('hs_cpaths'))
        exec(hs_feats.execstr('hs_feats'))
        exec(hs_tables.execstr('hs_tables'))
        exec(hs_dirs.execstr('hs_dirs'))
        cx2_kpts = hs_feats.cx2_kpts
        cx2_desc = hs_feats.cx2_kpts
        qcx = 1
        #cx  = 1
        # All of these functions operate on one qcx (except precompute I guess)
        exec(get_exec_src(precompute_index_1vM))
        exec(get_exec_src(assign_matches_1vM))
        #exec(get_exec_src(spatially_verify_matches))

        try: 
            __IPYTHON__
        except: 
            plt.show()
