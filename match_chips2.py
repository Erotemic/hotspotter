import drawing_functions2 as df2
import chip_compute2, feature_compute2, load_data2
import report_results2
import sys
import matplotlib.pyplot as plt
import numpy as np
from hotspotter.Parallelize import parallel_compute
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.helpers import Timer, get_exec_src, check_path, tic, toc
from drawing_functions2 import draw_matches, draw_kpts, tile_all_figures
from hotspotter.tpl.pyflann import FLANN
import hotspotter.tpl.cv2  as cv2
from itertools import chain
from numpy import linalg
from cvransac2 import H_homog_from_RANSAC, H_homog_from_DELSAC, H_homog_from_PCVSAC, H_homog_from_CV2SAC

__K__ = 2
__NUM_RERANK__     = 50
__1v1_RAT_THRESH__ = 1.5
__FLANN_PARAMS__ = {'algorithm' :'kdtree',
                    'trees'     :4,
                    'checks'    :128}
__FEAT_TYPE__ = 'HESAFF'
__xy_thresh_percent__ = .05

def runall_match(hs):
    with Timer(msg=None):
        flann_1vM = precompute_index_1vM(hs)
    #functools.partial
    cx2_res_1vM = __run_matching(hs, assign_matches_1vM, flann_1vM)
    cx2_res_1v1 = __run_matching(hs, assign_matches_1v1)

class HotspotterQueryResult(DynStruct):
    def __init__(self):
        super(HotspotterQueryResult, self).__init__()
        self.qcx    = -1
        self.cx2_fm = []
        self.cx2_fs = []
        self.cx2_fm_SV = []
        self.cx2_fs_SV = []

# Work function that is the basic matching pipeline. 
# specific functions need to be bound: 
# 
# fn_assign - assign feature matches
def __run_matching(hs, fn_assign, *args):
    cx2_cid    = hs.tables.cx2_cid
    cx2_kpts, cx2_desc = hs.get_feats(__FEAT_TYPE__)
    cx2_desc   = [d for (k,d) in cx2_feats]
    cx2_kpts   = [k for (k,d) in cx2_feats]
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
        (cx2_fm, cx2_fs) = fn_assign(qcx, cx2_cid, cx2_desc, *args)
        assign_times.append(toc(tt_A))
        # Spatially verify the assigned matches
        tt_V = tic('verify(qcx=%d)' % qcx)
        (cx2_fm_SV, cx2_fs_SV) = spatially_verify_1vX(qcx, cx2_kpts, cx2_fm, cx2_fs)
        verify_times.append(toc(tt_V))
        # Assign output to a query result
        res = cx2_res[qcx]
        res.qcx = qcx
        res.cx2_fm    = cx2_fm
        res.cx2_fs    = cx2_fs
        res.cx2_fm_SV = cx2_fm_SV
        res.cx2_fs_SV = cx2_fs_SV
    if len(skip_list) > 0:
        print('Skipped more queries than you should have: %r ' % skip_list)
    total_time = toc(tt_ALL)
    report_results2.report_results(cx2_res, hs.tables)
    return cx2_res

#@profile
def spatially_verify(kpts1, kpts2, fm, fs):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers '''
    # ugg transpose, I like row first, but ransac seems not to
    kpts1_m = kpts1[fm[:,0],:].T
    kpts2_m = kpts2[fm[:,1],:].T
    # -----------------------------------------------
    # TODO: SHOULD THIS HAPPEN HERE? (ISSUE XY_THRESH)
    # Get match threshold 10% of matching keypoint extent diagonal
    img1_extent = (kpts1_m[0:2,:].max(1) - kpts1_m[0:2,:].min(1))[0:2]
    xy_thresh1_sqrd = np.sum(img1_extent**2) * (__xy_thresh_percent__**2)
    # -----------------------------------------------
    H, inliers = H_homog_from_DELSAC(kpts2_m, kpts1_m, xy_thresh1_sqrd) 
    fm_SV = fm[inliers,:]
    fs_SV = fs[inliers,:]
    return fm_SV, fs_SV

#@profile
def spatially_verify_1vX(qcx, cx2_kpts, cx2_fm, cx2_fs):
    qkpts1 = cx2_kpts[qcx]
    cx2_cscore = np.array([np.sum(fs) for fs in cx2_fs])
    top_cx = cx2_cscore.argsort()[::-1]
    num_rerank = min(len(top_cx), __NUM_RERANK__)
    # -----------------------------------------------
    # TODO: SHOULD THIS HAPPEN HERE? (ISSUE XY_THRESH)
    #img1_extent = (qkpts1[:,0:2].max(0) - qkpts1[:,0:2].min(0))[0:2]
    #xy_thresh1_sqrd = np.sum(img1_extent**2) * __xy_thresh_percent__
    # -----------------------------------------------
    # Precompute output container
    cx2_fm_SV = [[] for _ in xrange(len(cx2_fm))]
    cx2_fs_SV = [[] for _ in xrange(len(cx2_fs))]
    # spatially verify the top __NUM_RERANK__ results
    for topx in xrange(num_rerank):
        cx    = top_cx[topx]
        kpts2 = cx2_kpts[cx]
        fm    = cx2_fm[cx]
        fs    = cx2_fs[cx]
        fm_SV, fs_SV = spatially_verify(qkpts1, kpts2, fm, fs)
        cx2_fm_SV[cx] = fm_SV
        cx2_fs_SV[cx] = fs_SV
    return cx2_fm_SV, cx2_fs_SV


#@profile
def precompute_index_1vM(hs):
    cx2_cid   = hs.tables.cx2_cid
    cx2_desc  = hs.feats.cx2_desc
    feat_dir  = hs.dirs.feat_dir
    feat_type = hs.feats.feat_type
    print('Precomputing one vs many information')
    cx2_nFeats = [len(k) for k in cx2_desc]
    _ax2_cx = [[cx_]*nFeats for (cx_, nFeats) in iter(zip(range(len(cx2_cid)), cx2_nFeats))]
    _ax2_fx = [range(nFeats) for nFeats in iter(cx2_nFeats)]
    ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    ax2_desc   = np.vstack(cx2_desc)
    # Build (or reload) one vs many flann index
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

# Feature scoring functions
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist+1, vdist+1)) 
def RATIO_fn(vdist, ndist): return np.divide(ndist+1, vdist+1)
def LNBNN_fn(vdist, ndist): return ndist - vdist 
score_fn = LNRAT_fn

#@profile
def assign_matches_1vM(qcx, cx2_cid, cx2_desc, flann_1vM):
    print('Assigning 1vM feature matches from qcx=%d to %d chips' % (qcx, len(cx2_cid)))
    isQueryIndexed = True
    qdesc = cx2_desc[qcx]
    K = __K__+1 if isQueryIndexed else __K__
    # Find each query descriptor's K+1 nearest neighbors
    (qfx2_ax, qfx2_dists) = flann_1vM.nn_index(qdesc, K+1, **__FLANN_PARAMS__)
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
    num_qf = len(qdesc)
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

def assign_matches_1v1(qcx, cx2_cid, cx2_desc):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    qdesc = cx2_desc[qcx]
    flann_1v1 = FLANN()
    flann_1v1.build_index(qdesc, **__FLANN_PARAMS__)
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, desc in enumerate(cx2_desc):
        sys.stdout.write('.')
        sys.stdout.flush()
        if cx == qcx: continue
        (fm, fs) = match_1v1(qdesc, desc, flann_1v1)
        cx2_fm[cx] = fm
        cx2_fs[cx] = fs
    sys.stdout.write('DONE')
    flann_1v1.delete_index()
    return cx2_fm, cx2_fs

def match_1v1(qdesc, desc, flann_1v1=None):
    (fx2_qfx, fx2_dist) = flann_1v1.nn_index(desc, 2, **__FLANN_PARAMS__)
    # Lowe's ratio test
    fx2_ratio = np.divide(fx2_dist[:,1]+1, fx2_dist[:,0]+1)
    fx, = np.where(fx2_ratio > 1.5)
    qfx = fx2_qfx[fx,0]
    fm = np.array(zip(qfx, fx))
    fs = fx2_ratio[fx]
    return (fm, fs)

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
        __FEAT_TYPE__ = 'SIFT'
        hs_feats.set_feat_type(__FEAT_TYPE__)
        cx2_kpts = hs_feats.cx2_kpts
        cx2_desc = hs_feats.cx2_kpts
        qcx = 1
        #cx  = 1
        # All of these functions operate on one qcx (except precompute I guess)
        exec(get_exec_src(precompute_index_1vM))
        exec(get_exec_src(assign_matches_1vM))
        #exec(get_exec_src(spatially_verify_1vX))

        try: 
            __IPYTHON__
        except: 
            plt.show()
