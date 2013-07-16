import drawing_functions2 as df2
import sys
import matplotlib.pyplot as plt
import numpy as np
from hotspotter.Parallelize import parallel_compute
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.helpers import Timer, get_exec_src, check_path
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


def EXPERIMENT(hs_tables, hs_feats):
    cx2_feats_hesaff = hs_feats.cx2_feats_hesaff
    cx2_feats_sift   = hs_feats.cx2_feats_sift
    cx2_feats_freak  = hs_feats.cx2_feats_freak

    runall_match(cx2_feats_sift,   hs_tables)
    runall_match(cx2_feats_hesaff, hs_tables)
    #runall_match(cx2_feats_freak,  hs_tables)


def runall_match(cx2_feats, hs_tables):
    cx2_cid    = hs_tables.cx2_cid

    cx2_desc   = [d for (k,d) in cx2_feats]
    cx2_kpts   = [k for (k,d) in cx2_feats]

    funcs_1v1 = {'fn_precomp_args'        : precompute_args_1v1, 
                 'fn_assign_feat_matches' : assign_feat_matches_1v1}
    funcs_1vM = {'fn_precomp_args'        : precompute_args_1vM, 
                 'fn_assign_feat_matches' : assign_feat_matches_1vM}

    tmparg = DynStruct(copy_dict=funcs_1vM)
    exec(tmparg.execstr('tmparg'))
    
    cx2_res_1vM = __runall(cx2_desc, cx2_kpts, **funcs_1vM)
    #cx2_res_1v1 = __runall(cx2_desc, cx2_kpts, **funcs_1v1)


class HotspotterQueryResult(DynStruct):
    def __init__(self):
        super(HotspotterQueryResult, self).__init__()
        self.qcx    = -1
        self.cx2_fm = []
        self.cx2_fs = []

def __runall(cx2_desc, cx2_kpts, fn_precomp_args=None, fn_assign_feat_matches=None):
    with Timer(msg=None):
        assign_args = fn_precomp_args(cx2_cid, cx2_desc)
    cx2_res = [HotspotterQueryResult() for _ in xrange(len(cx2_cid))]
    with Timer(msg=None):
        for qcx, qcid in enumerate(cx2_cid):
            res = cx2_res[qcx]
            res.qcx = qcx
            if qcid == 0: continue
            with Timer(msg=None):
                matches_scores = fn_assign_feat_matches(qcx, *assign_args)
                res.cx2_fm = matches_scores[0]
                res.cx2_fs = matches_scores[1]
    return cx2_res

def spatially_verify(qcx, cx2_kpts, cx2_fm, cx2_fs):
    qkpts = cx2_kpts[qcx]
    cx2_cscore = np.array([np.sum(fs) for fs in cx2_fs])
    top_cx = cx2_cscore.argsort()[::-1]

    topx = 0 # for
    cx = top_cx[topx]

    kpts = cx2_kpts[cx]
    fm12 = cx2_fm[cx]
    mx1  = fm12[:,0]
    mx2  = fm12[:,1]

    # ugg transpose, put in an assert to keep things sane. I like row first, 
    # but ransac seems not to
    kpts1_m = qkpts[mx1,:].T
    kpts2_m =  kpts[mx2,:].T
    assert kpts1_m.shape[0] == 5 and kpts2_m.shape[0] == 5, 'needs ellipses'
    # Get match threshold 10% of matching keypoint extent diagonal
    img2_extent = (kpts2_m[0:2,:].max(1) - kpts2_m[0:2,:].min(1))[0:2]
    xy_thresh_sqrd = np.sum(img2_extent**2)/100

    #H, inliers = H_homog_from_DELSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)
    # Show what it did
    #fm12_SV = fm12[inliers,:]

def precompute_args_1vM(cx2_cid, cx2_desc, hs_dirs):
    feat_dir = hs_dirs.feat_dir
    print('Precomputing one vs many information')
    cx2_nFeats = [len(k) for k in cx2_desc]
    _ax2_cx = [[cx_]*nFeats for (cx_, nFeats) in iter(zip(range(len(cx2_cid)), cx2_nFeats))]
    _ax2_fx = [range(nFeats) for nFeats in iter(cx2_nFeats)]
    ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    ax2_desc   = np.vstack(cx2_desc)

    flann_1vM = FLANN()
    flann_1vM_path = feat_dir + '/flann_1vM.index'
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
    flann_1vM.ax2_desc = ax2_desc # dont let this loose scope
    return cx2_cid, cx2_desc, ax2_cx, ax2_fx, flann_1vM

# Feature scoring functions
def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist+1, vdist+1)) 
def RATIO_fn(vdist, ndist): return np.divide(ndist+1, vdist+1)
def LNBNN_fn(vdist, ndist): return ndist - vdist 

score_fn = LNRAT_fn
def assign_feat_matches_1vM(qcx, cx2_cid, cx2_desc, ax2_cx, ax2_fx, flann_1vM):
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
    qfx2_cx = ax2_cx[qfx2_ax[:,0:K]]
    qfx2_fx = ax2_fx[qfx2_ax[:,0:K]]
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

def precompute_args_1v1(cx2_cid, cx2_desc):
    return cx2_cid, cx2_desc

def assign_feat_matches_1v1(qcx, cx2_cid, cx2_desc):
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
        (fx2_qfx, fx2_dist) = flann_1v1.nn_index(desc, 2, **__FLANN_PARAMS__)
        # Lowe's ratio test
        fx2_ratio = np.divide(fx2_dist[:,1]+1, fx2_dist[:,0]+1)
        fx, = np.where(fx2_ratio > 1.5)
        qfx = fx2_qfx[fx,0]
        cx2_fm[cx] = np.array(zip(qfx, fx))
        cx2_fs[cx] = fx2_ratio[fx]
    sys.stdout.write('DONE')
    flann_1v1.delete_index()
    return cx2_fm, cx2_fs

def FREAK_assign_feat_matches_1v1(qcx, cx2_cid, cx2_freak):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    qfreak = cx2_freak[qcx]
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, freak in enumerate(cx2_freak):
        sys.stdout.write('.')
        sys.stdout.flush()
        m = matcher.match(freak, qfreak)
        if cx == qcx: continue
        (fx2_qfx, fx2_dist) = flann_1v1.nn_index(freak, 2, **__FLANN_PARAMS__)
        # Lowe's ratio test
        fx2_ratio = np.divide(fx2_dist[:,1]+1, fx2_dist[:,0]+1)
        fx, = np.where(fx2_ratio > __1v1_RAT_THRESH__)
        qfx = fx2_qfx[fx,0]
        cx2_fm[cx] = np.array(zip(qfx, fx))
        cx2_fs[cx] = fx2_ratio[fx]
    sys.stdout.write('DONE')
    flann_1v1.delete_index()
    return cx2_fm, cx2_fs

def report_results(cx2_res, hs_tables):
    cx2_cid = hs_tables.cx2_cid
    cx2_nx  = hs_tables.cx2_nx
    cx2_top_truepos_rank  = np.zeros(len(cx2_cid)) - 100
    cx2_top_truepos_score = np.zeros(len(cx2_cid)) - 100
    cx2_top_trueneg_rank  = np.zeros(len(cx2_cid)) - 100
    cx2_top_trueneg_score = np.zeros(len(cx2_cid)) - 100
    cx2_top_score         = np.zeros(len(cx2_cid)) - 100
    for qcx, qcid in enumerate(cx2_cid):
        qnx = cx2_nx[qcx]
        res = cx2_res[qcx]
        # The score is the sum of the feature scores
        cx2_score = np.array([np.sum(fs) for fs in res.cx2_fs])
        top_cx = np.argsort(cx2_score)[::-1]
        top_score = cx2_score[top_cx]
        top_nx = cx2_nx[top_cx]
        # Remove query from true positives ranks
        _truepos_ranks, = np.where(top_nx == qnx)
        # Get TRUE POSTIIVE ranks
        truepos_ranks = _truepos_ranks[top_cx[_truepos_ranks] != qcx]
        # Get BEST True Positive and BEST True Negative
        if len(truepos_ranks) > 0:
            top_truepos_rank = truepos_ranks.min()
            bot_truepos_rank = truepos_ranks.max()
            true_neg_range   = np.arange(0, bot_truepos_rank+2)
            top_trueneg_rank = np.setdiff1d(true_neg_range, truepos_ranks).min()
            top_trupos_score = top_score[top_truepos_rank]
        else:
            top_trueneg_rank = 0
            top_truepos_rank = np.NAN
            top_trupos_score = np.NAN
        # Append stats to output
        cx2_top_truepos_rank[qcx]  = top_truepos_rank
        cx2_top_truepos_score[qcx] = top_trupos_score
        cx2_top_trueneg_rank[qcx]  = top_trueneg_rank
        cx2_top_trueneg_score[qcx] = top_score[top_trueneg_rank]
        cx2_top_score[qcx]         = top_score[0]
    # difference between the top score and the actual best score
    cx2_score_disp = cx2_top_score - cx2_top_true_score
    #
    # Easy to digest results
    num_chips = len(cx2_top_truepos_rank)
    num_with_gtruth = (1 - np.isnan(cx2_top_truepos_rank)).sum()
    num_rank_less5 = (cx2_top_truepos_rank < 5).sum()
    num_rank_less1 = (cx2_top_truepos_rank < 1).sum()
    
    # Display ranking results
    rankres_str = ('#TTP = top true positive #TTN = top true negative\n')
    rankres_header = '#CID, TTP RANK, TTN RANK, TTP SCORE, TTN SCORE, SCORE DISP, NAME\n'
    rankres_str += rankres_header
    todisp = np.vstack([cx2_cid,
                        cx2_top_truepos_rank,
                        cx2_top_trueneg_rank,
                        cx2_top_truepos_score,
                        cx2_top_trueneg_score,
                        cx2_score_disp, 
                        cx2_nx]).T
    for (cid, ttpr, ttnr, ttps, ttns, sdisp, nx) in todisp:
        rankres_str+=('%4d, %8.0f, %8.0f, %9.2f, %9.2f, %10.2f, %s\n' %\
              (cid, ttpr, ttnr, ttps, ttns, sdisp, nx2_name[nx]) )
    rankres_str += rankres_header

    rankres_str += '#Num Chips: %d \n' % num_chips
    rankres_str += '#Num Chips with at least one match: %d \n' % num_with_gtruth
    rankres_str += '#Ranks <= 5: %d / %d\n' % (num_rank_less5, num_with_gtruth)
    rankres_str += '#Ranks <= 1: %d / %d\n' % (num_rank_less1, num_with_gtruth)
    
    print(rankres_str)
    result_csv = 'results_ground_truth_rank.csv'
    with open(result_csv, 'w') as file:
        file.write(rankres_str)
    os.system('gvim '+result_csv)

def unpack_freak(cx2_desc):
    cx2_unpacked_freak = []
    for descs in cx2_desc:
        unpacked_desc = []
        for d in descs:
            bitstr = ''.join([('{0:#010b}'.format(byte))[2:] for byte in d])
            d_bool = np.array([int(bit) for bit in bitstr],dtype=bool)
            unpacked_desc.append(d_bool)
        cx2_unpacked_freak.append(unpacked_desc)

def show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12, fignum=1):
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    rchip1 = cv2.imread(cx2_rchip_path[qcx])
    rchip2 = cv2.imread(cx2_rchip_path[cx])
    kpts1  = cx2_kpts[qcx]
    kpts2  = cx2_kpts[cx]

    rchip1_kpts = draw_kpts(rchip1, kpts1[fm12[:,0],:])
    rchip2_kpts = draw_kpts(rchip2, kpts2[fm12[:,1],:])
    img = draw_matches(rchip1_kpts, rchip2_kpts, qkpts, kpts, fm12, vert=True)
    fig = plt.figure(fignum)
    fig.clf()
    plt.imshow(img)
    fig.show()

if __name__ == '__main__':
    import chip_compute2, feature_compute2, load_data2
    from multiprocessing import freeze_support
    freeze_support()
    # --- CHOOSE DATABASE --- #
    db_dir = load_data2.MOTHERS
    # --- LOAD DATA --- #
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)
    # --- LOAD CHIPS --- #
    hs_cpaths = chip_compute2.load_chip_paths(hs_dirs, hs_tables)
    # --- LOAD FEATURES --- #
    hs_feats  = feature_compute2.load_chip_features(hs_dirs, hs_tables, hs_cpaths)

    ## DEV ONLY CODE ##
    __DEV_MODE__ = True
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
        #cx2_feats = cx2_feats_hesaff
        cx2_feats = cx2_feats_sift
        cx2_desc  = [d for (k,d) in cx2_feats]
        cx2_kpts  = [k for (k,d) in cx2_feats]
        qcx = 1
        #cx  = 1
        # All of these functions operate on one qcx (except precompute I guess)
        exec(get_exec_src(precompute_args_1vM))
        exec(get_exec_src(assign_feat_matches_1vM))
        #exec(get_exec_src(spatially_verify))

        #sys.exit(1)

        #import imp
        #import cvransac2
        #imp.reload(cvransac2)
        #from cvransac2 import H_homog_from_RANSAC, H_homog_from_DELSAC, H_homog_from_PCVSAC

        #http://scikit-image.org/docs/dev/api/skimage.transform.html#estimate-transform
        #with Timer(msg=None):
            #H1, inliers1 = H_homog_from_RANSAC(kpts1_m, kpts2_m, xy_thresh_sqrd) 
        #with Timer(msg=None):
            #H2, inliers2 = H_homog_from_DELSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)
        #with Timer(msg=None):
            #H3, inliers3 = H_homog_from_PCVSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)

        #H1, inliers1 = H_homog_from_RANSAC(kpts1_m, kpts2_m, xy_thresh_sqrd) 
        #H2, inliers2 = H_homog_from_DELSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)
        #H3, inliers3 = H_homog_from_PCVSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)

        #rchip1 = cv2.imread(cx2_rchip_path[qcx])
        #rchip2 = cv2.imread(cx2_rchip_path[cx])

        #import skimage.transform
        #rchip1_H1 = skimage.transform.fast_homography(rchip1, H1)
        #rchip1_H2 = skimage.transform.fast_homography(rchip1, H2)
        #rchip1_H3 = skimage.transform.fast_homography(rchip1, H3)
        #import hotspotter.tpl.cv2 as cv2
        
        # http://stackoverflow.com/questions/8181872/finding-homography-and-warping-perspective
        #H = findHomography( src2Dfeatures, dst2Dfeatures, outlierMask, RANSAC, 3);

        
        #rchip1_H = cv2.warpPerspective(rchip2, H, rchip2.shape[0:2], cv2.INTER_LANCZOS4)
        #rchip1_H= cv2.warpPerspective(rchip2, H, rchip2.size(), cv2.INTER_LANCZOS4)
        #rchip2_H1 = cv2.warpPerspective(rchip2, linalg.inv(H1), rchip1.size(), cv2.INTER_LANCZOS4)
        #rchip2_H2 = cv2.warpPerspective(rchip2, linalg.inv(H2), rchip1.size(), cv2.INTER_LANCZOS4)
        #rchip2_H3 = cv2.warpPerspective(rchip2, linalg.inv(H3), rchip1.size(), cv2.INTER_LANCZOS4)


        #cmd = 'H1, inliers1 = H_homog_from_RANSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)'
        #from hotspotter.helpers import profile
        #profile(cmd)

        #H1, inliers1 = H_homog_from_RANSAC(kpts1_m, kpts2_m, xy_thresh_sqrd) 
        # Show what it did
        #fm12_SV1 = fm12[inliers1, :]
        #fm12_SV2 = fm12[inliers2, :]
        #fm12_SV3 = fm12[inliers3, :]

        #show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12,     fignum=0)
        #show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12_SV1, fignum=1)
        #show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12_SV2, fignum=2)
        #show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12_SV3, fignum=3)

        #tile_all_figures()

        try: 
            __IPYTHON__
        except: 
            plt.show()
