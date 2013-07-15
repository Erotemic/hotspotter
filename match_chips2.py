from PCV.geometry import homography, warp
from hotspotter.Parallelize import parallel_compute
from drawing_functions2 import draw_matches, draw_kpts
from hotspotter.tpl.pyflann import FLANN
import hotspotter.tpl.cv2  as cv2
from itertools import chain

flann_params = {'algorithm' :'kdtree',
                'trees'     :4,
                'checks'    :128}

if __name__ == '__main__':
    import chip_compute2
    import feature_compute2
    import load_data2
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

def unpack_freak(cx2_desc):
    cx2_unpacked_freak = []
    for descs in cx2_desc:
        unpacked_desc = []
        for d in descs:
            bitstr = ''.join([('{0:#010b}'.format(byte))[2:] for byte in d])
            d_bool = np.array([int(bit) for bit in bitstr],dtype=bool)
            unpacked_desc.append(d_bool)
        cx2_unpacked_freak.append(unpacked_desc)

        

def EXPERIMENT(hs_tables, hs_feats):
    __K__ = 2

    cx2_hesaff_feats = hs_feats.cx2_hesaff_feats
    cx2_sift_feats   = hs_feats.cx2_sift_feats
    cx2_freak_feats  = hs_feats.cx2_freak_feats

    cx2_freak  = [d for (k,d) in cx2_freak_feats]
    cx2_freak_kpt  = [k for (k,d) in cx2_freak_feats]

    #cx2_feats  = cx2_hesaff_feats
    cx2_feats  = cx2_freak_feats

    cx2_cid    = hs_tables.cx2_cid

    cx2_desc   = [d for (k,d) in cx2_feats]
    cx2_kpts   = [k for (k,d) in cx2_feats]
    cx2_nFeats = [len(k) for k in cx2_kpts]

    ## <1vM Aggregate Info> ##
    _ax2_cx = [[cx_]*nFeats for (cx_, nFeats) in iter(zip(range(len(cx2_cid)), cx2_nFeats))]
    _ax2_fx = [range(nFeats) for nFeats in iter(cx2_nFeats)]
    ax2_cx  = np.array(list(chain.from_iterable(_ax2_cx)))
    ax2_fx  = np.array(list(chain.from_iterable(_ax2_fx)))
    ax2_desc   = np.vstack(cx2_desc)
    flann_1vM = FLANN()
    flann_1vM.build_index(ax2_desc, **flann_params)
    ## </1vM Aggregate Info> ##
    # Query information
    qcx = 1

    with Timer(msg=None):
        cx2_fm_1vM, cx2_fs_1vM = assign_feat_matches_1vM(qcx, cx2_cid, cx2_desc, ax2_cx, ax2_fx, flann_1vM)
    with Timer(msg=None):
        cx2_fm_1v1, cx2_fs_1v1 = assign_feat_matches_1v1(qcx, cx2_cid, cx2_desc)
    with Timer(msg=None):
        cx2_fm_1v1, cx2_fs_1v1 = FREAK_assign_feat_matches_1v1(qcx, cx2_cid, cx2_freak)


    cmd = 'assign_feat_matches_1v1(qcx, cx2_cid, cx2_desc)'

    cx2_score_1vM = [np.sum(fs) for fs in cx2_fs_1vM]
    cx2_score_1v1 = [np.sum(fs) for fs in cx2_fs_1v1]
    cx2_num_fm_1v1  = [len(_) for _ in cx2_fs_1vM]
    cx2_num_fm_1vM  = [len(_) for _ in cx2_fs_1v1]


def assign_feat_matches_1vM(qcx, cx2_cid, cx2_desc, ax2_cx, ax2_fx, flann_1vM):
    print('Assigning 1vM feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    qdesc = cx2_desc[qcx]
    # if query is indexed in FLANN
    isQueryIndexed = True
    K = __K__+1 if isQueryIndexed else __K__
    (qfx2_ax, qfx2_dists) = flann_1vM.nn_index(qdesc, K+1, **flann_params)
    vote_dists = qfx2_dists[:, 0:K]
    norm_dists = qfx2_dists[:, K] # K+1th descriptor for normalization

    # Feature scoring functions
    def LNRAT_fn(vdist, ndist): return np.log(np.divide(ndist+1, vdist+1)) 
    def RATIO_fn(vdist, ndist): return np.divide(ndist+1, vdist+1)
    def LNBNN_fn(vdist, ndist): return ndist - vdist 
    #LNBNN_score = np.array([norm_dists - _vdist.T for _vdist in vote_dists.T]).T
    #LNRAT_score = np.array([np.log((norm_dists+1) / (_vdist.T+1)) for _vdist in vote_dists.T]).T
    #RATIO_score = np.array([(norm_dists+1) / (_vdist.T+1) for _vdist in vote_dists.T]).T
    score_fn = LNRAT_fn
    qfx2_score = np.array([score_fn(_vdist.T, norm_dists) for _vdist in vote_dists.T]).T

    # Vote for the appropriate indexes
    qfx2_cx = ax2_cx[qfx2_ax[:,0:K]]
    qfx2_fx = ax2_fx[qfx2_ax[:,0:K]]

    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    iter_matches = iter(zip(qfx2_cx.flat, qfx2_fx.flat, qfx2_score.flat))
    for qfx, (cx,fx,score) in enumerate(iter_matches):
        if qcx == cx: continue # dont vote for yourself
        cx2_fm[cx].append((qfx,fx))
        cx2_fs[cx].append(score)
    return cx2_fm, cx2_fs

def assign_feat_matches_1v1(qcx, cx2_cid, cx2_desc):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    qdesc = cx2_desc[qcx]
    flann_1v1 = FLANN()
    flann_1v1.build_index(qdesc, **flann_params)
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, desc in enumerate(cx2_desc):
        sys.stdout.write('.')
        sys.stdout.flush()
        if cx == qcx: continue
        (fx2_qfx, fx2_dist) = flann_1v1.nn_index(desc, 2, **flann_params)
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
        (fx2_qfx, fx2_dist) = flann_1v1.nn_index(freak, 2, **flann_params)
        # Lowe's ratio test
        fx2_ratio = np.divide(fx2_dist[:,1]+1, fx2_dist[:,0]+1)
        fx, = np.where(fx2_ratio > 1.5)
        qfx = fx2_qfx[fx,0]
        cx2_fm[cx] = np.array(zip(qfx, fx))
        cx2_fs[cx] = fx2_ratio[fx]
    sys.stdout.write('DONE')
    flann_1v1.delete_index()
    return cx2_fm, cx2_fs

rchip_path = cx2_rchip_path[0]
sift_path = cx2_sift_path[0]
sift = fc2.load_features(sift_path)

kpts, desc = sift

qcx = 1


def FLANN_Searcher(object): 
    def __init__(self, qdesc):
        self.flann = FLANN()

        self.flann.build_index(qdesc, **flann_params)
    def neareset(desc2, K=1):
        (idx21, dists21) = flann.nn_index(desc2, K, **flann_params)
        idx21.shape   =  (desc2.shape[0], K)
        dists21.shape =  (desc2.shape[0], K)
        flann.delete_index()
        return idx21.T, dists21.T
    #flann.save_index(path)
    #flann.load_index(path, qdesc)
    
def one_vs_one(qcx, cx2):
    kpts1, qdesc = cx2_sift_feats[qcx]
    kpts2, desc2 = cx2_sift_feats[cx2]
    idx21, dists = flann_nearest(qdesc, desc2, K=2)
    ratio = dists[1,:] / dists[0,:]
    mx2, = np.where(ratio > 1.5)
    mx1 = idx21[0, mx2]
    matches12 = np.array(zip(mx1, mx2))

    rchip1 = cv2.imread(cx2_rchip_path[qcx])
    rchip2 = cv2.imread(cx2_rchip_path[cx2])
    # Homogonize and transpose for PCV

    inlier_matches_12 = PCV_ransac(kpts1, kpts2, matches12)
    kpts_img1 = draw_kpts(rchip1, kpts1_m)
    figure(1)
    imshow(kpts_img1)

    kpts_img2 = draw_kpts(rchip1, kpts2_m)
    figure(2)
    imshow(kpts_img2)

    figure(3)
    match_img = draw_matches(rchip1, rchip2,
                             kpts1,   kpts2, 
                             inlier_matches12, vert=True)
    imshow(match_img)

def PCV_ransac(kpts1, kpts2, matches12):
    # Get xy points
    xy1_m = kpts1[matches12[:,0],0:2] 
    xy2_m = kpts2[matches12[:,1],0:2] 
    # Homogonize points
    num_m = len(matches12)
    fp = np.hstack([xy1_m, np.ones((num_m,1))]).T
    tp = np.hstack([xy2_m, np.ones((num_m,1))]).T
    # Get match threshold 10% of image diagonal
    img2_extent = (kpts2.min(0) - kpts2.max(0))[0:2]
    match_theshold = np.sqrt(np.sum(img2_extent**2))/10
    # Get RANSAC inliers
    maxiter = 1000
    model = homography.RansacModel() 
    try: 
        H_12, inliers = homography.H_from_ransac(fp,tp,model, maxiter,match_theshold)
        inlier_matches12 = matches12[inliers,:]
    except ValueError as ex:
        print(ex)
        inlier_matches12 = []
    return inlier_matches12

def desc_matcher(qcx, cx2):
    ''' BruteForce, BruteForce-L1, BruteForce-Hamming,
    BruteForceHamming(2), FlannBased '''
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')

    matches = matcher.match(qdesc, desc2)
    return matches

    # 1vM Agg info graveyard
    #cx2_max_ax = np.cumsum(cx2_nFeats)
    #_ax2_cid   = [[cid_]*nFeats for (cid_, nFeats) in iter(zip(cx2_cid, cx2_nFeats))]
    #ax2_cid    = np.array(list(chain.from_iterable(_ax2_cid)))
    # dont need no cid stuff here
    #qfx2_cx = np.array([cx2_cid.index(cid) for cid in qfx2_cid.flat])
    #qfx2_cx.shape = qfx2_cid.shape

#print('Baseline SIFT matching')
#print('len(qdesc) = %d' % len(qdesc))
#print('len(desc2) = %d' % len(desc2))
#print('len(matches) = %d' % len(matches))

    #delta = 2000
    #im_12 = warp.panorama(H_12,rchip1,rchip2,delta,delta)
    
