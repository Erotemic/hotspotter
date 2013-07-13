from PCV.geometry import homography, warp
from drawing_functions2 import draw_matches, draw_kpts
from hotspotter.tpl.pyflann import FLANN
import hotspotter.tpl.cv2  as cv2

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

rchip_path = cx2_rchip_path[0]
sift_path = cx2_sift_path[0]
sift = fc2.load_features(sift_path)

kpts, desc = sift

cx1 = 1
cx2 = 2

def flann_nearest(desc1, desc2, K=1):
    flann = FLANN()
    flann_params = {'algorithm':'kdtree',
                    'trees':4,
                    'checks':128}
    flann.build_index(desc1, **flann_params)
    (idx21, dists21) = flann.nn_index(desc2, K, **flann_params)
    idx21.shape   =  (desc2.shape[0], K)
    dists21.shape =  (desc2.shape[0], K)
    flann.delete_index()
    return (np.transpose(idx21), np.transpose(dists21))
    #flann.save_index(path)
    #flann.load_index(path, desc1)
    
#print('Baseline SIFT matching')
#print('len(desc1) = %d' % len(desc1))
#print('len(desc2) = %d' % len(desc2))
#print('len(matches) = %d' % len(matches))
def one_vs_one(cx1, cx2):
    kpts1, desc1 = cx2_sift_feats[cx1]
    kpts2, desc2 = cx2_sift_feats[cx2]
    idx21, dists = flann_nearest(desc1, desc2, K=2)
    ratio = dists[1,:] / dists[0,:]
    mx2, = np.where(ratio > 1.5)
    mx1 = idx21[0, mx2]
    matches12 = np.array(zip(mx1, mx2))

    rchip1 = cv2.imread(cx2_rchip_path[cx1])
    rchip2 = cv2.imread(cx2_rchip_path[cx2])
    # Homogonize and transpose for PCV

    inlier_matches_12 = PCV_ransac(kpts1, kpts2, matches12)

    #delta = 2000
    #im_12 = warp.panorama(H_12,rchip1,rchip2,delta,delta)
    
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

def desc_matcher(cx1, cx2):
    ''' BruteForce, BruteForce-L1, BruteForce-Hamming,
    BruteForceHamming(2), FlannBased '''
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    matches = matcher.match(desc1, desc2)
    return matches
