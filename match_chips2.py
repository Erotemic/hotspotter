''' BruteForce
BruteForce-L1
BruteForce-Hamming
BruteForceHamming(2)
FlannBased ''' 
#raw_matches = matcher.match(descriptors2  descriptors4)
#img_matches = cv2.DRAW_MATCHES_FLAGS_DEFAULT(im2  keypoints2  im4  keypoints4  raw_matches)

rchip_path = cx2_rchip_path[0]
sift_path = cx2_sift_path[0]
sift = fc2.load_features(sift_path)

kpts, desc = sift
from feature_compute2 import *
detector = common_detector
extractor = sift_extractor

rchip = cv2.imread(rchip_path)
_cvkpts = detector.detect(rchip)  
print_cvkpt(_cvkpts)

cx1 = 1
cx2 = 2




from hotspotter.tpl.pyflann import FLANN
import hotspotter.tpl.cv2  as cv2
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
    
from PCV.geometry import homography, warp
import pylab
pylab.set_cmap('gray')
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


# adapted from:
# http://jayrambhia.com/blog/sift-keypoint-matching-using-python-opencv/
def draw_matches(rchip1, rchip2, kpts1, kpts2, matches12, vert=False):
    h1, w1 = rchip1.shape[0:2]
    h2, w2 = rchip2.shape[0:2]
    woff = 0; hoff = 0 # offsets 
    if vert:
        wB = max(w1, w2); hB = h1+h2; hoff = h1
    else: 
        hB = max(h1, h2); wB = w1+w2; woff = w1
    # Concat images
    match_img = np.zeros((hB, wB, 3), np.uint8)
    match_img[0:h1, 0:w1, :] = rchip1
    match_img[hoff:(hoff+h2), woff:(woff+w2), :] = rchip2
    # Draw lines
    for kx1, kx2 in iter(matches12):
        pt1 = (int(kpts1[kx1,0]),      int(kpts1[kx1,1]))
        pt2 = (int(kpts2[kx2,0])+woff, int(kpts2[kx2,1])+hoff)
        match_img = cv2.line(match_img, pt1, pt2, (255, 0, 0))
    return match_img
    
def draw_kpts(_rchip, _kpts):
    kpts_img = np.copy(_rchip)
    # Draw circles
    for (x,y,a,d,c) in iter(_kpts):
        center = (int(x), int(y))
        radius = int(3*np.sqrt(1/a))
        kpts_img = cv2.circle(kpts_img, center, radius, (255, 0, 0))
    return kpts_img


def desc_matcher(cx1, cx2):
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    matches = matcher.match(desc1, desc2)
    return matches

    xy_thresh2 = np.sum(np.array(rchip2.shape[0:2])**2)

    Haffine_from_points()
    model = homography.RansacModel()
    num_m = len(matches12)
