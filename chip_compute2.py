from __future__ import division
from PIL import Image
from hotspotter.Parallelize import parallel_compute
from hotspotter.algo.imalgos import histeq
from hotspotter.helpers import ensure_path, mystats, myprint
import load_data2
import numpy as np
import os, sys
import feature_compute2 as fc2

__DBG_INFO__ = False

# =======================================
# Parallelizable Work Functions          
# =======================================
def compute_chip(img_path, chip_path, roi, new_size):
    ''' 
    Crops chip from image
    Converts to grayscale
    Resizes to standard size
    Equalizes the histogram
    Saves as png
    '''
    # Read image
    img = Image.open(img_path)
    [img_w, img_h] = [ gdim - 1 for gdim in img.size ]
    # Ensure ROI is within bounds
    [roi_x, roi_y, roi_w, roi_h] = [ max(0, cdim) for cdim in roi]
    roi_x2 = min(img_w, roi_x + roi_w)
    roi_y2 = min(img_h, roi_y + roi_h)
    # Crop out ROI: left, upper, right, lower
    raw_chip = img.crop((roi_x, roi_y, roi_x2, roi_y2))
    # Scale chip, but do not rotate
    chip = raw_chip.convert('L').resize(new_size, Image.ANTIALIAS)
    # Preprocessing based on preferences
    chip = histeq(chip)
    # Save chip to disk
    chip.save(chip_path, 'PNG')
    return True

def rotate_chip(chip_path, rchip_path, theta):
    ''' reads chip, rotates, and saves'''
    chip = Image.open(chip_path)
    degrees = theta * 180. / np.pi
    rchip = chip.rotate(degrees, resample=Image.BICUBIC, expand=1)
    rchip.save(rchip_path, 'PNG')

# =======================================
# Main Script 
# =======================================
if __name__ == '__main__':
    # <Support for windows>
    from multiprocessing import freeze_support
    freeze_support()
    # </Support for windows>

    # --- LOAD DATA --- #
    db_dir = load_data2.MOTHERS
    hs_tables = load_data2.load_csv_tables(db_dir)
    exec(hs_tables.execstr('hs_tables'))
    print(hs_tables)

    # --- CREATE COMPUTED DIRS --- #
    img_dir      = db_dir + '/images'
    internal_dir = db_dir + '/.hs_internals'
    internal_sym = db_dir + '/Shortcut-to-hs_internals'
    chip_dir     = internal_dir + '/computed/chips'
    rchip_dir    = internal_dir + '/computed/temp'
    feat_dir     = internal_dir + '/computed/feats'

    ensure_path(internal_dir)
    ensure_path(chip_dir)
    ensure_path(rchip_dir)
    ensure_path(feat_dir)
    
    if not os.path.islink(internal_sym):
        from hotspotter.helpers import symlink
        symlink(internal_dir, internal_sym, noraise=True)

    # --- BUILD TASK INFORMATION --- #
    # Full image path
    cx2_img_path  = [ img_dir+'/'+gx2_gname[gx]   for gx  in cx2_gx ]
  
    # Paths to chip, rotated chip, and chip features 
    cx2_chip_path   = [ chip_dir+'/CID_%d.png'     % cid for cid in cx2_cid]
    cx2_rchip_path  = [rchip_dir+'/CID_%d.rot.png' % cid for cid in cx2_cid]
    cx2_hesaff_path = [ feat_dir+'/CID_%d_hesaff.npz' % cid for cid in cx2_cid]
    cx2_sift_path   = [ feat_dir+'/CID_%d_sift.npz'   % cid for cid in cx2_cid]
    cx2_freak_path  = [ feat_dir+'/CID_%d_freak.npz'  % cid for cid in cx2_cid]

    # Normalized chip size
    At = 500.0 ** 2 # target area
    def _target_resize(w, h):
        ht = np.sqrt(At * h / w)
        wt = w * ht / h
        return (int(round(wt)), int(round(ht)))
    cx2_chip_sz = [_target_resize(float(w), float(h)) for (x,y,w,h) in cx2_roi]
    cx2_imgchip_sz = [(float(w), float(h)) for (x,y,w,h) in cx2_roi]

    if __DBG_INFO__:
        cx2_chip_sf = [(w/wt, h/ht)
                    for ((w,h),(wt,ht)) in zip(cx2_imgchip_sz, cx2_chip_sz)]
        cx2_sf_ave = [(sf1 + sf2) / 2 for (sf1, sf2) in cx2_chip_sf]
        cx2_sf_err = [np.abs(sf1 - sf2) for (sf1, sf2) in cx2_chip_sf]
        myprint(mystats(cx2_sf_ave),lbl='ave scale factor')
        myprint(mystats(cx2_sf_err),lbl='ave scale factor error')

    # --- COMPUTE CHIPS --- # 
    parallel_compute(compute_chip,
                     arg_list=[cx2_img_path,
                               cx2_chip_path,
                               cx2_roi,
                               cx2_chip_sz])
    # --- ROTATE CHIPS --- # 
    parallel_compute(rotate_chip,
                     arg_list=[cx2_chip_path,
                               cx2_rchip_path,
                               cx2_theta])

    # --- COMPUTE FEATURES --- # 
    # Hessian Affine Features
    print('Computing features')
    #parallel_compute(fc2.compute_hesaff, [cx2_rchip_path, cx2_hesaff_path])
    parallel_compute(fc2.compute_sift,   [cx2_rchip_path, cx2_sift_path])
    #parallel_compute(fc2.compute_freak,  [cx2_rchip_path, cx2_freak_path]) 
    # --- LOAD FEATURES --- # 
    print('Loading features')
    #cx2_hesaff_feats = parallel_compute(fc2.load_features, [cx2_hesaff_path])
    cx2_sift_feats  = parallel_compute(fc2.load_features, [cx2_sift_path], 1)
    #cx2_sift_kpts, cx2_sift_desc = \
    #     ([ k for k,d in cx2_sift_feats ], [ d for k,d in cx2_sift_feats ])

    #cx2_freak_feats   = parallel_compute(fc2.load_features, [cx2_freak_path])
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
    xy1_m = kpts1[matches12[:,0],0:2]
    xy2_m = kpts2[matches12[:,1],0:2]

    rchip1 = cv2.imread(cx2_rchip_path[cx1])
    rchip2 = cv2.imread(cx2_rchip_path[cx2])
    # Homogonize and transpose for PCV
    num_m = len(matches12)
    fp = np.hstack([kpts1_m[:,0:2], np.ones((num_m,1))]).T
    tp = np.hstack([kpts2_m[:,0:2], np.ones((num_m,1))]).T
    
    model = homography.RansacModel() 
    H_12 = homography.H_from_ransac(fp,tp,model) #im 1 to 2 


    kpts_img1 = draw_kpts(rchip1, kpts1_m)
    figure(1)
    imshow(kpts_img1)
    kpts_img2 = draw_kpts(rchip1, kpts2_m)
    figure(2)
    imshow(kpts_img2)
    matching_img = draw_matches(rchip1, rchip2, kpts1, kpts2, matches12, draw_vert=False)
    imshow(matching_img)

# adapted from:
# http://jayrambhia.com/blog/sift-keypoint-matching-using-python-opencv/
def draw_matches(rchip1, rchip2, kpts1, kpts2, matches12, draw_vert=False):
    h1, w1 = rchip1.shape[0:2]
    h2, w2 = rchip2.shape[0:2]
    woff = 0; hoff = 0 # offsets 
    if vert: wB = max(w1, w2); hB = h1+h2; hoff = h1
    else:    hB = max(h1, h2); wB = w1+w2; woff = w1
    # Concat images
    match_img = np.zeros((hB, wB, 3), np.uint8)
    match_img[0:h1, 0:w1, :] = rchip1
    match_img[hoff:(hoff+h2), woff:(woff+w2), :] = rchip2
    # Draw lines
    for kx1, kx2 in iter(matches12):
        pt1 = (int(kpts1[kx1,0]), int(kpts1[kx1,1]))
        pt2 = (int(kpts2[kx2,0])+woff, int(kpts2[kx2,1])+hoff)
        match_img = cv2.line(new_img, pt1, pt2, (255, 0, 0))
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
