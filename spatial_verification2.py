from helpers import printWARN, printINFO
from warnings import catch_warnings, simplefilter 
import cv2
import numpy.linalg as linalg
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
#skimage.transform
# http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
# skimage.transform.fast_homography(im, H)
def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

# Generate 6 degrees of freedom homography transformation
def compute_homog(xyz_norm1, xyz_norm2):
    'Computes homography from normalized (0 to 1) point correspondences'
    num_pts = xyz_norm1.shape[1]
    Mbynine = np.zeros((2*num_pts,9), dtype=np.float32)
    for ilx in xrange(num_pts): # Loop over inliers
        # Concatinate all 2x9 matrices into an Mx9 matrix
        u2      =     xyz_norm2[0,ilx]
        v2      =     xyz_norm2[1,ilx]
        (d,e,f) =    -xyz_norm1[:,ilx]
        (g,h,i) =  v2*xyz_norm1[:,ilx]
        (j,k,l) =     xyz_norm1[:,ilx]
        (p,q,r) = -u2*xyz_norm1[:,ilx]
        Mbynine[ilx*2:(ilx+1)*2,:]  = np.array(\
            [(0, 0, 0, d, e, f, g, h, i),
             (j, k, l, 0, 0, 0, p, q, r) ] )
    # Solve for the nullspace of the Mbynine
    try:
        (_U, _s, V) = linalg.svd(Mbynine)
    except MemoryError as ex:
        printWARN('warning 35 ransac:'+repr(ex))
        print('Singular Value Decomposition Ran Out of Memory. Trying with a sparse matrix')
        MbynineSparse = sparse.lil_matrix(Mbynine)
        (_U, _s, V) = sparse_linalg.svds(MbynineSparse)
    # Rearange the nullspace into a homography
    h = V[-1] # (transposed in matlab)
    H = np.vstack( ( h[0:3],  h[3:6],  h[6:9]  ) )
    return H

def homography_inliers(kpts1_m,
                       kpts2_m,
                       xy_thresh=1,
                       scale_thresh=2.0,
                       min_num_inliers=4):

    if not 'xy_thresh' in vars():
        xy_thresh = .05
    if not 'scale_thresh' in vars():
        scale_thresh = .05
    if not 'min_num_inliers' in vars():
        min_num_inliers = 4
    scale_thresh_high = scale_thresh_factor ** 2
    scale_thresh_low  = 1.0/scale_thresh_high
    # Not enough data
    if kpts1_m.shape[1] < min_num_inliers or kpts1_m.shape[1] == 0: 
        return  None
    # keypoint xy coordinates shape=(dim, num)
    def normalize_points(x_m, y_m):
        'Returns a transformation to normalize points to mean=0, stddev=1'
        mean_x = x_m.mean() # center of mass
        mean_y = y_m.mean()
        sx = 1 /x_m.std()   # average xy magnitude
        sy = 1 / y_m.std()
        tx = -mean_x * sx
        ty = -mean_y * sy
        T = np.array([(sx, 0, tx),
                    (0, sy, ty),
                    (0,  0,  1)])
        x_norm = (x_m - mean_x) * sx
        y_norm = (y_m - mean_y) * sy
        return x_norm, y_norm, T
    x1_m, y1_m, T1 = normalize_points(kpts1_m[0], kpts1_m[1])
    x2_m, y2_m, T2 = normalize_points(kpts2_m[0], kpts2_m[1])
    # keypoint ellipses matrix [a 0; c d]
    acd1_m = kpts1_m[2:5] 
    acd2_m = kpts2_m[2:5]
    # Estimate affine correspondence
    aff_inliers = affine_inliers(x1_m, y1_m, acd1_m, 
                                 x2_m, y2_m, acd2_m,
                                 xy_thresh, 
                                 scale_thresh_low,
                                 scale_thresh_high)
    # Cannot find good affine correspondence
    if len(aff_inliers) < min_num_inliers:
        return np.eye(3), aff_inliers


    try: 
        H_prime = compute_homog(xyz_norm1, xyz_norm2)
        H = linalg.solve(T2, H_prime).dot(T1)                # Unnormalize
    except linalg.LinAlgError as ex:
        printWARN('Warning 285 '+repr(ex), )
        return np.eye(3), aff_inliers

    # Estimate final inliers
    xy1_mHt = transform_xy(H, xy1_m)                        # Transform Kpts1 to Kpts2-space
    sqrd_dist_error = np.sum( (xy1_mHt - xy2_m)**2, axis=0) # Final Inlier Errors
    inliers = sqrd_dist_error < xy_thresh_sqrd
    return H, inliers

# This new function is much faster .035 vs .007
    # EXPLOITS LOWER TRIANGULAR MATRIXES
    # Precompute the determinant of matrix 2 (a*d - b*c), but b = 0
    # Need the inverse of acd2_m:  1/det * [(d, -b), (-c, a)]
    # Precompute lower triangular affine tranforms inv2_m (dot) acd1_m
    # [(a2*a1), (c2*a1+d2*c1), (d2*d1)]
def affine_inliers(x1_m, y1_m, acd1_m,
                   x2_m, y2_m, acd2_m, 
                   xy_thresh, 
                   scale_thresh_high,
                   scale_thresh_low):
    '''Estimates inliers deterministically using elliptical shapes'''
    def det_acd(acd):
        return acd[0] * acd[2]
    def inv_acd(acd, det):
        return np.array((acd[2], -acd[1], acd[0])) / det
    def dot_acd(acd1, acd2): 
        a = (acd1[0] * acd2[0])
        c = (acd1[1] * acd2[0] + acd1[2] * acd2[1])
        d = (acd1[2] * acd2[2])
        return np.array([a, c, d])
    def xy_error_acd(x1, y1, x2, y2):
        return (x1 - x2)**2 + (x1 - y1)**2
    best_inliers = []
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    inv2_m = inv_acd(acd2_m, det2_m)
    A = dot_acd(inv2_m, acd1_m)
    Adet = det_acd(A)
    # Enumerate All Hypothesis (Match transformations)
    mx = 1
    for mx in xrange(len(x1_m)): 
        A11 = A[0,mx]
        A21 = A[1,mx]
        A22 = A[2,mx]
        # Translate x1_m -> x1_mt
        x1_mt = x2_m[mx] + (A11 * (x1_m - x1_m[mx]))
        y1_mt = y2_m[mx] + (A21 * (x1_m - x1_m[mx])) + (A22 * (y1_m - y1_m[mx]))
        # Get transformed determinant det(A) * det(acd) 
        det1_mt = det1_m * Adet[mx]
        # Check Error in position and scale (most of these are squared)
        xy_err    = xy_error_acd(x1_m, y1_mt, x2_m, y2_m) 
        scale_err = det1_mt / det2_m
        # Test xy and scale inliers
        xy_inliers = xy_err < xy_thresh
        #scale_inliers = np.logical_and(scale_err > scale_thresh_low,
                                       #scale_err < scale_thresh_high)
        #hypothesis_inliers, = np.where(np.logical_and(xy_inliers, scale_inliers))
        hypothesis_inliers, = np.where(xy_inliers)
        # See if more inliers than previous best
        if len(hypothesis_inliers) > len(best_inliers):
            best_inliers = hypothesis_inliers
    return best_inliers

