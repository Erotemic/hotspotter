from helpers import printWARN, printINFO
from warnings import catch_warnings, simplefilter 
import cv2
import numpy.linalg as linalg
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
# skimage.transform
# http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
# skimage.transform.fast_homography(im, H)
def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

# Generate 6 degrees of freedom homography transformation
def compute_homog(x1_m, y1_m, x2_m, y2_m):
    'Computes homography from normalized (0 to 1) point correspondences'
    num_pts = xyz_norm1.shape[1]
    Mbynine = np.zeros((2*num_pts,9), dtype=np.float32)
    for ix in xrange(num_pts): # Loop over inliers
        # Concatinate all 2x9 matrices into an Mx9 matrix
        u2      =     x2_m[ix]
        v2      =     y2_m[ix]
        (d,e,f) =  (   -x1_m[ix],    -y1_m[ix],  -1)
        (g,h,i) =  ( v2*x1_m[ix],  v2*y1_m[ix],  v2)
        (j,k,l) =  (    x1_m[ix],     y1_m[ix],   1)
        (g,h,i) =  (-u2*x1_m[ix], -u2*y1_m[ix], -u2)
        Mbynine[ix*2:(ix+1)*2,:]  = np.array((
            (0, 0, 0, d, e, f, g, h, i),
            (j, k, l, 0, 0, 0, p, q, r)))
    # Solve for the nullspace of the Mbynine
    try:
        (_U, _s, V) = linalg.svd(Mbynine)
    except MemoryError as ex:
        printWARN('Caught MemErr %r during full SVD. Trying sparse SVD.' % (ex))
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
        return None
    # keypoint xy coordinates shape=(dim, num)
    def normalize_xy_points(x_m, y_m):
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
    # Get corresponding points and shapes
    _x1_m, _y1_m, acd1_m = breakup_kpts(kpts1[fm[:, 0]].T)
    _x2_m, _y2_m, acd2_m = breakup_kpts(kpts2[fm[:, 1]].T)
    x1_m, y1_m, T1 = normalize_xy_points(_x1_m, _y1_m)
    x2_m, y2_m, T2 = normalize_xy_points(_x2_m, _y2_m)
    # Estimate affine correspondence
    aff_inliers = __affine_inliers(x1_m, y1_m, acd1_m, 
                                 x2_m, y2_m, acd2_m,
                                 xy_thresh, 
                                 scale_thresh_low,
                                 scale_thresh_high)
    # Cannot find good affine correspondence
    if len(aff_inliers) < min_num_inliers:
        return None
    try: 
        H_prime = compute_homog(xyz_norm1, xyz_norm2)
        # Computes ax = b
        # x = linalg.solve(a, b)
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
    
def breakup_kpts(kpts_5xN):
    _xs   = kpts_5xN[0]
    _ys   = kpts_5xN[1]
    _acds = kpts_5xN[2:5] 
    return _xs, _ys, _acds

def affine_inliers(kpts1, kpts2, fm, xy_thresh, scale_thresh):
    scale_thresh_low  = scale_thresh ** 2
    scale_thresh_high = 1.0 / scale_thresh_low
    # Get matching keypoints (x, y, ellipse_acd)
    x1_m, y1_m, acd1_m = breakup_kpts(kpts1[fm[:, 0]].T)
    x2_m, y2_m, acd2_m = breakup_kpts(kpts2[fm[:, 1]].T)
    inliers, Aff = __affine_inliers

# --------------------------------
# Linear algebra functions on lower triangular matrices
def det_acd(acd):
    'Lower triangular determinant'
    return acd[0] * acd[2]
def inv_acd(acd, det):
    'Lower triangular inverse'
    return np.array((acd[2], -acd[1], acd[0])) / det
def dot_acd(acd1, acd2): 
    'Lower triangular dot product'
    a = (acd1[0] * acd2[0])
    c = (acd1[1] * acd2[0] + acd1[2] * acd2[1])
    d = (acd1[2] * acd2[2])
    return np.array([a, c, d])
def xy_error_acd(x1, y1, x2, y2):
    'Aligned points spatial error'
    return (x1 - x2)**2 + (x1 - y1)**2
def inv_sqrtm_acd(acd):
    eps = 1e-9
    a = acd[0]
    c = acd[1]
    d = acd[2]
    _a = 1.0 / np.sqrt(a) 
    _c = (c / np.sqrt(d) - c / np.sqrt(d)) / (a - d + eps)
    _d = 1.0 / np.sqrt(d)
    return _a, _c, _d


# --------------------------------

def __affine_inliers(x1_m, y1_m, acd1_m,
                     x2_m, y2_m, acd2_m, xy_thresh, 
                     scale_thresh_high, scale_thresh_low):
    'Estimates inliers deterministically using elliptical shapes'
    best_inliers = []
    best_Aff = None
    # Get keypoint scales (determinant)
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    # Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    inv2_m = inv_acd(acd2_m, det2_m)
    Aff_list = dot_acd(inv2_m, acd1_m)
    # Compute scale change of all transformations 
    detAff_list = det_acd(Aff_list)
    # Test all hypothesis 
    for mx in xrange(len(x1_m)): 
        A11 = Aff_list[0,mx]
        A21 = Aff_list[1,mx]
        A22 = Aff_list[2,mx]
        Adet = detAff_list[mx]
        x1_hypoth = x1_m[mx]
        x2_hypoth = x2_m[mx]
        y1_hypoth = y1_m[mx]
        y2_hypoth = y2_m[mx]

        # Transform x1_m -> x1_mt
        x1_mt = x2_hypoth + (A11 * (x1_m - x1_hypoth))
        # Transform y1_m -> y1_mt
        y1_mt = y2_hypoth + (A21 * (x1_m - x1_hypoth)) + (A22 * (y1_m - y1_m[mx]))
        # Get transformed determinant det(A) * det(acd) 
        det1_mt = det1_m * Adet
        # Find xy-(squared)-error 
        xy_err    = xy_error_acd(x1_m, y1_mt, x2_m, y2_m) 
        # Find scale-(squared)-error
        scale_err = det1_mt / det2_m
        # Test xy inliers
        xy_inliers = xy_err < xy_thresh
        # Test scale inliers
        scale_inliers = np.logical_and(scale_err > scale_thresh_low,
                                       scale_err < scale_thresh_high)
        #hypothesis_inliers, = np.where(np.logical_and(xy_inliers, scale_inliers))
        hypothesis_inliers, = np.where(xy_inliers)
        # See if more inliers than previous best
        if len(hypothesis_inliers) > len(best_inliers):
            best_inliers = hypothesis_inliers
            best_Aff       = A_list[:, mx]
    return best_inliers, best_Aff

def test_realdata2():
    import load_data2
    import params
    import draw_func2 as df2
    import helpers
    import spatial_verification2 as sv2
    params.reload_module()
    load_data2.reload_module()
    df2.reload_module()
    df2.reset()
    sv2.reload_module()
    # Parameters
    xy_thresh = params.__XY_THRESH__
    scale_thresh = params.__SCALE_THRESH__
    # Pick out some data
    (hs, qcx, cx, fm, rchip1, rchip2, kpts1, kpts2) =\
            load_data2.get_sv_test_data()
    # Draw Data
    df2.reload_module()
    df2.ELL_COLOR = (1, 0, 0)
    df2.ELL_LINEWIDTH = 2
    df2.ELL_ALPHA = .5
    df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm, fs=None,
                      all_kpts=False, draw_lines=False, doclf=True,
                      title='Assigned matches')
    # Affine matching tests
    scale_thresh_low  = scale_thresh ** 2
    scale_thresh_high = 1.0 / scale_thresh_low
    # Get corresponding keypoints (x, y, ellipse_acd)
    x1_m, y1_m, acd1_m = breakup_kpts(kpts1[fm[:, 0]].T)
    x2_m, y2_m, acd2_m = breakup_kpts(kpts2[fm[:, 1]].T)
    # -----------------------------------------------
    # Get match threshold 10% of matching keypoint extent diagonal
    aff_inliers1 = sv2.affine_inliers(kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh)
    #aff_inliers1 = sv2.__affine_inliers(x1_m, y1_m, x2_m, y2_m, 
                                        #acd1_m, acd2_m, xy_thresh_sqrd, scale_thresh)

    H_prime = spatial_verification.compute_homog(xyz_norm1, xyz_norm2)
    H = linalg.solve(T2, H_prime).dot(T1)                # Unnormalize

    Hdet = linalg.det(H)

    # Estimate final inliers
    acd1_m   = kpts1_m[2:5,:] # keypoint shape matrix [a 0; c d] matches
    acd2_m   = kpts2_m[2:5,:]
    # Precompute the determinant of lower triangular matrix (a*d - b*c); b = 0
    det1_m = acd1_m[0] * acd1_m[2]
    det2_m = acd2_m[0] * acd2_m[2]

    # Matrix Multiply xyacd matrix by H
    # [[A, B, X],      
    #  [C, D, Y],      
    #  [E, F, Z]] 
    # dot 
    # [(a, 0, x),
    #  (c, d, y),
    #  (0, 0, 1)] 
    # = 
    # [(a*A + c*B + 0*E,   0*A + d*B + 0*X,   x*A + y*B + 1*X),
    #  (a*C + c*D + 0*Y,   0*C + d*D + 0*Y,   x*C + y*D + 1*Y),
    #  (a*E + c*F + 0*Z,   0*E + d*F + 0*Z,   x*E + y*F + 1*Z)]
    # =
    # [(a*A + c*B,               d*B,         x*A + y*B + X),
    #  (a*C + c*D,               d*D,         x*C + y*D + Y),
    #  (a*E + c*F,               d*F,         x*E + y*F + Z)]
    # # IF x=0 and y=0
    # =
    # [(a*A + c*B,               d*B,         0*A + 0*B + X),
    #  (a*C + c*D,               d*D,         0*C + 0*D + Y),
    #  (a*E + c*F,               d*F,         0*E + 0*F + Z)]
    # =
    # [(a*A + c*B,               d*B,         X),
    #  (a*C + c*D,               d*D,         Y),
    #  (a*E + c*F,               d*F,         Z)]
    # --- 
    #  A11 = a*A + c*B
    #  A21 = a*C + c*D
    #  A31 = a*E + c*F
    #  A12 = d*B
    #  A22 = d*D
    #  A32 = d*F
    #  A31 = X
    #  A32 = Y
    #  A33 = Z
    #
    # det(A) = A11*(A22*A33 - A23*A32) - A12*(A21*A33 - A23*A31) + A13*(A21*A32 - A22*A31)

    det1_mAt = det1_m * Hdet
    # Check Error in position and scale
    xy_sqrd_err = (x1_mAt - x2_m)**2 + (y1_mAt - y2_m)**2
    scale_sqrd_err = det1_mAt / det2_m
    # Check to see if outliers are within bounds
    xy_inliers = xy_sqrd_err < xy_thresh_sqrd
    s1_inliers = scale_sqrd_err > scale_thresh_low
    s2_inliers = scale_sqrd_err < scale_thresh_high
    _inliers, = np.where(np.logical_and(np.logical_and(xy_inliers, s1_inliers), s2_inliers))

    xy1_mHt = transform_xy(H, xy1_m)                        # Transform Kpts1 to Kpts2-space
    sqrd_dist_error = np.sum( (xy1_mHt - xy2_m)**2, axis=0) # Final Inlier Errors
    inliers = sqrd_dist_error < xy_thresh_sqrd



    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers1], kpts2_m.T[aff_inliers1], title=title, fignum=2, vert=False)
    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers2], kpts2_m.T[aff_inliers2], title=title, fignum=3, vert=False)
    df2.present(wh=(600,400))

