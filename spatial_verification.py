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
    #assert xyz_norm1.shape == xyz_norm2.shape, ''
    #assert xyz_norm1.shape[0] == 3, ''
    Mbynine = np.zeros((2*num_pts,9), dtype=np.float32)
    for ilx in xrange(num_pts): # Loop over inliers
        # Concatinate all 2x9 matrices into an Mx9 matrcx
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
        # sparse seems to be 4 times slower
        # the SVD itself is actually 37 times slower
        print('Singular Value Decomposition Ran Out of Memory. Trying with a sparse matrix')
        MbynineSparse = sparse.lil_matrix(Mbynine)
        (_U, _s, V) = sparse_linalg.svds(MbynineSparse)
    # Rearange the nullspace into a homography
    h = V[-1,:] # (transposed in matlab)
    H = np.vstack( ( h[0:3],  h[3:6],  h[6:9]  ) )
    return H
# 
def _homogonize_pts(xy):
    'Adds a 3rd dimension of ones to xy-position vectors'
    #assert xy.shape[0] == 2, ''
    xyz = np.vstack([xy, np.ones(xy.shape[1])]);
    return xyz
#
def _normalize_pts(xyz):
    'Returns a transformation to normalize points to mean=0, stddev=1'
    num_xyz = xyz.shape[1]
    com = np.sum(xyz,axis=1) / num_xyz # center of mass
    with catch_warnings():
        simplefilter("ignore")
        sx  = num_xyz / np.sum(abs(xyz[0,:]-com[0]))  # average xy magnitude
        sy  = num_xyz / np.sum(abs(xyz[1,:]-com[1])) 
    tx  = -com[0]*sx
    ty  = -com[1]*sy
    T = np.array([(sx, 0, tx), (0, sy, ty), (0, 0, 1)])
    return T
#
def homogo_normalize_pts(xy):
    'Homoginize points for stable homography estimation'
    xyz = _homogonize_pts(xy)
    T   = _normalize_pts(xyz)
    xyz_norm = T.dot(xyz)
    return (xyz_norm, T)

# This new function is much faster .035 vs .007
def aff_inliers_from_ellshape2(kpts1_m, kpts2_m, xy_thresh_sqrd):
    '''Estimates inliers deterministically using elliptical shapes'''
    # EXPLOITS LOWER TRIANGULAR MATRIXES
    best_inliers = []
    x1_m    = kpts1_m[0,:] # keypoint xy coordinates matches
    y1_m    = kpts1_m[1,:] # keypoint xy coordinates matches
    x2_m    = kpts2_m[0,:]
    y2_m    = kpts2_m[1,:]
    acd1_m   = kpts1_m[2:5,:] # keypoint shape matrix [a 0; c d] matches
    acd2_m   = kpts2_m[2:5,:]
    # Precompute the determinant of matrix 2 (a*d - b*c), but b = 0
    det1_m = acd1_m[0] * acd1_m[2]
    det2_m = acd2_m[0] * acd2_m[2]
    # Need the inverse of acd2_m:  1/det * [(d, -b), (-c, a)]
    inv2_m = np.array((acd2_m[2], -acd2_m[1], acd2_m[0])) / det2_m
    # Precompute lower triangular affine tranforms inv2_m (dot) acd1_m
    # [(a2*a1), (c2*a1+d2*c1), (d2*d1)]
    H_aff12_a = (inv2_m[0] * acd1_m[0])
    H_aff12_c = (inv2_m[1] * acd1_m[0] + inv2_m[2] * acd1_m[1])
    H_aff12_d = (inv2_m[2] * acd1_m[2])
    H_det_list = H_aff12_a * H_aff12_d
    scale_thresh_high = 2.0 ** 2
    scale_thresh_low  = 1.0/scale_thresh_high
    # Enumerate All Hypothesis (Match transformations)
    for mx in xrange(len(x1_m)): 
        x1 = x1_m[mx]
        y1 = y1_m[mx]
        x2 = x2_m[mx]
        y2 = y2_m[mx]
        Ha = H_aff12_a[mx]
        Hc = H_aff12_c[mx]
        Hd = H_aff12_d[mx]
        Hdet = H_det_list[mx]
        # Translate and transform xy-positions using H_aff12
        x1_mAt = x2 + (H_aff12_a * (x1_m - x1))
        y1_mAt = y2 + (H_aff12_c * (x1_m - x1)) + (H_aff12_d * (y1_m - y1))
        # Get transformed determinant
        det1_mAt = det1_m * Hdet
        # Check Error in position and scale
        xy_sqrd_err = (x1_mAt - x2_m)**2 + (y1_mAt - y2_m)**2
        scale_sqrd_err = det1_mAt / det2_m
        # Check to see if outliers are within bounds
        xy_inliers = xy_sqrd_err < xy_thresh_sqrd
        s1_inliers = scale_sqrd_err > scale_thresh_low
        s2_inliers = scale_sqrd_err < scale_thresh_high
        _inliers, = np.where(np.logical_and(np.logical_and(xy_inliers, s1_inliers), s2_inliers))
        # See if more inliers than previous best
        if len(_inliers) > len(best_inliers):
            best_inliers = _inliers
    return best_inliers

def aff_inliers_from_ellshape(kpts1_m, kpts2_m, xy_thresh_sqrd):
    '''Estimates inliers deterministically using elliptical shapes'''
    best_inliers = []
    xy1_m    = kpts1_m[0:2,:] # keypoint xy coordinates matches
    xy2_m    = kpts2_m[0:2,:]
    acd1_m   = kpts1_m[2:5,:] # keypoint shape matrix [a 0; c d] matches
    acd2_m   = kpts2_m[2:5,:]
    # Enumerate All Hypothesis (Match transformations)
    num_m = xy1_m.shape[1]
    for mx in xrange(num_m): 
        xy1  = xy1_m[:,mx].reshape(2,1) #  XY Positions
        xy2  = xy2_m[:,mx].reshape(2,1) 
        A1   = np.insert(acd1_m[:,mx], [1.], 0.).reshape(2,2)
        A2   = np.insert(acd2_m[:,mx], [1.], 0.).reshape(2,2)
        # Compute Affine Tranform 
        # from img1 to img2 = (E2\E1) 
        H_aff12  = linalg.inv(A2).dot(A1)
        # Translate and transform XY-Positions
        xy1_mAt = xy2 + H_aff12.dot( (xy1_m - xy1) ) 
        xy_err_sqrd = sum( np.power(xy1_mAt - xy2_m, 2) , 0)
        _inliers, = np.where(xy_err_sqrd < xy_thresh_sqrd)
        # See if more inliers than previous best
        if len(_inliers) > len(best_inliers):
            best_inliers = _inliers
    return best_inliers


def transform_xy(H3x3, xy):
    xyz = _homogonize_pts(xy)
    H_xyz = H3x3.dot(xyz)
    with catch_warnings():
        simplefilter("ignore")
        H_xy = H_xyz[0:2,:] / H_xyz[2,:]
    return H_xy

def H_homog_from_CV2SAC(kpts1_m, kpts2_m, xy_thresh_sqrd):
    xy1_m   = kpts1_m[0:2,:]
    xy2_m   = kpts2_m[0:2,:]

    #method = 0 # a regular method using all the points
    #method = cv2.LMEDS # Least-Median robust method
    method = cv2.RANSAC # RANSAC-based robust method

    H, inliers = cv2.findHomography(xy1_m.T, xy2_m.T, method, np.sqrt(xy_thresh_sqrd))
    H = H if not H is None else np.eye(3)
    return H, np.array(inliers, dtype=bool).flatten()

def H_homog_from_DELSAC(kpts1_m, kpts2_m, xy_thresh_sqrd):
    ' Deterministic Elliptical Sample Consensus'
    return __H_homog_from(kpts1_m, kpts2_m, xy_thresh_sqrd, aff_inliers_from_ellshape2)

def __H_homog_from(kpts1_m, kpts2_m, xy_thresh_sqrd, func_aff_inlier):
    ''' RanSaC: 
        Ransom Sample Consensus Inlier Generator 
        - Object retrieval fast, Philbin1, Chum1, et al 
        input: matching 
    '''
    assert kpts1_m.shape[1] == kpts2_m.shape[1], 'RanSaC works on matches!'
    assert kpts1_m.shape[0] == 5 and kpts2_m.shape[0] == 5, 'RanSaC works on ellipses!'

    num_m = kpts1_m.shape[1] # num matches

    # min number of matches to compute transform
    min_num_inliers = 3 
    if num_m < min_num_inliers or num_m == 0: 
        return  None

    # Estimate initial inliers with some RANSAC variant
    aff_inliers = func_aff_inlier(kpts1_m, kpts2_m, xy_thresh_sqrd)

    # If we cannot estimate a good correspondence 
    if len(aff_inliers) < min_num_inliers:
        return np.eye(3), aff_inliers

    # Homogonize+Normalize
    xy1_m    = kpts1_m[0:2,:] 
    xy2_m    = kpts2_m[0:2,:]
    (xyz_norm1, T1) = homogo_normalize_pts(xy1_m[:,aff_inliers]) 
    (xyz_norm2, T2) = homogo_normalize_pts(xy2_m[:,aff_inliers])

    # Compute Normalized Homog
    #__AFFINE_OVERRIDE__ = False
    #if __AFFINE_OVERRIDE__:
        #printINFO('Affine Override')
        ##src = _homogonize_pts(xy1_m[:,aff_inliers])
        ##dst = _homogonize_pts(xy2_m[:,aff_inliers])
        ##H_ = H_affine_from_points(src, dst)
        #src = np.float32(xy1_m[:,aff_inliers].T)
        #dst = np.float32(xy2_m[:,aff_inliers].T)
        #fullAffine = True
        #H_2x3 = cv2.estimateRigidTransform(src[0:3,:], dst[0:3,:], fullAffine)
        #if H_2x3 == None:
            #H_2x3 = np.array(((1.,0.,0.),(0.,1.,0.)))

        #H = np.vstack([H_2x3, ([0.,0.,1.],)])
    #else: 
        # H = cv2.getPerspectiveTransform(xy1_m[:,aff_inliers], xy2_m[:,aff_inliers])
    try: 
        H_prime = compute_homog(xyz_norm1, xyz_norm2)
        H = linalg.solve(T2, H_prime).dot(T1)                # Unnormalize
    except linalg.LinAlgError as ex:
        printWARN('Warning 285 '+repr(ex), )
        return np.eye(3), aff_inliers

    # Estimate final inliers
    acd1_m   = kpts1_m[2:5,:] # keypoint shape matrix [a 0; c d] matches
    acd2_m   = kpts2_m[2:5,:]
    # Precompute the determinant of matrix 2 (a*d - b*c), but b = 0
    det1_m = acd1_m[0] * acd1_m[2]
    det2_m = acd2_m[0] * acd2_m[2]

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
    return H, inliers

def test():
    weird_A = np.array([(2.7, 4.2, 10),
                        (1.7, 3.1, 20),
                        (2.2, 2.4, 1.1)])

    num_mat = 2000
    frac_in = .5
    num_inl = int(num_mat * frac_in)
    kpts1_m = np.random.rand(5, num_mat)
    kpts2_m = np.random.rand(5, num_mat)

    kpts2_cor = kpts1_m[:, 0:num_inl]

    x_cor = kpts2_cor[0]
    y_cor = kpts2_cor[1]
    z_cor = np.ones(num_inl)
    a_cor = kpts2_cor[2]
    zeros = np.zeros(num_inl)
    c_cor = kpts2_cor[3]
    d_cor = kpts2_cor[4]

    kpts2_mats = np.array([(a_cor, zeros, x_cor),
                           (c_cor, d_cor, y_cor), 
                           (zeros, zeros, z_cor)]).T

    # Do some weird transform on some keypoints
    import scipy.linalg
    for x in xrange(num_inl):
        mat = weird_A.dot(kpts2_mats[x].T)
        mat /= mat[2,2]
        kpts2_m[0,x] = mat[0,2]
        kpts2_m[1,x] = mat[1,2]
        kpts2_m[2,x] = mat[0,0]
        kpts2_m[3,x] = mat[1,0]
        kpts2_m[4,x] = mat[1,2]

    # Find some inliers? 
    xy_thresh_sqrd = 7
    H, inliers = H_homog_from_DELSAC(kpts1_m, kpts2_m, xy_thresh_sqrd)


if __name__ == '__main__':
    print 'Testing spatial_verification.py'
    print test_realdata()

def test_realdata():
    import load_data2
    import params
    import drawing_functions2 as df2
    import helpers
    import spatial_verification
    #params.reload_module()
    #load_data2.reload_module()
    #df2.reload_module()

    db_dir = load_data2.MOTHERS
    hs = load_data2.HotSpotter(db_dir)
    assign_matches = hs.matcher.assign_matches
    qcx = 0
    cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    # Get chips
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    # Get feature matches 
    kpts1_m = kpts1[fm[:, 0], :].T
    kpts2_m = kpts2[fm[:, 1], :].T
    # -----------------------------------------------
    # Get match threshold 10% of matching keypoint extent diagonal
    xy_thresh = params.__XY_THRESH__
    img1_extent = (kpts1_m[0:2, :].max(1) - kpts1_m[0:2, :].min(1))[0:2]
    xy_thresh_sqrd = np.sum(img1_extent**2) * (xy_thresh**2)
    
    title='(qx%r v cx%r)\n #match=%r' % (qcx, cx, len(fm))
    df2.show_matches2(rchip1, rchip2, kpts1,  kpts2, fm, fs, title=title)

    np.random.seed(6)
    subst = helpers.random_indexes(len(fm),len(fm))
    kpts1_m = kpts1[fm[subst, 0], :].T
    kpts2_m = kpts2[fm[subst, 1], :].T

    df2.reload_module()
    df2.SHOW_LINES = True
    df2.ELL_LINEWIDTH = 2
    df2.LINE_ALPHA = .5
    df2.ELL_ALPHA  = 1
    df2.reset()
    df2.show_keypoints(rchip1, kpts1_m.T, fignum=0, plotnum=121)
    df2.show_keypoints(rchip2, kpts2_m.T, fignum=0, plotnum=122)
    df2.show_matches2(rchip1, rchip2, kpts1_m.T,  kpts2_m.T, title=title, fignum=1, vert=False)

    spatial_verification.reload_module()
    with helpers.Timer():
        best_inliers1 = spatial_verification.aff_inliers_from_ellshape2(kpts1_m, kpts2_m, xy_thresh_sqrd)
    with helpers.Timer():
        best_inliers2 = spatial_verification.aff_inliers_from_ellshape(kpts1_m, kpts2_m, xy_thresh_sqrd)

    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers1], kpts2_m.T[best_inliers1], title=title, fignum=2, vert=False)
    df2.show_matches2(rchip1, rchip2, kpts1_m.T[best_inliers2], kpts2_m.T[best_inliers2], title=title, fignum=3, vert=False)
    df2.present(wh=(600,400))

def test_realdata2():
    import load_data2
    import params
    import drawing_functions2 as df2
    import helpers
    import spatial_verification
    #params.reload_module()
    #load_data2.reload_module()
    #df2.reload_module()

    db_dir = load_data2.MOTHERS
    hs = load_data2.HotSpotter(db_dir)
    assign_matches = hs.matcher.assign_matches
    qcx = 0
    cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    # Get chips
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    # Get feature matches 
    kpts1_m = kpts1[fm[:, 0], :].T
    kpts2_m = kpts2[fm[:, 1], :].T
    # -----------------------------------------------
    # Get match threshold 10% of matching keypoint extent diagonal
    xy_thresh = params.__XY_THRESH__
    img1_extent = (kpts1_m[0:2, :].max(1) - kpts1_m[0:2, :].min(1))[0:2]
    xy_thresh_sqrd = np.sum(img1_extent**2) * (xy_thresh**2)
    
    title='(qx%r v cx%r)\n #match=%r' % (qcx, cx, len(fm))
    df2.show_matches2(rchip1, rchip2, kpts1,  kpts2, fm, fs, title=title)

    np.random.seed(6)
    subst = helpers.random_indexes(len(fm),len(fm))
    kpts1_m = kpts1[fm[subst, 0], :].T
    kpts2_m = kpts2[fm[subst, 1], :].T

    df2.reload_module()
    df2.SHOW_LINES = True
    df2.ELL_LINEWIDTH = 2
    df2.LINE_ALPHA = .5
    df2.ELL_ALPHA  = 1
    df2.reset()
    df2.show_keypoints(rchip1, kpts1_m.T, fignum=0, plotnum=121)
    df2.show_keypoints(rchip2, kpts2_m.T, fignum=0, plotnum=122)
    df2.show_matches2(rchip1, rchip2, kpts1_m.T,  kpts2_m.T, title=title, fignum=1, vert=False)

    spatial_verification.reload_module()
    with helpers.Timer():
        aff_inliers1 = spatial_verification.aff_inliers_from_ellshape2(kpts1_m, kpts2_m, xy_thresh_sqrd)
    with helpers.Timer():
        aff_inliers2 = spatial_verification.aff_inliers_from_ellshape(kpts1_m, kpts2_m, xy_thresh_sqrd)

    # Homogonize+Normalize
    xy1_m    = kpts1_m[0:2,:] 
    xy2_m    = kpts2_m[0:2,:]
    (xyz_norm1, T1) = spatial_verification.homogo_normalize_pts(xy1_m[:,aff_inliers1]) 
    (xyz_norm2, T2) = spatial_verification.homogo_normalize_pts(xy2_m[:,aff_inliers1])

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

