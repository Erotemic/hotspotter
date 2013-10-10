from __future__ import division
from helpers import printWARN, printINFO
import warnings
import cv2
import numpy.linalg as linalg
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
'''skimage.transform
http://stackoverflow.com/questions/11462781/
fast-2d-rigid-body-transformations-in-numpy-scipy
skimage.transform.fast_homography(im, H)'''

def reload_module():
    import imp, sys
    print('[sv1] Reloading: '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

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
        (U, S, Vct) = linalg.svd(Mbynine)
    except MemoryError as ex:
        warnings.warn('warning 35 ransac:'+repr(ex), category=UserWarning)
        # sparse seems to be 4 times slower
        # the SVD itself is actually 37 times slower
        print('Singular Value Decomposition Ran Out of Memory. Trying with a sparse matrix')
        MbynineSparse = sparse.lil_matrix(Mbynine)
        (U, S, Vct) = sparse_linalg.svds(MbynineSparse)
    # Rearange the nullspace into a homography
    h = Vct[-1]
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #OLD WAY WAS WRONG. NOT STD
        '''
        sx  = num_xyz / np.sum(abs(xyz[0,:]-com[0]))  # average xy magnitude
        sy  = num_xyz / np.sum(abs(xyz[1,:]-com[1])) 
        '''
        sx  = 1 / np.sqrt(np.sum(abs(xyz[0,:]-com[0])**2)/num_xyz)  # average xy magnitude
        sy  = 1 / np.sqrt(np.sum(abs(xyz[1,:]-com[1])**2)/num_xyz)
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
    best_Aff = (1, 0, 1, 0, 0, 0)
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
    scale_thresh_high = 2.0
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
        x1_mAt = x2 + (Ha * (x1_m - x1))
        y1_mAt = y2 + (Hc * (x1_m - x1)) + (Hd * (y1_m - y1))
        # Get transformed determinant
        det1_mAt = det1_m * Hdet
        # Check Error in position and scale
        xy_sqrd_err = (x1_mAt - x2_m)**2 + (y1_mAt - y2_m)**2
        scale_sqrd_err = det1_mAt / det2_m
        # Check to see if outliers are within bounds
        xy_inliers = xy_sqrd_err < xy_thresh_sqrd
        s1_inliers = scale_sqrd_err > scale_thresh_low
        s2_inliers = scale_sqrd_err < scale_thresh_high
        _inliers, = np.where(
            np.logical_and(xy_inliers,
                           np.logical_and(s2_inliers, s1_inliers)))
        # See if more inliers than previous best
        if len(_inliers) > len(best_inliers):
            best_inliers = _inliers
            best_Aff = (Ha, Hc, Hd, x1, y1, x2, y2)
    (Ha, Hc, Hd, x1, y1, x2, y2) = best_Aff
    best_Aff = np.array([(Ha,  0,  x2-Ha*x1      ),
                         (Hc, Hd,  y2-Hc*x1-Hd*y1),
                         ( 0,  0,               1)])
    return best_Aff, best_inliers

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
            best_Aff = H_aff12
    return best_inliers, best_Aff


def transform_xy(H3x3, xy):
    xyz = _homogonize_pts(xy)
    H_xyz = H3x3.dot(xyz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

def H_homog_from_DELSAC(kpts1_m, kpts2_m,
                        xy_thresh,
                        scale_thresh_high,
                        scale_thresh_low):
    ' Deterministic Elliptical Sample Consensus'
    #=====
    # BUG
    FIX = True
    if not FIX:
        img1_extent = (kpts1_m[0:2, :].max(1) - kpts1_m[0:2, :].min(1))[0:2]
        xy_thresh_sqrd = np.sum(img1_extent**2) * xy_thresh
    else: # FIX
        img2_extent = (kpts2_m[0:2, :].max(1) - kpts2_m[0:2, :].min(1))[0:2]
        diag_len = np.sum(np.array(img2_extent, dtype=np.float64)**2)
        xy_thresh_sqrd = diag_len * xy_thresh
    #=====
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
    Aff, aff_inliers = func_aff_inlier(kpts1_m, kpts2_m, xy_thresh_sqrd)


    # If we cannot estimate a good correspondence 
    if len(aff_inliers) < min_num_inliers:
        return np.eye(3), aff_inliers
    # Homogonize+Normalize
    xy1_m = kpts1_m[0:2, :] 
    xy2_m = kpts2_m[0:2, :]
    (xyz_norm1, T1) = homogo_normalize_pts(xy1_m[:, aff_inliers]) 
    (xyz_norm2, T2) = homogo_normalize_pts(xy2_m[:, aff_inliers])

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
        warnings.warn('Warning 285 '+repr(ex), category=UserWarning)
        return np.eye(3), aff_inliers

    # Estimate final inliers
    xy1_mHt = transform_xy(H, xy1_m)                        # Transform Kpts1 to Kpts2-space
    sqrd_dist_error = np.sum( (xy1_mHt - xy2_m)**2, axis=0) # Final Inlier Errors
    inliers = sqrd_dist_error < xy_thresh_sqrd
    return H, inliers, Aff, aff_inliers

def compare():
    sv1.reload_module()
    sv2.reload_module()
    kpts1_m = kpts1[fm[:, 0], :].T
    kpts2_m = kpts2[fm[:, 1], :].T
    with helpers.Timer('sv1') as t: 
        hinlier_tup1 = sv1.H_homog_from_DELSAC(kpts1_m, kpts2_m,
                                               xy_thresh, 
                                               scale_thresh_high,
                                               scale_thresh_low)
    with helpers.Timer('sv2') as t: 
        hinlier_tup2 = sv2.homography_inliers(kpts1, kpts2, fm,
                                              xy_thresh, 
                                              scale_thresh_high,
                                              scale_thresh_low)
    
    H1, inliers1, Aff1, aff_inliers1 = hinlier_tup1
    H2, inliers2, Aff2, aff_inliers2 = hinlier_tup2
    print('Aff1=\n%r' % Aff1)
    print('Aff2=\n%r' % Aff2)
    print('num aff_inliers sv1: %r ' % len(aff_inliers1))
    print('num aff_inliers sv2: %r ' % len(aff_inliers2))
    print('num inliers sv1: %r ' % inliers1.sum())
    print('num inliers sv2: %r ' % len(inliers2))
    print('H1=\n%r' % H1)
    print('H2=\n%r' % H2)
    args_ = [rchip1, rchip2, kpts1, kpts2]
    df2.show_matches2(*args_+[fm[aff_inliers1]], fignum=1, title='sv1 affine')
    df2.show_matches2(*args_+[fm[inliers1]],     fignum=2, title='sv1 homog')
    df2.show_matches2(*args_+[fm[aff_inliers2]], fignum=3, title='sv2 affine')
    df2.show_matches2(*args_+[fm[inliers2]],     fignum=4, title='sv2 homog')
    df2.present(num_rc=(2,2), wh=(800,500))

if __name__ == '__main__':
    print 'Testing spatial_verification.py'
