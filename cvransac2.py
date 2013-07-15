import numpy as np
from numpy import linalg
from warnings import catch_warnings, simplefilter 
import scipy.sparse
import scipy.sparse.linalg

#skimage.transform
# http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
# skimage.transform.fast_homography(im, H)

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
    except MemoryError:
        # TODO: is sparse calculation faster than not?
        # sparse seems to be 4 times slower
        # the SVD itself is actually 37 times slower
        print('Singular Value Decomposition Ran Out of Memory. Trying with a sparse matrix')
        MbynineSparse = scipy.sparse.lil_matrix(Mbynine)
        (_U, _s, V) = scipy.sparse.linalg.svds(MbynineSparse)
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
#
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
        # Transform XY-Positions
        xy1_mAt = xy2 + H_aff12.dot( (xy1_m - xy1) ) 
        xy_err_sqrd = sum( np.power(xy1_mAt - xy2_m, 2) , 0)
        _inliers, = np.where(xy_err_sqrd < xy_thresh_sqrd)
        # See if more inliers than previous best
        if len(_inliers) > len(best_inliers):
            best_inliers = _inliers
    return best_inliers

def aff_inliers_from_randomsac(kpts1_m, kpts2_m, xy_thresh_sqrd, nIter=500, nSamp=3):
    best_inliers = []
    xy1_m    = kpts1_m[0:2,:] # keypoint xy coordinates matches
    xy2_m    = kpts2_m[0:2,:]
    num_m = xy1_m.shape[1]
    match_indexes = np.arange(0,num_m)
    xyz1_m = _homogonize_pts(xy1_m)
    xyz2_m = _homogonize_pts(xy2_m)
    for iterx in xrange(nIter):
        np.random.shuffle(match_indexes)
        selx = match_indexes[:nSamp]
        fp = xyz1_m[:,selx]
        tp = xyz2_m[:,selx]
        H_aff12  = H_affine_from_points(fp, tp)
        # Transform XY-Positions
        xyz1_mAt = H_aff12.dot(xyz1_m)  
        xy_err_sqrd = sum( np.power(xyz1_mAt - xyz2_m, 2) , 0)
        _inliers, = np.where(xy_err_sqrd < xy_thresh_sqrd)
        # See if more inliers than previous best
        if len(_inliers) > len(best_inliers):
            best_inliers = _inliers
    return best_inliers

# From PCV (python computer vision) code
def H_homog_from_points(fp,tp):
    """ Find homography H, such that fp is mapped to tp using the 
        linear DLT method. Points are conditioned automatically.  """
    # condition points (important for numerical reasons)
    # --from points--
    fp_mean = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0:2,2] = -fp_mean[0:2] / maxstd
    fp = np.dot(C1,fp)
    # --to points--
    tp_mean = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0:2,2] = -tp_mean[0:2] / maxstd
    tp = np.dot(C2,tp)
    # create matrix for linear method, 2 rows for each correspondence pair
    num_matches = fp.shape[1]
    A = np.zeros((2*num_matches,9))
    for i in xrange(num_matches):        
        A[2*i] =   [        -fp[0][i],         -fp[1][i],       -1,
                                    0,                 0,        0,
                    tp[0][i]*fp[0][i], tp[0][i]*fp[1][i], tp[0][i]]

        A[2*i+1] = [                0,                 0,        0,
                            -fp[0][i],         -fp[1][i],       -1,
                    tp[1][i]*fp[0][i], tp[1][i]*fp[1][i], tp[1][i]]
    U,S,V = linalg.svd(A)
    H = V[8].reshape((3,3))    
    # decondition
    H = np.dot(linalg.inv(C2),np.dot(H,C1))
    # normalize and return
    return H / H[2,2]

# From PCV (python computer vision) code
def H_affine_from_points(fp,tp):
    """ Find H, affine transformation, such that tp is affine transf of fp. """
    # condition points
    # --from points--
    fp_mean = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0:2,2] = -fp_mean[0:2] / maxstd
    fp_cond = np.dot(C1,fp)
    # --to points--
    tp_mean = np.mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0:2,2] = -tp_mean[0:2] / maxstd
    tp_cond = np.dot(C2, tp)
    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)
    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    tmp2 = np.concatenate((C.dot(linalg.pinv(B)),np.zeros((2,1))), axis=1) 
    H = np.vstack((tmp2,[0,0,1]))
    # decondition
    H = (linalg.inv(C2)).dot(H.dot(C1))
    return H / H[2,2]

def transform_xy(H3x3, xy):
    xyz = _homogonize_pts(xy)
    H_xyz = H3x3.dot(xyz)
    with catch_warnings():
        simplefilter("ignore")
        H_xy = H_xyz[0:2,:] / H_xyz[2,:]
    return H_xy

def H_homog_from_PCVSAC(kpts1_m, kpts2_m, xy_thresh_sqrd):
    'Python Computer Visions Random Sample Consensus'
    from PCV.geometry import homography
    # Get xy points
    xy1_m = kpts1_m[0:2,:] 
    xy2_m = kpts2_m[0:2,:] 
    # Homogonize points
    fp = np.vstack([xy1_m, np.ones((1,xy1_m.shape[1]))])
    tp = np.vstack([xy2_m, np.ones((1,xy2_m.shape[1]))])
    # Get match threshold 10% of image diagonal
    # Get RANSAC inliers
    model = homography.RansacModel() 
    try: 
        H, inliers = homography.H_from_ransac(fp,tp,model, 500, np.sqrt(xy_thresh_sqrd))
    except ValueError as ex:
        print(ex)
        H = np.eye(3)
        inliers = []
    return H, inliers

def H_homog_from_DELSAC(kpts1_m, kpts2_m, xy_thresh_sqrd):
    ' Deterministic Elliptical Sample Consensus'
    return __H_homog_from(kpts1_m, kpts2_m, xy_thresh_sqrd, aff_inliers_from_ellshape)

def H_homog_from_RANSAC(kpts1_m, kpts2_m, xy_thresh_sqrd):
    ' Random Sample Consensus'
    return __H_homog_from(kpts1_m, kpts2_m, xy_thresh_sqrd, aff_inliers_from_randomsac)

def __H_homog_from(kpts1_m, kpts2_m, xy_thresh_sqrd, func_aff_inlier):
    ''' RanSaC: 
        Ransom Sample Consensus Inlier Generator 
        - Object retrieval fast, Philbin1, Chum1, et al 
        input: matching 
    '''
    #assert kpts1_m.shape[1] == kpts2_m.shape[1], 'RanSaC works on matches!'
    #assert kpts1_m.shape[0] == 5 and kpts2_m.shape[0] == 5, 'RanSaC works on ellipses!'

    num_m = kpts1_m.shape[1] # num matches

    # min number of matches to compute transform
    min_num_inliers = 3 
    if num_m < min_num_inliers: return  np.eye(3), ones(num_m, dtype=np.uint32)
    if num_m == 0: return np.eye(3), []

    # Estimate initial inliers with some RANSAC variant
    aff_inliers = func_aff_inlier(kpts1_m, kpts2_m, xy_thresh_sqrd)

    # If we cannot estimate a good correspondence 
    if len(aff_inliers) < min_num_inliers:
        return aff_inliers 

    # Homogonize+Normalize
    xy1_m    = kpts1_m[0:2,:] 
    xy2_m    = kpts2_m[0:2,:]
    (xyz_norm1, T1) = homogo_normalize_pts(xy1_m[:,aff_inliers]) 
    (xyz_norm2, T2) = homogo_normalize_pts(xy2_m[:,aff_inliers])

    # Compute Normalized Homog
    try: 
        H_prime = compute_homog(xyz_norm1, xyz_norm2)
        H = linalg.solve(T2, H_prime).dot(T1)                # Unnormalize
    except linalg.LinAlgError:
        return None

    # Estimate final inliers
    xy1_mHt = transform_xy(H, xy1_m)                        # Transform Kpts1 to Kpts2-space
    sqrd_dist_error = np.sum( (xy1_mHt - xy2_m)**2, axis=0) # Final Inlier Errors
    inliers = sqrd_dist_error < xy_thresh_sqrd
    return H, inliers

