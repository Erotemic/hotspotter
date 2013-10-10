from pylab import find
import numpy as np
from numpy import linalg, zeros, array, vstack, ones, sum, power, uint32, insert, matrix
from warnings import catch_warnings, simplefilter 

#skimage.transform
# http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
# skimage.transform.fast_homography(im, H)

# Generate 6 degrees of freedom homography transformation
def compute_homog(xyz_norm1, xyz_norm2):
    'Computes homography from normalized (0 to 1) point correspondences'
    num_pts = xyz_norm1.shape[1]
    assert xyz_norm1.shape == xyz_norm2.shape, ''
    assert xyz_norm1.shape[0] == 3, ''
    Mbynine = zeros((2*num_pts,9), dtype=np.float32)
    for ilx in xrange(num_pts): # Loop over inliers
        # Concatinate all 2x9 matrices into an Mx9 matrcx
        u2      =     xyz_norm2[0,ilx]
        v2      =     xyz_norm2[1,ilx]
        (d,e,f) =    -xyz_norm1[:,ilx]
        (g,h,i) =  v2*xyz_norm1[:,ilx]
        (j,k,l) =     xyz_norm1[:,ilx]
        (p,q,r) = -u2*xyz_norm1[:,ilx]
        Mbynine[ilx*2:(ilx+1)*2,:]  = array(\
            [(0, 0, 0, d, e, f, g, h, i),
             (j, k, l, 0, 0, 0, p, q, r) ] )
    # Solve for the nullspace of the Mbynine
    try:
        (_U, _s, V) = linalg.svd(Mbynine)
    except MemoryError:
        # TODO: is sparse calculation faster than not?
        print('Singular Value Decomposition Ran Out of Memory. Trying with a sparse matrix')
        import scipy.sparse
        import scipy.sparse.linalg
        MbynineSparse = scipy.sparse.lil_matrix(Mbynine)
        (_U, _s, V) = scipy.sparse.linalg.svds(MbynineSparse)
        #import gc
        #gc.collect()
        #print('Singular Value Decomposition Ran Out of Memory.'+\
              #'Trying to free some memory with garbage collection')
        #(_U, _s, V) = linalg.svd(Mbynine)
        #import pdb
        #pdb.set_trace()

    # Rearange the nullspace into a homography
    h = V[-1,:] # (transposed in matlab)
    H = vstack( ( h[0:3],  h[3:6],  h[6:9]  ) )
    return H
# 
def _homogonize_pts(xy):
    'Adds a 3rd dimension of ones to xy-position vectors'
    assert xy.shape[0] == 2, ''
    xyz = vstack([xy, ones(xy.shape[1])]);
    return xyz
#
def _normalize_pts(xyz):
    'Returns a transformation to normalize points to mean=0, stddev=1'
    num_xyz = xyz.shape[1]
    com = sum(xyz,axis=1) / num_xyz # center of mass
    with catch_warnings():
        simplefilter("ignore")
        sx  = num_xyz / sum(abs(xyz[0,:]-com[0]))  # average xy magnitude
        sy  = num_xyz / sum(abs(xyz[1,:]-com[1])) 
    tx  = -com[0]*sx
    ty  = -com[1]*sy
    T = array([(sx, 0, tx), (0, sy, ty), (0, 0, 1)])
    return T
#
def homogo_normalize_pts(xy):
    'Homoginize points for stable homography estimation'
    xyz = _homogonize_pts(xy)
    T   = _normalize_pts(xyz)
    xyz_norm = T.dot(xyz)
    return (xyz_norm, T)
#
def get_affine_inliers_RANSAC(num_m, xy1_m, xy2_m,\
                              acd1_m, acd2_m, xy_thresh_sqrd, sigma_thresh_sqrd=None):
    '''Computes initial inliers by iteratively computing affine transformations
    between matched keypoints'''
    aff_inliers = []
    # Enumerate All Hypothesis (Match transformations)
    for mx in xrange(num_m): 
        xy1  = xy1_m[:,mx].reshape(2,1) #  XY Positions
        xy2  = xy2_m[:,mx].reshape(2,1) 
        A1   = matrix(insert(acd1_m[:,mx], [1.], 0.)).reshape(2,2)
        A2   = matrix(insert(acd2_m[:,mx], [1.], 0.)).reshape(2,2)
        # Compute Affine Tranform 
        # from img1 to img2 = (E2\E1) 
        Aff  = linalg.inv(A2).dot(A1)
        #
        # Transform XY-Positions
        xy1_mAt = xy2 + Aff.dot( (xy1_m - xy1) ) 
        xy_err_sqrd = sum( power(xy1_mAt - xy2_m, 2) , 0)
        _inliers = find(xy_err_sqrd < xy_thresh_sqrd)
        #
        # Transform Ellipse Geometry (solved on paper)
        if not sigma_thresh_sqrd is None:
            scale1_mAt = (acd1_m[0]*Aff[0,0]) *\
                         (acd1_m[1]*Aff[1,0]+acd1_m[2]*Aff[1,1])
            scale2_m   = acd2_m[0] * acd2_m[2]
            scale_err  = np.abs(scale1_mAt - scale2_m)
            _inliers_scale = find(scale_err < sigma_thresh_sqrd)
            _inliers = np.bitwise_and(_inliers, _inliers_scale)
        #If this hypothesis transformation is better than the ones we have
        #previously seen then set it as the best
        if len(_inliers) > len(aff_inliers):
            aff_inliers = _inliers
            #bst_xy_err  = xy_err_sqrd 
    return aff_inliers

def homog_warp_shape(H3x3, acd):
    #acd = np.array([(1,2,3,4,5,6,7,8,9,0),(9,8,7,6,5,4,3,2,1,0),(9,1,8,2,7,3,0,5,4,6)])
    #H3x3 = np.matrix('[1, 2, 3; 4, 5, 6; 7, 8, 9]')
    num_shapes = acd.shape[1]
    # Allocate Space for a return matrix and a stacked operation matrix
    shape2x2 = np.empty((num_shapes, 2,2))
    shape3x3 = np.zeros((num_shapes, 3,3))
    # Fill the operation matrix, to do the multiply in one operation
    shape3x3[:,2,2] = 1
    shape3x3[:,0,0] = acd[0]
    shape3x3[:,1,0] = acd[1]
    shape3x3[:,1,1] = acd[2]
    shape3x3.shape = (3*num_shapes, 3)
    # Transform Stacked Matrix
    H_shape3x3 = H3x3.dot(np.transpose(shape3x3)).getA()
    # Insert Warped Shape components into return array
    # This discards the transformation information (unsure if this is the right
    # way to go about it)
    shape2x2[:,0,0] = H_shape3x3[0,0::3]
    shape2x2[:,0,1] = H_shape3x3[0,1::3]
    shape2x2[:,1,0] = H_shape3x3[1,0::3]
    shape2x2[:,1,1] = H_shape3x3[1,1::3]
    # Return an array of 2x2 shapes
    return shape2x2


def homog_warp(H3x3, xy):
    xyz = _homogonize_pts(xy)
    H_xyz = H3x3.dot(xyz)
    with catch_warnings():
        simplefilter("ignore")
        H_xy = H_xyz[0:2,:] / H_xyz[2,:]
    return H_xy

def ransac(fpts1_match, fpts2_match,\
            xy_thresh_sqrd = None,\
            sigma_thresh   = None,\
            theta_thresh   = None):
    ''' RanSaC: 
        Ransom Sample Consensus Inlier Generator 
        - Object retrieval fast, Philbin1, Chum1, et al '''

    assert len(fpts1_match) == len(fpts2_match), 'RanSaC works on matches!'

    num_m = fpts1_match.shape[1] # num matches

    nInlier_thresh = 3 
    if num_m < nInlier_thresh:
        # there are not enough matches to be spatially invalid
        inliers = ones(num_m, dtype=uint32)
        return inliers

    #print('''
    #RANSAC Resampling matches on %r chips 
    #* xy_thresh2   = %r * (chip diagonal length) 
    #* theta_thresh = %r (orientation)
    #* sigma_thresh = %r (scale)''' % (num_m, xy_thresh_sqrd, theta_thresh, sigma_thresh))

    if num_m == 0:
        return zeros(num_m, dtype=uint32)

    xy1_m    = fpts1_match[0:2,:] # keypoint xy coordinates matches
    xy2_m    = fpts2_match[0:2,:]
    acd1_m   = fpts1_match[2:5,:] # keypoint shape matrix [a 0; c d] matches
    acd2_m   = fpts2_match[2:5,:]

    # Compute affine inliers using exhaustive ransac
    aff_inliers = get_affine_inliers_RANSAC(num_m, xy1_m, xy2_m,\
                                            acd1_m, acd2_m, xy_thresh_sqrd, sigma_thresh_sqrd=None)
    if len(aff_inliers) < nInlier_thresh:
        # Cannot establish a better correspondence
        return aff_inliers
    # Homogonize+Normalize
    (xyz_norm1, T1) = homogo_normalize_pts(xy1_m[:,aff_inliers]) 
    (xyz_norm2, T2) = homogo_normalize_pts(xy2_m[:,aff_inliers])

    # Compute Normalized Homog
    try: 
        H_prime = compute_homog(xyz_norm1, xyz_norm2)
        # Unnormalize
        H = linalg.solve(T2, H_prime).dot(T1)                
        # Kpts1 in Kpts2-space
        xy1_mHt = homog_warp(H, xy1_m) 
        # Final Inlier Errors
        sqrd_dist_error = sum( (xy1_mHt - xy2_m)**2, axis=0)
        inliers = sqrd_dist_error < xy_thresh_sqrd
    except linalg.LinAlgError:
        #logdbg('linalg.LinAlgError: Returning None')
        return None
    return  inliers
