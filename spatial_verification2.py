from helpers import printWARN, printINFO
import warnings
import cv2
import numpy.linalg as linalg
import numpy as np
import scipy 
import scipy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
# skimage.transform
# http://stackoverflow.com/questions/11462781/fast-2d-rigid-body-transformations-in-numpy-scipy
# skimage.transform.fast_homography(im, H)
def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

SV_DTYPE = np.float64

# Generate 6 degrees of freedom homography transformation
def compute_homog(x1_mn, y1_mn, x2_mn, y2_mn):
    'Computes homography from normalized (0 to 1) point correspondences'
#with helpers.Timer('computehomog'):
    num_pts = len(x1_mn)
    Mbynine = np.zeros((2*num_pts,9), dtype=SV_DTYPE)
    for ix in xrange(num_pts): # Loop over inliers
        # Concatinate all 2x9 matrices into an Mx9 matrix
        u2      = x2_mn[ix]
        v2      = y2_mn[ix]
        (d,e,f) = (   -x1_mn[ix],    -y1_mn[ix],  -1)
        (g,h,i) = ( v2*x1_mn[ix],  v2*y1_mn[ix],  v2)
        (j,k,l) = (    x1_mn[ix],     y1_mn[ix],   1)
        (p,q,r) = (-u2*x1_mn[ix], -u2*y1_mn[ix], -u2)
        Mbynine[ix*2]   = (0, 0, 0, d, e, f, g, h, i)
        Mbynine[ix*2+1] = (j, k, l, 0, 0, 0, p, q, r)
    # Solve for the nullspace of the Mbynine
    try:
        (U, S, Vct) = linalg.svd(Mbynine)
    except MemoryError as ex:
        printWARN('Caught MemErr %r during full SVD. Trying sparse SVD.' % (ex))
        MbynineSparse = sparse.lil_matrix(Mbynine)
        (U, S, Vct) = sparse_linalg.svds(MbynineSparse)
    except linalg.LinAlgError as ex2:
        return np.eye(3)
    # Rearange the nullspace into a homography
    h = Vct[-1] # (transposed in matlab)
    H = np.vstack( ( h[0:3],  h[3:6],  h[6:9]  ) )
    return H
'''
if not 'xy_thresh' in vars():
    xy_thresh = .002
if not 'scale_thresh' in vars():
    scale_thresh = 2
if not 'min_num_inliers' in vars():
    min_num_inliers = 4
'''

'''
from hotspotter.spatial_verification2 import split_kpts
x_m, y_m = x2_m, y2_m
'''
def calc_diaglen_sqrd(x_m, y_m):
    x_extent_sqrd = (x_m.max() - x_m.min()) ** 2
    y_extent_sqrd = (y_m.max() - y_m.min()) ** 2
    diaglen_sqrd = x_extent_sqrd + y_extent_sqrd
    return diaglen_sqrd

def split_kpts(kpts5xN):
    'breakup keypoints into position and shape'
    _xs   = np.array(kpts5xN[0], dtype=SV_DTYPE)
    _ys   = np.array(kpts5xN[1], dtype=SV_DTYPE)
    _acds = np.array(kpts5xN[2:5], dtype=SV_DTYPE)
    return _xs, _ys, _acds

def normalize_xy_points(x_m, y_m):
    'Returns a transformation to normalize points to mean=0, stddev=1'
    mean_x = x_m.mean() # center of mass
    mean_y = y_m.mean()
    sx = 1.0 / x_m.std()  # average xy magnitude
    sy = 1.0 / y_m.std()
    T = np.array([(sx, 0, -mean_x * sx),
                    (0, sy, -mean_y * sy),
                    (0,  0,  1)])
    x_norm = (x_m - mean_x) * sx
    y_norm = (y_m - mean_y) * sy
    return x_norm, y_norm, T

def homography_inliers(kpts1, kpts2, fm, 
                       xy_thresh,
                       scale_thresh_high,
                       scale_thresh_low,
                       min_num_inliers=4):
    if len(fm) < min_num_inliers:
        return None
    # Not enough data
    # Estimate affine correspondence convert to SV_DTYPE
    fx1_m, fx2_m = fm[:, 0], fm[:, 1]
    x1_m, y1_m, acd1_m = split_kpts(kpts1[fx1_m, :].T)
    x2_m, y2_m, acd2_m = split_kpts(kpts2[fx2_m, :].T)
    # Get diagonal length
    diaglen_sqrd = calc_diaglen_sqrd(x2_m, y2_m)
    xy_thresh_sqrd = diaglen_sqrd * xy_thresh
    Aff, aff_inliers = __affine_inliers(x1_m, y1_m, acd1_m, fm[:, 0],
                                        x2_m, y2_m, acd2_m, fm[:, 1],
                                        xy_thresh_sqrd, 
                                        scale_thresh_high,
                                        scale_thresh_low)
    # Cannot find good affine correspondence
    if len(aff_inliers) < min_num_inliers:
        return None
    # Get corresponding points and shapes
    (x1_ma, y1_ma, acd1_m) = (x1_m[aff_inliers], y1_m[aff_inliers],
                              acd1_m[:,aff_inliers])
    (x2_ma, y2_ma, acd2_m) = (x2_m[aff_inliers], y2_m[aff_inliers],
                              acd2_m[:,aff_inliers])
    # Normalize affine inliers
    x1_mn, y1_mn, T1 = normalize_xy_points(x1_ma, y1_ma)
    x2_mn, y2_mn, T2 = normalize_xy_points(x2_ma, y2_ma)
    H_prime = compute_homog(x1_mn, y1_mn, x2_mn, y2_mn)
    try: 
        # Computes ax = b # x = linalg.solve(a, b)
        H = linalg.solve(T2, H_prime).dot(T1) # Unnormalize
    except linalg.LinAlgError as ex:
        printWARN('Warning 285 '+repr(ex), )
        return np.eye(3), aff_inliers

    ((H11, H12, H13),
     (H21, H22, H23),
     (H31, H32, H33)) = H
    # Transform kpts1 to kpts2
    x1_mt = H11*(x1_m) + H12*(y1_m) + H13
    y1_mt = H21*(x1_m) + H22*(y1_m) + H23
    z1_mt = H31*(x1_m) + H32*(y1_m) + H33
    # --- Find (Squared) Error ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy_err = (x1_mt/z1_mt - x2_m)**2 + (y1_mt/z1_mt - y2_m)**2 
    # Estimate final inliers
    inliers, = np.where(xy_err < xy_thresh_sqrd)
    return H, inliers, Aff, aff_inliers

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
    return np.array((a, c, d))
# --------------------------------
'''
fx1_m  = np.array( (1, 2, 3, 4, 5))
x1_m   = np.array( (1, 2, 1, 4, 5))
y1_m   = np.array( (1, 2, 1, 4, 5))
acd1_m = np.array(((1, 1, 1, 1, 1),
                   (0, 0, 0, 0, 0),
                   (1, 1, 1, 1, 1)))

fx2_m  = np.array( (1, 2, 3, 2, 5))
x2_m   = np.array( (1, 2, 1, 4, 5))
y2_m   = np.array( (1, 2, 1, 4, 5))
acd2_m = np.array(((1, 1, 1, 1, 1),
                   (0, 0, 0, 0, 0),
                   (1, 1, 1, 1, 1)))

'''
#---
# Ensure that a feature doesn't have multiple assignments
'''
fx1_uq, fx1_ux, fx1_ui = np.unique(fx1_m, return_index=True, return_inverse=True)
fx2_uq, fx2_ux, fx2_ui = np.unique(fx2_m, return_index=True, return_inverse=True)

fx_m = fx2_m
inliers_flag = hypo_inliers
error = xy_err
def dupl_items(list_):
    seen = set()
    seen_add = seen.add
    seen_twice = set( x for x in list_ if x in seen or seen_add(x) )
    return list( seen_twice )
'''
def remove_multiassignments(fx_m, inliers, error):
    '''
    I think this works, but I haven't integrated it yet.
    Also, is probably going to be slow.
    Try pip install hungarian instead
    '''
    # Get the inlier feature indexes
    ilx_to_fx = fx_m[inliers]
    ux_to_fx, ilx2_to_ux = np.unique(ilx_to_fx, return_inverse=True)
    # Find which fx are duplicates
    dupl_ux = dupl_items(ilx2_to_ux)
    # This is the list of multi-assigned feature indexes
    fx_mulass = ux_to_fx[dupl_ux]
    ilx_mulass = [np.where(fx_m == fx)[0] for fx in fx_mulass]
    # For each multi-assigned inlier, pick one
    ilx2_flag = np.ones(len(inliers), dtype=np.bool)
    for ilx_list in ilx_mulass:
        ilx2_flag[ilx_list] = False
        keepx = np.argmin(error[inliers[ilx_list]])
        ilx2_flag[ilx_list[keepx]] = True
    return inliers[ilx_to_flag]

def flag_unique(list_):
    seen = set([])
    seen_add = seen.add
    return np.array([False if fx in seen or seen_add(fx) else True 
                     for fx in list_], dtype=np.bool)

def __affine_inliers(x1_m, y1_m, acd1_m, fx1_m,
                     x2_m, y2_m, acd2_m, fx2_m,
                     xy_thresh_sqrd, 
                     scale_thresh_high, scale_thresh_low):
    '''Estimates inliers deterministically using elliptical shapes
    1_m = img1_matches; 2_m = img2_matches
    x and y are locations, acd are the elliptical shapes. 
    fx are the original feature indexes (used for making sure 1 keypoint isn't assigned to 2)
    '''
#with helpers.Timer('enume all'):
    fx1_uq, fx1_ui = np.unique(fx1_m, return_inverse=True)
    fx2_uq, fx2_ui = np.unique(fx2_m, return_inverse=True)
    best_inliers = []
    num_best_inliers = 0
    best_mx  = None
    # Get keypoint scales (determinant)
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    # Compute all transforms from kpts1 to kpts2 (enumerate all hypothesis)
    inv2_m = inv_acd(acd2_m, det2_m)
    # The transform from kp1 to kp2 is given as:
    # A = inv(A2).dot(A1)
    Aff_list = dot_acd(inv2_m, acd1_m)
    # Compute scale change of all transformations 
    detAff_list = det_acd(Aff_list)
    # Test all hypothesis
    for mx in xrange(len(x1_m)):
        # --- Get the mth hypothesis ---
        Aa = Aff_list[0, mx]
        Ac = Aff_list[1, mx]
        Ad = Aff_list[2, mx]
        Adet = detAff_list[mx]
        x1_hypo = x1_m[mx]
        y1_hypo = y1_m[mx]
        x2_hypo = x2_m[mx]
        y2_hypo = y2_m[mx]
        # --- Transform from kpts1 to kpts2 ---
        x1_mt   = x2_hypo + Aa*(x1_m - x1_hypo)
        y1_mt   = y2_hypo + Ac*(x1_m - x1_hypo) + Ad*(y1_m - y1_hypo)
        # --- Find (Squared) Error ---
        xy_err    = (x1_mt - x2_m)**2 + (y1_mt - y2_m)**2 
        scale_err = Adet * det2_m / det1_m 
        # --- Determine Inliers ---
        xy_inliers_flag = xy_err < xy_thresh_sqrd 
        scale_inliers_flag = np.logical_and(scale_err > scale_thresh_low,
                                            scale_err < scale_thresh_high)
        hypo_inliers_flag = np.logical_and(xy_inliers_flag, scale_inliers_flag)
        #---
        #---------------------------------
        # TODO: More sophisticated scoring
        # Currently I'm using the number of inliers as a transformations'
        # goodness. Also the way I'm accoutning for multiple assignment
        # does not take into account any error reporting
        #---------------------------------
        '''
        unique_assigned1 = flag_unique(fx1_ui[hypo_inliers_flag])
        unique_assigned2 = flag_unique(fx2_ui[hypo_inliers_flag])
        unique_assigned_flag = np.logical_and(unique_assigned1,
                                              unique_assigned2)
        hypo_inliers = np.where(hypo_inliers_flag)[0][unique_assigned_flag]
        '''
        hypo_inliers = np.where(hypo_inliers_flag)[0]
        #---
        num_hypo_inliers = len(hypo_inliers)
        # --- Update Best Inliers ---
        if num_hypo_inliers > num_best_inliers:
            best_mx = mx 
            best_inliers = hypo_inliers
            num_best_inliers = num_hypo_inliers
    if not best_mx is None: 
        (Aa, Ac, Ad) = Aff_list[:, best_mx]
        (x1, y1, x2, y2) = (x1_m[best_mx], y1_m[best_mx],
                            x2_m[best_mx], y2_m[best_mx])
        best_Aff = np.array([(Aa,  0,  x2-Aa*x1      ),
                             (Ac, Ad,  y2-Ac*x1-Ad*y1),
                             ( 0,  0,               1)])
    else: 
        best_Aff = np.eye(3)
    return best_Aff, best_inliers


def show_inliers(hs, qcx, cx, inliers, title='inliers', **kwargs):
    import load_data2 as ld2
    df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm[inliers], title=title, **kwargs_)


def test():
    from __init__ import *
    from spatial_verification2 import *
    reload()
    df2.reset()
    xy_thresh         = params.__XY_THRESH__
    scale_thresh_high = params.__SCALE_THRESH_HIGH__
    scale_thresh_low  = params.__SCALE_THRESH_LOW__
    # Pick out some data
    if not 'hs' in vars():
        (hs, qcx, cx, fm, fs, 
         rchip1, rchip2, kpts1, kpts2) = ld2.get_sv_test_data()
    #df2.update()
    x1_m, y1_m, acd1_m = split_kpts(kpts1[fm[:, 0]].T)
    x2_m, y2_m, acd2_m = split_kpts(kpts2[fm[:, 1]].T)
    fx1_m = fm[:, 0]
    fx2_m = fm[:, 1]
    x2_extent = x2_m.max() - x2_m.min()
    y2_extent = y2_m.max() - y2_m.min()
    img2_extent = np.array([x2_extent, y2_extent])
    img2_diaglen_sqrd = x2_extent**2 + y2_extent**2
    xy_thresh_sqrd = img2_diaglen_sqrd * xy_thresh
    # -----------------------------------------------
    # Get match threshold 10% of matching keypoint extent diagonal
    #aff_inliers1, Aff1 = sv2.affine_inliers(kpts1, kpts2, fm, xy_thresh, scale_thresh)
    '''
    # Draw assigned matches
    args_ = [rchip1, rchip2, kpts1, kpts2]
    df2.show_matches2(*args_+[fm], fs=None,
                      all_kpts=False, draw_lines=False,
                      doclf=True, title='Assigned matches')
    # Draw affine inliers
    df2.show_matches2(*args_+[fm[aff_inliers1]], fs=None,
                      all_kpts=False, draw_lines=False, doclf=True,
                      title='Assigned matches')
    '''
    df2.update()

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
    import multiprocessing as mp
    mp.freeze_support()
    print('__main__ = spatial_verification2.py')
