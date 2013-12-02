#USE_AUTOJIT = True
#AUTOJIT_CALLS = 0
#if USE_AUTOJIT:
    #from numba import autojit
    #def try_autojit(fn, *args, **kwargs):
        #global AUTOJIT_CALLS
        #AUTOJIT_CALLS += 1 
        #return autojit(fn, *args, **kwargs)
#else:
    #def try_autojit(fn):
        #global AUTOJIT_CALLS
        #AUTOJIT_CALLS -= 1 
        #return fn

# --------------------------------
# Warping / Transformation stuff
# 
def keypoint_scale(fx2_kp):
    'the scale of a keypoint is sqrt(1/det(A))'
    fx2_acd = fx2_kp[:,2:5].T
    fx2_det = det_acd(fx2_acd)
    fx2_scale = np.sqrt(1.0/fx2_det)
    return fx2_scale
def border_coordinates(img):
    'specified in (x,y) coordinates'
    (img_h, img_w) = img.shape[0:2]
    tl = (0, 0)
    tr = (img_w-1, 0)
    bl = (0, img_h-1)
    br = (img_w-1, img_w-1)
    return np.array((tl, tr, bl, br)).T
def homogonize(coord_list):
    'input: list of (x,y) coordinates'
    ones_vector = np.ones((1, coord_list.shape[1]))
    coord_homog = np.vstack([np.array(coord_list), ones_vector])
    return coord_homog 
def transform_coord_list(coord_list, M):
    coord_homog  = homogonize(coord_list)
    Mcoord_homog = M.dot(coord_homog)
    Mcoord_list  = np.vstack((Mcoord_homog[0] / Mcoord_homog[2],
                              Mcoord_homog[1] / Mcoord_homog[2]))
    return Mcoord_list
def minmax_coord_list(coord_list):
    minx, miny = coord_list.min(1)
    maxx, maxy = coord_list.max(1)
    return (minx, maxx, miny, maxy)

def transformed_bounds(img, M):
    coord_list  = border_coordinates(img)
    Mcoord_list = transform_coord_list(coord_list, M)
    (minx, maxx, miny, maxy) = minmax_coord_list(Mcoord_list)
    return (minx, maxx, miny, maxy) 
#---


def sqrt_inv(fx2_kp):
    # numba approved
    x, y, a, c, d = fx2_kp.T
    aIS = 1/np.sqrt(a) 
    bIS = c/(-np.sqrt(a)*d - a*np.sqrt(d))
    dIS = 1/np.sqrt(d)
    kpts_iter = iter(zip(x,y,aIS,bIS,dIS))
    kptsIS = [np.array([( a_, b_, x_),
                        ( 0 , d_, y_),
                        ( 0 , 0 , 1)])
              for (x_,y_,a_,b_,d_) in kpts_iter ]
    return kptsIS

def keypoint_axes(fx2_kp):
    raise NotImplemented('this doesnt work')
    num_kpts = len(fx2_kp)
    (a,c,d) = fx2_kp[:,2:5].T
    # sqrtm(inv(A))
    aIS = 1/np.sqrt(a) 
    bIS = -c/(np.sqrt(a)*d + a*np.sqrt(d))
    cIS = np.zeros(num_kpts)
    dIS = 1/np.sqrt(d)
    # Build lower triangular matries that maps unit circles to ellipses
    abcdIS = np.vstack([aIS, bIS, cIS, dIS]).T.reshape(num_kpts, 2, 2)
    # Get major and minor axies of ellipes. 
    eVals, eVecs = zip(*[np.linalg.eig(cir2ell) for cir2ell in abcdIS])


def keypoint_radius(fx2_kp):
    scale_m = keypoint_scale(fx2_kp)
    radius_m = 3*np.sqrt(3*scale_m)
    # I'm not sure which one is right. 
    #radius_m = 3*np.sqrt(3)*scale_m
    return radius_m




###



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



def compare1():
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

def spatially_verify(kpts1, kpts2, rchip_size2, fm, fs, xy_thresh,
                     shigh_thresh, slow_thresh, use_chip_extent):
    '''1) compute a robust transform from img2 -> img1
       2) keep feature matches which are inliers 
       returns fm_V, fs_V, H '''
    # Return if pathological
    min_num_inliers   = 4
    if len(fm) < min_num_inliers:
        return (np.empty((0, 2)), np.empty((0, 1)))
    # Get homography parameters
    if use_chip_extent:
        diaglen_sqrd = rchip_size2[0]**2 + rchip_size2[1]**2
    else:
        x_m = kpts2[fm[:,1],0].T
        y_m = kpts2[fm[:,1],1].T
        diaglen_sqrd = calc_diaglen_sqrd(x_m, y_m)
    # Try and find a homography
    sv_tup = homography_inliers(kpts1, kpts2, fm, xy_thresh, 
                                    shigh_thresh, slow_thresh,
                                    diaglen_sqrd, min_num_inliers)
    if sv_tup is None:
        return (np.empty((0, 2)), np.empty((0, 1)))
    # Return the inliers to the homography
    (H, inliers, Aff, aff_inliers) = sv_tup
    fm_V = fm[inliers, :]
    fs_V = fs[inliers]
    return fm_V, fs_V
####
    #test2(0, 1)
    #test2(0, 2)
    #test2(0, 3)
    #test2(0, 4)
    #test2(0, 5)
    #test2(0, 6)
    #test2(0, 6)
    if 'AUTOJIT_CALLS' in vars():
        print('autojit calls: '+str(AUTOJIT_CALLS))
