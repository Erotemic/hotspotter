# TODO: For Spatial Nearest Neighbors
# 1) Change kpts to be in the range 0 to 1
# 2) Make a quick lookup table for chip size
'''
K = 3
qfx2_xy1 = np.array([(.1, .1), (.2, .2), (.3, .3)])
qfx2_xy2 = np.array([((.1, .1), (.2, .2), (.3, .3)), 
                     ((.1, .1), (.2, .2), (.3, .3)), 
                     ((.1, .1), (.2, .2), (.3, .3)), 
                     ((.1, .1), (.2, .2), (.3, .3)) ])
arr = qfx2_xy2
'''

def desc_nearest_neighbors(desc, vsmany_args, K=None):
    vsmany_flann = vsmany_args.vsmany_flann
    ax2_cx       = vsmany_args.ax2_cx
    ax2_fx       = vsmany_args.ax2_fx
    isQueryIndexed = True
    K = params.__VSMANY_K__ if K is None else K
    checks   = params.VSMANY_FLANN_PARAMS['checks']
    # Find each query descriptor's k+1 nearest neighbors
    (qfx2_ax, qfx2_dists) = vsmany_flann.nn_index(desc, K, checks=checks)
    qfx2_cx = ax2_cx[qfx2_ax]
    qfx2_fx = ax2_fx[qfx2_ax]
    return (qfx2_cx, qfx2_fx, qfx2_dists) 

def tile_before_axis(arr, axis, num):
    repl = tuple([num] + ([1]*len(arr.shape)))
    rollax = np.rollaxis
    tile  = np.tile
    return rollax(rollax(tile(rollax(arr, axis, 0), repl), 0, axis+2), 0, axis+2)
    #roll1 = np.rollaxis(arr, axis, 0)
    #tarr = np.tile(roll1, repl)
    #tarr2 = np.rollaxis(tarr, 0, axis+2)
    #tarr3 = np.rollaxis(tarr2, 0, axis+2)
    #print(arr.shape)
    #print(roll1.shape)
    #print(tarr.shape)
    #print(tarr2.shape)
    #print(tarr3.shape)

def snn_testdata(hs):
    hs.ensure_matcher(match_type='vsmany')
    vsmany_args = hs.matcher.vsmany_args
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    qcx = 0
    qfx2_desc = cx2_desc[qcx]
    qfx2_kpts = cx2_kpts[qcx]
    cx2_rchip_size = hs.get_cx2_rchip_size()
    qchipdiag = np.sqrt((np.array(cx2_rchip_size[qcx])**2).sum())
    data_flann = vsmany_args.vsmany_flann
    dx2_desc = vsmany_args.ax2_desc
    dx2_cx = vsmany_args.ax2_cx
    dx2_fx = vsmany_args.ax2_fx
    K = 3
    checks = 128

def spatial_nearest_neighbors(qfx2_desc, qfx2_kpts, qchipdiag, dx2_desc, dx2_kpts, dx2_cx, data_flann, K, checks):
    'K Spatial Nearest Neighbors'
    nQuery, dim = qfx2_desc.shape
    # Get nearest neighbors #.2690s 
    (qfx2_dx, qfx2_dists) = data_flann.nn_index(qfx2_desc, K, checks=checks)
    # Get matched chip sizes #.0300s
    qfx2_cx = dx2_cx[qfx2_dx]
    qfx2_fx = dx2_fx[qfx2_dx]
    qfx2_chipsize2 = np.array([cx2_rchip_size[cx] for cx in qfx2_cx.flat])
    qfx2_chipsize2.shape = (nQuery, K, 2)
    qfx2_chipdiag2 = np.sqrt((qfx2_chipsize2**2).sum(2))
    # Get query relative xy keypoints #.0160s / #.0180s (+cast)
    qfx2_xy1 = np.array(qfx2_kpts[:, 0:2], np.float)
    qfx2_xy1[:,0] /= qchipdiag
    qfx2_xy1[:,1] /= qchipdiag
    # Get database relative xy keypoints
    qfx2_xy2 = np.array([cx2_kpts[cx][fx, 0:2] for (cx, fx) in
                         izip(qfx2_cx.flat, qfx2_fx.flat)], np.float)
    qfx2_xy2.shape = (nQuery, K, 2)
    qfx2_xy2[:,:,0] /= qfx2_chipdiag2
    qfx2_xy2[:,:,1] /= qfx2_chipdiag2
    # Get the relative distance # .0010s
    qfx2_K_xy1 = np.rollaxis(np.tile(qfx2_xy1, (K, 1, 1)), 1)
    qfx2_xydist = ((qfx2_K_xy1 - qfx2_xy2)**2).sum(2)
    qfx2_dist_valid = qfx2_xydist < .5

    # Do scale for funzies
    qfx2_det1 = np.array(qfx2_kpts[:, [2,4]], np.float).prod(1)
    qfx2_det1 = np.sqrt(1.0/qfx2_det1)
    qfx2_K_det1 = np.rollaxis(np.tile(qfx2_det1, (K, 1)), 1)
    qfx2_det2 = np.array([cx2_kpts[cx][fx, [2,4]] for (cx, fx) in
                          izip(qfx2_cx.flat, qfx2_fx.flat)], np.float).prod(1)
    qfx2_det2.shape = (nQuery, K)
    qfx2_det2 = np.sqrt(1.0/qfx2_det2)
    qfx2_scaledist = qfx2_det2 / qfx2_K_det1

    qfx2_scale_valid = np.bitwise_and(qfx2_scaledist > .5, qfx2_scaledist < 2)
    
    # All neighbors are valid
    qfx2_valid = np.bitwise_and(qfx2_dist_valid, qfx2_scale_valid)
    return qfx2_dx, qfx2_dists, qfx2_valid

def reciprocal_nearest_neighbors(qfx2_desc, dx2_desc, data_flann, K, checks):
    'K Reciprocal Nearest Neighbors'
    nQuery, dim = qfx2_desc.shape
    # Assign query features to K nearest database features
    (qfx2_dx, qfx2_dists) = data_flann.nn_index(qfx2_desc, K, checks=checks)
    # Assign those nearest neighbors to K nearest database features
    qx2_nn = dx2_desc[qfx2_dx]
    qx2_nn.shape = (nQuery*K, dim)
    (_nn2_dx, nn2_dists) = data_flann.nn_index(qx2_nn, K, checks=checks)
    # Get the maximum distance of the reciprocal neighbors
    nn2_dists.shape = (nQuery, K, K)
    qfx2_maxdist = nn2_dists.max(2)
    # Test if nearest neighbor distance is less than reciprocal distance
    qfx2_valid = qfx2_dists < qfx2_maxdist
    return qfx2_dx, qfx2_dists, qfx2_valid 

def assign_matches_vsmany_BINARY(qcx, cx2_desc):
    return None

