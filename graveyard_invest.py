_stdoutlock = None
def start_stdout():
    global _stdoutlock
    _stdoutlock = helpers.RedirectStdout(autostart=True)
def stop_stdout():
    global _stdoutlock
    _stdoutlock.stop()

#rss = helpers.RedirectStdout(autostart=True)
#rss.stop()
def where_did_vsone_matches_go(hs, qcx, fnum=1):
    vsone_cfg  = mc3.vsone_cfg()
    vsmany_cfg = mc3.vsmany_cfg()
    mc3.unify_cfgs([vsone_cfg, vsmany_cfg])
    for qcx, ocxs, notes in qon_list:
        __where_did_vsone_matches_go(hs, qcx, fnum=fnum)
        fnum += 1
    return fnum

def __where_did_vsone_matches_go(hs, qcx, fnum, vsone_cfg, vsmany_cfg):
    '''Finds a set of vsone matches and a set of vsmany matches. 
    displays where the vsone matches are in the vsmany ranked lists'''
    # Ground truth matches
    gt_cxs = hs.get_other_cxs(qcx)
    # Query Features
    cx2_rchip_size = hs.get_cx2_rchip_size()
    get_features = quick_get_features_factory(hs)
    rchip1, qfx2_kp1, qfx2_desc1, qcid = get_features(qcx)
    # Get/Show vsone matches
    res_vsone = mc3.query_database(hs, qcx, vsone_cfg)
    gt2_fm_V = res_vsone.cx2_fm_V[gt_cxs]
    # Get vsmany assigned matches (no spatial verification)
    hs.ensure_matcher(match_type='vsmany')
    vsmany_args = hs.matcher.vsmany_args
    (qfx2_cx, qfx2_fx, qfx2_dists) = mc2.desc_nearest_neighbors(qfx2_desc1, vsmany_args, K)
    # Find where the matches to the correct images are
    print('[invest]  Finding where the vsone matches went for %s' % hs.vs_str(qcx, qcid))
    k_inds  = np.arange(0, K)
    qf_inds = np.arange(0, len(qfx2_cx))
    kxs, qfxs = np.meshgrid(k_inds, qf_inds)
    for gtx, ocx in enumerate(gt_cxs):
        rchip2, fx2_kp2, fx2_desc2, ocid = get_features(ocx)
        rchip_size2 = cx2_rchip_size[ocx]
        print('[invest] Checking matches to ground truth %r / %r %s' % 
              (gtx+1, len(gt_cxs), hs.vs_str(ocx, ocid)))
        # Get vsone indexes
        vsone_fm_V = gt2_fm_V[gtx]
        # Find correct feature and rank indexes: fx and kx
        vsmany_qfxs, vsmany_kxs = np.where(qfx2_cx == ocx)
        # Get comparisons to vsone
        qfx_kx_tup = zip(vsmany_qfxs, vsmany_kxs)
        vsmany_fxs = np.array([qfx2_fx[qfx, kx] for qfx, kx in qfx_kx_tup])
        def cast_uint32(arr):
            return np.array(arr, dtype=np.uint32)
        vsmany_fm  = cast_uint32(np.vstack(map(cast_uint32,(vsmany_qfxs, vsmany_fxs))).T)
        vsmany_fs  = vsmany_kxs # use k as score
        # Intersect vsmany with vsone_V
        fm_intersect, vsone_ix, vsmany_ix = helpers.intersect2d(vsone_fm_V, vsmany_fm)
        print(vsmany_ix)
        isecting_vsmany_fm = vsmany_fm[vsmany_ix]
        isecting_vsmany_fs = vsmany_kxs[vsmany_ix] # use k as score
        isecting_kxs = vsmany_kxs[vsmany_ix]
        # Spatially verify the vsmany matches 
        vsmany_fm_V, vsmany_fs_V = mc2.spatially_verify2(qfx2_kp1, fx2_kp2,
                                                         vsmany_fm, vsmany_fs,
                                                         rchip_size2=rchip_size2)
        # Intersect vsmany_V with vsone_V
        fm_V_intersect, vsoneV_ix, vsmanyV_ix = helpers.intersect2d(vsone_fm_V, vsmany_fm_V)
        isecting_vsmany_fm_V = vsmany_fm[vsmanyV_ix]
        print('[invest]   VSONE had %r verified matches to this image ' % (len(vsone_fm_V)))
        print('[invest]   In the top K=%r in this image...' % (K))
        print('[invest]   VSMANY had %r assignments to this image.' % (len(vsmany_qfxs)))
        print('[invest]   VSMANY had %r unique assignments to this image' % (len(np.unique(qfxs))))
        print('[invest]   VSMANY had %r verified assignments to this image' % (len(vsmany_fm_V)))
        print('[invest]   There were %r / %r intersecting matches in VSONE_V and VSMANY' % 
              (len(fm_intersect), len(vsone_fm_V)))
        print('[invest]   There were %r / %r intersecting verified matches in VSONE_V and VSMANY_V' % 
              (len(fm_V_intersect), len(vsone_fm_V)))
        print('[invest]   Distribution of kxs: '+helpers.printable_mystats(kxs))
        print('[invest]   Distribution of intersecting kxs: '+helpers.printable_mystats(isecting_kxs))
        # Visualize the intersecting matches 
        def _show_matches_helper(fm, fs, plotnum, title):
            df2.show_matches2(rchip1, rchip2, qfx2_kp1, fx2_kp2, fm, fs,
                              fignum=fnum, plotnum=plotnum, title=title, 
                              draw_pts=False)
        _show_matches_helper(vsmany_fm, vsmany_fs, (1,2,1), 'vsmany matches')
        #_show_matches_helper(vsmany_fm_V, vsmany_fs_V, (1,3,2), 'vsmany verified matches')
        _show_matches_helper(isecting_vsmany_fm, isecting_vsmany_fs, (1,2,2),
                             'intersecting vsmany K=%r matches' % (K,))
        df2.set_figtitle('vsmany K=%r qid%r vs cid%r'  % (K, qcid, ocid))
        # Hot colorscheme is black->red->yellow->white
        print('[invest] black->red->yellow->white')
        fnum+=1
    return fnum
