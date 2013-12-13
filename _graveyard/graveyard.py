if not 'hs' in vars():
    hs = ld2.HotSpotter()
    hs.load_all(params.GZ)
    qcx = 111
    cx = 305
    # Database descriptor + keypoints
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    def get_features(cx):
        rchip = hs.get_chip(cx)
        rchip_size = cx2_rchip_size[cx]
        fx2_kp   = cx2_kpts[cx]
        fx2_scale = sv2.keypoint_scale(fx2_kp)
        fx2_desc = cx2_desc[cx]
        return rchip, rchip_size, fx2_kp, fx2_scale, fx2_desc
    # Query features
    rchip1, rchip_size1, fx2_kp1, fx2_scale1, fx2_desc1 = get_features(qcx)
    # Result features
    rchip2, rchip_size2, fx2_kp2, fx2_scale2, fx2_desc2 = get_features(cx)
    # Vsmany index
    vsmany_index = hs.matcher._Matcher__vsmany_index
    #c2.precompute_index_vsmany(hs)
    #qcx2_res = mc2.run_matching(hs)

    #params.__MATCH_TYPE__ = 'bagofwords'
    #hs.load_matcher()
    #resBOW = mc2.build_result_qcx(hs, qcx)
    #df2.show_match_analysis(hs, resBOW, N=5, fignum=1)
    
    params.__MATCH_TYPE__ = 'vsmany'
    hs.load_matcher()
    params.__VSMANY_SCORE_FN__ = 'LNRAT'
    resLNRAT = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resLNRAT, N=5, fignum=1)

    params.__VSMANY_SCORE_FN__ = 'LNBNN'
    resLNBNN = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resLNBNN, N=5, fignum=2)

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    resRATIO = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resRATIO, N=5, fignum=3)

    params.__VSMANY_SCORE_FN__ = 'RATIO'
    
    params.__MATCH_TYPE__ = 'vsone'
    hs.load_matcher()
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=4)
    df2.present()

    #allres = init_allres(hs, qcx2_res, SV, oxford=oxford)

    def get_vsmany_all_data():
        vsmany_all_assign = mc2.assign_matches_vsmany(qcx, cx2_desc, vsmany_index)
        cx2_fm, cx2_fs, cx2_score = vsmany_all_assign
        vsmany_all_svout = mc2.spatially_verify_matches(qcx, cx2_kpts, cx2_rchip_size, cx2_fm, cx2_fs)
        return vsmany_all_assign, vsmany_all_svout

    def get_vsmany_data(vsmany_all_data, cx):
        ' Assigned matches (vsmany)'
        vsmany_all_assign, vsmany_all_svout = vsmany_all_data
        vsmany_cx_assign = map(lambda _: _[cx],  vsmany_all_assign)
        vsmany_cx_svout  = map(lambda _: _[cx],  vsmany_all_svout)
        return vsmany_cx_assign, vsmany_cx_svout

    def get_vsone_data(cx):
        ' Assigned matches (vsone)'
        vsone_flann, checks = mc2.get_vsone_flann(fx2_desc1)
        fm, fs = mc2.match_vsone(fx2_desc2, vsone_flann, checks)
        fm_V, fs_V, H = mc2.spatially_verify(fx2_kp1, fx2_kp2, rchip_size2, fm, fs, qcx, cx)
        score = fs.sum(); score_V = fs_V.sum()
        vsone_cx_assign = fm, fs, score
        vsone_cx_svout = fm_V, fs_V, score_V
        return vsone_cx_assign, vsone_cx_svout

    # Assign + Verify
    params.__USE_CHIP_EXTENT__ = False
    vsmany_all_data = get_vsmany_all_data()
    vsmany_data = get_vsmany_data(vsmany_all_data, cx)
    vsone_data = get_vsone_data(cx)

    params.__USE_CHIP_EXTENT__ = True
    vsmany_all_data2 = get_vsmany_all_data()
    vsmany_data2 = get_vsmany_data(vsmany_all_data2, cx)
    vsone_data2 = get_vsone_data(cx)

    def show_matchers_compare(vsmany_data, vsone_data, fignum=0, figtitle=''):
        vsmany_cx_assign, vsmany_cx_svout = vsmany_data
        vsone_cx_assign, vsone_cx_svout = vsone_data
        # Show vsmany
        fm, fs, score = vsmany_cx_assign
        fm_V, fs_V, score_V = vsmany_cx_svout
        plot_kwargs = dict(all_kpts=False, fignum=fignum)
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm, fs,
                          plotnum=(2,2,1), title='vsmany assign', **plot_kwargs)
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm_V, fs_V,
                          plotnum=(2,2,2),  title='vsmany verified', **plot_kwargs)
        # Show vsone
        fm, fs, score      = vsone_cx_assign
        fm_V, fs_V, score_V = vsone_cx_svout
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm, fs,
                          plotnum=(2,2,3), title='vsone assign', **plot_kwargs)
        df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm_V, fs_V,
                          plotnum=(2,2,4), title='vsone verified', **plot_kwargs)
        df2.set_figtitle(figtitle)

    show_matchers_compare(vsmany_data, vsone_data, fignum=1, figtitle='kpt extent')
    show_matchers_compare(vsmany_data2, vsone_data2, fignum=2, figtitle='chip extent')
    df2.update()
    fx2_feature = hs.get_feature_fn(qcx)

def target_dsize(img, M):
    # Get img bounds under transformation
    (minx, maxx, miny, maxy) = sv2.transformed_bounds(rchip, M)
    Mw, Mh = (maxx-minx, maxy-miny)
    # If any border forced below, return a translation to append to M
    tx = -min(0, minx)
    ty = -min(0, miny)
    # Round to integer size
    dsize = tuple(map(int, np.ceil((Mw, Mh))))
    return dsize, tx, ty

    def get_features(cx):
        rchip = hs.get_chip(cx)
        rchip_size = cx2_rchip_size[cx]
        fx2_kp   = hs.feats.cx2_kpts[cx]
        fx2_scale = sv2.keypoint_scale(fx2_kp)
        fx2_desc = hs.feats.cx2_desc[cx]
        return rchip, rchip_size, fx2_kp, fx2_scale, fx2_desc
    rchip1, rchip_size1, fx2_kp1, fx2_scale1, fx2_desc1 = get_features(qcx)
    rchip2, rchip_size2, fx2_kp2, fx2_scale2, fx2_desc2 = get_features(cx)


    #--------------
    def free_some_memory(hs):
        print('[hs] Releasing matcher memory')
        import gc
        helpers.memory_profile()
        print("[hs] HotSpotter Referrers: "+str(gc.get_referrers(hs)))
        print("[hs] Matcher Referrers: "+str(gc.get_referrers(hs.matcher)))
        print("[hs] Desc Referrers: "+str(gc.get_referrers(hs.feats.cx2_desc)))
        #reffers = gc.get_referrers(hs.feats.cx2_desc) #del reffers
        del hs.feats.cx2_desc
        del hs.matcher
        gc.collect()
        helpers.memory_profile()
        ans = raw_input('[hs] good?')
