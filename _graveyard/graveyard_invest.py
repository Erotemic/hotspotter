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



#-------------


    #return qcx_list, ocid_list, note_list
    #qcx_list = hs.cid2_cx(qcid_list)
    #print('qcid_list = %r ' % qcid_list)
    #print('qcx_list = %r ' % qcid_list)
    #print('[get_hard_cases]\n %r\n %r\n %r\n' % (qcx_list, ocid_list, note_list))
@helpers.__DEPRICATED__
def get_hard_cases(hs):
    return get_cases(hs, with_hard=True, with_gt=False, with_nogt=False)

    #print('qcid_list = %r ' % qcid_list)
    #print('qcx_list = %r ' % qcid_list)
    #print('[get_hard_cases]\n %r\n %r\n %r\n' % (qcx_list, ocid_list, note_list))


def view_all_history_names_in_db(hs, db):
    qcx_list = zip(*get_hard_cases(hs))[0]
    nx_list = hs.tables.cx2_nx[qcx_list]
    unique_nxs = np.unique(nx_list)
    print('unique_nxs = %r' % unique_nxs)
    names = hs.tables.nx2_name[unique_nxs]
    print('names = %r' % names)
    helpers.ensuredir('hard_names')
    for nx in unique_nxs:
        viz.plot_name(hs, nx, fignum=nx)
        df2.save_figure(fpath='hard_names', usetitle=True)
    helpers.vd('hard_names')


    #return qcx_list, ocid_list, note_list
    #qcx_list = hs.cid2_cx(qcid_list)
    #print('qcid_list = %r ' % qcid_list)
    #print('qcx_list = %r ' % qcid_list)
    #print('[get_hard_cases]\n %r\n %r\n %r\n' % (qcx_list, ocid_list, note_list))
@helpers.__DEPRICATED__
def get_hard_cases(hs):
    return get_cases(hs, with_hard=True, with_gt=False, with_nogt=False)

    #print('qcid_list = %r ' % qcid_list)
    #print('qcx_list = %r ' % qcid_list)
    #print('[get_hard_cases]\n %r\n %r\n %r\n' % (qcx_list, ocid_list, note_list))


def view_all_history_names_in_db(hs, db):
    qcx_list = zip(*get_hard_cases(hs))[0]
    nx_list = hs.tables.cx2_nx[qcx_list]
    unique_nxs = np.unique(nx_list)
    print('unique_nxs = %r' % unique_nxs)
    names = hs.tables.nx2_name[unique_nxs]
    print('names = %r' % names)
    helpers.ensuredir('hard_names')
    for nx in unique_nxs:
        viz.plot_name(hs, nx, fignum=nx)
        df2.save_figure(fpath='hard_names', usetitle=True)
    helpers.vd('hard_names')
    



def quick_get_features_factory(hs):
    'builds a factory function'
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_cid  = hs.tables.cx2_cid 
    def get_features(cx):
        rchip = hs.get_chip(cx)
        fx2_kp = cx2_kpts[cx]
        fx2_desc = cx2_desc[cx]
        cid = cx2_cid[cx]
        return rchip, fx2_kp, fx2_desc, cid
    return get_features

def show_vsone_matches(hs, qcx, fnum=1):
    hs.ensure_matcher(match_type='vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    return res_vsone, fnum

def get_qfx2_gtkrank(hs, qcx, q_cfg): 
    '''Finds how deep possibily correct matches are in the ANN structures'''
    import matching_functions as mf
    dx2_cx   = q_cfg.data_index.ax2_cx
    gt_cxs   = hs.get_other_cxs(qcx)
    qcx2_nns = mf.nearest_neighbors(hs, [qcx], q_cfg)
    filt2_weights = mf.weight_neighbors(hs, qcx2_nns, q_cfg)
    K = q_cfg.nn_cfg.K
    (qfx2_dx, _) = qcx2_nns[qcx]
    qfx2_weights = filt2_weights['lnbnn'][qcx]
    qfx2_cx = dx2_cx[qfx2_dx[:, 0:K]]
    qfx2_gt = np.in1d(qfx2_cx.flatten(), gt_cxs)
    qfx2_gt.shape = qfx2_cx.shape
    qfx2_gtkrank = np.array([helpers.npfind(isgt) for isgt in qfx2_gt])
    qfx2_gtkweight = [0 if rank == -1 else weights[rank] 
                      for weights, rank in zip(qfx2_weights, qfx2_gtkrank)]
    qfx2_gtkweight = np.array(qfx2_gtkweight)
    return qfx2_gtkrank, qfx2_gtkweight

def measure_k_rankings(hs):
    'Reports the k match of correct feature maatches for each problem case'
    import match_chips3 as mc3
    K = 500
    q_cfg = mc3.QueryConfig(K=K, Krecip=0,
                                roidist_thresh=None, lnbnn_weight=1)
    id2_qcxs, id2_ocids, id2_notes = get_hard_cases(hs)
    id2_rankweight = [get_qfx2_gtkrank(hs, qcx, q_cfg) for qcx in id2_qcxs]
    df2.rrr()
    df2.reset()
    for qcx, rankweight, notes in zip(id2_qcxs, id2_rankweight, id2_notes):
        ranks, weights = rankweight
        ranks[ranks == -1] = K+1
        title = 'q'+hs.cxstr(qcx) + ' - ' + notes
        print(title)
        df2.figure(fignum=qcx, doclf=True, title=title)
        #draw_support
        #label=title
        df2.draw_hist(ranks, nbins=100, weights=weights) # FIXME
        df2.legend()
    print(len(id2_qcxs))
    df2.present(num_rc=(4,5), wh=(300,250))

def measure_cx_rankings(hs):
    ' Reports the best chip ranking over each problem case'
    import match_chips3 as mc3
    q_cfg = ds.QueryConfig(K=500, Krecip=0,
                                roidist_thresh=None, lnbnn_weight=1)
    id2_qcxs, id2_ocids, id2_notes = get_hard_cases(hs)
    id2_bestranks = []
    for id_ in xrange(len(id2_qcxs)):
        qcx = id2_qcxs[id_]
        reses = mc3.execute_query_safe(hs, q_cfg, [qcx])
        gt_cxs = hs.get_other_cxs(qcx)
        res = reses[2][qcx]
        cx2_score = res.get_cx2_score(hs)
        top_cxs  = cx2_score.argsort()[::-1]
        gt_ranks = [helpers.npfind(top_cxs == gtcx) for gtcx in gt_cxs]
        bestrank = min(gt_ranks)
        id2_bestranks += [bestrank]
    print(id2_bestranks)

param1 = 'K'
param2 = 'xy_thresh'
assign_alg = 'vsmany'
nParam1=1 
fnum = 1
nParam2=1
cx_list='gt1'
        #fnum = vary_query_params(hs, qcx, 'ratio_thresh', 'xy_thresh', 'vsone', 4, 4, fnum, cx_list='gt1')
    #if '2' in args.tests:
        #hs.ensure_matcher(match_type='vsmany')
        #fnum = vary_query_params(hs, qcx, 'K', 'xy_thresh', 'vsmany', 4, 4, fnum, cx_list='gt1') 

def linear_logspace(start, stop, num, base=2):
    return 2 ** np.linspace(np.log2(start), np.log2(stop), num)


def quick_assign_vsmany(hs, qcx, cx, K): 
    print('[invest] Performing quick vsmany')
    cx2_desc = hs.feats.cx2_desc
    vsmany_args = hs.matcher.vsmany_args
    cx2_fm, cx2_fs, cx2_score = mc2.assign_matches_vsmany(qcx, cx2_desc, vsmany_args)
    fm = cx2_fm[cx]
    fs = cx2_fs[cx]
    return fm, fs

def quick_assign_vsone(hs, qcx, cx, **kwargs):
    print('[invest] Performing quick vsone')
    cx2_desc       = hs.feats.cx2_desc
    vsone_args     = mc2.VsOneArgs(cxs=[cx], **kwargs)
    vsone_assigned = mc2.assign_matches_vsone(qcx, cx2_desc, vsone_args)
    (cx2_fm, cx2_fs, cx2_score) = vsone_assigned
    fm = cx2_fm[cx] ; fs = cx2_fs[cx]
    return fm, fs

