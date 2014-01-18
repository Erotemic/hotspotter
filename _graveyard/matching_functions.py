    #printDBG('Executing nnsearch K+Knorm=%r; checks=%r' % (K + Knorm, checks))
    #printDBG('flann = %r' % flann)
    #qcx2_nns = {qcx:func(qcx) for qcx in qcxs}
    #printDBG('[mf] Finding nearest neighbors of qcx=%r' % (qcx,))
    #printDBG('[mf] qfx2_desc.shape nearest neighbors of qcx=%r' % (qcx,))
    #helpers.printvar2('qfx2_desc', '.shape')


    #SPEEDprint('[mf] * computing %s weights' % nnfilter)
    #nnfilter_fn = eval('nn_filters.nn_' + nnfilter + '_weight')


    #printDBG('[mf] unique(dx2_cx) = %r ' % (np.unique(dx2_cx),))
    #printDBG('[mf] --------------')
    #printDBG('[mf] * scoring q' + hs.cidstr(qcx))
    #printDBG('[mf] * qcx  = %r ' % (qcx,))
    #printDBG('[mf] * unique(qfx2_cx) = %r ' % (np.unique(qfx2_cx),))
    #printDBG('[mf] * Removed %d/%d self-votes' % ((True - qfx2_notself_vote).sum(), qfx2_notself_vote.size))
    #printDBG('[mf] * %d/%d valid neighbors ' % (qfx2_valid.sum(), qfx2_valid.size))



    #printDBG(filt2_tw)
    #nValid  = qfx2_valid.sum()
    #printDBG('[mf] * \\ weight=%r' % weight)
    #nPassed = (True - qfx2_passed).sum()
    #nAdded = nValid - qfx2_valid.sum()
    #print(str(sign * qfx2_weights))
    #printDBG('[mf] * \\ *thresh=%r, nFailed=%r, nFiltered=%r' % (sign * thresh, nPassed, nAdded))
    #printDBG('[mf] * filt=%r ' % filt)
    #printDBG('[mf] * thresh=%r ' % thresh)
    #printDBG('[mf] * sign=%r ' % sign)
    #if isinstance(thresh, (int, float)) or not weight == 0:
        #printDBG('[mf] * \\ qfx2_weights = %r' % helpers.printable_mystats(qfx2_weights.flatten()))
            #qfx2_valid  = np.bitwise_and(qfx2_valid, qfx2_passed)


#s2coring_func  = [LNBNN, PlacketLuce, TopK, Borda]
#load_precomputed(cx, qdat)
    #elif score_method == 'nsum':
        #cx2_score, nx2_score = score_chipmatch_nsum(hs, qcx, chipmatch, qdat)
    #elif score_method == 'nunique':
        #cx2_score, nx2_score = score_chipmatch_nunique(hs, qcx, chipmatch, qdat)
    # Autoremove chips which are not the top scoring in their name
    #if hs.prefs.display_prefs.:
    #cx2_score = vr2.enforce_one_name_per_cscore(hs, cx2_score, chipmatch)
#SPEEDprint('[mf] * Scoring chipmatch: %s cx=%r' % (score_method, qcx))
#if USE_2_to_1:
    #sv_tup = sv2.homography_inliers(kpts2, kpts1, np.hstack(fm[:, 1], fm[:, 0]), xy_thresh, max_scale,
                                    #min_scale, dlen_sqrd, min_nInliers, #just_affine)
#else
#if not USE_2_to_1:
#if not use_chip_extent or USE_1_to_2:

# qcx2_chipmatch = matchesSVER

##============================
# Conversion to cx2 -> qfx2
#============================
@profile
def chipmatch2_neighbors(hs, qcx2_chipmatch, qdat):
    raise NotImplemented('almost')
    qcx2_nns = {}
    K = qdat.cfg.nn_cfg.K
    for qcx in qcx2_chipmatch.iterkeys():
        nQuery = len(hs.feats.cx2_kpts[qcx])
        # Stack the feature matches
        (cx2_fm, cx2_fs, cx2_fk) = qcx2_chipmatch[qcx]
        cxs = np.hstack([[cx] * len(cx2_fm[cx]) for cx in xrange(len(cx2_fm))])
        fms = np.vstack(cx2_fm)
        # Get the individual feature match lists
        qfxs = fms[:, 0]
        fxs  = fms[:, 0]
        fss  = np.hstack(cx2_fs)
        fks  = np.hstack(cx2_fk)
        # Rebuild the nearest neigbhor matrixes
        qfx2_cx = -np.ones((nQuery, K), ds.X_DTYPE)
        qfx2_fx = -np.ones((nQuery, K), qr.X_DTYPE)
        qfx2_fs = -np.ones((nQuery, K), qr.FS_DTYPE)
        qfx2_valid = np.zeros((nQuery, K), np.bool)
        # Populate nearest neigbhor matrixes
        for qfx, k in izip(qfxs, fks):
            assert qfx2_valid[qfx, k] is False
            qfx2_valid[qfx, k] = True
        for cx, qfx, k in izip(cxs, qfxs, fks):
            qfx2_cx[qfx, k] = cx
        for qfx, fx, k in izip(qfxs, fxs, fks):
            qfx2_fx[qfx, k] = fx
        for qfx, fs, k in izip(qfxs, fss, fks):
            qfx2_fs[qfx, k] = fs
        nns = (qfx2_cx, qfx2_fx, qfx2_fs, qfx2_valid)
        qcx2_nns[qcx] = nns
    return qcx2_nns

