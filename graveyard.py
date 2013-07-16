

# 1vM Agg info graveyard
#cx2_max_ax = np.cumsum(cx2_nFeats)
#_ax2_cid   = [[cid_]*nFeats for (cid_, nFeats) in iter(zip(cx2_cid, cx2_nFeats))]
#ax2_cid    = np.array(list(chain.from_iterable(_ax2_cid)))
# dont need no cid stuff here
#qfx2_cx = np.array([cx2_cid.index(cid) for cid in qfx2_cid.flat])
#qfx2_cx.shape = qfx2_cid.shape
#print('Baseline SIFT matching')
#print('len(qdesc) = %d' % len(qdesc))
#print('len(desc2) = %d' % len(desc2))
#print('len(matches) = %d' % len(matches))
#delta = 2000
#im_12 = warp.panorama(H_12,rchip1,rchip2,delta,delta)
#cx2_score_1vM = [np.sum(fs) for fs in cx2_fs_1vM]
#ax2_cx, ax2_fx, ax2_desc, flann_1vM = aggregate_1vM(cx2_cid, cx2_desc)
#matches_scores = assign_feat_matches_1vM(qcx, cx2_cid, cx2_desc, ax2_cx, ax2_fx, flann_1vM)
#cx2_num_fm_1v1  = [len(_) for _ in cx2_fs_1vM]

#rchip_path = cx2_rchip_path[0]
#sift_path = cx2_sift_path[0]
#sift = fc2.load_features(sift_path)

#kpts, desc = sift

#qcx = 1


#def FLANN_Searcher(object): 
    #def __init__(self, qdesc):
        #self.flann = FLANN()

        #self.flann.build_index(qdesc, **__FLANN_PARAMS__)
    #def neareset(desc2, K=1):
        #(idx21, dists21) = flann.nn_index(desc2, K, **__FLANN_PARAMS__)
        #idx21.shape   =  (desc2.shape[0], K)
        #dists21.shape =  (desc2.shape[0], K)
        #flann.delete_index()
        #return idx21.T, dists21.T
    ##flann.save_index(path)
    ##flann.load_index(path, qdesc)

def desc_matcher(qcx, cx2):
    ''' BruteForce, BruteForce-L1, BruteForce-Hamming,
    BruteForceHamming(2), FlannBased '''
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')

    matches = matcher.match(qdesc, desc2)
    return matches
#####
# DIRECTION 2 of __test_homog():
####
    #with Timer(msg=testname+' SV21'):
    #H21, inliers21 = func_homog(kpts2_m, kpts1_m, xy_thresh12_sqrd) 
    #print(' * num inliers21 = %d' % inliers21.sum())
    #fm1_SV2 = fm12[inliers21,:]
    #df2.show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm1_SV2, fignum=fignum+1, title=testname+' SV2')
    #df2.imshow(rchip1_H2, fignum=fignum+2, title=testname+' warped querychip1')
    #df2.imshow(rchip2_H2, fignum=fignum+3, title=testname+' warped reschip2')
    #print H2
    #rchip1_H2 = cv2.warpPerspective(rchip1, inv(H2), rchip2.shape[0:2][::-1])
    #rchip2_H2 = cv2.warpPerspective(rchip2,     H2, rchip1.shape[0:2][::-1])


def FREAK_assign_feat_matches_1v1(qcx, cx2_cid, cx2_freak):
    print('Assigning 1v1 feature matches from cx=%d to %d chips' % (qcx, len(cx2_cid)))
    qfreak = cx2_freak[qcx]
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    for cx, freak in enumerate(cx2_freak):
        sys.stdout.write('.')
        sys.stdout.flush()
        m = matcher.match(freak, qfreak)
        if cx == qcx: continue
        (fx2_qfx, fx2_dist) = flann_1v1.nn_index(freak, 2, **__FLANN_PARAMS__)
        # Lowe's ratio test
        fx2_ratio = np.divide(fx2_dist[:,1]+1, fx2_dist[:,0]+1)
        fx, = np.where(fx2_ratio > __1v1_RAT_THRESH__)
        qfx = fx2_qfx[fx,0]
        cx2_fm[cx] = np.array(zip(qfx, fx))
        cx2_fs[cx] = fx2_ratio[fx]
    sys.stdout.write('DONE')
    flann_1v1.delete_index()
    return cx2_fm, cx2_fs

def unpack_freak(cx2_desc):
    cx2_unpacked_freak = []
    for descs in cx2_desc:
        unpacked_desc = []
        for d in descs:
            bitstr = ''.join([('{0:#010b}'.format(byte))[2:] for byte in d])
            d_bool = np.array([int(bit) for bit in bitstr],dtype=bool)
            unpacked_desc.append(d_bool)
        cx2_unpacked_freak.append(unpacked_desc)
