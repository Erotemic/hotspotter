

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
