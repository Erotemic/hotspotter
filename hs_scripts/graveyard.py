

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


# SPATIAL VERIFICATION PARAMS SETUP
img1_extent = (kpts1_m[0:2,:].max(1) - kpts1_m[0:2,:].min(1))[0:2]
img2_extent = (kpts2_m[0:2,:].max(1) - kpts2_m[0:2,:].min(1))[0:2]
xy_thresh12_sqrd = np.sum(img1_extent**2) * (__xy_thresh_percent__**2)
xy_thresh21_sqrd = np.sum(img2_extent**2) * (__xy_thresh_percent__**2)

__PRINT_THRESH_INFO__ = False
if __PRINT_THRESH_INFO__:
    print('---------------------------------------')
    print(' * Threshold is %.1f%% of diagonal length' % (__xy_thresh_percent__*100))
    print('Computing the xy_threshold:')
    print(' * img1_extent = %r ' % img1_extent)
    print(' * img2_extent = %r ' % img2_extent)
    print(' * img1_diag_len = %.2f ' % np.sqrt(np.sum(img1_extent**2)))
    print(' * img2_diag_len = %.2f ' % np.sqrt(np.sum(img2_extent**2)))
    print(' * xy_thresh12_sqrd=%.2f' % np.sqrt(xy_thresh12_sqrd))
    print(' * xy_thresh21_sqrd=%.2f' % np.sqrt(xy_thresh21_sqrd))
    print('---------------------------------------')


def gen_subset_split(full_set, M, K):
    np.random.seed(0) # repeatibility
    seen = set([])
    split_list = []
    for kx in xrange(K):
        np.random.shuffle(full_set)
        failsafe = 0
        while True: 
            np.random.shuffle(full_set)
            subset = tuple(full_set[0:M])
            if not subset in seen: 
                seen.add(subset)
                compliment = tuple(np.setdiff1d(full_set, subset))
                yield (compliment, subset)
                break
            failsafe += 1
            if failsafe > 100:
                break

def test_entropy_internals(desc):
    fig = df2.figure(1, doclf=True)
    max_bw = 5
    for ix in range(max_bw):
        bw_factor = (ix + 1)**2
        print('bw=%d' % bw_factor)
        prob_x1 = _hist_prob_x(desc, bw_factor)
        prob_x2 = _gkde_prob_x(desc, bw_factor)
        entropy1 = [-(px * np.log2(px)).sum() for px in prob_x1]
        entropy2 = [-(px * np.log2(px)).sum() for px in prob_x2]
        x = sorted(entropy1)
        y = sorted(entropy2)
        fig = df2.figure(1, plotnum=(max_bw, 2, ix*2+1), title='sorted bw=%d' % bw_factor)
        plt.plot(x, y)
        fig = df2.figure(1, plotnum=(max_bw, 2, ix*2+2), title='scatter bw=%d' % bw_factor)
        plt.plot(entropy1, entropy2, 'go')
    fig.tight_layout()
    df2.update()




    # Renormalize descriptor to have an l2 norm of 1
    desc1 = np.array(desc1, dtype=float) 
    l2norm1 = np.sqrt((desc1**2).sum(1))
    desc1 /= l2norm1[:, np.newaxis]
    desc2 = np.array(desc2, dtype=float)
    l2norm2 = np.sqrt((desc2**2).sum(1))
    desc2 /= l2norm2[:, np.newaxis]
    desc_hist = np.histogram(desc1[0], bins=32, density=True)[0]
    def check(desc):
        norm = np.sqrt((desc**2).sum(1))
        print('norm: %r ' % norm)
        print('shape: %r ' % norm.shape)
        print('mean: %r ' % np.mean(norm))
        print('std: %r ' % np.std(norm))
    check(desc1)
    check(desc2)
    print('DESC1: %r ' % np.sqrt((desc1**2).sum(1)))
    print('DESC2: %r ' % np.sqrt((desc2**2).sum(1)))
    print('DESC1: %r ' % np.sqrt((desc1**2).sum(0)))
    print('DESC2: %r ' % np.sqrt((desc2**2).sum(0)))

    print rank
orgres.qcxs
orgres.cxs



    def get_sort_and_x(scores):
        scores = np.array(scores)
        scores_sortx = scores.argsort()[::-1]
        scores_sort  = scores[scores_sortx]
        return scores_sort, scores_sortx
    tt_sort, tt_sortx = get_sort_and_x(allres.top_true.scores)
    tf_sort, tf_sortx = get_sort_and_x(allres.top_false.scores)



    #orgres = allres.top_true
    #qcx, cx, score, rank = orgres.iter().next()
    #res = qcx2_res[qcx]
    #fm = res.cx2_fm_V[cx]
    ## Get matching descriptors
    #desc1 = cx2_desc[qcx][fm[:,0]]
    #desc2 = cx2_desc[cx ][fm[:,1]]


def leave_out(expt_func=None, **kwargs):
    '''
    do with TF-IDF on the zebra data set. 
    Let M be the total number of *animals* (not images and not chips) in an experimental data set. 
    Do a series of leave-M-out (M >= 1) experiments on the TF-IDF scoring,
    where the "left out" M are M different zebras, 
    so that there are no images of these zebras in the images used to form the vocabulary.
    The vocabulary is formed from the remaining N-M animals.
    Test how well TF-IDF recognition does with these M animals. 
    Repeat for different subsets of M animals.
    import experiments as expt
    from experiments import *
    '''
    # ---
    # Testing should have animals I have seen and animals I haven't seen. 
    # Make sure num descriptors -per- word is about the same as Oxford 
    # ---
    # Notes from Monday: 
    # 1) Larger training set (see how animals in training do vs animals out of training)
    # 2) More detailed analysis of failures
    # 3) Aggregate scores across different pictures of the same animal
    if not 'expt_func' in vars() or expt_func is None:
        expt_func = run_experiment
    # Load tables
    hs = ld2.HotSpotter(ld2.DEFAULT, load_basic=True)
    # Grab names
    nx2_name   = hs.tables.nx2_name
    cx2_nx     = hs.tables.cx2_nx
    nx2_cxs    = np.array(hs.get_nx2_cxs())
    nx2_nChips = np.array(map(len, nx2_cxs))
    num_uniden = nx2_nChips[0] + nx2_nChips[1] 
    nx2_nChips[0:3] = 0 # remove uniden names
    # Seperate singleton / multitons
    multiton_nxs, = np.where(nx2_nChips > 1)
    singleton_nxs, = np.where(nx2_nChips == 1)
    all_nxs = np.hstack([multiton_nxs, singleton_nxs]) 
    print('[expt] There are %d names' % len(all_nxs))
    print('[expt] There are %d multiton names' % len(multiton_nxs))
    print('[expt] There are %d singleton names' % len(singleton_nxs))
    print('[expt] There are %d unidentified animals' % num_uniden)
    # 
    multiton_cxs = nx2_cxs[multiton_nxs]
    singleton_cxs = nx2_cxs[singleton_nxs]
    multiton_nChips = map(len, multiton_cxs)
    print('[expt] multion #cxs stats: %r' % helpers.printable_mystats(multiton_nChips))
    # Find test/train splits
    num_names = len(multiton_cxs)

    # How to generate samples/splits for names
    num_nsplits = 3
    nsplit_size = (num_names//num_nsplits)

    # How to generate samples/splits for chips
    csplit_size = 1 # number of indexed chips per Jth experiment

    # Generate name splits
    kx2_name_split = far_appart_splits(multiton_nxs, nsplit_size, num_nsplits)
    result_map = {}
    kx = 0
    # run K experiments
    all_cxs = nx2_cxs[list(all_nxs)]
    for kx in xrange(num_nsplits):
        print('***************')
        print('[expt] Leave M=%r names out iteration: %r/%r' % (nsplit_size, kx+1, num_nsplits))
        print('***************')
        # Get name splits
        (test_nxs, train_nxs) = kx2_name_split[kx]
        # Lock in training set
        # train_nxs
        train_cxs_list = nx2_cxs[list(train_nxs)]
        train_samp = np.hstack(train_cxs_list)
        # 
        # Choose test / index smarter
        #test_samp = np.hstack(test_cxs_list)    # Test on half
        #indx_samp = np.hstack([test_samp, train_samp]) # Search on all
        #
        # Generate chip splits
        test_cxs_list = nx2_cxs[list(test_nxs)]
        test_nChip = map(len, test_cxs_list)
        print('[expt] testnames #cxs stats: %r' % helpers.printable_mystats(test_nChip))
        test_cx_splits  = []
        for ix in xrange(len(test_cxs_list)):
            cxs = test_cxs_list[ix]
            num_csplits = len(cxs)//csplit_size
            cxs_splits = far_appart_splits(cxs, csplit_size, num_csplits)
            test_cx_splits.append(cxs_splits)
        max_num_csplits = max(map(len, test_cx_splits))
        # Put them into experiment sets
        jx2_test_cxs = [[] for _ in xrange(max_num_csplits)]
        jx2_index_cxs = [[] for _ in xrange(max_num_csplits)]
        for ix in xrange(len(test_cx_splits)):
            cxs_splits = test_cx_splits[ix]
            for jx in xrange(max_num_csplits):
                if jx >= len(cxs_splits): 
                    break
                #ix_test_cxs, ix_index_cxs = cxs_splits[jx]
                ix_index_cxs, ix_test_cxs = cxs_splits[jx]
                jx2_test_cxs[jx].append(ix_test_cxs)
                jx2_index_cxs[jx].append(ix_index_cxs)
        jx = 0
        for jx in xrange(max_num_csplits): # run K*J experiments
            # Lock in test and index set
            #all_cxs # np.hstack(jx2_test_cxs[jx])
            indx_samp = np.hstack(jx2_index_cxs[jx]+[train_samp])
            # Run all the goddamn queries (which have indexed ground truth)
            test_samp = hs.get_cxs_in_sample(indx_samp)
            # Set samples
            hs.set_samples(test_samp, train_samp, indx_samp)
            mj_label = '[LNO:%r/%r;%r/%r]' % (kx+1, num_nsplits, jx+1, max_num_csplits)
            # Run experiment
            print('[expt] <<<<<<<<')
            print('[expt] Run expt_func()')
            print('[expt] M=%r, J=%r' % (nsplit_size,csplit_size))
            print(mj_label)
            #rss = helpers.RedirectStdout('[expt %d/%d]' % (kx, K)); rss.start()
            expt_locals = expt_func(hs, pprefix=mj_label, **kwargs)
            print('[expt] Finished expt_func()')
            print('[expt] mth iteration: %r/%r' % (kx+1, num_nsplits))
            print('[expt] jth iteration: %r/%r' % (jx+1, max_num_csplits))
            print('[expt] >>>>>>>>')
            result_map[kx] = expt_locals['allres']
            #rss.stop(); rss.dump()
    return locals()
'''
this is interesting
0 - 1 = -1 
0 - 0 - 1 = -1? idk, why?
   (x - y) =    (z)
-1*(x - y) = -1*(z)
  -(x + y) =   -(z)
    -x + y = -z

let x=0
let y=1
let z=-1
   (0 - 1) =    (-1)
-1*(0 - 1) = -1*(-1)
  -(0 + 1) =   -(-1)
    -0 + 1 =    --1
    -0 + 1 = 1
         1 = 1 + 0
         1 = 1

let x=0
let a=0
let y=1
let z=-1
   (a - x - y) =    (z)
-1*(a - x - y) = -1*(z)
  -(a - x + y) =   -(z)
    -a - x + y = -z

   (0 - 0 - 1) =    (-1)
-1*(0 - 0 - 1) = -1*(-1)
  -(0 - 0 + 1) =   -(-1)
    -0 - 0 + 1 = --1
'''
