def precompute_bag_of_words2(hs):
    print('__PRECOMPUTING_BAG_OF_WORDS__')
    # Build (or reload) one vs many flann index
    feat_dir  = hs.dirs.feat_dir
    cache_dir = hs.dirs.cache_dir
    num_clusters = params.__VOCAB_SIZE__
    # Compute words
    ax2_cx, ax2_fx, ax2_desc = aggregate_descriptors_1vM(hs)
    ax2_wx, words = algos.precompute_akmeans(ax2_desc, 
                                             num_clusters,
                                             force_recomp=False, 
                                             cache_dir=cache_dir)
    # Build a NN index for the words
    flann_words = pyflann.FLANN()
    algorithm = 'default'
    flann_words_params = flann_words.build_index(words, algorithm=algorithm)
    print(' * bag of words is using '+linear+' NN search')
    # Compute Inverted File
    wx2_axs = [[] for _ in xrange(num_clusters)]
    for ax, wx in enumerate(ax2_wx):
        wx2_axs[wx].append(ax)
    wx2_cxs = [[ax2_cx[ax] for ax in ax_list] for ax_list in wx2_axs]
    wx2_fxs = [[ax2_fx[ax] for ax in ax_list] for ax_list in wx2_axs]
    # Create visual-word-vectors for each chip
    # Build bow using coorindate list coo matrix
    # The term frequency (TF) is implicit in the coo format
    coo_cols = ax2_wx
    coo_rows = ax2_cx
    coo_values = np.ones(len(ax2_cx), dtype=np.uint8)
    coo_format = (coo_values, (coo_rows, coo_cols))
    coo_cx2_bow = scipy.sparse.coo_matrix(
        coo_format, dtype=np.float, copy=True)
    # Normalize each visual vector
    csr_cx2_bow = scipy.sparse.csr_matrix(coo_cx2_bow, copy=False)
    csr_cx2_bow = sklearn.preprocessing.normalize(
        csr_cx2_bow, norm=BOW_NORM, axis=1, copy=False)
    # Calculate inverse document frequency (IDF)
    # chip indexes (cxs) are the documents
    num_chips = np.float(len(hs.tables.cx2_cid))
    wx2_df  = [len(set(cx_list)) for cx_list in wx2_cxs]
    wx2_idf = np.log2(num_chips / np.array(wx2_df, dtype=np.float))
    wx2_idf.shape = (1, wx2_idf.size)
    # Preweight the bow vectors
    idf_sparse = scipy.sparse.csr_matrix(wx2_idf)
    cx2_bow = scipy.sparse.vstack(
        [row.multiply(idf_sparse) for row in csr_cx2_bow], format='csr')
    # Renormalize
    cx2_bow = sklearn.preprocessing.normalize(
        cx2_bow, norm=BOW_NORM, axis=1, copy=False)
    # Return vocabulary

    wx2_fxs = np.array(wx2_fxs)
    wx2_cxs = np.array(wx2_cxs)
    wx2_idf = np.array(wx2_idf)
    wx2_idf.shape = (wx2_idf.size, )
    vocab = Vocabulary(words, flann_words, cx2_bow, wx2_idf, wx2_cxs, wx2_fxs)
    return vocab

def truncate_vvec(vvec, thresh):
    shape = vvec.shape
    vvec_flat = vvec.toarray().ravel()
    vvec_flat = vvec_flat * (vvec_flat > thresh)
    vvec_trunc = scipy.sparse.csr_matrix(vvec_flat)
    vvec_trunc = sklearn.preprocessing.normalize(
        vvec_trunc, norm=BOW_NORM, axis=1, copy=False)
    return vvec_trunc

def test__():
    vvec_flat = np.array(vvec.toarray()).ravel()
    qcx2_wx   = np.flatnonzero(vvec_flat)
    cx2_score = (cx2_bow.dot(vvec.T)).toarray().ravel()

    cx2_nx = hs.tables.cx2_nx
    qnx = cx2_nx[qcx]
    other_cx, = np.where(cx2_nx == qnx)

    vvec2 = truncate_vvec(vvec, .04)
    vvec2_flat = vvec2.toarray().ravel()

    cx2_score = (cx2_bow.dot(vvec.T)).toarray().ravel()
    top_cxs = cx2_score.argsort()[::-1]

    cx2_score2 = (cx2_bow.dot(vvec2.T)).toarray().ravel()
    top_cxs2 = cx2_score2.argsort()[::-1]

    comp_str = ''
    tst2s2_list = np.vstack([top_cxs, cx2_score, top_cxs2, cx2_score2]).T
    for t, s, t2, s2 in tst2s2_list:
        m1 = [' ', '*'][t in other_cx]
        m2 = [' ', '*'][t2 in other_cx] 
        comp_str += m1 + '%4d %4.2f %4d %4.2f' % (t, s, t2, s2) + m2 + '\n'
    print comp_str
    df2.close_all_figures()
    df2.show_signature(vvec_flat, fignum=2)
    df2.show_signature(vvec2_flat, fignum=3)
    df2.present()
    #df2.show_histogram(qcx2_wx, fignum=1)
    pass

        #_qfs = vvec_flat[wx]
            #_fs = cx2_bow[cx, wx]
            #fs  = _qfs * _fs
    #print helpers.toc(t)


    #t = helpers.tic()
    #cx2_fm = [[] for _ in xrange(len(cx2_cid))]
    #cx2_fs = [[] for _ in xrange(len(cx2_cid))]
    #qfx2_cxlist = wx2_cxs[qfx2_wx] 
    #qfx2_fxlist = wx2_fxs[qfx2_wx] 
    #qfx2_fs = wx2_idf[qfx2_wx] 
    #for qfx, (fs, cx_list, fx_list) in enumerate(zip(qfx2_fs, 
                                                     #qfx2_cxlist, 
                                                     #qfx2_fxlist)):
        #for (cx, fx) in zip(cx_list, fx_list): 
            #if cx == qcx: continue
            #fm  = (qfx, fx)
            #cx2_fm[cx].append(fm)
            #cx2_fs[cx].append(fs)
    #cx2_fm_ = cx2_fm
    #cx2_fs_ = cx2_fs
    #print helpers.toc(t)
    #t = helpers.tic()

# the inverted file (and probably tf-idf scores too)
# Compute word assignments of database descriptors
def __assign_tfidf_vvec(cx2_desc, words, flann_words):
    '''
    Input: cx2_desc 
    Description: 
        Assigns the descriptors in each chip to a visual word
        Makes a sparse vector weighted by chip term frequency
        Compute the iverse document frequency 
    Output: cx2_vvec, wx2_idf
    '''
    # Compute inverse document frequency (IDF)
    num_train = len(train_cxs)
    wx2_idf   = __idf(wx2_cxs, num_train)
    # Preweight the bow vectors
    cx2_vvec  = algos.sparse_multiply_rows(csr_cx2_vvec, wx2_idf)
    # Normalize
    cx2_vvec = algos.sparse_normalize_rows(cx2_vvec)
    return cx2_vvec, wx2_idf

def __compute_vocabulary(hs, train_cxs):
    # hotspotter data
    cx2_cid  = hs.tables.cx2_cid
    cx2_desc = hs.feats.cx2_desc
    # training information
    tx2_cid  = cx2_cid[train_cxs]
    tx2_desc = cx2_desc[train_cxs]



#------------------- convert
@helpers.__DEPRICATED__
def __read_ox_gtfile_OLD(gt_fpath, gt_name, quality):
    ox_chip_info_list = []
    with open(gt_fpath,'r') as file:
        line_list = file.read().splitlines()
        for line in line_list:
            if line == '': continue
            fields = line.split(' ')
            gname = fields[0].replace('oxc1_','')+'.jpg'
            if gname.find('paris_') >= 0: 
                # PARIS HACK >:(
                #Because they just cant keep their paths consistent 
                paris_hack = gname[6:gname.rfind('_')]
                gname = paris_hack+'/'+gname
            if gname in corrupted_gname_list: continue
            if len(fields) > 1: #quality == query
                roi = map(lambda x: int(round(float(x))),fields[1:])
            else: # quality in ['good','ok','junk']
                gpath = os.path.join(img_dpath, gname)
                (w,h) = Image.open(gpath).size
                roi = [0,0,w,h]
            ox_chip_info = (gt_name, gname, roi, quality, gt_fpath)
            ox_chip_info_list.append(ox_chip_info)
    return ox_chip_info_list
                     
@helpers.__DEPRICATED__
def convert_from_oxford_sytle_OLD(db_dir):
    hs_dirs, hs_tables = load_data2.load_csv_tables(db_dir)

    # Get directories for the oxford groundtruth
    oxford_gt_dpath      = join(db_dir, 'oxford_style_gt')
    corrupted_file_fpath = join(db_dir, 'corrupted_files.txt')

    # Check for corrupted files (Looking at your Paris Buildings Dataset)
    corrupted_gname_list = []
    if helpers.checkpath(corrupted_file_fpath):
        with open(corrupted_file_fpath) as f:
            corrupted_gname_list = f.read().splitlines()
        corrupted_gname_list = set(corrupted_gname_list)
    print('Loading Oxford Style Images')

    # Recursively get relative path of all files in img_dpath
    img_dpath  = join(db_dir, 'images')
    gname_list = [join(relpath(root, img_dpath), fname).replace('\\','/').replace('./','')\
                    for (root,dlist,flist) in os.walk(img_dpath)
                    for fname in flist]

    print('Loading Oxford Style Names and Chips')
    # Read the Oxford Style Groundtruth files
    gt_fname_list = os.listdir(oxford_gt_dpath)

    print('There are %d ground truth files' % len(gt_fname_list))
    total_chips = 0
    quality2_nchips = {'good':0,'bad':0,'ok':0,'junk':0,'query':0}
    gtname2_nchips = {}
    qtx2_nchips = []
    ox_chip_info_list = []
    for gt_fname in iter(gt_fname_list):
        #Get gt_name, quality, and num from fname
        (gt_name, quality, num) = __oxgtfile2_oxgt_tup(gt_fname)
        gt_fpath = join(oxford_gt_dpath, gt_fname)
        ox_chip_info_sublist = __read_ox_gtfile_OLD(gt_fpath, gt_name, quality)
        num_subchips = len(ox_chip_info_sublist)
        ox_chip_info_list.extend(ox_chip_info_sublist)
        # Sanity
        # count number of chips
        quality2_nchips[quality] += num_subchips
        total_chips              += num_subchips
        if not gt_name in gtname2_nchips.keys():
            gtname2_nchips[gt_name] = 0
        gtname2_nchips[gt_name] += num_subchips
        qtx2_nchips.append(num_subchips)

    import numpy as np
    print('Total num: %d ' % total_chips)
    print('Quality breakdown: '+repr(quality2_nchips).replace(',',',\n    '))
    print('GtName breakdown: '+repr(gtname2_nchips).replace(',',',\n    '))
    print('Quality Sanity Total %d: ' % np.array(map(int, quality2_nchips.values())).sum())
    print('Gt_Name Sanity Total %d: ' % np.array(map(int, gtname2_nchips.values())).sum())

    gtx2_name  = [tup[0] for tup in iter(ox_chip_info_list)]
    gtx2_gname = [tup[1] for tup in iter(ox_chip_info_list)]

    unique_gnames = set(gtx2_gname)

    print('Total chips: %d ' % len(gtx2_gname))
    print('Unique chip gnames: %d ' % len(unique_gnames))

    gname2_info = {}
    query_images = []
    for (name, gname, roi,  quality, fpath) in iter(ox_chip_info_list):
        if quality == 'query':
            query_images.append(gname)
        if not gname in gname2_info.keys():
            gname2_info[gname] = []
        gname2_info[gname].append((name, quality, roi, fpath))

    for gname in gname2_info.keys():
        info_list = gname2_info[gname]
        print '-------------'
        print '\n    '.join(map(repr, [gname]+info_list))


    print('This part prints out which files the query images exist in')
    import subprocess
    os.chdir(oxford_gt_dpath)
    for qimg in query_images:
        print('-----')
        qimg = qimg.replace('.jpg','')
        cmd = ['grep', qimg, '*.txt']
        cmd2 = ' '.join(cmd)
        os.system(cmd2)

    print('Num query images: ' + str(len(query_images)))
    print('Num unique query images: ' + str(len(set(query_images))))


    print('Total images: %d ' % len(gname_list))
    print('Total unique images: %d ' % len(set(gname_list)))

    len(ox_chip_info_list)
    print '\n'.join((repr(cinfo) for cinfo in iter(ox_chip_info_list)))
    # HACKISH Duplicate detection. Eventually this should actually be in the codebase
    print('Detecting and Removing Duplicate Ground Truth')
    dup_cx_list = []
    for nx in nm.get_valid_nxs():
        cx_list = array(nm.nx2_cx_list[nx])
        gx_list = cm.cx2_gx[cx_list]
        (unique_gx, unique_x) = np.unique(gx_list, return_index=True)
        name = nm.nx2_name[nx]
        for gx in gx_list[unique_x]:
            bit = False
            gname = gm.gx2_gname[gx]
            x_list = pylab.find(gx_list == gx)
            cx_list2  = cx_list[x_list]
            roi_list2 = cm.cx2_roi[cx_list2]
            roi_hash = lambda roi: roi[0]+roi[1]*10000+roi[2]*100000000+roi[3]*1000000000000
            (_, unique_x2) = np.unique(map(roi_hash, roi_list2), return_index=True)
            non_unique_x2 = np.setdiff1d(np.arange(0,len(cx_list2)), unique_x2)
            for nux2 in non_unique_x2:
                cx_  = cx_list2[nux2]
                dup_cx_list += [cx_]
                roi_ = roi_list2[nux2]
                print('Duplicate: cx=%4d, gx=%4d, nx=%4d roi=%r' % (cx_, gx, nx, roi_) )
                print('           Name:%s, Image:%s' % (name, gname) )
                bit = True
            if bit:
                print('-----------------')
    for cx in dup_cx_list:
        cm.remove_chip(cx)

#---------end convert
