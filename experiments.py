# We have 5,202,157 descriptors
# They claim to have 16,334,970 descriptors

def reload_modules():
    import imp
    imp.reload(mc2)
    imp.reload(helpers)

def get_oxsty_mAP_score(hs, cx2_res, SV=False):
    query_mAP_list = []
    for qcx in (hs.test_sample_cx):
        res = cx2_res[qcx]
        mAP = get_oxsty_mAP_score_from_res(hs, res, SV)
        query_mAP_list.append(mAP)

    total_mAP = np.mean(np.array(query_mAP_list))
    return total_mAP, query_mAP_list

import subprocess
from subprocess import PIPE
import os
from os.path import join

def get_oxsty_mAP_score_from_res(hs, res, SV):
    # find oxford ground truth directory
    cwd = os.getcwd()
    oxford_gt_dir = join(hs.dirs.db_dir, 'oxford_style_gt')
    # build groundtruth query
    qcx = res.qcx
    qnx = hs.tables.cx2_nx[qcx]
    oxnum_px = np.where(hs.tables.px2_propname == 'oxnum')[0][0]
    #oxnum_px  = hs.tables.px2_propname.index('oxnum')
    cx2_oxnum = hs.tables.px2_cx2_prop[oxnum_px]
    qoxnum = cx2_oxnum[qcx]
    qname  = hs.tables.nx2_name[qnx]
    #groundtruth_query = 
    # build ranked list
    cx2_score, cx2_fm, cx2_fs = res.get_info(SV)
    top_cx = cx2_score.argsort()[::-1]
    top_gx = hs.tables.cx2_gx[top_cx]
    top_gname = hs.tables.gx2_gname[top_gx]
    # build mAP args
    ground_truth_query = qname+'_'+qoxnum
    # build ranked list of gnames (remove duplicates)
    seen = set([])
    ranked_list = []
    for gname in iter(top_gname):
        gname_ = gname.replace('.jpg','')
        if not gname_ in seen: 
            seen.add(gname_)
            ranked_list.append(gname_)
    ranked_list2 = [gname.replace('.jpg','') for gname in top_gname]
    SV_aug = ['','SV_'][SV]
    ranked_list_fname = join(hs.dirs.result_dir, 'ranked_list_'+SV_aug+ground_truth_query+'.txt')
    helpers.write_to(ranked_list_fname, '\n'.join(ranked_list))
    # execute external mAP code
    os.chdir(oxford_gt_dir)
    args = ('../compute_ap', ground_truth_query, ranked_list_fname)
    cmdstr  = ' '.join(args)
    print('Executing: %r ' % cmdstr)
    proc = subprocess.Popen(args, stdout=PIPE, stderr=PIPE)
    (out, err) = proc.communicate()
    return_code = proc.returncode
    os.chdir(cwd)
    mAP = float(out.strip())
    return mAP

def oxford_bag_of_words_reproduce_philbin07():
    import load_data2
    import match_chips2 as mc2
    import params
    import numpy as np
    import helpers

    # TODO: No resize chips
    # TODO: Orientation assignment / Mikj detectors

    # I guess no orientation
    #These three models take advantage of the fact that images are usually
    #displayed on the web with the correct (upright) orientation. For this
    #reason, we have not allowed for in-plane image rotations.

    '''
    We ﬁnd that the two-way transfer error with scale threshold performs the
    best on the data. In cases where a simpler one-way transfer error sufﬁces,
    we can speed up veriﬁcation when there is a high visual-word
    multiplicity between the two images (usually for smaller vocabularies). We
    merely need to ﬁnd the spatially closest matching feature in the target to a
    particular query feature and check whether this single distance is less than
    the threshold. This is done using a 2D k-d tree, to provide logarithmic
    time search. In all experiments where “spatial” is speciﬁed, we have
    used our spatial veriﬁcation procedure to re-rank up to the top 1000
    images. 
    
    We consider spatial veriﬁcation to be “successful” if we ﬁnd a
    transformation with at least 4 inlier correspondences. We re-rank the
    images by scoring them equal to the sum of the idf values for the inlier
    words andh place spatially veriﬁed images above unveriﬁed ones in the
    ranking. 
    
    We abort the re-ranking early (after considering fewer than 1000
    images) if we process 20 images in a row without a successful spatial
    veriﬁcation. We ﬁnd this gives a good trade-off between speed and accuracy
    '''

    helpers.__PRINT_CHECKS__ = True
    # The vocab sizes run by philbin et al 
    vocab_sizes = [1e4, 2e4, 5e4, 1e6, 1.25e6] 

    rerank_nums = [100,200,400,800]

    dof = [3,4,5]

    # They use 8 trees in for their AKMEANS. Unsure how many checks
    philbin_params = {'algorithm':'kdtree',
                      'trees'    :8,
                      'checks'   :64}

    params.__NUM_WORDS__ = vocab_sizes[0]
    params.__NUM_RERANK__ = rerank_nums[0]
    params.__FLANN_ONCE_PARAMS__ = philbin_params
    dbdir = load_data2.OXFORD
    hs = mc2.HotSpotter(dbdir, load_matcher=False)

    db_sample_cx = hs.database_sample_cx
    tr_sample_cx = hs.train_sample_cx
    te_sample_cx = hs.test_sample_cx

    assert db_sample_cx == tr_sample_cx
    assert len(set(te_sample_cx)) == 55

    print('Database shape: '+str(np.vstack(hs.feats.cx2_desc[db_sample_cx]).shape))

    hs.use_matcher('bagofwords')
    cx2_res = mc2.run_matching(hs)

    total_mAP, mAP_list       = get_oxsty_mAP_score(hs, cx2_res, SV=False)
    total_mAP_SV, mAP_list_SV = get_oxsty_mAP_score(hs, cx2_res, SV=True)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    oxford_bag_of_words()
