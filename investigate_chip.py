#exec(open('__init__.py').read())
#exec(open('_research/investigate_chip.py').read())
from __future__ import division
import numpy as np
import load_data2 as ld2
import draw_func2 as df2
import match_chips2 as mc2
import cv2
import helpers
import spatial_verification2 as sv2
import sys
import params

from _research import dump_groundtruth

def history_entry(database='', cx=-1, ocxs=[], cid=None, notes=''):
    return (database, cx, ocxs, cid, notes)

# A list of poster child examples. (curious query cases)
HISTORY = [
    history_entry('TOADS', 32),
    history_entry('GZ', 111, [305]),
    history_entry('GZ', 1046, notes='viewpoint'),
    history_entry('GZ', 111, [305]),
    history_entry('MOTHERS',   1),
]

def temporary_names(cx_list, nx_list, zeroed_cx_list=[], zeroed_nx_list=[]):
    '''
    Test Input: 
        nx_list = np.array([(1, 5, 6), (2, 4, 0), (1, 1,  1), (5, 5, 5)])
        cx_list = np.array([(2, 3, 4), (5, 6, 7), (8, 9, 10), (4, 5, 5)])
        zeroed_nx_list = []
        zeroed_cx_list = [3]
    Test Output:
    '''
    zeroed_cx_list = set(zeroed_cx_list)
    tmp_nx_list = []
    for ix, (cx, nx) in enumerate(zip(cx_list.flat, nx_list.flat)):
        if cx in zeroed_cx_list:
            tmp_nx_list.append(0)
        elif nx in zeroed_nx_list:
            tmp_nx_list.append(0)
        elif nx >= 2:
            tmp_nx_list.append(nx)
        else:
            tmp_nx_list.append(-cx)
    tmp_nx_list = np.array(tmp_nx_list)
    tmp_nx_list = tmp_nx_list.reshape(cx_list.shape)
    return tmp_nx_list


def positional_scoring_rule(qfx2_candx, score_vec):
    cand_score = np.zeros(num_cands)
    for qfx in xrange(len(qfx2_candx)):
        partial_order = qfx2_candx[qfx]
        partial_order = partial_order[partial_order != -1]
        for ix, candx in enumerate(partial_order):
            cand_score[candx] += score_vec[ix]
    print('score_vec = %r' % (score_vec,))
    print('cand_score = %r' % (cand_score,))
    print('ranked list = %r' % (cand_score.argsort()[::-1]))
    print('----')
    return cand_score

def voting_rule_borda(qfx2_candx):
    borda_vector = np.arange(0,K)[::-1]
    cand_score = positional_scoring_rule(qfx2_candx, borda_vector)
    return cand_score.argsort()[::-1]

def voting_rule_plurality(qfx2_candx):
    plurality_vector = np.zeros(K)
    plurality_vector[0] = 1
    cand_score = positional_scoring_rule(qfx2_candx, plurality_vector)
    return

def voting_rule_topk(qfx2_candx):
    top_k_vector = np.ones(K)
    cand_score = positional_scoring_rule(qfx2_candx, top_k_vector)
    return cand_score.argsort()[::-1]

def pairwise_voting(candidate_ids, qfx2_candx):
    ''' e.g.
    candidate_ids = [0,1,2]
    qfx2_candx = np.array([(0, 1, 2), (1, 2, 0)])
    '''
    num_cands = len(candidate_ids)
    def generate_pairwise_votes(partial_order, compliment_order):
        pairwise_winners = [partial_order[rank:rank+1] 
                           for rank in xrange(0, len(partial_order))]
        pairwise_losers  = [np.hstack((compliment_order, partial_order[rank+1:]))
                           for rank in xrange(0, len(partial_order))]
        pairwise_vote_list = [helpers.cartesian((pwinners, plosers)) for pwinners, plosers
                                    in zip(pairwise_winners, pairwise_losers)]
        pairwise_votes = np.vstack(pairwise_vote_list)
        return pairwise_votes

    def make_PL_matrix():
        pairiwse_wins = np.zeros((num_cands, num_cands))
        num_voters = 0
        for qfx in xrange(len(qfx2_candx)):
            partial_order = qfx2_candx[qfx]
            partial_order = partial_order[partial_order != -1]
            if len(partial_order) == 0: continue
            num_voters += 1
            compliment_order = np.setdiff1d(candidate_ids, partial_order)
            pairwise_votes = generate_pairwise_votes(partial_order, compliment_order)
            def sum_win(i,j): pairiwse_wins[i, j] += 1 # pairiwse wins on off-diagonal
            def sum_loss(i,j): pairiwse_wins[i, j] -= 1 # pairiwse wins on off-diagonal
            [ sum_win(i, j) for i, j in iter(pairwise_votes)]
            [sum_loss(j, j) for i, j in iter(pairwise_votes)]
        # Divide num voters
        PLmatrix = pairiwse_wins / num_voters # = P(D) = Placket Luce GMoM function
        return PLmatrix

    PLmatrix = make_PL_matrix()
    #viz_PL_matrix(PLmatrix)
    return PLmatrix

def viz_PL_matrix(PLmatrix):
    # Show the matrix
    PLmatrix2 = PLmatrix.copy()
    #np.fill_diagonal(PLmatrix2, 0)
    fig = df2.plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111)
    cax = ax.imshow(PLmatrix2, interpolation='nearest', cmap='jet')
    ax.set_xticks(candidate_ids[::3])
    ax.set_yticks(candidate_ids[::3])
    correct_candx = nx2_candx[cx2_nx[qcx]]
    ax.set_xlabel('candiate ids')
    ax.set_ylabel('candiate ids.')
    ax.set_title('Correct ID=%r' % (correct_candx))
    #plt.set_cmap('jet', plt.cm.jet,norm = LogNorm())
    fig.colorbar(cax, orientation='horizontal')

import scipy.optimize
from numpy import linalg
import numpy as np
def optimize():
    '''
    candidate_ids = [0,1,2]
    qfx2_candx = np.array([(0,1,2), (1,0,2)])
    M = pairwise_voting(candidate_ids, qfx2_candx)
    M = array([[-0.5,  0.5,  1. ],
               [ 0.5, -0.5,  1. ],
               [ 0. ,  0. , -2. ]])
    viz_PL_matrix(M)
    '''
    m = M.shape[0]
    x0 = np.ones(m)/np.sqrt(m)
    f   = lambda x, M: linalg.norm(M.dot(x))
    con = lambda x: linalg.norm(x) - 1
    cons = {'type':'eq', 'fun': con}
    res = scipy.optimize.minimize(f, x0, args=(M,), constraints=cons)
    x = res['x']
    xnorm = linalg.norm(x)
    gamma = np.abs(x / xnorm)
    print('x = %r' % (x,))
    print('xnorm = %r' % (xnorm,))
    print('gamma = %r' % (gamma,))
    return gamma

def PlacketLuce(vote, gamma):
    ''' e.g. 
    gamma = optimize()
    vote = np.arange(len(gamma))
    np.random.shuffle(vote)
    pr = PlacketLuce(vote, gamma)
    print(vote)
    print(pr)
    print('----')
    '''
    m = len(vote)-1
    pl_term = lambda x: gamma[vote[x]] / gamma[vote[x:]].sum()
    prob = np.prod([pl_term(x) for x in xrange(m)])
    return prob

def optimize2():
    from numpy import linalg
    x = linalg.solve(M, np.zeros(M.shape[0]))
    x /= linalg.norm(x)
    

def build_pairwise_comparisons(hs, qcx, qfx2_ax, vsmany_index):
    import itertools
    cx2_nx = hs.tables.cx2_nx
    ax2_cx   = vsmany_index.ax2_cx
    ax2_fx   = vsmany_index.ax2_fx
    qfx2_cx  = ax2_cx[qfx2_ax[:, 0:K]]
    qfx2_nx  = temporary_names(qfx2_cx, cx2_nx[qfx2_cx], zeroed_cx_list=[qcx])
    alts_cxs = np.unique(qfx2_cx.flatten())
    alts_nxs = np.setdiff1d(np.unique(qfx2_nx.flatten()), [0])

    # Apply temporary candidate labels
    nx2_candx = {nx:candx for candx, nx in enumerate(alts_nxs)}
    nx2_candx[0] = -1
    qfx2_candx = np.copy(qfx2_nx)
    old_shape = qfx2_candx.shape 
    qfx2_candx.shape = (qfx2_candx.size,)
    for i in xrange(len(qfx2_candx)):
        qfx2_candx[i] = nx2_candx[qfx2_candx[i]]
    qfx2_candx.shape = old_shape
    candidate_ids = np.arange(0, len(alts_nxs))
    PL_matrix = pairiwse_voting(candidate_ids, qfx2_candx)


K = 1
def quick_assign_vsmany(hs, qcx, cx, K): 
    #if hs.isindexed(qcx)
    K += 1
    desc1 = hs.feats.cx2_desc[qcx]
    vsmany_index = hs.matcher._Matcher__vsmany_index
    vsmany_flann = vsmany_index.vsmany_flann
    ax2_cx       = vsmany_index.ax2_cx
    ax2_fx       = vsmany_index.ax2_fx
    print('[invest] Quick vsmany over %s indexed descriptors. K=%r' %
          (helpers.commas(len(ax2_cx)), K))
    checks       = params.VSMANY_FLANN_PARAMS['checks']
    (qfx2_ax, qfx2_dists) = vsmany_flann.nn_index(desc1, K+1, checks=checks)
    vote_dists = qfx2_dists[:, 0:K]
    norm_dists = qfx2_dists[:, K] # k+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([mc2.LNBNN_fn(_vdist.T, norm_dists)
                           for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    qfx2_cx = ax2_cx[qfx2_ax[:, 0:K]]
    qfx2_fx = ax2_fx[qfx2_ax[:, 0:K]]
    # Build feature matches
    num_qf = len(desc1)
    qfx2_qfx = np.tile(np.arange(num_qf).reshape(num_qf, 1), (1, K)) 
    iter_matches = iter(zip(qfx2_qfx.flat, qfx2_cx.flat,
                            qfx2_fx.flat, qfx2_score.flat))
    fm, fs = ([], [])
    for qfx, cx_, fx, score in iter_matches:
        if cx != cx_: continue
        fm.append((qfx, fx))
        fs.append(score)
    fm = mc2.fix_fm(fm)
    fs = mc2.fix_fs(fs)
    return fm, fs

def quick_assign_vsone(hs, qcx, cx, ratio_thresh=1.2, burst_thresh=None):
    print('[invest] Performing quick vsone')
    desc1 = hs.feats.cx2_desc[qcx]
    desc2 = hs.feats.cx2_desc[cx]
    vsone_flann, checks = mc2.get_vsone_flann(desc1)
    fm, fs = mc2.match_vsone(desc2, vsone_flann, checks, ratio_thresh, burst_thresh)
    return fm, fs

def top_matching_features(res, axnum=None, match_type=''):
    cx2_fs = res.cx2_fs_V
    cx_fx_fs_list = []
    for cx in xrange(len(cx2_fs)):
        fx2_fs = cx2_fs[cx]
        for fx in xrange(len(fx2_fs)):
            fs = fx2_fs[fx]
            cx_fx_fs_list.append((cx, fx, fs))

    cx_fx_fs_sorted = np.array(sorted(cx_fx_fs_list, key=lambda x: x[2])[::-1])

    sorted_score = cx_fx_fs_sorted[:,2]
    fig = df2.figure(0)
    df2.plot(sorted_score)


param1 = 'K'
param2 = 'xy_thresh'
assign_alg = 'vsmany'
nParam1=1 
fnum = 1
nParam2=1
cx_list='gt1'
def vary_query_params(hs, qcx, param1='ratio_thresh', param2='xy_thresh',
                      assign_alg='vsone', nParam1=3, nParam2=3, fnum=1,
                      cx_list='gt'):
    possible_variations = {
                        # mean , #sigma  #props
        'K'            : (1, 5, 'int', 'pos'),
        'ratio_thresh' : (1.6,   .001,  'pos'),  
        'xy_thresh'    : (0.001, 0.1, 'pos'), 
        'scale_min'    : (0.5,   0.25, 'pos'),
        'scale_max'    : (2.0,   0.5,  'neg')
    }
    param_ranges = {
        'param1'    : [param1]+[list(possible_variations[param1])],
        'param2'    : [param2]+[list(possible_variations[param2])]
    }
    # Ground truth matches
    if cx_list == 'gt':
        cx_list = hs.get_groundtruth_cxs(qcx)
    if cx_list == 'gt1':
        gt_list = hs.get_groundtruth_cxs(qcx)
        cx_list = gt_list[0:1]
    #cx = cx_list[0]
    for cx in cx_list:
        fnum = vary_two_params(hs, qcx, cx, param_ranges, assign_alg,
                               nParam1, nParam2, fnum)
    return fnum


def linear_logspace(start, stop, num, base=2):
    return 2 ** np.linspace(np.log2(start), np.log2(stop), num)

def vary_two_params(hs, qcx, cx, param_ranges, assign_alg, nParam1=3, nParam2=3, fnum=1):
    # Query Features
    cx2_rchip_size = hs.get_cx2_rchip_size()
    get_features = quick_get_features_factory(hs)
    rchip1, fx2_kp1, fx2_desc1, cid1 = get_features(qcx)
    rchip2, fx2_kp2, fx2_desc2, cid2 = get_features(cx)
    rchip_size1 = cx2_rchip_size[qcx]
    rchip_size2 = cx2_rchip_size[cx]

    possible_assign_fns = {'vsone'   : quick_assign_vsone, 
                           'vsmany'  : quick_assign_vsmany, }
    #possible_space_fns = {'lin' : np.linspace,
                          #'log' : linear_logspace}
    quick_assign_fn = possible_assign_fns[assign_alg]

    # Varied Parameters
    def get_param(key, nParam):
        param = param_ranges[key][0]
        param_info = param_ranges[key][1]
        param_type = 'float' if len(param_info) <= 2 else param_info[2]
        #space_fn = possible_space_fns['lin' if len(param_info) <= 3 else param_info[3]]
        #param_range = list(param_info[0:2]) + [nParam]
        npnormal = np.random.normal
        mean = param_info[0]
        std  = param_info[0]
        if 'pos' in param_info:
            random_steps = list(mean + np.abs(npnormal(0, std, nParam-1))) 
        elif 'neg' in param_info:
            random_steps = list(mean - np.abs(npnormal(0, std, nParam-1))) 
        else:
            random_steps = list(mean + npnormal(0, std, nParam-1))
        # Sample the mean and a gaussian neighborhood around the mean
        param_steps = [mean] + random_steps
        if param_type == 'int':
            param_steps = map(int, map(round, param_steps))
        return param, param_steps, nParam
    param1, param1_steps, nParam1 = get_param('param1', nParam1)
    param2, param2_steps, nParam2 = get_param('param2', nParam2)
    nRows = nParam1
    nCols = nParam2+1

    print('[invest] Varying parameters %r: nRows=%r, nCols=%r' % (assign_alg, nRows, nCols))
    print('[invest] %r = %r ' % (param1, param1_steps))
    print('[invest] %r = %r ' % (param2, param2_steps))
    # Assigned Features with param1
    for rowx, param1_value in enumerate(param1_steps):
        assign_args = {param1:param1_value}
        (fm, fs) = quick_assign_fn(hs, qcx, cx, **assign_args)
        def _show_matches_helper(fm, fs, rowx, colx, title):
            plotnum = (nRows, nCols, rowx*nCols+colx)
            #print('rowx=%r, colx=%r, plotnum=%r' % (rowx, colx, plotnum))
            df2.show_matches2(rchip1, rchip2, fx2_kp1, fx2_kp2, fm, fs, fignum=fnum,
                            plotnum=plotnum, title=title, draw_pts=False)
        # Plot the original assigned matches
        title = param1+'=%.3e' % param1_value
        _show_matches_helper(fm, fs, rowx, 1, '')
        ax = df2.plt.gca()
        ylabel_args = dict(rotation='horizontal',
                           verticalalignment='bottom',
                           horizontalalignment='right')
        ax.set_ylabel(title, **ylabel_args)
        #if rowx == nRows - 1:
        def _set_xlabel(label):
            #if False or rowx == 0:
                #ax = df2.plt.gca()
                #ax.set_title(label)
            if rowx == nRows - 1:
                ax = df2.plt.gca()
                ax.set_xlabel(label)

        df2.plt.subplots_adjust(left=0.05, right=1.0,
                                bottom=0.1, top=0.85,
                                wspace=0.01, hspace=0.01)
        _set_xlabel(assign_alg)
        # Spatially verify with params2
        for colx, param2_value in enumerate(param2_steps):
            sv_args = {'rchip_size2':rchip_size2, param2:param2_value}
            fm_V, fs_V = mc2.spatially_verify2(fx2_kp1, fx2_kp2, fm, fs, **sv_args)
            # Plot the spatially verified matches
            title = param2 + '=%.3e' % param2_value #helpers.commas(param2_value, 3)
            _show_matches_helper(fm_V, fs_V, rowx, colx+2, '')
            _set_xlabel(title)

    df2.set_figtitle(assign_alg+' vary '+param1+' and '+param2+' \n qcid=%r, cid=%r' % (cid1, cid2))
    fnum += 1
    return fnum

def quick_get_features_factory(hs):
    'builds a factory function'
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_cid = hs.tables.cx2_cid 
    def get_features(cx):
        rchip = hs.get_chip(cx)
        fx2_kp = cx2_kpts[cx]
        fx2_desc = cx2_desc[cx]
        cid = cx2_cid[cx]
        return rchip, fx2_kp, fx2_desc, cid
    return get_features

def show_vsone_matches(hs, qcx, fnum=1):
    set_matcher_type(hs, 'vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    return res_vsone, fnum

def where_did_vsone_matches_go(hs, qcx, fnum=1, K=100):
    '''Finds a set of vsone matches and a set of vsmany matches. 
    displays where the vsone matches are in the vsmany ranked lists'''
    # Ground truth matches
    gt_cxs = hs.get_other_cxs(qcx)
    # Query Features
    cx2_rchip_size = hs.get_cx2_rchip_size()
    get_features = quick_get_features_factory(hs)
    rchip1, qfx2_kp1, qfx2_desc1, qcid = get_features(qcx)
    # Get/Show vsone matches
    res_vsone, fnum = show_vsone_matches(hs, qcx, fnum)
    gt2_fm_V = res_vsone.cx2_fm_V[gt_cxs]
    # Get vsmany assigned matches (no spatial verification)
    set_matcher_type(hs, 'vsmany')
    vsmany_index = hs.matcher._Matcher__vsmany_index
    (qfx2_cx, qfx2_fx, qfx2_dists) = mc2.desc_nearest_neighbors(qfx2_desc1, vsmany_index, K)
    # Find where the matches to the correct images are
    print('[invest]  Finding where the vsone matches went for qcx=%r, qcid=%r' % (qcx, qcid))
    k_inds  = np.arange(0, K)
    qf_inds = np.arange(0, len(qfx2_cx))
    kxs, qfxs = np.meshgrid(k_inds, qf_inds)
    for gtx, ocx in enumerate(gt_cxs):
        rchip2, fx2_kp2, fx2_desc2, ocid = get_features(ocx)
        rchip_size2 = cx2_rchip_size[ocx]
        print('[invest] Checking matches to ground truth %r / %r cx=%r, cid=%r' % 
              (gtx+1, len(gt_cxs), ocx, ocid))
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

def set_matcher_type(hs, match_type):
    print('[invest] Setting matcher type to: '+str(match_type))
    params.__MATCH_TYPE__ = match_type
    hs.load_matcher()

def ensure_matcher_type(hs, match_type):
    if hs.matcher is None or hs.matcher.type != match_type:
        return set_matcher_type(hs, match_type)


def plot_name(hs, qcx, fnum=1):
    print('[invest] Plotting name')
    dump_groundtruth.plot_name_cx(hs, qcx, fignum=fnum)
    return fnum+1

def compare_matching_methods(hs, qcx, fnum=1):
    print('[invest] Comparing match methods')
    # VSMANY matcher
    set_matcher_type(hs, 'vsmany')
    vsmany_score_options = ['LNRAT', 'LNBNN', 'RATIO']
    vsmany_index = hs.matcher._Matcher__vsmany_index
    vsmany_results = {}
    for score_type in vsmany_score_options:
        params.__VSMANY_SCORE_FN__ = score_type
        res_vsmany = mc2.build_result_qcx(hs, qcx)
        df2.show_match_analysis(hs, res_vsmany, N=5, fignum=fnum, figtitle=' LNRAT')
        vsmany_results[score_type] = res_vsmany
        fnum+=1
    # BAGOFWORDS matcher
    set_matcher_type(hs, 'bagofwords')
    resBOW = mc2.build_result_qcx(hs, qcx)
    df2.show_match_analysis(hs, resBOW, N=5, fignum=fnum, figtitle=' bagofwords')
    fnum+=1
    # VSONE matcher
    set_matcher_type(hs, 'vsone')
    res_vsone = mc2.build_result_qcx(hs, qcx, use_cache=True)
    df2.show_match_analysis(hs, res_vsone, N=5, fignum=fnum, figtitle=' vsone')
    fnum+=1
    # Extra 
    df2.show_match_analysis(hs, vsmany_results['LNBNN'], N=20, fignum=fnum,
                                figtitle=' LNBNN More', show_query=False)
    fnum+=1
    return fnum

if __name__ == '__main__':
    if not 'hs' in vars():
        # Grab an example
        current = len(HISTORY) - 1
        (db, qcx, ocxs, cid, notes) = HISTORY[current]
        db_dir = eval('params.'+db)
        # Load hotspotter
        hs = ld2.HotSpotter()
        hs.load_all(db_dir, matcher=False)
        hs.set_samples()

    fnum = 1

    #fnum = plot_name(hs, qcx, fnum)
    #fnum = compare_matching_methods(hs, qcx, fnum)
    #fnum = vary_query_params(hs, qcx, 'ratio_thresh', 'xy_thresh', 'vsone', 2, 2, fnum, cx_list='gt1')
    set_matcher_type(hs, 'vsmany')
    fnum = vary_query_params(hs, qcx, 'K', 'xy_thresh', 'vsmany', 1, 1, fnum, cx_list='gt1')
    #fnum = where_did_vsone_matches_go(hs, qcx, fnum, K=100)
    #fnum = where_did_vsone_matches_go(hs, qcx, fnum, K=1000)

    df2.update()

exec(df2.present(**df2.OooScreen2()))
