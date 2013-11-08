import matplotlib
import textwrap
import draw_func2 as df2
import sys
import vizualizations as viz
import matplotlib
import numpy as np
from numpy import linalg
import helpers
import scipy.optimize
import scipy
import params
import match_chips2 as mc2

def reload_module():
    import imp, sys
    print('[reload] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def build_voters_profile(hs, qcx, K):
    cx2_nx = hs.tables.cx2_nx
    hs.ensure_matcher(match_type='vsmany')
    K += 1
    desc1 = hs.feats.cx2_desc[qcx]
    vsmany_args = hs.matcher.vsmany_args
    vsmany_flann = vsmany_args.vsmany_flann
    ax2_cx       = vsmany_args.ax2_cx
    ax2_fx       = vsmany_args.ax2_fx
    print('[invest] Building voter preferences over %s indexed descriptors. K=%r' %
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
    qfx2_nx  = temporary_names(qfx2_cx, cx2_nx[qfx2_cx], zeroed_cx_list=[qcx])
    voters_profile = (qfx2_nx, qfx2_cx, qfx2_fx, qfx2_score)
    return voters_profile

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

def build_pairwise_votes(candidate_ids, qfx2_candx):
    '''
    Divides full rankings over alternatives into pairwise rankings. 
    Assumes that the breaking has already been applied.
    e.g.
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
    pairiwse_wins = np.zeros((num_cands, num_cands))
    num_voters = len(qfx2_candx)
    progstr = helpers.make_progress_fmt_str(num_voters, lbl='[voting] building P(d)')
    for ix, qfx in enumerate(xrange(num_voters)):
        helpers.print_(progstr % (ix+1))
        partial_order = qfx2_candx[qfx]
        partial_order = partial_order[partial_order != -1]
        if len(partial_order) == 0: continue
        compliment_order = np.setdiff1d(candidate_ids, partial_order)
        pairwise_votes = generate_pairwise_votes(partial_order, compliment_order)
        def sum_win(ij): pairiwse_wins[ij[0], ij[1]] += 1 # pairiwse wins on off-diagonal
        def sum_loss(ij): pairiwse_wins[ij[1], ij[1]] -= 1 # pairiwse wins on off-diagonal
        map(sum_win,  iter(pairwise_votes))
        map(sum_loss, iter(pairwise_votes))
    # Divide num voters
    PLmatrix = pairiwse_wins / num_voters # = P(D) = Placket Luce GMoM function
    return PLmatrix


def optimize(M):
    '''
    candidate_ids = [0,1,2]
    qfx2_candx = np.array([(0,1,2), (1,0,2)])
    M = PLmatrix
    M = pairwise_voting(candidate_ids, qfx2_candx)
    M = array([[-0.5,  0.5,  1. ],
               [ 0.5, -0.5,  1. ],
               [ 0. ,  0. , -2. ]])
    '''
    print(r'[vote] x = argmin_x ||Mx||_2, s.t. ||x||_2 = 1')
    m = M.shape[0]
    x0 = np.ones(m)/np.sqrt(m)
    f   = lambda x, M: linalg.norm(M.dot(x))
    con = lambda x: linalg.norm(x) - 1
    cons = {'type':'eq', 'fun': con}
    print('[vote] running optimization')
    with helpers.Timer() as t:
        res = scipy.optimize.minimize(f, x0, args=(M,), constraints=cons)
    x = res['x']
    xnorm = linalg.norm(x)
    gamma = np.abs(x / xnorm)
    print('[voting_rules] x = %r' % (x,))
    print('[voting_rules] xnorm = %r' % (xnorm,))
    print('[voting_rules] gamma = %r' % (gamma,))
    return gamma

def optimize2():
    x = linalg.solve(M, np.zeros(M.shape[0]))
    x /= linalg.norm(x)

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

#----

def viz_votingrule_table(ranked_candiates, ranked_scores, correct_candx, title, fnum):
    num_top = 5
    correct_rank = np.where(ranked_candiates == correct_candx)[0]
    if len(correct_rank) > 0:
        correct_rank = correct_rank[0]
    correct_score = ranked_scores[correct_rank]
    np.set_printoptions(precision=1)
    top_cands  = ranked_candiates[0:num_top]
    top_scores = ranked_scores[0:num_top]
    print('[vote] top%r ranked cands = %r' % (num_top, top_scores))
    print('[vote] top%r ranked scores = %r' % (num_top, top_cands))
    print('[vote] correct candid = %r ' % correct_candx)
    print('[vote] correct ranking / score = %r / %r ' % (correct_rank, correct_score))
    print('----')
    np.set_printoptions(precision=8)

    plt = df2.plt
    df2.figure(fignum=fnum, doclf=True, subplot=(1,1,1))
    ax=plt.gca()
    #plt.plot([10,10,14,14,10],[2,4,4,2,2],'r')
    col_labels=map(lambda x: '%8d' % x, np.arange(num_top)+1)
    row_labels=['cand ids        ',
                'cand scores     ',
                'correct ranking ',
                'correct score   ']
    table_vals=[map(lambda x: '%8d' % x, top_cands),
                map(lambda x: '%8.2f' % x, top_scores),
                ['%8d' % (correct_rank)]  + ['        '] * (num_top-1), 
                ['%8.2f' % correct_score] + ['        '] * (num_top-1)]

    #matplotlib.table.Table
    # the rectangle is where I want to place the table
    #the_table = plt.table(cellText=table_vals,
                    #rowLabels=row_labels,
                    #colLabels=col_labels,
                    #colWidths = [0.1]*num_top,
                    #loc='center')
    def latex_table(row_labels, col_labels, table_vals):
        #matplotlib.rc('text', usetex=True)
        #print('col_labels=%r' % col_labels)
        #print('row_labels=%r' % row_labels)
        #print('table_vals=%r' % table_vals)
        nRows = len(row_labels)
        nCols = len(col_labels)
        def tableline(list_, rowlbl): 
            return rowlbl + ' & '+(' & '.join(list_))+'\\\\'
        collbl = tableline(col_labels, ' '*16)
        col_strs = [collbl, '\hline'] + [tableline(rowvals, rowlbl) for rowlbl, rowvals in zip(row_labels, table_vals)]
        col_split = '\n'
        body = col_split.join(col_strs)
        col_placement = ' c || '+(' | '.join((['c']*nCols)))
        latex_str = textwrap.dedent(r'''
        \begin{tabular}{%s}
        %s
        \end{tabular}
        ''') % (col_placement, helpers.indent(body))
        print(latex_str)
        plt.text(0, 0, latex_str, fontsize=14, 
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 fontname='Courier New')
                 #family='monospaced')
        #print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
        #print(matplotlib.font_manager.findfont('Courier'))
        #fontname
        r'''
        \begin{tabular}{ c || c | c | c | c | c} 
                        & 1      & 2      & 3      & 4      &      5\\
        \hline
        cand ids        & 3      & 38     & 32     & 40     &      5\\
        cand scores     & 4512.0 & 4279.0 & 4219.0 & 4100.0 & 3960.0\\
        correct ranking & 25     &        &        &        &       \\
        correct score   & 1042.0 &        &        &        &       \\
        \end{tabular}
        '''
    latex_table(row_labels, col_labels, table_vals)
    df2.set_figtitle(title)


def voting_rule(candidate_ids, qfx2_candx, qfx2_weight=None, rule='borda',
                correct_candx=None, fnum=1):
    K = qfx2_candx.shape[1]
    if rule == 'borda':
        score_vec = np.arange(0,K)[::-1]
    if rule == 'plurality':
        score_vec = np.zeros(K); score_vec[0] = 1
    if rule == 'topk':
        score_vec = np.ones(K)
    score_vec = np.array(score_vec, dtype=np.int)
    print('----')
    title = 'Rule=%s Weighted=%r ' % (rule, not qfx2_weight is None)
    print('[vote] ' + title)
    print('[vote] score_vec = %r' % (score_vec,))
    cand_score = weighted_positional_scoring_rule(candidate_ids, qfx2_candx, score_vec, qfx2_weight)
    ranked_candiates = cand_score.argsort()[::-1]
    ranked_scores    = cand_score[ranked_candiates]
    viz_votingrule_table(ranked_candiates, ranked_scores, correct_candx, title, fnum)
    return ranked_candiates, ranked_scores


def weighted_positional_scoring_rule(candidate_ids, qfx2_candx, score_vec, qfx2_weight=None):
    num_cands = len(candidate_ids)
    cand_score = np.zeros(num_cands)
    if qfx2_weight is None: 
        qfx2_weight = np.ones(qfx2_candx.shape)
    for qfx in xrange(len(qfx2_candx)):
        partial_order = qfx2_candx[qfx]
        weights       = qfx2_weight[qfx]
        # Remove impossible votes
        weights       = weights[partial_order != -1]
        partial_order = partial_order[partial_order != -1]
        for ix, candx in enumerate(partial_order):
            cand_score[candx] += weights[ix] * score_vec[ix]
    return cand_score


def _normalize_voters_profile(hs, qcx, voters_profile):
    cx2_nx = hs.tables.cx2_nx
    (qfx2_nx, qfx2_cx, qfx2_fx, qfx2_score) = voters_profile
    # Apply temporary candidate labels
    alts_cxs = np.unique(qfx2_cx.flatten())
    alts_nxs = np.setdiff1d(np.unique(qfx2_nx.flatten()), [0])
    nx2_candx = {nx:candx for candx, nx in enumerate(alts_nxs)}
    nx2_candx[0] = -1
    qfx2_candx = np.copy(qfx2_nx)
    old_shape = qfx2_candx.shape 
    qfx2_candx.shape = (qfx2_candx.size,)
    for i in xrange(len(qfx2_candx)):
        qfx2_candx[i] = nx2_candx[qfx2_candx[i]]
    qfx2_candx.shape = old_shape
    candidate_ids = np.arange(0, len(alts_nxs))
    correct_candx = nx2_candx[cx2_nx[qcx]]
    qfx2_weight   = qfx2_score
    return candidate_ids, qfx2_candx, qfx2_weight, correct_candx

def viz_PLmatrix(PLmatrix, qfx2_candx=None, correct_candx=None, candidate_ids=None, fnum=1):
    if candidate_ids is None:
        candidate_ids = []
    if correct_candx is None: 
        correct_candx = -1
    if qfx2_candx is None:
        num_voters = -1
    else:
        num_voters = len(qfx2_candx)
    # Separate diagonal and off diagonal 
    PLdiagonal = np.diagonal(PLmatrix)
    PLdiagonal.shape = (len(PLdiagonal), 1)
    PLoffdiag = PLmatrix.copy(); np.fill_diagonal(PLoffdiag, 0)
    # Build a figure
    fig = df2.plt.gcf()
    fig.clf()
    # Show the off diagonal
    colormap = 'hot'
    ax = fig.add_subplot(121)
    cax = ax.imshow(PLoffdiag, interpolation='nearest', cmap=colormap)
    stride = int(np.ceil(np.log10(len(candidate_ids)))+1)*10
    ax.set_xticks(candidate_ids[::stride])
    ax.set_yticks(candidate_ids[::stride])
    ax.set_xlabel('candiate ids')
    ax.set_ylabel('candiate ids.')
    ax.set_title('Off-Diagonal')
    fig.colorbar(cax, orientation='horizontal')
    # Show the diagonal
    ax = fig.add_subplot(122)
    def duplicate_cols(M, nCols):
        return np.tile(M, (1, nCols))
    nCols = len(PLdiagonal) / 2
    cax2 = ax.imshow(duplicate_cols(PLdiagonal, nCols), interpolation='nearest', cmap=colormap)
    ax.set_title('diagonal')
    ax.set_xticks([])
    ax.set_yticks(candidate_ids[::stride])
    df2.set_figtitle('Correct ID=%r' % (correct_candx))
    fig.colorbar(cax2, orientation='horizontal')
    fig.subplots_adjust(left=0.05, right=.99,
                        bottom=0.01, top=0.88,
                        wspace=0.01, hspace=0.01)
    #plt.set_cmap('jet', plt.cm.jet,norm = LogNorm())


def apply_voting_rules(hs, qcx, voters_profile, fnum=1):
    normal_profile = _normalize_voters_profile(hs, qcx, voters_profile)
    candidate_ids, qfx2_candx, qfx2_weight, correct_candx = normal_profile
    m = len(candidate_ids)
    n = len(qfx2_candx)
    k = len(qfx2_candx.T)
    bigo_breaking = helpers.int_comma_str((m+k)*k*n)
    bigo_gmm = helpers.int_comma_str(int(m**2.376))
    bigo_gmm3 = helpers.int_comma_str(int(m**3))
    print('[voting] m = num_candidates = %r ' % len(candidate_ids))
    print('[voting] n = num_voters = %r ' % len(qfx2_candx))
    print('[voting] k = top_k_breaking = %r ' % len(qfx2_candx.T))
    print('[voting] Computing breaking O((m+k)*k*n) = %s' % bigo_breaking)
    print('[voting] Computing GMM breaking O(m^{2.376}) = %s < %s' % (bigo_gmm, bigo_gmm3))
    #---
    borda_ranking              = voting_rule(candidate_ids, qfx2_candx, None, 'borda', correct_candx, fnum)
    fnum += 1
    plurality_ranking          = voting_rule(candidate_ids, qfx2_candx, None, 'plurality', correct_candx, fnum)
    fnum += 1
    topk_ranking               = voting_rule(candidate_ids, qfx2_candx, None, 'topk', correct_candx, fnum)
    fnum += 1
    weighted_borda_ranking     = voting_rule(candidate_ids, qfx2_candx, qfx2_weight, 'borda', correct_candx, fnum)
    fnum += 1
    weighted_plurality_ranking = voting_rule(candidate_ids, qfx2_candx, qfx2_weight, 'plurality', correct_candx, fnum)
    fnum += 1
    weighted_topk_ranking      = voting_rule(candidate_ids, qfx2_candx, qfx2_weight, 'topk', correct_candx, fnum)
    fnum += 1
    #---

    #PLmatrix = build_pairwise_votes(candidate_ids, qfx2_candx)
    #viz_PLmatrix(PLmatrix, qfx2_candx, correct_candx, candidate_ids, fnum)
    #gamma = optimize(PLmatrix)

    df2.update()



if __name__ == '__main__':
    pass

