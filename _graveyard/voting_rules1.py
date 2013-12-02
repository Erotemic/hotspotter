
#_________________
# OLD

def build_voters_profile(hs, qcx, K):
    '''This is too similar to assign_matches_vsmany right now'''
    cx2_nx = hs.tables.cx2_nx
    hs.ensure_matcher(match_type='vsmany', K=K)
    K += 1
    cx2_desc = hs.feats.cx2_desc
    cx2_kpts = hs.feats.cx2_kpts
    cx2_rchip_size = hs.get_cx2_rchip_size()
    desc1 = cx2_desc[qcx]
    args = hs.matcher.vsmany_args
    vsmany_flann = args.vsmany_flann
    ax2_cx       = args.ax2_cx
    ax2_fx       = args.ax2_fx
    print('[invest] Building voter preferences over %s indexed descriptors. K=%r' %
          (helpers.commas(len(ax2_cx)), K))
    nn_args = (args, qcx, cx2_kpts, cx2_desc, cx2_rchip_size, K+1)
    nn_result = mc2.vsmany_nearest_neighbors(*nn_args)
    (qfx2_ax, qfx2_dists, qfx2_valid) = nn_result
    vote_dists = qfx2_dists[:, 0:K]
    norm_dists = qfx2_dists[:, K] # k+1th descriptor for normalization
    # Score the feature matches
    qfx2_score = np.array([mc2.LNBNN_fn(_vdist.T, norm_dists)
                           for _vdist in vote_dists.T]).T
    # Vote using the inverted file 
    qfx2_cx = ax2_cx[qfx2_ax[:, 0:K]]
    qfx2_fx = ax2_fx[qfx2_ax[:, 0:K]]
    qfx2_valid = qfx2_valid[:, 0:K]
    qfx2_nx = temporary_names(qfx2_cx, cx2_nx[qfx2_cx], zeroed_cx_list=[qcx])
    voters_profile = (qfx2_nx, qfx2_cx, qfx2_fx, qfx2_score, qfx2_valid)
    return voters_profile

#def filter_alternative_frequencies2(alternative_ids1, qfx2_altx1, correct_altx, max_cands=32):

def filter_alternative_frequencies(alternative_ids1, qfx2_altx1, correct_altx, max_cands=32):
    'determines the alternatives who appear the most and filters out the least occuring'
    alternative_ids = alternative_ids.copy()
    qfx2_altx    = qfx2_altx.copy()
    altx2_freq = np.bincount(qfx2_altx.flatten()+1)[1:]
    smallest_altx = altx2_freq.argsort()
    smallest_cfreq = altx2_freq[smallest_altx]
    smallest_thresh = len(smallest_cfreq) - max_cands
    print('Current num alternatives = %r. Truncating to %r' % (len(altx2_freq), max_cands))
    print('Frequency stats: '+str(helpers.mystats(altx2_freq[altx2_freq != 0])))
    print('Correct alternative frequency = %r' % altx2_freq[correct_altx])
    print('Correct alternative frequency rank = %r' % (np.where(smallest_altx == correct_altx)[0],)) 
    if smallest_thresh > -1:
        freq_thresh = smallest_cfreq[smallest_thresh]
        print('Truncating at rank = %r' % smallest_thresh)
        print('Truncating at frequency = %r' % freq_thresh)
        to_remove_altx, = np.where(altx2_freq <= freq_thresh)
        qfx2_remove = np.in1d(qfx2_altx.flatten(), to_remove_altx)
        qfx2_remove.shape = qfx2_altx.shape
        qfx2_altx[qfx2_remove] = -1
        keep_ids = True - np.in1d(alternative_ids, alternative_ids[to_remove_altx])
        alternative_ids = alternative_ids[keep_ids]
    return alternative_ids, qfx2_altx

def temporary_names(cx_list, nx_list, zeroed_cx_list=[], zeroed_nx_list=[]):
    '''Test Input: 
        nx_list = np.array([(1, 5, 6), (2, 4, 0), (1, 1,  1), (5, 5, 5)])
        cx_list = np.array([(2, 3, 4), (5, 6, 7), (8, 9, 10), (4, 5, 5)])
        zeroed_nx_list = []
        zeroed_cx_list = [3]
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

def build_pairwise_votes(alternative_ids, qfx2_altx):
    '''
    Divides full rankings over alternatives into pairwise rankings. 
    Assumes that the breaking has already been applied.
    e.g.
    alternative_ids = [0,1,2]
    qfx2_altx = np.array([(0, 1, 2), (1, 2, 0)])
    '''
    nAlts = len(alternative_ids)
    def generate_pairwise_votes(partial_order, compliment_order):
        pairwise_winners = [partial_order[rank:rank+1] 
                           for rank in xrange(0, len(partial_order))]
        pairwise_losers  = [np.hstack((compliment_order, partial_order[rank+1:]))
                           for rank in xrange(0, len(partial_order))]
        pairwise_vote_list = [helpers.cartesian((pwinners, plosers)) for pwinners, plosers
                                    in zip(pairwise_winners, pairwise_losers)]
        pairwise_votes = np.vstack(pairwise_vote_list)
        return pairwise_votes
    pairwise_mat = np.zeros((nAlts, nAlts))
    nVoters = len(qfx2_altx)
    progstr = helpers.make_progress_fmt_str(nVoters, lbl='[voting] building P(d)')
    for ix, qfx in enumerate(xrange(nVoters)):
        helpers.print_(progstr % (ix+1))
        partial_order = qfx2_altx[qfx]
        partial_order = partial_order[partial_order != -1]
        if len(partial_order) == 0: continue
        compliment_order = np.setdiff1d(alternative_ids, partial_order)
        pairwise_votes = generate_pairwise_votes(partial_order, compliment_order)
        def sum_win(ij): pairwise_mat[ij[0], ij[1]] += 1 # pairiwse wins on off-diagonal
        def sum_loss(ij): pairwise_mat[ij[1], ij[1]] -= 1 # pairiwse wins on off-diagonal
        map(sum_win,  iter(pairwise_votes))
        map(sum_loss, iter(pairwise_votes))
    # Divide num voters
    PLmatrix = pairwise_mat / nVoters # = P(D) = Placket Luce GMoM function
    return PLmatrix


def optimize(M):
    '''
    alternative_ids = [0,1,2]
    qfx2_altx = np.array([(0,1,2), (1,0,2)])
    M = PLmatrix
    M = pairwise_voting(alternative_ids, qfx2_altx)
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

def viz_votingrule_table(ranked_candiates, ranked_scores, correct_altx, title, fnum):
    num_top = 5
    correct_rank = np.where(ranked_candiates == correct_altx)[0]
    if len(correct_rank) > 0:
        correct_rank = correct_rank[0]
    correct_score = ranked_scores[correct_rank]
    np.set_printoptions(precision=1)
    top_cands  = ranked_candiates[0:num_top]
    top_scores = ranked_scores[0:num_top]
    print('[vote] top%r ranked cands = %r' % (num_top, top_scores))
    print('[vote] top%r ranked scores = %r' % (num_top, top_cands))
    print('[vote] correct candid = %r ' % correct_altx)
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


def voting_rule(alternative_ids, qfx2_altx, qfx2_weight=None, rule='borda',
                correct_altx=None, fnum=1):
    K = qfx2_altx.shape[1]
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
    alt_score = weighted_positional_scoring_rule(alternative_ids, qfx2_altx, score_vec, qfx2_weight)
    ranked_candiates = alt_score.argsort()[::-1]
    ranked_scores    = alt_score[ranked_candiates]
    viz_votingrule_table(ranked_candiates, ranked_scores, correct_altx, title, fnum)
    return ranked_candiates, ranked_scores


def weighted_positional_scoring_rule(alternative_ids, 
                                     qfx2_altx, score_vec,
                                     qfx2_weight=None):
    nAlts = len(alternative_ids)
    alt_score = np.zeros(nAlts)
    if qfx2_weight is None: 
        qfx2_weight = np.ones(qfx2_altx.shape)
    for qfx in xrange(len(qfx2_altx)):
        partial_order = qfx2_altx[qfx]
        weights       = qfx2_weight[qfx]
        # Remove impossible votes
        weights       = weights[partial_order != -1]
        partial_order = partial_order[partial_order != -1]
        for ix, altx in enumerate(partial_order):
            alt_score[altx] += weights[ix] * score_vec[ix]
    return alt_score


def _normalize_voters_profile(hs, qcx, voters_profile):
    '''Applies a temporary labeling scheme'''
    cx2_nx = hs.tables.cx2_nx
    (qfx2_nx, qfx2_cx, qfx2_fx, qfx2_score, qfx2_valid) = voters_profile
    # Apply temporary alternative labels
    alts_cxs = np.unique(qfx2_cx[qfx2_valid].flatten())
    alts_nxs = np.setdiff1d(np.unique(qfx2_nx[qfx2_valid].flatten()), [0])
    nx2_altx = {nx:altx for altx, nx in enumerate(alts_nxs)}
    nx2_altx[0] = -1
    qfx2_altx = np.copy(qfx2_nx)
    old_shape = qfx2_altx.shape 
    qfx2_altx.shape = (qfx2_altx.size,)
    for i in xrange(len(qfx2_altx)):
        qfx2_altx[i] = nx2_altx[qfx2_altx[i]]
    qfx2_altx.shape = old_shape
    alternative_ids = np.arange(0, len(alts_nxs))
    correct_altx = nx2_altx[cx2_nx[qcx]] # Ground truth labels
    qfx2_weight   = qfx2_score
    return alternative_ids, qfx2_altx, qfx2_weight, correct_altx

def viz_PLmatrix(PLmatrix, qfx2_altx=None, correct_altx=None, alternative_ids=None, fnum=1):
    if alternative_ids is None:
        alternative_ids = []
    if correct_altx is None: 
        correct_altx = -1
    if qfx2_altx is None:
        nVoters = -1
    else:
        nVoters = len(qfx2_altx)
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
    stride = int(np.ceil(np.log10(len(alternative_ids)))+1)*10
    correct_id = alternative_ids[correct_altx]
    alternative_ticks = sorted(alternative_ids[::stride].tolist() + [correct_id])
    ax.set_xticks(alternative_ticks)
    ax.set_xticklabels(alternative_ticks)
    ax.set_yticks(alternative_ticks)
    ax.set_yticklabels(alternative_ticks)
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
    ax.set_yticks(alternative_ticks)
    ax.set_yticklabels(alternative_ticks)
    df2.set_figtitle('Correct ID=%r' % (correct_id))
    fig.colorbar(cax2, orientation='horizontal')
    fig.subplots_adjust(left=0.05, right=.99,
                        bottom=0.01, top=0.88,
                        wspace=0.01, hspace=0.01)
    #plt.set_cmap('jet', plt.cm.jet,norm = LogNorm())



def test_voting_rules(hs, qcx, K, fnum=1):
    voters_profile = build_voters_profile(hs, qcx, K)
    normal_profile = _normalize_voters_profile(hs, qcx, voters_profile)
    alternative_ids, qfx2_altx, qfx2_weight, correct_altx = normal_profile
    #alternative_ids, qfx2_altx = filter_alternative_frequencies(alternative_ids, qfx2_altx, correct_altx)
    m = len(alternative_ids)
    n = len(qfx2_altx)
    k = len(qfx2_altx.T)
    bigo_breaking = helpers.int_comma_str((m+k)*k*n)
    bigo_gmm = helpers.int_comma_str(int(m**2.376))
    bigo_gmm3 = helpers.int_comma_str(int(m**3))
    print('[voting] m = num_alternatives = %r ' % len(alternative_ids))
    print('[voting] n = nVoters = %r ' % len(qfx2_altx))
    print('[voting] k = top_k_breaking = %r ' % len(qfx2_altx.T))
    print('[voting] Computing breaking O((m+k)*k*n) = %s' % bigo_breaking)
    print('[voting] Computing GMoM breaking O(m^{2.376}) < O(m^3) = %s < %s' % (bigo_gmm, bigo_gmm3))
    #---
    def voting_rule_(weighting, rule_name, fnum):
        ranking = voting_rule(alternative_ids, qfx2_altx, weighting, rule_name, correct_altx, fnum)
        return ranking, fnum + 1
    #weighted_topk_ranking, fnum      = voting_rule_(qfx2_weight, 'topk', fnum)
    #weighted_borda_ranking, fnum     = voting_rule_(qfx2_weight, 'borda', fnum)
    #weighted_plurality_ranking, fnum = voting_rule_(qfx2_weight, 'plurality', fnum)
    #topk_ranking, fnum               = voting_rule_(None, 'topk', fnum)
    #borda_ranking, fnum              = voting_rule_(None, 'borda', fnum)
    #plurality_ranking, fnum          = voting_rule_(None, 'plurality', fnum)
    #---

    PLmatrix = build_pairwise_votes(alternative_ids, qfx2_altx)
    viz_PLmatrix(PLmatrix, qfx2_altx, correct_altx, alternative_ids, fnum)
    # Took 52 seconds on bakerstreet with (41x41) matrix
    gamma = optimize(PLmatrix)             # (41x41) -> 52 seconds
    gamma = optimize(PLmatrix[:-1,:-1])    # (40x40) -> 83 seconds
    gamma = optimize(PLmatrix[:-11,:-11])  # (30x30) -> 45 seconds)
    gamma = optimize(PLmatrix[:-21,:-21])  # (20x20) -> 21 seconds)
    gamma = optimize(PLmatrix[:-31,:-31])  # (10x10) ->  4 seconds)
    gamma = optimize(PLmatrix[:-36,:-36])  # ( 5x 5) ->  2 seconds)

    def PlacketLuceWinnerProb(gamma):
        nAlts = len(gamma)
        mask = np.ones(nAlts, dtype=np.bool)
        ax2_prob = np.zeros(nAlts)
        for ax in xrange(nAlts):
            mask[ax] = False
            ax2_prob[ax] = gamma[ax] / np.sum(gamma[mask])
            mask[ax] = True
        ax2_prob = ax2_prob / ax2_prob.sum()
        return ax2_prob
    ax2_prob = PlacketLuceWinnerProb(gamma)
    pl_ranking = ax2_prob.argsort()[::-1]
    pl_confidence = ax2_prob[pl_ranking]
    correct_rank = np.where(pl_ranking == correct_altx)[0][0]
    ranked_altxconf = zip(pl_ranking, pl_confidence)
    print('Top 5 Ranked altx/confidence = %r' % (ranked_altxconf[0:5],))
    print('Correct Rank=%r altx/confidence = %r' % (correct_rank, ranked_altxconf[correct_rank],))

    df2.update()


    #b = np.zeros(4)
    #b[-1] = 1
    #[- + +]
    #[+ - +]  x = b
    #[+ + -]      1
    #[1 0 0]
    #X = np.vstack([M,[1,0,0]])
    #print(X)
    #print(b)
    #x = linalg.solve(X, b)

def test():
    from numpy import linalg
    linalg.lstsq
    '''
    Test Data:
    K = 5
    votes = [(3,2,1,4), (4,1,2,3), (4, 2, 3, 1), (1, 2, 3, 4)]
    qfx2_utilities = [[(nx, nx, nx**3, k) for k, nx in enumerate(vote)] for vote in votes]
    M, altx2_nx= _utilities2_pairwise_breaking(qfx2_utilities)

    from numpy.linalg import svd, inv
    from numpy import eye, diag, zeros
    #Because s is sorted, and M is rank deficient, the value s[-1] should be 0
    np.set_printoptions(precision=2, suppress=True, linewidth=80)
    #The svd is: 
    #u * s * v = M
    u.dot(diag(s)).dot(v) = M

    #u is unitary: 
    inv(u).dot(u) == eye(len(s))
    
    diag(s).dot(v) == inv(u).dot(M)

    u.dot(diag(s)) == M.dot(inv(v))
    And because s[-1] is 0
    u.dot(diag(s))[:,-1:] == zeros((len(s),1))

    Because we want to find Mx = 0

    So flip the left and right sides
    M.dot(inv(v)[:,-1:]) == u.dot(diag(s))[:,-1:] 

    And you find
    M = M
    x = inv(v)[:,-1:]
    0 = u.dot(diag(s))[:,-1:] 
    
    So we have the solution to our problem as x = inv(v)[:,-1:]

    Furthermore it is true that 
    inv(v)[:,-1:].T == v[-1:,:]
    because v is unitary and the last vector in v corresponds to a singular
    vector because M is rank m-1
    
    ALSO: v.dot(inv(v)) = eye(len(s)) so
    v[-1].dot(inv(v)[:,-1:]) == 1
    
    this means that v[-1] is non-zero, and v[-1].T == inv(v[:,-1:])

    So all of this can be done as...
     '''

    # We could also say
    def eq(M1, M2):
        print(str(M1)+'\n = \n'+str(M2))
    # Compute SVD
    (u, s_, v) = linalg.svd(M)
    s = diag(s_)
    #---
    print('-------')
    print('M =\n%s' % (M,))
    print('-------')
    print('u =\n%s' % (u,))
    print('-------')
    print('s =\n%s' % (s,))
    print('-------')
    print('v =\n%s' % (v,))
    print('-------')
    print('u s v = M')
    eq(u.dot(s).dot(v), M)
    # We want to find Mx = 0
    print('-------')
    print('The last value of s is zeros because M is rank m-1 and s is sorted')
    print('s =\n%s' % (s,))
    print('-------')
    print('Therefore the last column of u.dot(s) is zeros')
    print('v is unitary so v.T = inv(v)')
    print('u s = M v.T')
    eq(u.dot(s), M.dot(v.T))
    print('-------')
    print('We want to find Mx = 0, and the last column of LHS corresponds to this')
    print('u s = M v.T')
    eq(u.dot(s), M.dot(v.T))

    # The right column u.dot(s) is 

    #Ok, so v[-1] can be negative, but that's ok
    # its unitary, we can just negate it. 
    # or we can take the absolute value or l2 normalize it
    # x = v[-1] = inv(v)[:,-1]
    # so
    # x.dot(x) == 1
    # hmmmm
    # I need to find a way to proove 
    # components of x are all negative or all 
    # positive
    # Verify s is 0
    x = v[-1]
