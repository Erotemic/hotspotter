from hotspotter.algo.spatial_functions import ransac
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.other.helpers import alloc_lists
from hotspotter.other.logger import logdbg, logerr
from numpy import spacing as eps
from os.path import join
import os
import cPickle
import numpy as np

# -----------------
class RawResults(DynStruct):
    ' Intermediate class for result storage '

    def __init__(rr, qcx, qcid,  qnid,  qfpts, qfdsc, qchip_size, dbid=''):
        super(RawResults, rr).__init__()
        # Database query information
        rr.dbid  = dbid # dbid of the queried database. ''=current
        # Chip query Information
        rr.qcx   = qcx
        rr.qcid  = qcid
        rr.qnid  = qnid
        rr.qfpts = qfpts
        rr.qfdsc = qfdsc
        rr.qchip_size = qchip_size
        # Result Information
        rr.cx2_cscore_ = []
        rr.cx2_nscore  = []
        rr.cx2_fs_     = []
        rr.cx2_fs      = []
        rr.cx2_fm      = []

    def presave_clean(rr):
        'Save some memory by not saving query descriptors.'
        rr.qfdsc = None
        rr.qfpts = None

    def rr_fpath(rr, qhs):
        return join(qhs.qm.rr_dpath, 'rr_' + rr.dbid + '_' + \
                    (qhs.qm.rr_fnamefmt % rr.qcid) + '.pkl')

    def save_result(rr, qhs):
        'Saves the result to the given database'
        with open(qm.rr_fpath(rr, qhs), 'wb') as rr_file:
            cPickle.dump(rr, rr_file.__dict__)

    def load_result(rr, qhs):
        'Loads the result from the given database'
        try:
            with open(qm.rr_fpath(rr, qhs), 'rb') as rr_file:
                rr.__dict__ = cPickle.load(rr_file)
        except EOFError:
            os.remove(qm.rr_fpath(rr, qhs))
            logwarn('Result was corrupted for CID=%d' % rr.qcid)
# --- /RawResults ---


# -----------------
class QueryManager(AbstractManager):
    ' Handles Searching the Vocab Manager'
    def __init__(qm, hs):
        super(QueryManager, qm).__init__(hs)
        qm.rr = None # Reference to the last RawResult for debugging
        qm.rr_fnamefmt = None # Filename Format for saving RawResults
        qm.rr_dpath = None

    #@depends_algo
    #@depends_sample
    def update_rr_fnamefmt(qm):
        depends = ['chiprep', 'preproc', 'model', 'query']
        algo_suffix = qm.hs.am.get_algo_suffix(depends)
        samp_suffix = qm.hs.vm.get_samp_suffix()
        qm.rr_dpath = em.hs.iom.ensure_computed_directory('query_results')
        qm.rr_fnamefmt = 'cid%07d'+samp_suffix+algo_suffix

    def query_and_save(qm, qcid, qhs=None):
        'Query this database using the cid of the database: query_hs'
        if query_hs is None:
            dbid = ''
            qhs = qm.hs
        else:
            dbid = qm.hs.get_dbid()
        # Create RawResults
        qcx            = qhs.cm.cid2_cx[qcid]
        qnid           = qhs.cm.cx2_nid(qcx)
        (qfpts, qfdsc) = qhs.get_feats(qcx)
        qchip_size     = qhs.cm.cx2_chip_size(qcx)
        rr = RawResults(qcx, qcid, qnid, qfpts, qfdsc, qchip_size, dbid)
        # Populate RawResults
        qm.repopulate_raw_results(rr, qhs)

    def repopulate_raw_results(qm, rr, qhs):
        # Get Query Parameters
        hs = qm.hs
        query_pref = qhs.am.algo_prefs.query
        qindexed_bit   = (qhs is hs) and (rr.qcx in hs.vm.get_train_cx())
        K              = query_pref.k
        method         = query_pref.method
        # Add 1 to k and remove query from matches if query exists in results
        if qindexed_bit:
            K += 1
            logdbg('Query qcx=%d is in database ' % qcx)

        xy_thresh    = query_pref.spatial_thresh
        sigma_thresh = query_pref.sigma_thresh
        num_rerank   = query_pref.num_rerank

        qm.assign_feature_matches_1vM(rr, K, method)
        qm.spatial_rerank(rr)
        qm.compute_scores(rr)
        return rr

    def  cx2_res(qm, qcx):
        'Driver function for the query pipeliene'
        hs = qm.hs
        cm = hs.cm
        (qfpts, qfdsc) = cm.get_feats(qcx)
        qcid           = cm.cx2_cid[qcx]
        qnid           = cm.cx2_nid(qcx)
        logdbg('Querying QCID=%d' % qcid)
        rr = RawResults(qcx, qcid,  qnid,  qfpts, qfdsc)
        qm.rr = rr
        qm.repopulate_raw_results(rr)
        res = QueryResult(hs, rr)
        return res

    def  assign_feature_matches_1vM(qm, rr, K, method):
        '''Assigns each query feature to its K nearest database features
        with a similarity-score. Each feature votes for its assigned
        chip with this weight.'''
        logdbg('Assigning feature matches and initial scores')
        # Get managers
        hs = qm.hs
        cm = qm.hs.cm
        nm = qm.hs.nm
        vm = qm.hs.vm
        # Get intermediate results
        qcx   = rr.cx
        qcid  = rr.qcid
        qfdsc = rr.qfdsc
        qfpts = rr.qfpts

        num_qf         = qfpts.shape[0]
        # define: Prefix K = list of K+1 nearest; k = K nearest
        # Everything is done in a flat manner, and reshaped at the end.
        (qfx2_Kwxs, qfx2_Kdists) = vm.nearest_neighbors(qfdsc, K+1)

        # ---
        # Candidate score the nearest neighbor matches
        #
        # p - pth nearest ; o - k+1th nearest
        score_fn_dict = {
            'DIFF'  : lambda p, o: o - p,
            'RAT'   : lambda p, o: o / p+eps(1),
            'LNRAT' : lambda p, o: np.log2(o / p+eps(1)),
            'COUNT' : lambda p, o: 1,
            'NDIST' : lambda p, o: 10**16 - p,
            'TFIDF' : lambda wx2_tf, wx_idf, wx: wx2_tf[wx] * wx_idf[wx] }
        score_fn = score_fn_dict[method]
        if method == 'TFIDF':
                # The wx2_qtf could really be per k or as agged across all K
                w_histo = bincount(qfx2_wxs, minlength=vm.numWords())
                wx2_qtf = np.array(w_histo, dtype=np.float32) / num_qf
                qfx2_vweight = score_fn(wx2_qtf, vm.wx2_idf, qfx2_wxs)
        else:
            # Distances to the 0-K results
            p_vote = qfx2_Kdists[:, 0:K]
            # Distance to the K+1th result
            o_norm = np.tile(qfx2_Kdists[:, -1].reshape(num_qf, 1), (1, K))
            # Use score method to get weight
            qfx2_kweight = np.array([score_fn(p, o) for (p, o) in \
                                     iter(zip(p_vote.flat, o_norm.flat))],
                                    dtype=np.float32)
            qfx2_kweight.shape = (num_qf, K)


        # ---
        # Use the scores to cast weighted votes for database chips
        #

        # Remove the query from results
        # query feature index 2 agg descriptor indexes -> cids -> self_query_bit -> clean_axs
        #
        # Feature Matches -> Chip Ids
        qfx2_Kaxs_   = vm.wx2_axs[qfx2_Kwxs]
        qfx2_Kcids_  = [vm.ax2_cid[axs] for axs in qfx2_Kaxs_.flat]
        # Test if each FeatureMatch-ChipId is the Query-ChipId.
        qfx2_Ksqbit_ = [qcid != cids for cids in qfx2_Kcids_]
        # Remove FeatureMatches to the Query-ChipId
        qfx2_Kaxs    = [np.array(axs)[sqbit].tolist() for (axs, sqbit) in\
                        iter(zip(qfx2_Kaxs_.flat, qfx2_Ksqbit_))]

        # Clean Vote for Info
        qfx2_Kcxs = np.array([vm.ax2_cx(axs) for axs in qfx2_Kaxs])
        qfx2_Kfxs = np.array([vm.ax2_fx[axs] for axs in qfx2_Kaxs])
        qfx2_Knxs = np.array([cm.cx2_nx[cxs] for cxs in qfx2_Kcxs])
        if qfx2_Kfxs.size == 0:
            logerr('Cannot query when there is one chip in database')
        # Reshape Vote for Info
        qfx2_Kcxs = np.array(qfx2_Kcxs).reshape(num_qf, K+1)
        qfx2_Kfxs = np.array(qfx2_Kfxs).reshape(num_qf, K+1)
        qfx2_Knxs = np.array(qfx2_Knxs).reshape(num_qf, K+1)

        # Using the K=K+1 results, make k=K scores
        qfx2_kcxs_vote = qfx2_Kcxs[:, 0:K] # vote for cx
        qfx2_kfxs_vote = qfx2_Kfxs[:, 0:K] # vote for fx
        qfx2_knxs_vote = qfx2_Knxs[:, 0:K] # check with nx

        # Attempt to recover from problems where K is too small
        qfx2_knxs_norm = np.tile(qfx2_Knxs[:, K].reshape(num_qf, 1), (1, K))
        qfx2_knxs_norm[qfx2_knxs_norm == nm.UNIDEN_NX()] = 0 # Remove Unidentifieds from this test
        qfx2_kcxs_norm = np.tile(qfx2_Kcxs[:, K].reshape(num_qf, 1), (1, K))
        # If the normalizer has the same name, but is a different chip, there is a good chance
        # it is a correct match and was peanalized by the scoring function
        qfx2_normgood_bit = np.logical_and(qfx2_kcxs_vote != qfx2_kcxs_norm, \
                                        qfx2_knxs_vote == qfx2_knxs_norm)
        #qfx2_kweight[qfx2_normgood_bit] = 2


        # -----
        # Build FeatureMatches and FeaturesScores
        #
        cx2_fm  = alloc_lists(cm.max_cx + 1)
        cx2_fs_ = alloc_lists(cm.max_cx + 1)

        qfx2_qfx = np.tile(np.arange(0, num_qf).reshape(num_qf, 1), (1, K))
        # Add matches and scores
        for (qfx, qfs, cxs, fxs)\
                in iter(zip(qfx2_qfx.flat, \
                            qfx2_kweight.flat, \
                            qfx2_kcxs_vote.flat, \
                            qfx2_kfxs_vote.flat)):
            for (vote_cx, vote_fx) in iter(zip(cxs, fxs)):
                cx2_fm[vote_cx].append((qfx, vote_fx))
                cx2_fs_[vote_cx].append(qfs)

        # Convert correspondences to to numpy
        for cx in xrange(len(cx2_fs_)):
            num_m = len(cx2_fm[cx])
            cx2_fs_[cx] = np.array(cx2_fs_[cx], dtype=np.float32)
            cx2_fm[cx]  = np.array(cx2_fm[cx], dtype=np.uint32).reshape(num_m, 2)
        logdbg('Setting feature assignments')
        rr.cx2_fm  = cx2_fm
        rr.cx2_fs_ = cx2_fs_
        # --- end assign_feature_matches_1vM ---

    def compute_cscore(cx2_fs):
        return np.array([ np.sum(fs[fs > 0]) for fs in iter(cx2_fs) ], dtype=np.float32)

    def spatial_rerank(qm, rr, xy_thresh, sigma_thresh, num_rerank):
        '''Recalculates the votes for chips by setting the similarity-score of
         spatially invalid assignments to 0'''

        logdbg('Spatially Reranking')
        # hs managers
        cm = qm.hs.cm
        # intermediate results
        qcx     = rr.qcx
        qfpts   = rr.qfpts
        cx2_fm  = rr.cx2_fm
        cx2_fs_ = rr.cx2_fs_
        (w, h)   = rr.qchip_size

        # Sort by orderless score
        cx2_cscore_ = qm.compute_cscore(rr.cx2_fs_)
        # Get shortlist of top results
        top_cxs = cx2_cscore_.argsort()[::-1] #set diff with order
        invalids = set(cm.invalid_cxs())
        top_cxs = np.array([cx for cx in top_cxs \
                            if cx not in invalids], dtype=np.uint32)
        num_c      = len(top_cxs)
        num_rerank = min(num_c, num_rerank)
        # Build new feature scores
        cx2_fs = [arr.copy() for arr in cx2_fs_]
        if num_rerank == 0:
            logdbg('Breaking rerank. num_rerank = 0;  min(num_c, query_prefs.num_rerank)')
            rr.cx2_fs     = cx2_fs
            return False
        min_reranked_score = 2^30
        #Initialize the reranked scores as the normal scores
        cm.load_features(top_cxs) #[np.arange(num_rerank)])
        # For each tcx (top cx) in the shortlist
        (w, h) = rr.chip_size
        for tcx in top_cxs[0:num_rerank]:
            logdbg('Reranking qcx=%d vs tcx=%d' % (qcx, tcx))
            cid   = cm.cx2_cid[tcx]
            if cid <= 0: continue
            fx2_match  = cx2_fm[tcx]
            if len(fx2_match) == 0:
                logdbg('Breaking rerank. len(fx2_match) == 0')
                break
            tfpts      = cm.cx2_fpts[tcx]
            xy_thresh2 = (w*w + h*h) * xy_thresh**2
            # Find which matches are spatially constent
            fpts1_match  = qfpts[fx2_match[:, 0], :].transpose()
            fpts2_match  = tfpts[fx2_match[:, 1], :].transpose()
            #logdbg('''
            #RANSAC Resampling matches on %r chips
            #* xy_thresh2   = %r * (chip diagonal length)
            #* theta_thresh = %r (orientation)
            #* sigma_thresh = %r (scale)''' % (num_m, xy_thresh2, theta_thresh, sigma_thresh))
            inliers = ransac(fpts1_match, fpts2_match, xy_thresh2, None) #sigma_thresh
            # Remove the scores of chip pairs which generate RANSAC errors
            if inliers is None:
                cx2_fs[tcx][:] = -2 # -2 is the Degenerate Homography Code
                continue
            outliers = True - inliers
            # Set the score of previously valid outliers to 0
            cx2_fs[tcx][outliers] = np.minimum(cx2_fs[tcx][outliers], -1) # -1 is the Spatially Inconsistent Code
            # Get the new score of the database instance.
            # This is only used for a temp np.minimum calculation.
            tmp_rrscore = np.sum(cx2_fs[tcx], axis=0)
            if tmp_rrscore > 0:
                #Keep track of our worst reranked score.
                min_reranked_score = min(min_reranked_score, tmp_rrscore)
        # Non-Reranked scores should always be less than reranked np.ones.
        if num_rerank < num_c:
            logdbg('Rescaling '+str(num_c - num_rerank)+' non-reranked scores')
            all_nonreranked_scores = [s for scores in cx2_fs_[num_rerank:] for s in scores]
            if len(all_nonreranked_scores) == 0:
                max_unreranked_score = 1.
            else:
                max_unreranked_score = max(all_nonreranked_scores)
            unrr_scale_factor = .5 * min_reranked_score / max_unreranked_score
            for tcx in top_cxs[num_rerank:]:
                cx2_fs[tcx] = np.multiply(cx2_fs[tcx], unrr_scale_factor)
        else:
            logdbg('Entire training set was reranked')
        logdbg('Reranking done.')
        rr.cx2_fs     = cx2_fs
        return True

    def compute_scores(qm, rr):
        ' Aggregates the votes for chips into votes for animal names'
        logdbg('Aggregating Feature Scores ')
        nm, vm, cm = qm.hs.get_managers('nm', 'vm', 'cm')
        cx2_fm, cx2_fs = rr.dynget('cx2_fm', 'cx2_fs')

        cx2_nscore = -np.ones(cm.max_cx + 1, dtype=np.float32)
        cx2_cscore = qm.compute_cscore(cx2_fs)

        nx2_fx2_scores = {}
        # Analyze the freq of a keypoint matching a name later
        nx2_fx2_freq = {}
        for (cx, fm) in enumerate(cx2_fm):
            # Each keypoint votes for the highest scoring match
            # it had to a particular name. (it can vote for multiple names)
            fs = cx2_fs[cx]
            nx = cm.cx2_nx[cx]
            if nx == nm.UNIDEN_NX(): # UNIDEN is a unique name
                nx = -int(cx)
            if not nx in nx2_fx2_scores.keys():
                nx2_fx2_scores[nx] = {}
                nx2_fx2_freq[nx] = {}
            for qfs, (qfx, fx) in iter(zip(fs, fm)):
                if qfx in nx2_fx2_scores[nx].keys():
                    nx2_fx2_scores[nx][qfx] = max(nx2_fx2_scores[nx][qfx], qfs)
                    nx2_fx2_freq[nx][qfx] += 1
                else:
                    nx2_fx2_scores[nx][qfx] = qfs
                    nx2_fx2_freq[nx][qfx]   = 1

        for nx in nx2_fx2_scores.keys():
            fx2_scores = nx2_fx2_scores[nx]
            scores = np.array(fx2_scores.values())
            nscore = scores[scores > 0].sum()
            if nx < 0: # UNIDEN HACK. See -int(cx) above
                cx2_nscore[-nx] = nscore
            else:
                for cx in nm.nx2_cx_list[nx]:
                    cx2_nscore[cx] = nscore

        rr.cx2_cscore = cx2_cscore
        rr.cx2_nscore = cx2_nscore
# --- /QueryManager  ---


# ----------------
class QueryResult(AbstractManager):
    'Wrapper around raw results which computes the top results'
    def __init__(res, hs, rr):
        super(QueryResult, res).__init__(hs)
        logdbg('Constructing Query Result')
        res.rr = rr
        # Set Result Settings based on preferences
        result_prefs = hs.am.algo_prefs.results
        res.score_type         = result_prefs.score
        res.one_result_per_name = result_prefs.one_result_per_name
        # Return all matches higher than this threshold. Subject to...
        res.top_thresh         = result_prefs.match_threshold
        # And return between this many results
        res.num_top_min        = result_prefs.min_num_results
        res.num_top_max        = result_prefs.max_num_results
        # And add extra runners up for context
        res.num_extra_return   = result_prefs.extra_num_results

    def _get_num_top(res, xsort, scores):
        '''Helper function -
        Takes the scores by whatever type of result is requested
        Finds the number of results above a threshhold plus context'''
        # Calculate the number of results greater than threshold
        rev_top = (scores[xsort]>res.top_thresh)[::-1].argmax()
        num_top = len(xsort) - rev_top if rev_top > 0 else 0
        # Such that there are no less than num_top_min
        num_top = max(num_top, min(len(scores), res.num_top_min))
        # Such that there are no more than num_top_max
        num_top = min(num_top, min(len(scores), res.num_top_max))
        # With the extra added, without overflowing
        num_top = max(num_top, min(len(scores), num_top+res.num_extra_return))
        return num_top

    # Top chips
    def top_cx(res):
        'Returns a list of the top chip indexes'
        cx_sort = res.cx_sort()
        scores   = res.scores()
        num_top = res._get_num_top(cx_sort, scores)
        return cx_sort[0:num_top]

    # Top names and chips
    def top_nxcx(res):
        '''Returns of a tuple containing two lists:
            1: List of top name indexes,
            2: the chip indexes corresponding to the name'''
        nx_sort, nx_sort2_cx = res.nxcx_sort()
        nx_bestcx = np.empty(len(nx_sort2_cx), dtype=np.uint32)
        for nx, cxs in enumerate(nx_sort2_cx):
            nx_bestcx[nx] = cxs[0] if np.iterable(cxs) else cxs
        num_top = res._get_num_top(nx_bestcx, res.scores())
        return (nx_sort[0:num_top], nx_sort2_cx[0:num_top])

    def scores(res):
        'returns the cx2_xscore np.array'
        if    res.score_type == 'cscore': return res.rr.cx2_cscore
        # This is sort of like query expansion
        elif  res.score_type == 'nscore': return res.rr.cx2_nscore
    def cx_sort(res): # Sorted Chips
        'returns the valid cxs, sorted by scores, '
        cx_sort_ = res.scores().argsort()[::-1]
        invalids = set(res.hs.cm.invalid_cxs().tolist()+[res.rr.qcx])
        return np.array([cx for cx in cx_sort_ \
                         if cx not in invalids], dtype=np.uint32)
    def nxcx_sort(res): 
        '''
        Sorted Names and Chips
        Returns the a tuple of sorted nxs, with the top corresponding cx
        '''
        #TODO Chance nxcx to something better
        cm = res.hs.cm
        nm = res.hs.nm
        cx_sort = res.cx_sort()
        cx2_nx_sort = cm.cx2_nx[cx_sort]
        nx_sort = []; nx_sort2_cx = []; nx_unique = set()
        for cx, nx in iter(zip(cx_sort, cx2_nx_sort)):
            if not nx in nx_unique or nx == nm.UNIDEN_NX():
                nx_sort.append(nx)     # nx is not an index into nx_sort
                nx_sort2_cx.append(cx) # there may be duplicate UNIDEN nxs
                nx_unique.add(nx)      #
        return nx_sort, nx_sort2_cx

    def __str__(res):
        return res.result_str()

    def qcid2_(res, *dynargs): #change to query2_?
        cm = res.hs.cm
        return cm.cx2_(res.rr.qcx, *dynargs)

    def tcid2_(res, *dynargs): #change to top2_?
        'returns the specified results in args: '
        # Top scoring chips, regardless of name.
        if res.one_result_per_name:
            _, top_cx = res.top_nxcx()
        # Top scoring chips of each name
        else:
            top_cx = res.top_cx()
        cm = res.hs.cm
        dyngot = cm.cx2_(top_cx, *dynargs)
        for ix in xrange(len(dyngot)):
            # Just drop the scores in the unfilled spots
            if type(dyngot[ix]) != np.ndarray and dyngot[ix] == '__UNFILLED__':
                dyngot[ix] = res.scores()[top_cx]
        return dyngot

    def get_precision(res):
        raise NotImplementedError('implement ground truth scoring metrics here')
        pass

    def get_recall(res):
        raise NotImplementedError('implement ground truth scoring metrics here')
        pass

    def result_str(res):#, scores=None):
        # TODO: Move this to experiment output format. and use this instead
        dynargs = ('cid', 'nid', 'name')
        (qcid, qnid, qname) =  res.qcid2_(*dynargs)
        (tcid, tnid, tname, tscore) = res.tcid2_(*dynargs+('score', ))
        result_str = '''
        ___Query Result___
    qcid=%d ; qnid=%d ; qname=%s

    Top :
        cid:   %r
        nid:   %r
        name:  %r
        score: %r
            ''' % (qcid, qnid, qname, \
                   str(tcid), str(tnid), str(tname), str(tscore))
        return result_str
# --- /QueryResult ---
