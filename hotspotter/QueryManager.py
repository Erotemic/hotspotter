from hotspotter.other.logger            import logdbg, logerr
from hotspotter.other.helpers           import alloc_lists
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.algo.spatial_functions  import ransac
from numpy import \
        array, logical_and, ones, float32, arange, minimum,\
        multiply, transpose, log2, tile, ndarray, int32, uint32
import numpy as np
from numpy import spacing as eps
import pylab as pl

class RawResults(DynStruct):
    def __init__(rr, *args):
        super(RawResults, rr).__init__()
        rr.dynset(*args)
        rr.cx2_cscore_ = []
        rr.cx2_nscore = []
        rr.cx2_fs_ = []
        rr.cx2_fs = []
        rr.cx2_fm = []
    def presave_clean(rr):
        # Save some memory. We dont really need these.
        rr.qfdsc = []
        rr.qfpts = []


class QueryManager(AbstractManager):
    ' Handles Searching the Vocab Manager'
    def __init__(qm, hs):
        super( QueryManager, qm ).__init__( hs )
        qm.rr = None # Reference to the last RawResult for debugging
        qm.compute_cscore = lambda cx2_fs:\
                array( [ np.sum(fs[fs > 0]) for fs in iter(cx2_fs) ] , dtype=float32)

    def  cx2_res(qm,qcx):
        'Driver function for the query pipeliene'
        hs = qm.hs
        cm = hs.cm
        (qfpts, qfdsc) = cm.get_feats(qcx)
        qcid           = cm.cx2_cid[qcx]
        qnid           = cm.cx2_nid(qcx)
        qname          = cm.cx2_name(qcx)
        logdbg(' Querying QCID='+str(qcid) )
        rr = RawResults('qcx','qcid', 'qnid', 'qname','qfpts','qfdsc',\
                         qcx , qcid ,  qnid ,  qname , qfpts , qfdsc)
        qm.rr = rr
        qm.assign_feature_matches(rr)
        qm.spatial_rerank(rr)
        qm.compute_scores(rr)
        res = QueryResult(hs, rr)
        return res
    
    def  assign_feature_matches(qm, rr):
        '''Assigns each query feature to its K nearest database features
        with a similarity-score. Each feature votes for its assigned 
        chip with this weight.'''
        logdbg('Assigning feature matches and initial scores')
        (vm,cm,am,nm) = qm.hs.get_managers('vm','cm','am','nm')
        (qcx, qcid, qfdsc, qfpts) = rr.dynget('qcx', 'qcid', 'qfdsc','qfpts')

        qindexed_bit = qcx in vm.get_train_cx()
        num_qf         = qfpts.shape[0]
        K              = am.algo_prefs.query.k + 1 
        if qindexed_bit: 
            K += 1
            logdbg('Query qcx=%d is in database ' % qcx)

        # define: Prefix K = list of K+1 nearest; k = K nearest 
        # Everything is done in a flat manner, and reshaped at the end. 
        (qfx2_Kwxs, qfx2_Kdists) = vm.nearest_neighbors(qfdsc, K)

        # Score the nearest neighbor matches
        score_functions = \
        { # p - pth nearest ; o - k+1th nearest 
            'DIFF'  : lambda p,o: o - p, 
            'RAT'   : lambda p,o: o / p+eps(1),
            'LNRAT' : lambda p,o: log2(o / p+eps(1)),
            'COUNT' : lambda p,o: 1,
            'NDIST' : lambda p,o: 10**16 - p, 
            'TFIDF' : lambda wx2_tf, wx_idf, wx: wx2_tf[wx] * wx_idf[wx]
        }
        isTFIDF        = am.algo_prefs.query.method == 'TFIDF'
        score_function = score_functions[am.algo_prefs.query.method]
        if isTFIDF: # TF-IDF voting is a little different
                # The wx2_qtf could really be per k or as agged across all K
                w_histo = bincount(qfx2_wxs, minlength=vm.numWords() )
                wx2_qtf = array( w_histo ,dtype=float32) / num_qf
                qfx2_vweight = score_function(wx2_qtf, vm.wx2_idf, qfx2_wxs)
        else: 
            qfx2_kdists_vote = qfx2_Kdists[:,0:(K-1)]
            qfx2_kdists_norm = tile(qfx2_Kdists[:,-1].reshape(num_qf, 1), (1, K-1) )
            qfx2_kweight = array([score_function(vd,nd) \
                                for (vd, nd) in zip(qfx2_kdists_vote.flat, qfx2_kdists_norm.flat)], dtype=np.float32)
            qfx2_kweight.shape = (num_qf, K-1)

        # We have the scores, now who do we vote for? 
        
        # Remove yourself from the query
        #query feature index 2 agg descriptor indexes -> cids -> self_query_bit -> clean_axs
        qfx2_Kaxs_   = vm.wx2_axs[qfx2_Kwxs]
        qfx2_Kcids_  = [ vm.ax2_cid[axs] for axs  in qfx2_Kaxs_.flat ]
        qfx2_Ksqbit_ = [ qcid != cids    for cids in qfx2_Kcids_ ]
        qfx2_Kaxs    = [ np.array(axs)[sqbit].tolist() \
                        for (axs, sqbit) in zip( qfx2_Kaxs_.flat, qfx2_Ksqbit_) ]
        # Clean Vote for Info
        qfx2_Kcxs    = array([ vm.ax2_cx(axs)  for axs  in qfx2_Kaxs ])
        qfx2_Kfxs    = array([ vm.ax2_fx[axs]  for axs  in qfx2_Kaxs ])
        qfx2_Knxs    = array([ cm.cx2_nx[cxs]  for cxs  in qfx2_Kcxs ])
        if qfx2_Kfxs.size == 0:
            logerr('Cannot query when there is one chip in database')
        # Reshape Vote for Info
        qfx2_Kcxs    = array(qfx2_Kcxs).reshape(num_qf, K)
        qfx2_Kfxs    = array(qfx2_Kfxs).reshape(num_qf, K)
        qfx2_Knxs    = array(qfx2_Knxs).reshape(num_qf, K)
        
        qfx2_kcxs_vote = qfx2_Kcxs[:, 0:K-1] # vote for cx
        qfx2_kfxs_vote = qfx2_Kfxs[:, 0:K-1] # vote for fx
        qfx2_knxs_vote = qfx2_Knxs[:, 0:K-1] # check with nx

        qfx2_knxs_norm = tile(qfx2_Knxs[:, K-1].reshape(num_qf,1), (1,K-1))
        qfx2_knxs_norm[qfx2_knxs_norm == nm.UNIDEN_NX()] = 0 # Remove Unidentifieds from this test
        qfx2_kcxs_norm = tile(qfx2_Kcxs[:, K-1].reshape(num_qf,1), (1,K-1))
        # If the normalizer has the same name, but is a different chip, there is a good chance 
        # it is a correct match and was peanalized by the scoring function
        qfx2_normgood_bit = logical_and(qfx2_kcxs_vote != qfx2_kcxs_norm,\
                                        qfx2_knxs_vote == qfx2_knxs_norm)
        #qfx2_kweight[qfx2_normgood_bit] = 2

        # Allocate FeatureMatches and FeautureScores
        cx2_fm  = alloc_lists(cm.max_cx + 1)
        cx2_fs_ = alloc_lists(cm.max_cx + 1)

        qfx2_qfx = tile(arange(0,num_qf).reshape(num_qf,1), (1, K-1))
        # Add matches and scores
        for (qfx, qfs, cxs, fxs)\
                in iter(zip(qfx2_qfx.flat,\
                            qfx2_kweight.flat,\
                            qfx2_kcxs_vote.flat,\
                            qfx2_kfxs_vote.flat)):
            for (vote_cx,vote_fx) in iter(zip(cxs,fxs)):
                cx2_fm[vote_cx].append( (qfx, vote_fx) )
                cx2_fs_[vote_cx].append( qfs )

        # Convert to numpy
        for cx in xrange(len(cx2_fs_)):
            num_m = len(cx2_fm[cx])
            cx2_fs_[cx] = array(cx2_fs_[cx],dtype=float32)
            cx2_fm[cx]  = array(cx2_fm[cx],dtype=uint32).reshape(num_m,2)
        logdbg('Setting feature assignments')
        rr.cx2_fm         = cx2_fm
        rr.cx2_fs_        = cx2_fs_
        # --- end assign_feature_matches --- 

    def spatial_rerank(qm, rr):
        '''Recalculates the votes for chips by setting the similarity-score of
         spatially invalid assignments to 0'''

        logdbg('Spatially Reranking')
        cm, am = qm.hs.get_managers('cm','am')
        ( qcx , qfpts , cx2_fm , cx2_fs_ ) = rr.dynget\
        ('qcx','qfpts','cx2_fm','cx2_fs_')
        cx2_cscore_ = qm.compute_cscore(rr.cx2_fs_) 

        top_cxs = cx2_cscore_.argsort()[::-1] #set diff with order
        invalids = set(cm.invalid_cxs())
        top_cxs = array([cx for cx in top_cxs if cx not in invalids],dtype=uint32)
        
        num_c      = len(top_cxs)
        xy_thresh  = am.algo_prefs.query.spatial_thresh
        sigma_thresh = am.algo_prefs.query.sigma_thresh
        num_rerank = min(num_c, am.algo_prefs.query.num_rerank)

        cx2_fs = [arr.copy() for arr in cx2_fs_]
        if num_rerank == 0:
            logdbg('Breaking rerank. num_rerank = 0;  min(num_c, am.algo_prefs.query.num_rerank)')
        else: 
            min_reranked_score = 2^30
            #Initialize the reranked scores as the normal scores
            cm.load_features(top_cxs) #[arange(num_rerank)])
            # For each tcx (top cx) in the shortlist
            (w,h) = cm.cx2_chip_size(qcx)
            for tcx in top_cxs[0:num_rerank]:
                logdbg('Reranking qcx=%d vs tcx=%d' % (qcx, tcx))
                cid   = cm.cx2_cid[tcx]
                if cid <= 0: continue
                fx2_match  = cx2_fm[tcx]
                if len(fx2_match) == 0:
                    logdbg('Breaking rerank. len(fx2_match) == 0')
                    break 
                tfpts      = cm.cx2_fpts[tcx]
                xy_thresh2 = ( w*w + h*h ) * xy_thresh**2
                # Find which matches are spatially constent 
                fpts1_match  = transpose(qfpts[fx2_match[:,0],:])
                fpts2_match  = transpose(tfpts[fx2_match[:,1],:])
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
                cx2_fs[tcx][outliers] = minimum(cx2_fs[tcx][outliers],-1) # -1 is the Spatially Inconsistent Code
                # Get the new score of the database instance. 
                # This is only used for a temp minimum calculation. 
                tmp_rrscore = np.sum(cx2_fs[tcx],axis=0)
                if tmp_rrscore > 0:
                    #Keep track of our worst reranked score. 
                    min_reranked_score = min(min_reranked_score, tmp_rrscore)
            # Non-Reranked scores should always be less than reranked ones. 
            if num_rerank < num_c:
                logdbg('Rescaling '+str(num_c - num_rerank)+' non-reranked scores')
                all_nonreranked_scores = [s for scores in cx2_fs_[num_rerank:] for s in scores]
                if len(all_nonreranked_scores) == 0:
                    max_unreranked_score = 1.
                else:
                    max_unreranked_score = max(all_nonreranked_scores)
                unrr_scale_factor = .5 * min_reranked_score / max_unreranked_score
                for tcx in top_cxs[num_rerank:]:
                    cx2_fs[tcx] = multiply(cx2_fs[tcx], unrr_scale_factor)
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

        cx2_nscore = -ones(cm.max_cx + 1, dtype=float32)
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
            for qfs, (qfx, fx) in iter(zip(fs,fm)):
                if qfx in nx2_fx2_scores[nx].keys():
                    nx2_fx2_scores[nx][qfx] = max(nx2_fx2_scores[nx][qfx], qfs)
                    nx2_fx2_freq[nx][qfx] += 1
                else: 
                    nx2_fx2_scores[nx][qfx] = qfs
                    nx2_fx2_freq[nx][qfx]   = 1
        
        for nx in nx2_fx2_scores.keys():
            fx2_scores = nx2_fx2_scores[nx]
            scores = array(fx2_scores.values())
            nscore = np.sum(scores[scores > 0])
            if nx < 0: # UNIDEN HACK. See -int(cx) above
                cx2_nscore[-nx] = nscore
            else:
                for cx in nm.nx2_cx_list[nx]: 
                    cx2_nscore[cx] = nscore 

        rr.cx2_cscore = cx2_cscore
        rr.cx2_nscore = cx2_nscore

# ----------------
# END Query Manager 

class QueryResult(AbstractManager):
    def __init__(res, hs, rr):
        super( QueryResult, res ).__init__( hs )
        logdbg('Constructing Query Result')
        res.rr = rr
        # Set Result Settings based on preferences
        result_prefs = res.hs.am.algo_prefs.results
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
            1: List of top name indexes, 2: the chip indexes corresponding to the name'''
        nx_sort, nx_sort2_cx = res.nxcx_sort()
        nx_bestcx = np.empty(len(nx_sort2_cx),dtype=np.uint32)
        for nx, cxs in enumerate(nx_sort2_cx):
            nx_bestcx[nx] = cxs[0] if np.iterable(cxs) else cxs
        num_top = res._get_num_top(nx_bestcx, res.scores())
        return (nx_sort[0:num_top], nx_sort2_cx[0:num_top])    

    def scores(res):
        'returns the cx2_xscore array'
        if    res.score_type == 'cscore': return res.rr.cx2_cscore
        elif  res.score_type == 'nscore': return res.rr.cx2_nscore # This is sort of like query expansion
    def cx_sort(res): # Sorted Chips
        'returns the valid cxs, sorted by scores, '
        cx_sort_ = res.scores().argsort()[::-1]
        invalids = set(res.hs.cm.invalid_cxs().tolist()+[res.rr.qcx])
        return array([cx for cx in cx_sort_ if cx not in invalids],dtype=uint32)
    def nxcx_sort(res): #Sorted Names and Chips #TODO Chance nxcx to something better
        'Returns the a tuple of sorted nxs, with the top corresponding cx'
        cm, nm = res.hs.get_managers('cm','nm')
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
        if res.one_result_per_name: # Top scoring chips, regardless of name. 
            _, top_cx = res.top_nxcx() 
        else: # Top scoring chips of each name
            top_cx = res.top_cx()
        cm = res.hs.cm
        dyngot = cm.cx2_(top_cx, *dynargs)
        for ix in xrange(len(dyngot)):
            # Just drop the scores in the unfilled spots
            if type(dyngot[ix]) != ndarray and dyngot[ix] == '__UNFILLED__':
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
        dynargs =\
        ('cid', 'nid', 'name')
        (qcid , qnid , qname ) =  res.qcid2_(*dynargs)
        (tcid , tnid , tname , tscore ) = res.tcid2_(*dynargs+('score',))
        result_str = '''
        ___Query Result___
    qcid=%d ; qnid=%d ; qname=%s

    Top :
        cid:   %r
        nid:   %r
        name:  %r
        score: %r
            ''' % ( qcid, qnid, qname, str(tcid), str(tnid), str(tname), str(tscore))
        return result_str
    #--- end assign_feature_matches ---
    
