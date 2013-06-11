from hotspotter.HotSpotterAPI import HotSpotterAPI 
from hotspotter.algo.spatial_functions import ransac
from hotspotter.helpers import alloc_lists, Timer
from hotspotter.other.AbstractPrintable import AbstractManager, AbstractPrintable
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.QueryManager import QueryResult
from hotspotter.other.logger import logdbg, logerr, hsl, logmsg, logwarn
from numpy import spacing as eps
from os.path import join, realpath, exists
import os
import numpy as np
import os, sys
import cPickle
import pylab

from scipy.stats.kde import gaussian_kde
from numpy import linspace,hstack
from pylab import *
    
# --- PARAMETERS ---
method = 'COUNT'
remove_other_names = True
thresh_dict = {'LNRAT':5, 'COUNT':20}
__ENSURE_MODEL__           = True
__CHIPSCORE_PROBAILITIES__ = False
__SYMETRIC_MATCHINGS__     = False
__THRESHOLD_MATCHINGS__    = True
__FEATSCORE_STATISTICS__   = False

# --- DRIVERS ---
def query_db_vs_db(hsA, hsB):
    'Runs cross database queries / reloads cross database queries'
    vs_str = hsA.get_dbid()+' vs '+hsB.get_dbid()
    print 'Running '+vs_str
    query_cxs = hsA.cm.get_valid_cxs()
    total = len(query_cxs)
    cx2_rr = alloc_lists(total)
    for count, qcx in enumerate(query_cxs):
        with Timer() as t:
            print ('Query %d / %d   ' % (count, total)) + vs_str
            rr = hsB.qm.cx2_rr(qcx, hsA)
            cx2_rr[count] = rr
    return cx2_rr

def print_dbvs_list(dbvs_list):
    for tup in dbvs_list:
        print(tup[0].get_dbid()+' vs '+tup[1].get_dbid())

def visualize_all_results(dbvs_list, count2rr_list, symx_list, result_dir, thresh):
    i = 0
    # Skip 2 places do do symetrical matching
    matching_pairs_list = []
    symdid_list = [] # keeps track of symetric matches already computed
    if not exists(result_dir):
        os.mkdir(result_dir)
    for i in range(len(dbvs_list)):
        # Database handles.
        hsA, hsB = dbvs_list[i]
        if hsA is hsB:
            results_name = (hsA.get_dbid()+' vs self')
        else:
            results_name = (hsA.get_dbid()+' vs '+hsB.get_dbid())
        print('Visualizing: '+results_name)
        # Symetric results.
        count2rr_AB = count2rr_list[i]

        # Visualize the probability of a chip score.
        if __CHIPSCORE_PROBAILITIES__:
            output_dir = join(result_dir, 'chipscore_probabilities')
            if not exists(output_dir):
                os.mkdir(output_dir)
            fig1 = viz_chipscore_pdf(hsA, hsB, count2rr_AB, i)
            fig1.savefig(join(output_dir, results_name+'_chipscore.png'), format='png')

        # Visualize the individual descriptor match statistics
        if __FEATSCORE_STATISTICS__:
            output_dir = join(result_dir, 'featscore_stats')
            if not exists(output_dir):
                os.mkdir(output_dir)
            fig_sd1, fig_fs1 = viz_featscore_stats(hsA, hsB, count2rr_AB, i)
            fig_sd1.savefig(join(output_dir, results_name+'_scalediff.png'), format='png')
            fig_fs1.savefig(join(output_dir, results_name+'_featscore.png'), format='png')
            
        # Visualize chips which have a results with a high score
        if __THRESHOLD_MATCHINGS__:
            output_dir = join(result_dir, 'threshold_matches')
            if not exists(output_dir):
                os.mkdir(output_dir)
            viz_threshold_matchings(hsA, hsB, count2rr_AB, thresh, output_dir)

        # Visualize chips which symetrically.
        if __SYMETRIC_MATCHINGS__:
            if (hsA, hsB) in symdid_list or (hsB, hsA) in symdid_list:
                continue
            symdid_list.apend((hsA, hsB))
            output_dir = join(result_dir, 'symetric_matches')
            symx = symx_list[count]
            count2rr_BA = count2rr_list[symx]
            if not exists(output_dir):
                os.mkdir(output_dir)
            matching_pairs = get_symetric_matchings(hsA, hsB, count2rr_AB, count2rr_BA)
            matching_pairs_list.append(matching_pairs)
            viz_symetric_matchings(matching_pairs, results_name, output_dir)


# --- VISUALIZATIONS ---
def viz_symetric_matchings(matching_pairs, results_name, output_dir='symetric_matches'):
    print('  * Visualizing '+str(len(matching_pairs))+' matching pairs')
    for cx, cx2, match_pos, match_pos1, res1, res2 in matching_pairs:
        for res, suffix in zip((res1,res2), ('AB','BA')):
            res.visualize()
            fignum = 0
            fig = figure(num=fignum, figsize=(19.2,10.8))
            fig.show()
            fig.canvas.set_window_title('Symetric Matching: '+str(cx)+' '+str(cx2))
            fig_fname = results_name+\
                    '__symmpos_'+str(match_pos)+'_'+str(match_pos1)+\
                    '__cx_'+str(cx)+'_'+str(cx2)+\
                    suffix+\
                    '.png'
            fig_fpath = realpath(join(output_dir, fig_fname))
            print('      * saving to '+fig_fpath)
            fig.savefig(fig_fpath, format='png')
            fig.clf()

def viz_threshold_matchings(hsA, hsB, count2rr_AB, match_threshold = 20, output_dir='threshold_matches'):
    'returns database, cx, database cx'
    import numpy as np
    valid_cxsB = hsB.cm.get_valid_cxs()
    num_found = 0
    
    for count in xrange(len(count2rr_AB)):
        rr = count2rr_AB[count]
        cx = rr.qcx
        res = QueryResult(hsB, rr, hsA)
        # Set matching threshold
        res.top_thresh = match_threshold
        res.num_top_min = 0
        res.num_top_max = 5
        res.num_extra_return = 0
        # See if there are matches
        top_cxs = res.top_cx()
        top_scores = res.scores()[top_cxs]
        if len(top_cxs) > 0:
            tsstr = str(top_scores[0])
            res.visualize()
            results_name = res.rr.dbid +' vs '+ res.rr.qdbid
            fig_fname = results_name+'_score'+tsstr+'_cx'+str(cx)+'.png'
            print('  * Threshold Match: '+str(res))
            fig = figure(0)
            fig.savefig(realpath(join(output_dir, fig_fname)), format='png')
            num_found += 1
    print('  * Visualized '+str(num_found)+' above thresh: '+str(match_threshold))            

def get_symetric_matchings(hsA, hsB, count2rr_AB, count2rr_BA):
    'returns database, cx, database cx'
    import numpy as np
    sym_match_thresh = 5

    matching_pairs = []
    valid_cxsB = hsB.cm.get_valid_cxs()
    lop_thresh = 10
    for count in xrange(len(count2rr_AB)):
        rr = count2rr_AB[count]
        cx = rr.qcx
        res = QueryResult(hsB, rr, hsA)
        top_cxs = res.top_cx()
        top_scores = res.scores()[top_cxs]
        level_of_promise = len(top_scores) > 0 and top_scores[0]
        if level_of_promise > lop_thresh:
            lop_thresh = lop_thresh + (0.2 * lop_thresh)
            print('    * Checking dbA cx='+str(cx)+' \n'+\
                  '      top_cxs='+str(top_cxs)+'\n'+\
                  '      top_scores='+str(top_scores))
        match_pos1 = -1
        for tcx, score in zip(top_cxs, top_scores):
            match_pos1 += 1
            count = (valid_cxsB == tcx).nonzero()[0]
            rr2  = count2rr_BA[count]
            res2 = QueryResult(hsA, rr2, hsB)
            top_cxs2    = res2.top_cx()
            top_scores2 = res2.scores()[top_cxs2]
            if level_of_promise > lop_thresh:
                print('      * topcxs2 = '+str(top_cxs2))
                print('      * top_scores2 = '+str(top_scores2))
            # Check if this pair has eachother in their top 5 results
            match_pos_arr = (top_cxs2 == cx).nonzero()[0]
            if len(match_pos_arr) == 0: continue
            match_pos = match_pos_arr[0]
            print('  * Symetric Match: '+str(cx)+' '+str(tcx)+'   match_pos='+str(match_pos)+', '+str(match_pos1))
            
            matching_pairs.append((cx, tcx, match_pos, match_pos1, res, res2))
    return matching_pairs

def viz_featscore_stats(hsA, hsB, count2rr_AB, fignum):
    num_queries = len(count2rr_AB)
    inlier_score_list = []
    outlier_score_list = []
    inlier_scale_pairs = []
    outlier_scale_pairs = []
    # Get Data
    print('Aggregating featscore info for  '+str(num_queries)+' queries')
    for count in xrange(num_queries):
        rr = count2rr_AB[count]
        qcx = rr.qcx
        qname = hsA.cm.cx2_name(qcx)
        # Get query features
        qfpts, _ = hsA.cm.get_feats(qcx)
        for cx in xrange(len(rr.cx2_fm)):
            # Get feature matching indexes and scores
            feat_matches = rr.cx2_fm[cx]
            feat_scores_SC  = rr.cx2_fs[cx]
            feat_scores_all = rr.cx2_fs_[cx]
            name = hsB.cm.cx2_name(cx)

            if len(feat_matches) == 0: continue

            # Get database features
            fpts, _ = hsB.cm.get_feats(cx)

            # Separate into inliers / outliers
            outliers = (feat_scores_SC == -1)
            inliers = True - outliers

            # Get info about matching scores
            outlier_scores = feat_scores_all[outliers]
            inlier_scores  = feat_scores_all[inliers]

            # Append score info
            inlier_score_list.extend(inlier_scores.tolist())
            outlier_score_list.extend(outlier_scores.tolist())

            # Get info about matching keypoint shape
            inlier_matches  = feat_matches[inliers]
            outlier_matches = feat_matches[outliers]

            inlier_qfpts  = qfpts[inlier_matches[:,0]]
            outlier_qfpts = qfpts[outlier_matches[:,0]]
            inlier_fpts   =  fpts[inlier_matches[:,1]]
            outlier_fpts  =  fpts[outlier_matches[:,1]]

            # Get the scales of matching keypoints as their sqrt(1/determinant)
            aQI,_,dQI = inlier_qfpts[:,2:5].transpose()
            aDI,_,dDI = inlier_fpts[:,2:5].transpose()
            inlier_scalesA = np.sqrt(1/np.multiply(aQI,dQI))
            inlier_scalesB = np.sqrt(1/np.multiply(aDI,dDI))

            aQO,_,dQO = outlier_qfpts[:,2:5].transpose()
            aDO,_,dDO = outlier_fpts[:,2:5].transpose()
            outlier_scalesA = np.sqrt(1/np.multiply(aQO,dQO))
            outlier_scalesB = np.sqrt(1/np.multiply(aDO,dDO))
            # Append to end of array
            outlier_scale_pairs.extend(zip(outlier_scalesA, outlier_scalesB))
            inlier_scale_pairs.extend(zip(inlier_scalesA, inlier_scalesB))

    # Do Visualization of Scale Differences
    out_scales = np.array(outlier_scale_pairs)
    in_scales = np.array(inlier_scale_pairs)
    out_scale_diff = np.abs(out_scales[:,0] - out_scales[:,1])
    in_scale_diff = np.abs(in_scales[:,0] - in_scales[:,1])

    outlier_args = {'label':'outlier','color':[1,0,0]}
    inlier_args  = {'label':'inlier', 'color':[0,0,1]}

    # Remove some extreme data
    in_scale_diff.sort() 
    out_scale_diff.sort() 
    subset_in  = in_scale_diff[0:int(len(in_scale_diff)*.88)]
    subset_out = out_scale_diff[0:int(len(out_scale_diff)*.88)]
    # Set up axes and labels: scalediff
    fig_scalediff = figure(num=42+100*fignum, figsize=(19.2,10.8))
    fig_scalediff.clf()
    title_str = 'Probability of feature scale differences (omitted largest 12%) \n' + \
        'queries from: '+hsA.get_dbid() + '\n' + \
        'results from: '+hsB.get_dbid()  + '\n' + \
        'scored with: '+hsB.am.algo_prefs.query.method
    xlabel('scale difference')
    ylabel('probability')
    title(title_str)
    fig_scalediff.canvas.set_window_title(title_str)
    inlier_args['label']  = 'P( scale_diff | inlier )'
    outlier_args['label'] = 'P( scale_diff | outlier )'
    # histogram 
    hist(subset_in, normed=1, alpha=.3, bins=100,  **inlier_args)
    hist(subset_out, normed=1, alpha=.3, bins=100, **outlier_args)
    # pdf
    sd_extent = (0, max(subset_in.max(), subset_out.max()))
    sd_domain = np.linspace(sd_extent[0], sd_extent[1], 100)
    subset_out_pdf = gaussian_kde(subset_out)
    subset_in_pdf = gaussian_kde(subset_in)
    plot(sd_domain, subset_in_pdf(sd_domain), **inlier_args) 
    plot(sd_domain, subset_out_pdf(sd_domain), **outlier_args) 
    
    legend()
    fig_scalediff.show()

    # data
    inlier_scores  = np.array(inlier_score_list)
    outlier_scores = np.array(outlier_score_list)
    # Set up axes and labels: fscores
    fig_scorediff = figure(num=43+100*fignum, figsize=(19.2,10.8))
    fig_scorediff.clf()
    title_str = 'Probability of feature scores \n' + \
        'queries from: '+hsA.get_dbid() + '\n' + \
        'results from: '+hsB.get_dbid()  + '\n' + \
        'scored with: '+hsB.am.algo_prefs.query.method
    xlabel('feature score ('+hsB.am.algo_prefs.query.method+')')
    ylabel('probability')
    title(title_str)
    inlier_args['label']  = 'P( fscore | inlier )'
    outlier_args['label'] = 'P( fscore | outlier )'
    # histogram 
    hist(inlier_scores,  normed=1, alpha=.3, bins=100, **inlier_args)
    hist(outlier_scores, normed=1, alpha=.3, bins=100, **outlier_args)
    # pdf
    fx_extent = (0, max(inlier_scores.max(), outlier_scores.max()))
    fs_domain = np.linspace(fx_extent[0], fx_extent[1], 100)
    inscore_pdf = gaussian_kde(inlier_scores)
    outscore_pdf = gaussian_kde(outlier_scores)
    plot(fs_domain, outscore_pdf(fs_domain), **outlier_args) 
    plot(fs_domain, inscore_pdf(fs_domain),  **inlier_args) 

    legend()
    fig_scorediff.show()

    return fig_scorediff, fig_scalediff


def viz_chipscore_pdf(hsA, hsB, cxA2rrB, fignum):
    ''' displays a pdf of how likely matching scores are '''
    num_results = 5  # ensure there are 5 top results 
    num_queries = len(cxA2rrB)
    # Get top scores of result
    chipscore_data = -np.ones((num_queries, num_results))
    for count in xrange(num_queries):
        rr = cxA2rrB[count]
        res = QueryResult(hsB, rr, hsA)
        res.force_num_top(num_results) 
        top_scores = res.top_scores()
        chipscore_data[count, 0:len(top_scores)] = top_scores
    # Prepare Plot
    score_range = (0, round(chipscore_data.max()+1))
    title_str = 'Probability of chip-scores \n' + \
            'queries from: '+hsA.get_dbid() + '\n' + \
            'results from: '+hsB.get_dbid()  + '\n' + \
            'scored with: '+hsB.am.algo_prefs.query.method
    fig = figure(num=fignum, figsize=(19.2,10.8))
    fig.clf()
    xlabel('chip-score')
    ylabel('probability')
    title(title_str)
    fig.canvas.set_window_title(title_str)
    # Compute pdf of top scores
    for tx in xrange(num_results):
        # --- plot info
        rank = tx + 1
        scores = chipscore_data[:,tx] 
        chipscore_pdf = gaussian_kde(scores)
        chipscore_domain = linspace(score_range[0], score_range[1], 100)
        rank_label = 'P(chip-score | chip-rank = #'+str(rank)+')'
        line_color = get_cmap('gist_rainbow')(tx/float(num_results))
        # --- plot agg
        figure(fignum)
        #hist(scores, normed=1, range=score_range, bins=max_score/2, alpha=.3, label=rank_label) 
        plot(chipscore_domain, chipscore_pdf(chipscore_domain), color=line_color, label=rank_label) 
    legend()
    fig.show()
    return fig


def high_score_matchings():
    'Visualizes each query against its top matches'
    pass


def scoring_metric_comparisons():
    'Plots a histogram of match scores of differnt types'
    pass


def spatial_consistent_match_comparisons():
    'Plots a histogram of spatially consistent match scores vs inconsistent'
    pass


# MAIN ENTRY POINT
if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    print "RUNNING EXPERIMENTS"
    #hsl.enable_global_logs()
    workdir = '/media/SSD_Extra/'
    if sys.platform == 'win32':
        workdir = 'D:/data/work/Lionfish/'
    bajo_bonito = workdir+'LF_Bajo_bonito'
    optimizas   = workdir+'LF_OPTIMIZADAS_NI_V_E'
    westpoint   = workdir+'LF_WEST_POINT_OPTIMIZADAS'

    # Dependents of parameters 
    result_dir = 'results_'+method
    thresh = 10 if not method in thresh_dict.keys() else thresh_dict[method] 

    # Build list of all databases to run experiments on
    hsdb_list = []
    dbpath_list = [bajo_bonito, optimizas, westpoint]
    for dbpath in dbpath_list:
        hsdb = HotSpotterAPI(dbpath)
        hsdb.am.algo_prefs.query.remove_other_names = remove_other_names 
        hsdb.am.algo_prefs.query.method = method
        hsdb.dm.draw_prefs.ellipse_bit = True 
        hsdb.dm.draw_prefs.figsize = (19.2,10.8)
        hsdb.dm.draw_prefs.fignum = 0
        hsdb_list.append(hsdb)

    # Delete precomputed results
    if len(sys.argv) > 1 and sys.argv[1] == '--delete':
        for hsdb in hsdb_list:
            hsdb.delete_precomputed_results()
        sys.exit(0)

    # Check what is in the query results directory
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        for hsdb in hsdb_list:
            print hsdb.db_dpath
            rawrr_dir  = hsdb.db_dpath+'/.hs_internals/computed/query_results'
            result_list = os.listdir(rawrr_dir)
            result_list.sort()
            for fname in result_list:
                print '  '+fname
        sys.exit(0)

    if __ENSURE_MODEL__:
        for hsdb in hsdb_list:
            hsdb.ensure_model()
    else: # The sample set things its 0 if you dont at least do this
        for hsdb in hsdb_list:
            hsdb.vm.sample_train_set()

    # Get all combinations of database pairs
    dbvs_list = []
    symx_list = [] # list of symetric matches
    for hsdbA in hsdb_list:
        # Reflexive case
        symx_list += [len(dbvs_list)]
        dbvs_list.append((hsdbA, hsdbA))
        for hsdbB in hsdb_list:
            # Nonreflexive cases
            if hsdbA is hsdbB: continue
            dbtupAB = (hsdbA, hsdbB)
            dbtupBA = (hsdbB, hsdbA)
            if dbtupAB in dbvs_list: continue
            assert not dbtupBA in dbvs_list
            cur_post = len(dbvs_list)
            symx_list += [cur_pos+1, cur_post]
            dbvs_list.append(dbtupAB)
            dbvs_list.append(dbtupBA)

    # Compute / Load all query results. Then visualize
    count2rr_list = [query_db_vs_db(hsA,ahsB) for hsA, hsB in dbvs_list]
    visualize_all_results(dbvs_list, count2rr_list, symx_list, result_dir, thresh)
