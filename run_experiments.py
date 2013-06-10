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

def print_dbvslist(dbvslist):
    for tup in dbvslist:
        print(tup[0].get_dbid()+' vs '+tup[1].get_dbid())

def visualize_all_results(dbvslist, cx2rr_list):
    if not exists('results'):
        os.mkdir('results')
    i = 0
    # Skip 2 places do do symetrical matching
    matching_pairs_list = []
    for i in range(len(dbvslist))[::2]:
        # Database handles.
        hsA, hsB = dbvslist[i]
        results_name = (hsA.get_dbid()+' vs '+hsB.get_dbid())
        print('Visualizing: '+results_name)
        hsA.dm.draw_prefs.figsize = (19.2,10.8)
        # Symetric results.
        cx2rr_dirAB = cx2rr_list[i]
        cx2rr_dirBA = cx2rr_list[i+1]
        # Visualize the probability of a chip score.
        __CHIPSCORE_PROBAILITIES__ = True
        if __CHIPSCORE_PROBAILITIES__:
            if not exists('results/chipscore_probabilities'):
                os.mkdir('results/chipscore_probabilities')
            fig1 = viz_chipscore_pdf(hsA, hsB, cx2rr_dirAB, i)
            fig2 = viz_chipscore_pdf(hsB, hsA, cx2rr_dirBA, i+1)
            fig1.savefig(results_name+'_chipscoreAB.png', format='png')
            fig2.savefig(results_name+'_chipscoreBA.png', format='png')
        # Visualize chips which symetrically match across databases.
        __SYMETRIC_MATCHINGS__ = True
        if __SYMETRIC_MATCHINGS__:
            matching_pairs = get_symetric_matchings(hsA, hsB, cx2rr_dirAB, cx2rr_dirBA)
            matching_pairs_list.append(matching_pairs)
            viz_symetric_matchings(matching_pairs, results_name)

# --- VISUALIZATIONS ---
def viz_symetric_matchings(matching_pairs, results_name):
    output_dir = 'results/symetric_matches'
    if not exists(output_dir):
        os.mkdir(output_dir)
    print('  * Visualizing '+str(len(matching_pairs))+' matching pairs')
    for cx, cx2, match_pos, match_pos1, res1, res2 in matching_pairs:
        for res in (res1,res2):
            res.hs.dm.draw_prefs.ellipse_bit = True 
            res.hs.dm.draw_prefs.figsize=(19.2,10.8)
            res.visualize()
            fignum = res.hs.dm.draw_prefs.fignum
            fig = figure(num=fignum, figsize=(19.2,10.8))
            fig.show()
            fig.canvas.set_window_title('Symetric Matching: '+str(cx)+' '+str(cx2))
            fig_fname = results_name+\
                    '__symmpos_'+str(match_pos)+'_'+str(match_pos1)+\
                    '__cx_'+str(cx)+'_'+str(cx2)+\
                    'AB'+\
                    '.png'
            fig_fpath = realpath(join(output_dir, fig_fname))
            print('      * saving to '+fig_fpath)
            fig.savefig(fig_fpath, format='png')
            fig.clf()

def get_symetric_matchings(hsA, hsB, cx2rr_dirAB, cx2rr_dirBA):
    'returns database, cx, database cx'
    import numpy as np
    sym_match_thresh = 10

    matching_pairs = []
    valid_cxsB = hsB.cm.get_valid_cxs()
    lop_thresh = 10
    for count in xrange(len(cx2rr_dirAB)):
        rr = cx2rr_dirAB[count]
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
            rr2  = cx2rr_dirBA[count]
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

def viz_featscore_pdf(hsA, hsB, cx2rr_dirAB, fignum):
    chipscore_data = []
    num_queries = len(cx2rr_dirAB)
    for count in xrange(num_queries):
        rr = cx2rr_dirAB[count]
        top_scores = res.top_scores()
        chipscore_data[count, 0:len(top_scores)] = top_scores


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

    dbpath_list = [bajo_bonito, optimizas, westpoint]
    hsdb_list = []
    for dbpath in dbpath_list:
        hsdb = HotSpotterAPI(dbpath)
        hsdb_list.append(hsdb)

    if len(sys.argv) > 1 and sys.argv[1] == '--delete':
        for hsdb in hsdb_list:
            hsdb.delete_precomputed_results()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        for hsdb in hsdb_list:
            print hsdb.db_dpath
            result_dir  = hsdb.db_dpath+'/.hs_internals/computed/query_results'
            result_list = os.listdir(result_dir)
            result_list.sort()
            for fname in result_list:
                print '  '+fname
        sys.exit(0)

    __ENSURE_MODEL__ = False
    if __ENSURE_MODEL__:
        for hsdb in hsdb_list:
            hsdb.ensure_model()
    else: 
        # The sample set things its 0 if you dont at least do this
        for hsdb in hsdb_list:
            hsdb.vm.sample_train_set()
        

    # Get all combinations of database pairs
    dbvslist = []
    for hsdbA in hsdb_list:
        for hsdbB in hsdb_list:
            if not hsdbA is hsdbB:
                dbtup1 = (hsdbA, hsdbB)
                dbtup2 = (hsdbB, hsdbA)
                if not dbtup1 in dbvslist:
                    assert not dbtup2 in dbvslist
                    dbvslist.append(dbtup1)
                    dbvslist.append(dbtup2)

    cx2rr_list = [query_db_vs_db(hsA, hsB) for hsA, hsB in dbvslist]
    visualize_all_results(dbvslist, cx2rr_list)
