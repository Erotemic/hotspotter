from hotspotter.HotSpotterAPI import HotSpotterAPI 
from hotspotter.algo.spatial_functions import ransac
from hotspotter.helpers import alloc_lists, Timer, vd, join_mkdir
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
import matplotlib

from scipy.stats.kde import gaussian_kde
from numpy import linspace,hstack
from pylab import *
    
# --- PARAMETERS ---
dbid_list   = ['Lionfish/LF_Bajo_bonito',
               'Lionfish/LF_OPTIMIZADAS_NI_V_E',
               'Lionfish/LF_WEST_POINT_OPTIMIZADAS']


__readable_dbmap__ = {'LF_Bajo_bonito':'Bajo Bonito', 
                      'LF_OPTIMIZADAS_NI_V_E':'Optimizadas',
                      'LF_WEST_POINT_OPTIMIZADAS':'West Point'}
                    
#dbid_list = ['NAUT_Dan', 'WS_sharks']

workdir = '/media/SSD_Extra/'
if sys.platform == 'win32':
    workdir = 'D:/data/work/'

global __SANS_GT__

__cmd_mode__               = False
__cmd_run_mode__           = False
__method_2_matchthresh__   = {'LNRAT':12, 'COUNT':20}

__FIGSIZE__                = (19.2,10.8)
__K__                      = 1
__NUM_RANKS_CSPDF__        = 1
__METHOD__                 = 'LNRAT'

__SANS_GT__                = True
__THRESHOLD_MATCHINGS__    = True

__CHIPSCORE_PROBAILITIES__ = True
__INDIVIDUAL_CHIPSCORES__  = True
__AGGREGATE_CHIPSCORES__   = True
__ENSURE_MODEL__           = True

__FEATSCORE_STATISTICS__   = False
__SYMETRIC_MATCHINGS__     = False

# --- DRIVERS ---
def get_results_name(hsA, hsB):
    if hsA is hsB:
        results_name = (__readable_dbmap__[hsA.get_dbid()]+' vs self')
    else:
        results_name = (__readable_dbmap__[hsA.get_dbid()]+' vs '\
                        +__readable_dbmap__[hsB.get_dbid()]) 
    return results_name

def safe_savefig(fig, fpath):
    if fpath[-4:] == '.png':
        format = 'png'
    if fpath[-4:] == '.jpg':
        format = 'jpg'
    [full_path, sanatized_fname] = os.path.split(fpath)
    sanatized_fname = sanatized_fname.replace(' vs ','-vs-')
    sanatized_fname = sanatized_fname.replace('Bajo Bonito','BB')
    sanatized_fname = sanatized_fname.replace('Optimizadas','OP')
    sanatized_fname = sanatized_fname.replace('West Point','WP')
    sanatized_fname = sanatized_fname.replace(' ','')
    sanatized_fname = sanatized_fname.replace('_','-')
    sanatized_fpath = join(full_path, sanatized_fname)
    print('Saving Figure: '+str(sanatized_fpath))
    fig.savefig(sanatized_fpath, format=format)

def query_db_vs_db(hsA, hsB):
    'Runs cross database queries / reloads cross database queries'
    vs_str = get_results_name(hsA, hsB) 
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

def visualize_all_results(dbvs_list, count2rr_list, symx_list, result_dir):
    global __SANS_GT__
    
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    i = 0
    # Skip 2 places do do symetrical matching
    matching_pairs_list = []
    symdid_set = set([]) # keeps track of symetric matches already computed

    withindb_chipscore_tup_list = []
    crossdb_chipscore_tup_list = []

    # cross-database chipscores probabilities in one graph
    if __CHIPSCORE_PROBAILITIES__:
        output_dir = join_mkdir(result_dir, 'chipscore_probabilities')
        combocrossdb_fname = join(output_dir, 'combocrossdb-chipscore')
        fig = figure(0, figsize=__FIGSIZE__)
        fig.clf()
        title('Probability density of cross-database chipscore experiments')
        tmp_count = 0.0
        tmp_total = len(dbvs_list)+1
        for i in range(len(dbvs_list)):
            hsA, hsB = dbvs_list[i]
            if hsA is hsB: continue
            results_name = get_results_name(hsA, hsB) 
            print(' combining: '+results_name)
            # Symetric results.
            count2rr_AB = count2rr_list[i]
            chipscore_data, ischipscore_TP = get_chipscores(hsA, hsB, count2rr_AB)
            bigplot_chipscores(results_name,
                               chipscore_data,
                               ischipscore_TP,
                               'NA-fname',
                               labelaug=results_name+' ',
                               releaseaxis=True, 
                               releasetitle=False, 
                               color=get_cmap('Set1')(tmp_count/tmp_total),
                               plotTPFP=False,
                               sameplot=True,
                               holdon=True)
            tmp_count += 1
        legend(prop={'size':22})
        #fig.show()
        fig.tight_layout()
        safe_savefig(fig, combocrossdb_fname+'.png')

    for i in range(len(dbvs_list)):
        # Database handles.
        hsA, hsB = dbvs_list[i]
        results_name = get_results_name(hsA, hsB) 
        print('Individual Visualizations: '+results_name)
        # Symetric results.
        count2rr_AB = count2rr_list[i]

        if __CHIPSCORE_PROBAILITIES__:
            # Visualize the probability density of a chip score.
            output_dir = join_mkdir(result_dir, 'chipscore_probabilities')
            chipscore_fname = join(output_dir, results_name+'-chipscore')

            chipscore_data, ischipscore_TP = get_chipscores(hsA, hsB, count2rr_AB)
            if hsA is hsB: 
                withindb_chipscore_tup_list.append((chipscore_data, ischipscore_TP))
            else:
                crossdb_chipscore_tup_list.append((chipscore_data, ischipscore_TP))

            if __INDIVIDUAL_CHIPSCORES__: 
                print('  * Visualizing chipscore probabilities')
                bigplot_chipscores(results_name, chipscore_data, ischipscore_TP,
                                   chipscore_fname, plotTPFP=True, sameplot=True)

        if __FEATSCORE_STATISTICS__:
            print('  * Visualizing feature score statistics')
            # Visualize the individual feature match statistics
            output_dir = join_mkdir(result_dir, 'featmatch')
            (TP_inlier_score_list,
             TP_outlier_score_list,
             TP_inlier_scale_pairs,
             TP_outlier_scale_pairs,
             FP_inlier_score_list,
             FP_outlier_score_list,
             FP_inlier_scale_pairs,
             FP_outlier_scale_pairs) = get_featmatch_stats(hsA, hsB, count2rr_AB, i)
            fig_sd1, fig_fs1 = viz_featmatch_stats(outlier_scale_pairs, inlier_scale_pairs)
            fig_sd1.savefig(join(output_dir, results_name+'-scalediff.png'), format='png')
            fig_fs1.savefig(join(output_dir, results_name+'-fmatchscore.png'), format='png')

        if __THRESHOLD_MATCHINGS__:
            # Visualize chips which have a results with a high score
            output_dir = join_mkdir(result_dir, 'threshold_matches')
            if hsA is hsB:
                output_dir = join_mkdir(result_dir, 'threshold_matches', 'within_db')
            print('  * Visualizing threshold matchings')
            viz_threshold_matchings(hsA, hsB, count2rr_AB, output_dir)

        if __SYMETRIC_MATCHINGS__:
            # Visualize chips which symetrically.
            # Do not symetric match twice 
            if (hsA, hsB) in symdid_set: 
                print('  * Already computed symetric matchings')
                continue
            else: 
                symdid_set.add((hsA, hsB))
                symdid_set.add((hsB, hsA))
            output_dir = join_mkdir(result_dir, 'symetric_matches')
            # Find the symetric index
            symx = symx_list[count]
            count2rr_BA = count2rr_list[symx]
            print('  * Visualizing symetric matchings')
            matching_pairs = get_symetric_matchings(hsA, hsB, count2rr_AB, count2rr_BA)
            viz_symetric_matchings(matching_pairs, results_name, output_dir)
        # endfor
    print('Aggregate Visualizations: ')
    if __AGGREGATE_CHIPSCORES__:
        output_dir  = join_mkdir(result_dir, 'chipscore_probabilities')
        for holdon in [False, True]:
            fig = figure(0, figsize=__FIGSIZE__)
            fig.clf()
            within_lbl = ['within-db','within-db-sansgt'][__SANS_GT__]
            cross_lbl = 'cross-db'
            agg_chipscore_titlestr = \
                    'Probability desnity of chip-scores \n' + \
                    within_lbl+'-databases vs cross-databases\n' + \
                    'scored with: '+__METHOD__ + \
                    ' k='+str(__K__)

            title(agg_chipscore_titlestr)
            fig.canvas.set_window_title(agg_chipscore_titlestr)
            aggchipscore_color = [1,0,0]
            for expt_type, chipscore_tup_list in zip((within_lbl, cross_lbl),\
                                                     (withindb_chipscore_tup_list,\
                                                      crossdb_chipscore_tup_list)):
                print ('  * Visualizing '+expt_type+' Aggregate Chipscores')
                expt_type_full     = expt_type.replace('db', 'database')
                aggchipscore_fname = join(output_dir, expt_type+'-aggchipscore')
                chipscore_data_all, ischipscore_TP_all =  [list(t) for t in \
                                                            zip(*chipscore_tup_list)]
                aggchipscore_data = np.vstack(chipscore_data_all)
                aggischipscore_TP = np.vstack(ischipscore_TP_all)
                bigplot_chipscores(expt_type_full+' experiments',
                                aggchipscore_data,
                                aggischipscore_TP,
                                aggchipscore_fname,
                                releaseaxis=True,
                                color=aggchipscore_color,
                                releasetitle=(not holdon),
                                labelaug=expt_type+' ',
                                sameplot=True, 
                                holdon=holdon)
                aggchipscore_color = [0,0,1]
            if holdon == True:
                aggchipscore_fname = join(output_dir, within_lbl+'-'+cross_lbl+'aggchipscore')
                legend()
                fig.tight_layout()
                safe_savefig(fig, aggchipscore_fname+'.png')
    print("Vizualizations Complete")
            

# --- Big Plots ---

def bigplot_chipscores(results_name,
                       chipscore_data,
                       ischipscore_TP,
                       chipscore_fname,
                       sameplot=False,
                       holdon=False,
                       plotTPFP=False,
                       plotall=True,
                       releasetitle=None,
                       releaseaxis=None,
                       color=None,
                       **kwargs):

    if sameplot: 
        fignum = 0
        fig = figure(0, figsize=__FIGSIZE__)
        if not holdon:
            fig.clf()
        kwargs['holdon'] = True
        kwargs['color'] = [0,0,1] if color is None else color
        
        kwargs['releaseaxis'] = True if releaseaxis is None else releaseaxis
        kwargs['releasetitle'] = True if releasetitle is None else releasetitle
    else: 
        fignum=1
    if plotall:
        fig_chipscore  = viz_chipscores(results_name, chipscore_data,
                                    ischipscore_TP,restype='', fignum=fignum, **kwargs)
    if plotTPFP: # DO TP / FP on self queries and not __SANS_GT__
        if sameplot: 
            kwargs['color'] = [0,1,0] if color is None else color
            kwargs['releaseaxis'] = False
            kwargs['releasetitle'] = False
        else: 
            fignum = 1
        print('plotting true positives')
        fig_chipscoreTP = viz_chipscores(results_name, chipscore_data,
                                         ischipscore_TP, restype='TP', fignum=fignum,
                                        **kwargs)
        if sameplot: kwargs['color'] = [1,0,0]  if color is None else color
        else: fignum = 2
        print('plotting false positives')
        fig_chipscoreFP = viz_chipscores(results_name, chipscore_data,
                                         ischipscore_TP, restype='FP', fignum=fignum,
                                        **kwargs)
    if sameplot:
        if not holdon:
            fig.tight_layout()
            #fig.show()
            legend()
            safe_savefig(fig, chipscore_fname+'.png')
        return fig
    else:
        fig_chipscore.tight_layout()
        fig_chipscore.savefig(chipscore_fname+'.png', format='png')
        if plotTPFP:
            fig_chipscoreTP.tight_layout()
            fig_chipscoreTP.savefig(chipscore_fname+'TP.png', format='png')
            fig_chipscoreFP.tight_layout()
            fig_chipscoreFP.savefig(chipscore_fname+'FP.png', format='png')
            return fig_chipscore, fig_chipscoreTP, fig_chipscoreFP
        else:
            return fig_chipscore



# --- Visualizations ---

# Visualization of chips which match symetrically (in top x results)
def viz_symetric_matchings(hsA, hsB, matching_pairs, results_name, output_dir='symetric_matches'):
    print('  * Visualizing '+str(len(matching_pairs))+' matching pairs')
    for cx, cx2, match_pos, match_pos1, res1, res2 in matching_pairs:
        for res, suffix in zip((res1,res2), ('AB','BA')):
            fig.tight_layout()
            res.visualize()
            fignum = 0
            fig = figure(num=fignum, figsize=__FIGSIZE__)
            #fig.show()
            fig.canvas.set_window_title('Symetric Matching: '+str(cx)+' '+str(cx2))
            fig_fname = results_name+\
                    '__symmpos_'+str(match_pos)+'_'+str(match_pos1)+\
                    '__cx_'+str(cx)+'_'+str(cx2)+\
                    suffix+\
                    '.png'
            fig_fpath = realpath(join(output_dir, fig_fname))
            print('      * saving to '+fig_fpath)
            fig.tight_layout()
            safe_savefig(fig, fig_fpath)
            fig.clf()

# Visualization of images which match above a threshold 
def viz_threshold_matchings(hsA, hsB, count2rr_AB, output_dir='threshold_matches'):
    'returns database, cx, database cx'
    import numpy as np
    valid_cxsB = hsB.cm.get_valid_cxs()
    num_found = 0
    
    match_threshold = __method_2_matchthresh__.get(__METHOD__, 10)
    for count in xrange(len(count2rr_AB)):
        rr = count2rr_AB[count]
        qcx = rr.qcx
        res = QueryResult(hsB, rr, hsA)
        qname = res.qhs.cm.cx2_name(qcx)
        # Set matching threshold
        res.top_thresh = match_threshold
        res.num_top_min = 0
        res.num_top_max = 5
        res.num_extra_return = 0
        # See if there are matches
        top_cxs = res.top_cx()
        top_names = res.hs.cm.cx2_name(top_cxs)
        top_scores = res.scores()[top_cxs]
        if len(top_cxs) > 0:
            tsstr = str(int(round(top_scores[0])))
            res.visualize()
            results_name = get_results_name(res.qhs, res.hs)

            matching_names_set = set([name.replace('Lionfish','') for name in\
                                  (top_names+[qname])])
            matching_names = '_'.join(list(matching_names_set))
            fig_fname = results_name+'-score'+tsstr+'-cx'+str(qcx)+'-'+matching_names+'.jpg'
            #print('  * Threshold Match: '+str(res))
            fig = figure(0)
            fig.tight_layout()
            safe_savefig(fig, realpath(join(output_dir, fig_fname)))
            num_found += 1
    print('  * Visualized '+str(num_found)+' above thresh: '+str(match_threshold))            


# Visualization of feature score probability
def viz_fmatch_score(results_name, inlier_score_list, outlier_score_list, fignum=0):
    inlier_scores  = np.array(inlier_score_list)
    outlier_scores = np.array(outlier_score_list)
    # Set up axes and labels: fscores
    fig_scorediff = figure(num=fignum, figsize=__FIGSIZE__)
    fig_scorediff.clf()
    title_str = 'Probability density of feature scores \n' + \
        'datasets: '+results_name+'\n'+\
        'scored with: '+hsB.am.algo_prefs.query.method +\
        ' k='+str(hsB.am.algo_prefs.query.k)
    xlabel('feature score ('+hsB.am.algo_prefs.query.method+')')
    ylabel('probability density')
    title(title_str)
    inlier_args  = {'label':'P(fscore | inlier)',  'color':[0,0,1]}
    outlier_args = {'label':'P(fscore | outlier)', 'color':[1,0,0]}
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
    #fig_scorediff.show()
    return fig_scorediff

# Visualization of scale difference probability
def viz_fmatch_scalediff(hsA, hsB, outlier_scale_pairs, inlier_scale_pairs, fignum=0):
    out_scales = np.array(outlier_scale_pairs)
    in_scales = np.array(inlier_scale_pairs)
    out_scale_diff = np.abs(out_scales[:,0] - out_scales[:,1])
    in_scale_diff = np.abs(in_scales[:,0] - in_scales[:,1])
    # Remove some extreme data
    in_scale_diff.sort() 
    out_scale_diff.sort() 
    subset_in  = in_scale_diff[0:int(len(in_scale_diff)*.88)]
    subset_out = out_scale_diff[0:int(len(out_scale_diff)*.88)]
    # Set up axes and labels: scalediff
    fig_scalediff = figure(num=fignum, figsize=__FIGSIZE__)
    fig_scalediff.clf()
    title_str = 'Probability density of feature scale differences (omitted largest 12%) \n' + \
        'queries from: '+hsA.get_dbid() + '\n' + \
        'results from: '+hsB.get_dbid()  + '\n' + \
        'scored with: '+hsB.am.algo_prefs.query.method + \
        ' k='+str(hsB.am.algo_prefs.query.k)
    xlabel('scale difference')
    ylabel('probability density')
    title(title_str)
    fig_scalediff.canvas.set_window_title(title_str)
    inlier_args  = {'label':'P( scale_diff | inlier )',  'color':[0,0,1]}
    outlier_args = {'label':'P( scale_diff | outlier )', 'color':[1,0,0]}
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
    #fig_scalediff.show()
    return fig_scalediff


def viz_chipscores(results_name, chipscore_data, ischipscore_TP, restype='',
                   fignum=0, holdon=False, releaseaxis=None, releasetitle=None,
                   color=None, labelaug='', **kwargs):
    ''' Displays a pdf of how likely matching scores are.
    Input: chipscore_data - QxT np.array containing Q queries and T top scores
    Output: A matplotlib figure
    '''
    # Prepare Plot
    no_data = False
    if restype == 'TP':
        typestr = 'True Positive '
        if len(chipscore_data[ischipscore_TP]) == 0:
            no_data = True
        else:
            max_score = round(chipscore_data[ischipscore_TP].max()+1)
    elif restype == 'FP':
        typestr = 'False Positive '
        if len(chipscore_data[True - ischipscore_TP]) == 0:
            no_data = True
        else:
            max_score = round(chipscore_data[True - ischipscore_TP].max()+1)
    else:
        typestr = ''
        if len(chipscore_data) == 0:
            no_data = True
        else: 
            max_score = round(chipscore_data.max()+1)
    title_str = 'Probability density of '+typestr+'chip-scores \n' + \
            results_name + '\n' + \
            'scored with: '+ __METHOD__ + \
            ' k='+str(__K__)
    fig = figure(num=fignum, figsize=__FIGSIZE__)
    if not holdon:
        fig.clf()
    if not holdon or releaseaxis is None or releaseaxis:
        xlabel('chip-score')
        ylabel('probability density')
    if not holdon or releasetitle is None or releasetitle:
        title(title_str)
        fig.canvas.set_window_title(title_str)
    #
    num_queries, num_results = chipscore_data.shape
    do_true_pos  = restype == 'TP'
    do_false_pos = restype == 'FP'

    if no_data:
        return fig

    # Compute pdf of top scores
    for tx in xrange(num_results):
        # --- plot info
        rank = tx + 1

        if do_true_pos:
            isTP   = ischipscore_TP[:,tx]
            scores = chipscore_data[isTP,tx]
        elif do_false_pos:
            isFP   = True - ischipscore_TP[:,tx]
            scores = chipscore_data[isFP,tx]
        else:
            scores = chipscore_data[:,tx]

        chipscore_pdf = gaussian_kde(scores)
        chipscore_domain = linspace(0, max_score, 100)
        extra_given = '' if restype == '' else ', '+restype
        rank_label = labelaug+'P(chip-score | chip-rank = #'+str(rank)+extra_given+') '+\
                '#examples='+str(scores.size)
        if color is None:
            line_color = get_cmap('gist_rainbow')(tx/float(num_results))
        else: 
            line_color = color
        # --- plot agg
        #hist(scores, normed=1, range=(0,max_score), bins=max_score/2, alpha=.3, label=rank_label) 
        plot(chipscore_domain, chipscore_pdf(chipscore_domain), color=line_color, label=rank_label)
    if not holdon:
        legend()
    return fig

# --- Visualization Data ---

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

def get_featmatch_stats(hsA, hsB, count2rr_AB):
    num_queries = len(count2rr_AB)
    TP_inlier_score_list   = []
    TP_outlier_score_list  = []
    TP_inlier_scale_pairs  = []
    TP_outlier_scale_pairs = []
    # False positive in this case also encompases False Negative and unknown
    FP_inlier_score_list   = []
    FP_outlier_score_list  = []
    FP_inlier_scale_pairs  = []
    FP_outlier_scale_pairs = []
    # Get Data
    print('Aggregating featmatch info for  '+str(num_queries)+' queries')
    for count in xrange(num_queries):
        rr = count2rr_AB[count]
        res = QueryResult(hsA, rr, hsB)
        # Get the cxs which are ground truth
        gtcx_list = res.get_groundtruth_cxs()
        qcx = rr.qcx
        qname = hsA.cm.cx2_name(qcx)
        # Get query features
        qfpts, _ = hsA.cm.get_feats(qcx)
        for cx in xrange(len(rr.cx2_fm)):
            # Switch to whatever the correct list to append to is
            if not gtcx_list is None and cx in gtcx_list:
                inlier_score_list   = TP_inlier_score_list   
                outlier_score_list  = TP_outlier_score_list  
                inlier_scale_pairs  = TP_inlier_scale_pairs  
                outlier_scale_pairs = TP_outlier_scale_pairs 
            else:
                inlier_score_list   = FP_inlier_score_list   
                outlier_score_list  = FP_outlier_score_list  
                inlier_scale_pairs  = FP_inlier_scale_pairs  
                outlier_scale_pairs = FP_outlier_scale_pairs 

            # Get feature matching indexes and scores
            feat_matches = rr.cx2_fm[cx]
            feat_scores_SC  = rr.cx2_fs[cx]
            feat_scores_all = rr.cx2_fs_[cx]
            nx = []
            name = hsB.cm.cx2_name(cx)
            # continue if no feature matches
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
            aQO,_,dQO = outlier_qfpts[:,2:5].transpose()
            aDO,_,dDO = outlier_fpts[:,2:5].transpose()
            inlier_scalesA  = np.sqrt(1/np.multiply(aQI,dQI))
            inlier_scalesB  = np.sqrt(1/np.multiply(aDI,dDI))
            outlier_scalesA = np.sqrt(1/np.multiply(aQO,dQO))
            outlier_scalesB = np.sqrt(1/np.multiply(aDO,dDO))
            # Append to end of array
            outlier_scale_pairs.extend(zip(outlier_scalesA, outlier_scalesB))
            inlier_scale_pairs.extend(zip(inlier_scalesA, inlier_scalesB))
    return (TP_inlier_score_list,
            TP_outlier_score_list,
            TP_inlier_scale_pairs,
            TP_outlier_scale_pairs,
            FP_inlier_score_list,
            FP_outlier_score_list,
            FP_inlier_scale_pairs,
            FP_outlier_scale_pairs)


def get_chipscores(hsA, hsB, count2rr_AB):
    '''
    Input: Two database handles and queries from A to B
    Output: Matrix of chip-scores. As well as 
    '''
    num_results = __NUM_RANKS_CSPDF__  # ensure there are N top results 
    num_queries = len(count2rr_AB)
    # Get top scores of result
    chipscore_data = -np.ones((num_queries, num_results))
    # Indicator as to if chipscore was a true positive
    ischipscore_TP = np.zeros((num_queries, num_results), dtype=np.bool)
    for count in xrange(num_queries):
        rr = count2rr_AB[count]
        res = QueryResult(hsB, rr, hsA)
        res.force_num_top(num_results) 
        top_scores = res.top_scores()
        gtpos_full = res.get_groundtruth_ranks()
        gtpos_full = [] if gtpos_full is None else gtpos_full
        gtpos_list = []
        for gt_pos in gtpos_full:
            if gt_pos < num_results: gtpos_list.append(gt_pos)
        ischipscore_TP[count, gtpos_list] = 1
        chipscore_data[count, 0:len(top_scores)] = top_scores
    return chipscore_data, ischipscore_TP

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
    print "Running experiments"
    #hsl.enable_global_logs()


    global count2rr_list

    # Some arguments from command line
    if '--count'           in sys.argv: __METHOD__ = 'COUNT'
    if '--lnrat'           in sys.argv: __METHOD__ = 'LNRAT'
    if '--diff'            in sys.argv: __METHOD__ = 'DIFF'
    if '--k1'              in sys.argv: __K__ = 1
    if '--k5'              in sys.argv: __K__ = 5
    if '--threshold'       in sys.argv: __THRESHOLD_MATCHINGS__ = True
    if '--sans-gt'         in sys.argv: __SANS_GT__ = True
    if '--num-results-cs1' in sys.argv: __NUM_RANKS_CSPDF__ = 1
    if '--num-results-cs5' in sys.argv: __NUM_RANKS_CSPDF__ = 5
    if '--cmd'             in sys.argv: __cmd_mode__ = True
    if '--cmdrun'          in sys.argv: __cmd_run_mode__ = True
    if '--delete'          in sys.argv: __cmd_run_mode__ = True


        
    # IPython forced settings
    print("Checking forced configurations")
    HAS_FORCED_SETTINGS = '__forced_sans_gt__' in vars()
    if HAS_FORCED_SETTINGS:
        print('...forcing __SANS_GT__='+str(__forced_sans_gt__))
        __SANS_GT__ = __forced_sans_gt__
        if '__prev_forced_sans_gt__' in vars():
            print('...Checking if the value has changed')
            if __prev_forced_sans_gt__ != __forced_sans_gt__:
                print('...... It changed Whelp, we\'ve got to rerun')
                del count2rr_list
            else:
                print('...... no change')
        __prev_forced_sans_gt__ = __forced_sans_gt__


    print("Checking if count2rr_list needs to reload")
    NEED_TO_RELOAD = not 'count2rr_list' in vars()
    # Build list of all databases to run experiments on
    if NEED_TO_RELOAD:
        print('... Dang no count2rr_list. Are you not in IPython?')
        hsdb_list = []
        dbpath_list = [join(workdir, dbid) for dbid in dbid_list]
        __forced_sans_gt__ = __SANS_GT__
        for dbpath in dbpath_list:
            hsdb = HotSpotterAPI(dbpath)
            hsdb_list.append(hsdb)
    else:
        print('... Great! count2rr_list is already loaded')

    print("Setting database configurations")
    for hsdb in hsdb_list:
        hsdb.am.algo_prefs.query.remove_other_names = __SANS_GT__ 
        hsdb.am.algo_prefs.query.method = __METHOD__
        hsdb.am.algo_prefs.query.k = __K__
        hsdb.dm.draw_prefs.ellipse_bit = True 
        hsdb.dm.draw_prefs.figsize = (19.2,10.8)
        hsdb.dm.draw_prefs.fignum = 0

    # Command Line Argument: Delete precomputed results
    if '--delete' in sys.argv:
        for hsdb in hsdb_list:
            hsdb.delete_precomputed_results()
        sys.exit(0)
    # Command Line Argument: List Results 
    if '--list' in sys.argv:
        for hsdb in hsdb_list:
            print hsdb.db_dpath
            rawrr_dir  = hsdb.db_dpath+'/.hs_internals/computed/query_results'
            result_list = os.listdir(rawrr_dir)
            result_list.sort()
            for fname in result_list:
                print('  '+fname)
        sys.exit(0)

    if NEED_TO_RELOAD:
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
                cur_pos = len(dbvs_list)
                symx_list += [cur_pos+1, cur_pos]
                dbvs_list.append(dbtupAB)
                dbvs_list.append(dbtupBA)

        count2rr_list = [query_db_vs_db(hsA, hsB) for hsA, hsB in dbvs_list]

    # List the database list we are running on
    print('--- Database versus list ---')
    print('   DBX --- SYMX - hsA vs hsB ')
    for dbx, (dbtup, symx) in enumerate(zip(dbvs_list, symx_list)):
        if dbtup[0] is dbtup[1]: 
            print('     ' + str(dbx) + ' --- sx' + str(symx) + ' - ' + dbtup[0].get_dbid()+' vs self')
        else:
            print('     ' + str(dbx) + ' --- sx' + str(symx) + ' - ' + dbtup[0].get_dbid()+' vs '+dbtup[1].get_dbid())
    print('---')

    # Dependents of parameters 
    results_configstr = 'results_' + __METHOD__ + '_k' + str(__K__) + ['','_sansgt'][__SANS_GT__]
    results_root = join_mkdir('Results')
    result_dir = join_mkdir(results_root, results_configstr)
    print('\n\nOutputing results in: '+result_dir+'\n\n')
    
    if not __cmd_mode__:
        # Compute / Load all query results. Then visualize
        visualize_all_results(dbvs_list, count2rr_list, symx_list, result_dir)

    if __cmd_mode__ or __cmd_run_mode__:
        print('Entering interacitve mode. You\'ve got variables.')
        i = 0
        # Skip 2 places do do symetrical matching
        matching_pairs_list = []
        symdid_set = set([])  
        hsA, hsB = dbvs_list[i]
        results_name = get_results_name(hsA, hsB)
        count2rr_AB = count2rr_list[i]
        count = 0
        import IPython 
        IPython.embed()
