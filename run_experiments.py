from __future__ import division
from __future__ import print_function
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
import scipy
import matplotlib

from scipy.stats.kde import gaussian_kde
from numpy import linspace,hstack
from matplotlib import pyplot as plt    
        
font = {'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


def print(*args): 
    if len(args) == 0:
        sys.stdout.write('\n')
    elif len(args) == 1:
        sys.stdout.write(str(args[0])+'\n')
    else:
        sys.stdout.write(' '.join(args)+'\n')
    sys.stdout.flush()

# --- PARAMETERS ---
dbid_list   = ['Lionfish/LF_Bajo_bonito',
               'Lionfish/LF_OPTIMIZADAS_NI_V_E',
               'Lionfish/LF_WEST_POINT_OPTIMIZADAS']

__readable_dbmap__ = {'LF_Bajo_bonito':'Bajo Bonito', 
                      'LF_OPTIMIZADAS_NI_V_E':'Optimizadas',
                      'LF_WEST_POINT_OPTIMIZADAS':'West Point',
                      'NAUT_Dan':'NAUT_Dan',
                      'WS_sharks':'WS_sharks'}

__abbrev_dbmap__ = {'Bajo Bonito':'BB',
                    'Optimizadas':'OP',
                     'West Point':'WP',
                       'NAUT_Dan':'ND',
                      'WS_sharks':'WS'}
                    
#dbid_list = ['NAUT_Dan', 'WS_sharks']

workdir = '/media/SSD_Extra/'
if sys.platform == 'win32':
    workdir = 'D:/data/work/'

global __NOGT__
__NOGT__                = False

__cmd_mode__               = False
__cmd_run_mode__           = False
__method_2_matchthresh__   = {'LNRAT':5, 'COUNT':20}

__FIGSIZE__                = (19.2,10.8)
__K__                      = 5
__RESTRICT_TP__            = 5
__METHOD__                 = 'LNRAT'

__THRESHOLD_MATCHINGS__    = False
__CHIPSCORE_PROBAILITIES__ = True

__ENSURE_MODEL__           = True

__FEATSCORE_STATISTICS__   = False
__SYMETRIC_MATCHINGS__     = False


__all_exceptions__ = []
# --- DRIVERS ---
def get_results_name(hsA, hsB):
    if hsA is hsB:
        results_name = (__readable_dbmap__[hsA.get_dbid()]+' vs self')
    else:
        results_name = (__readable_dbmap__[hsA.get_dbid()]+' vs '\
                        +__readable_dbmap__[hsB.get_dbid()]) 
    return results_name

def myfigure(fignum, doclf=False, title=None):
    fig = plt.figure(fignum, figsize=__FIGSIZE__)
    if not 'user_stat_list' in fig.__dict__.keys() or doclf:
        fig.user_stat_list = []
        fig.user_notes = []
    if doclf:
        fig.clf()
        ax = plt.subplot(111)
    ax  = fig.get_axes()[0]
    if not title is None:
        ax.set_title(title)
        fig.canvas.set_window_title(title)
    return fig

def safe_savefig(fig, fpath, adjust_axes=False):
    ax  = fig.get_axes()[0]
    if adjust_axes and fig.user_stat_list != []:
        fig_min  = np.array(fig.user_stat_list).max(0)[0]
        fig_mean = np.array(fig.user_stat_list).max(0)[1]
        fig_std  = np.array(fig.user_stat_list).max(0)[2]
        fig_max  = np.array(fig.user_stat_list).max(0)[3]

        trunc_max = fig_mean+fig_std*1.5
        trunc_min = fig_min
        trunc_xticks = np.linspace(trunc_min, trunc_max,10)
        no_zero_yticks = ax.get_yticks()[ax.get_yticks() > 0]
        ax.set_xlim(trunc_min,trunc_max)
        ax.set_xticks(trunc_xticks)
        ax.set_yticks(no_zero_yticks)
    if len(fpath) < 4:
        fpath += '.JPEG'
    if fpath[-4:] == '.png':
        format = 'png'
    if fpath[-4:] == '.jpg':
        format = 'jpg'
    else: 
        fpath  += '.JPEG'
        format = 'JPEG'
    [full_path, sanatized_fname] = os.path.split(fpath)
    sanatized_fname = sanatized_fname.replace(' vs ','-vs-')
    for key, val in __abbrev_dbmap__.iteritems():
        sanatized_fname = sanatized_fname.replace(key, val)
    sanatized_fname = sanatized_fname.replace(' ','')
    sanatized_fname = sanatized_fname.replace('_','-')
    sanatized_fpath = join(full_path, sanatized_fname)
    #fig.show()
    #try: 
    if fig.user_stat_list != []:
        print('\n\n---\nSaving '+ax.get_title())
        print(' stats: ')
        for stat in fig.user_stat_list:
            print(str(stat))
        print('---\n\n')
    ax.legend(**{'fontsize':18})
    fig.tight_layout()
    fig.savefig(sanatized_fpath, format=format)



# ----- 
# DRIVERS


def visualize_all_results(vsdb_list, count2rr_list, symx_list, results_root, nogt):
    print('\n\nOutputing results in: '+results_root+'\n\n')
    print('--- Vizualizing All Results ---')
    print('Ground Truth: '+str(not nogt))

    within_lbl = ['within-db','within-db-nogt'][nogt]
    cross_lbl  = 'cross-db'
    within_label = within_lbl.replace('db','database')
    cross_label  = cross_lbl.replace('db','database')

    results_configstr = 'results_%s_k%d%s' %\
            (__METHOD__, __K__,  ['','_nogt'][nogt])
    result_dir = join_mkdir(results_root, results_configstr)
    print('\n\nAlso in: '+result_dir+'\n\n')

    hs_configstr = '-%s_k%d' % (__METHOD__, __K__)

    def vizualize_all_threshold_experiments():
        thresh_output_dir = join_mkdir(results_root, 'threshold_matches'+hs_configstr)
        exptx = 0
        cmap = plt.get_cmap('Set1')
        for exptx in range(len(vsdb_list)):
            # Database handles.
            hsA, hsB = vsdb_list[exptx]
            expt_lbl   = [within_lbl, cross_lbl][not hsA is hsB]
            results_name = get_results_name(hsA, hsB) 
            print('    --- ---')
            print('      - database  ='+results_name+' ---')
            print('      - expt_lbl  ='+expt_lbl+' ---')
            count2rr_AB = count2rr_list[exptx]
            if hsA is hsB or not nogt:
                # Visualize chips which have a results with a high score
                thresh_out_dir = join_mkdir(thresh_output_dir, expt_lbl)
                viz_threshold_matchings(hsA, hsB, count2rr_AB, thresh_out_dir)
    def vizualize_all_chipscores():
        chipscore_dir = join_mkdir(results_root, 'chipscore_frequencies'+hs_configstr)
        if nogt:
                Rank1TNFig = myfigure(200, doclf=True,
                                    title='Frequency of true negative chip scores')
                CrossAndTNFig = myfigure(400, doclf=True,
                                    title='True negatives and cross database queries')

        elif not nogt:
            AllTPFig   = myfigure(100, doclf=True,
                                title='Frequency of true positive chip scores')
            CrossFig   = myfigure(300, doclf=True,
                                title='Frequency of all cross-database chip scores')
        cross_db_scores = []
        true_pos_scores = []
        for exptx in range(total_expts):
            # Database handles.
            hsA, hsB = vsdb_list[exptx]
            expt_lbl   = [within_lbl, cross_lbl][not hsA is hsB]
            expt_color = cmap(exptx/float(total_expts))
            
            results_name = get_results_name(hsA, hsB) 
            print('    --- ---')
            print('      - database  ='+results_name+' ---')
            print('      - expt_lbl  ='+expt_lbl+' ---')

            count2rr_AB = count2rr_list[exptx]
            print('    * Visualizing chip score frequencies '+results_name)
            chipscore_data, ischipscore_TP = get_chipscores(hsA, hsB, count2rr_AB)
            if nogt and hsA is hsB:
                # First true negative - within db (nogt)
                viz_chipscores(chipscore_data, chipscore_mask=True - ischipscore_TP,
                                fig=Rank1TNFig, holdon=True,
                                color=expt_color, labelaug=results_name,
                                conditions='Rank=1, not TP')

                viz_chipscores(chipscore_data, chipscore_mask=True - ischipscore_TP,
                                fig=CrossAndTNFig, holdon=True,
                                color=expt_color, labelaug=results_name,
                                conditions='Rank=1, not TP')

            elif nogt and not hsA is hsB:
                top_scores = chipscore_data[:,0]
                cross_db_scores.append(top_scores)
                viz_chipscores(chipscore_data, fig=CrossAndTNFig,
                                color=expt_color, holdon=True,
                                labelaug=results_name, conditions='Rank<=%d' % __RESTRICT_TP__)

            if not nogt and hsA is hsB:
                # ALL TRUE POSITIVES - within a database
                top_scores = chipscore_data[:,0:__RESTRICT_TP__]
                top_mask   = ischipscore_TP[:,0:__RESTRICT_TP__]
                true_pos_scores.append(top_scores[top_mask])
                viz_chipscores(chipscore_data, chipscore_mask=ischipscore_TP,
                                fig=AllTPFig, holdon=True,
                                color=expt_color, labelaug=results_name,
                                conditions='Rank<=%d, TP' % __RESTRICT_TP__)
            if not nogt and not hsA is hsB:
                # Highest Interdatabase matches for each combination of db
                top_scores = chipscore_data[:,0]
                cross_db_scores.append(top_scores)
                viz_chipscores(chipscore_data, fig=CrossFig,
                                color=expt_color, holdon=True,
                                labelaug=results_name, conditions='Rank<=%d' % __RESTRICT_TP__)

        if nogt:
            figfpath=join(chipscore_dir, within_lbl+'-rank1-chipscore')
            safe_savefig(Rank1TNFig,figfpath, adjust_axes=True)

            figfpath=join(chipscore_dir, within_lbl+'-and-cross-chipscore')
            safe_savefig(CrossAndTNFig,figfpath, adjust_axes=True)
            
        elif not nogt:
            highest_cdscores = sort(np.hstack(cross_db_scores))[::-1][0:20]
            for c in highest_cdscores:
                print('There are %d/%d TPs with scores less than %d' %
                    (np.sum(np.hstack(true_pos_scores) < c ),
                    np.hstack(true_pos_scores).size, c))

            # Finalize Plots and save
            safe_savefig(AllTPFig,
                        join(chipscore_dir, within_lbl+'-top%dtp-chipscore' % __RESTRICT_TP__),
                        adjust_axes=True)
            safe_savefig(CrossFig,
                        join(chipscore_dir, 'crossdb-all-chipscores'),
                        adjust_axes=True)
    #------------
    if __THRESHOLD_MATCHINGS__:
        vizualize_all_threshold_experiments()
    if __CHIPSCORE_PROBAILITIES__:
        vizualize_all_chipscores()
    
    # ALL TRUE POSITIVES - within a database
    # First true negative - within db (nogt)
    # Highest Interdatabase matches for each combination of db
    # Smallest Gaussian window we can do
    # Show top 5-10 scoring cross-database matches give them rest

    #AllTPFig   = myfigure(100)
    #Rank1TNFig = myfigure(200)
    #CrossFig   = myfigure(300)

# --- Visualizations ---

def viz_chipscores(chipscore_data,
                   fignum         =None,
                   fig            =None,
                   chipscore_mask =None,
                   title          =None,
                   holdon         =True,
                   releaseaxis    =None,
                   color          =None,
                   labelaug       ='', 
                   conditions     ='',
                   **kwargs):
    ''' Displays a pdf of how likely matching scores are.
    Input: chipscore_data - QxT np.array containing Q queries and T top scores
    Output: A matplotlib figure
    '''
    #print('   !! Vizualizing Chip Scores ')
    #print('   !! chipscore_data = %r '% chipscore_data)
    #print('   !! fignum         = %r '% fignum)
    #print('   !! fig            = %r' % fig)
    #print('   !! chipscore_mask = %r' % chipscore_mask)
    #print('   !! title          = %r' % title)
    #print('   !! holdon         = %r' % holdon)
    #print('   !! releaseaxis    = %r' % releaseaxis)
    #print('   !! color          = %r' % repr(color))
    #print('   !! labelaug       = %r' % labelaug)
    #print('   !! conditions     = %r' % conditions)
    #print('   !! kwargs         = %r' % kwargs)

    # Check that there is data
    no_data = False
    if chipscore_mask is None:
        max_score = round(chipscore_data.max()+1)
    else:
        if len(chipscore_data[chipscore_mask]) == 0: 
            print('There is no data!')
            no_data = True
        else:
            max_score = round(chipscore_data[chipscore_mask].max()+1) 
    # Get Figure 
    if fig is None:
        fig = myfigure(fignum, doclf=not holdon)
    else: 
        fig
    ax = fig.get_axes()[0]

    if not holdon or releaseaxis is None or releaseaxis:
        ax.set_xlabel('chip score')
        ax.set_ylabel('frequency')
    if not title is None:
        ax.set_title(title, figure=fig)
        fig.canvas.set_window_title(title_str)
    #
    num_queries, num_results = chipscore_data.shape

    if no_data:
        return fig

    # Compute pdf of top scores
    rankless_list = None
    if conditions.find('Rank=1') > -1:
        print(' Rank=1 condition')
        num_results = 1
        rankless_list = []
    elif conditions.find('Rank<=%d' % __RESTRICT_TP__) > -1:
        print(' Rank<=R condition')
        num_results = __RESTRICT_TP__
        rankless_list = []
    elif conditions.find('All Ranks') > -1:
        rankless_list = []
        num_results = chipscore_data.shape[1]
        
    def __plot_scores(scores, fig):
        # Set up plot info (labels and colors)
        _condstr = '' if len(conditions) == 0 else ' | '+conditions
        scores_lbl = labelaug+' P(chip score%s) #examples=%d' % (_condstr, scores.size)
        if color is None:
            line_color = plt.get_cmap('gist_rainbow')(tx/float(num_results))
        else: line_color = color
        # Estimate pdf
        bw_factor = .1
        score_pdf = gaussian_kde(scores, bw_factor)
        # Plot the actual scores on near the bottom perterbed in Y
        pdfrange = score_pdf(scores).max() - score_pdf(scores).min() 
        perb   = (np.random.randn(len(scores))) * pdfrange/30.
        y_data = np.abs([pdfrange/50. for _ in scores]+perb)
        ax.plot(scores, y_data, 'o', color=line_color, figure=fig, alpha=.1)
        # Plot the estimated PDF of the scores
        x_data = linspace(-max_score, max_score, 500)
        ax.plot(x_data, score_pdf(x_data),
                color=line_color,
                label=scores_lbl)

    # --- plot info
    print('Plotting info from the top %d results' % num_results)
    for tx in xrange(num_results):
        rank = tx + 1
        if chipscore_mask is None:
            scores = chipscore_data[:,tx]
        else:
            mask   = chipscore_mask[:,tx]
            scores = chipscore_data[mask,tx]
        if not rankless_list is None:
            rankless_list.append(scores)
        #else: 
            #__plot_scores(scores, fig)

    if not rankless_list is None:
        #import pdb
        #pdb.set_trace()
        scores = np.hstack(rankless_list)
        score_stats =( scores.min(), scores.std(), scores.mean(), scores.max())
        print('    !! Conditions: '+conditions)
        print('    !! Scores (min=%.1f, mean=%.1f, std=%.1f, max=%.1f)' % score_stats)
        __plot_scores(scores, fig)
        fig.user_stat_list.append(score_stats)
    return fig



# Visualization of images which match above a threshold 
def viz_threshold_matchings(hsA, hsB, count2rr_AB, thresh_out_dir):
    'returns database, cx, database cx'
    import numpy as np
    valid_cxsB = hsB.cm.get_valid_cxs()
    num_matching = 0
    
    MATCH_THRESHOLD = __method_2_matchthresh__.get(__METHOD__, 10)
    results_name = get_results_name(hsA, hsB)
    print('  * Visualizing threshold matchings '+results_name+' give it some time to plot...')
    threshdb_out_dir = join_mkdir(thresh_out_dir, results_name)
    # For each query run in hsA vs hsB
    for count in xrange(len(count2rr_AB)):
        rr    = count2rr_AB[count]
        qcx   = rr.qcx
        res   = QueryResult(hsB, rr, hsA)
        qname = res.qhs.cm.cx2_name(qcx)
        # Set matching threshold
        res.top_thresh       = MATCH_THRESHOLD
        res.num_top_min      = 0
        res.num_top_max      = 5
        res.num_extra_return = 0
        # Check to see if any matched over the threshold
        top_cxs    = res.top_cx()
        top_names  = res.hs.cm.cx2_name(top_cxs)
        top_scores = res.scores()[top_cxs]
        if len(top_cxs) > 0:
            # Visualize the result
            # Stupid segfaults #if qcx == 113: #41: #import IPython #IPython.embed() # Get the scores
            num_matching += 1
            res.visualize()
            # Create a filename showing dataset, score, cx, match names
            matchname_set = \
                    set([name.replace('Lionfish','') for name in (top_names+[qname])])
            matchnames = '-'.join(list(matchname_set))
            scorestr = str(int(round(top_scores[0])))

            fig_fname = '-'.join([results_name,
                                 'score'+scorestr,
                                 'cx'+str(qcx),
                                 'MATCHES'+matchnames])+'.jpg'

            fig = myfigure(0)
            fig_fpath = join(threshdb_out_dir, fig_fname)
            sys.stdout.write('.')
            safe_savefig(fig, fig_fpath)
    print('  * Visualized %d above thresh: %f from expt: %s ' % (num_matching,
                                                                 MATCH_THRESHOLD,
                                                                 results_name))


# --- Visualization Data ---
def get_chipscores(hsA, hsB, count2rr_AB):
    '''
    Input: Two database handles and queries from A to B
    Output: Matrix of chip scores. As well as 
    '''
    num_results = -1  # ensure there are N top results 
    num_queries = len(count2rr_AB)
    # Get top scores of result
    chipscore_list = [[] for _ in xrange(num_queries)]
    # Indicator as to if chipscore was a true positive
    isscoreTP_list = [[] for _ in xrange(num_queries)]
    for count in xrange(num_queries):
        rr = count2rr_AB[count]
        res = QueryResult(hsB, rr, hsA)
        res.force_num_top(-1) 
        top_scores = res.top_scores()
        gtpos_full = res.get_groundtruth_ranks()
        gtpos_full = [] if gtpos_full is None else gtpos_full
        gtpos_list = []
        for gt_pos in gtpos_full:
            if num_results == -1 or gt_pos < num_results:
                gtpos_list.append(gt_pos)
        isscoreTP_list[count] = gtpos_list
        chipscore_list[count] = top_scores

    num_results = len(chipscore_list[0])
    assert np.all(num_results == np.array([len(_) for _ in chipscore_list])), \
            'There must be the same number of results'
    chipscore_data = np.array(chipscore_list)
    ischipscore_TP = np.zeros((num_queries, num_results), dtype=np.bool)
    for count in xrange(num_queries):
        gtpos_list = [isscoreTP_list[count]]
        ischipscore_TP[count][gtpos_list] = 1
    return chipscore_data, ischipscore_TP

# MAIN ENTRY POINT
if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    print('Running experiments')
    #hsl.enable_global_logs()

    global count2rr_list

    # Some arguments from command line
    if '--count'           in sys.argv: __METHOD__ = 'COUNT'
    if '--lnrat'           in sys.argv: __METHOD__ = 'LNRAT'
    if '--diff'            in sys.argv: __METHOD__ = 'DIFF'
    if '--k1'              in sys.argv: __K__ = 1
    if '--k5'              in sys.argv: __K__ = 5
    if '--threshold'       in sys.argv: __THRESHOLD_MATCHINGS__ = True
    if '--cmd'             in sys.argv: __cmd_mode__ = True
    if '--cmdrun'          in sys.argv: __cmd_run_mode__ = True
    if '--runcmd'          in sys.argv: __cmd_run_mode__ = True
    # Command Line Argument: Delete precomputed results
    if '--delete' in sys.argv:
        for hsdb in hsdb_list:
            hsdb.delete_precomputed_results()
        sys.exit(0)
    # Command Line Argument: List Results 
    if '--list' in sys.argv:
        for hsdb in hsdb_list:
            print(hsdb.db_dpath)
            rawrr_dir  = hsdb.db_dpath+'/.hs_internals/computed/query_results'
            result_list = os.listdir(rawrr_dir)
            result_list.sort()
            for fname in result_list:
                print('  '+fname)
        sys.exit(0)

    # does count2rr_list need to reload?
    print("does count2rr_list need to reload?")
    if not 'count2rr_list' in vars():
        print('... yes')
        hsdb_list   = []
        if __NOGT__:
            hsdb_nogt_list = []
        dbpath_list = [join(workdir, dbid) for dbid in dbid_list]
        def append_api_list(api_list):
            hsdb = HotSpotterAPI(dbpath)
            api_list.append(hsdb)
        for dbpath in dbpath_list:
            append_api_list(hsdb_list)
            if __NOGT__:
                append_api_list(hsdb_nogt_list)
        def build_db_comparisons(_dblist, nogt):
            # Assemble ALL hotspotter-database-api combinations
            _vslist = []
            _symlist = [] # list of symetric matches
            for hsA in _dblist:
                _symlist.append(len(_symlist))
                _vslist.append((hsA, hsA))
                if nogt is False: # dont run cross-database without gt
                    for hsB in _dblist:
                        # cross db matches
                        if not hsA is hsB and not (hsA, hsB) in _vslist:
                            _vslen = len(_vslist)
                            _symlist.extend([_vslen+1, _vslen])
                            _vslist.append((hsA, hsB))
                            _vslist.append((hsB, hsA))
            return _vslist, _symlist
        vsdb_list, sym_list = build_db_comparisons(hsdb_list, nogt=False)
        if __NOGT__:
            vsdb_nogt_list, sym_nogt_list = build_db_comparisons(hsdb_nogt_list, nogt=True)

        # Build list of all databases to run experiments on
        def api_list_set_prefs(_dblist, nogt):
            print("Setting database configurations")
            for hsdb in _dblist:
                hsdb.am.algo_prefs.query.remove_other_names = nogt 
                hsdb.am.algo_prefs.query.method             = __METHOD__
                hsdb.am.algo_prefs.query.k                  = __K__
                hsdb.dm.draw_prefs.ellipse_bit              = True 
                hsdb.dm.draw_prefs.figsize                  = (19.2,10.8)
                hsdb.dm.draw_prefs.fignum                   = 0
        api_list_set_prefs(hsdb_list, True)
        if __NOGT__:
            api_list_set_prefs(hsdb_nogt_list, False)

        # Set the preferences of the experiments
        def api_list_ensure_model(_dblist):
                for hsdb in _dblist:
                    hsdb.vm.sample_train_set()
                    if __ENSURE_MODEL__:
                        hsdb.ensure_model()
        api_list_ensure_model(hsdb_list)
        if __NOGT__:
            api_list_ensure_model(hsdb_nogt_list)

        # Run all combinations of queries
        def query_db_vs_db(hsA, hsB):
            'Runs cross database queries / reloads cross database queries'
            vs_str = get_results_name(hsA, hsB) 
            print('Running '+vs_str)
            query_cxs = hsA.cm.get_valid_cxs()
            total = len(query_cxs)
            cx2_rr = alloc_lists(total)
            for count, qcx in enumerate(query_cxs):
                #with Timer() as t:
                    #print(('Query %d / %d   ' % (count, total)) + vs_str)
                    rr = hsB.qm.cx2_rr(qcx, hsA)
                    cx2_rr[count] = rr
            return cx2_rr 

        count2rr_list      = [query_db_vs_db(hsA, hsB) for hsA, hsB in vsdb_list]
        if __NOGT__:
            count2rr_list_nogt = [query_db_vs_db(hsA, hsB) for hsA, hsB in vsdb_nogt_list]
    else:
        print('... Great! count2rr_list is already loaded')


    # List the database list we are running on
    print('--- Database versus list ---\n   DBX --- SYMX - hsA vs hsB ')
    for dbx, (dbtup, symx) in enumerate(zip(vsdb_list, sym_list)):
        if dbtup[0] is dbtup[1]: 
            print('     %d --- sx%d - %s vs self' %
                  (dbx, symx, dbtup[0].get_dbid()) )
        else:
            print('     %d --- sx%d - %s vs %s' % (dbx, symx, dbtup[0].get_dbid(), dbtup[1].get_dbid()) )
    print('---')

    # Dependents of parameters 
    results_root = join_mkdir('Results')
    
    if not __cmd_mode__:
        # Compute / Load all query results. Then visualize
        visualize_all_results(vsdb_list,
                              count2rr_list,
                              sym_list,
                              results_root,
                              nogt=False)
        visualize_all_results(vsdb_nogt_list,
                              count2rr_list_nogt,
                              sym_nogt_list, 
                              results_root,
                              nogt=True)
    try: 
        __force_no_embed__
    except: 
        __force_no_embed__ = False

    if (__cmd_mode__ or __cmd_run_mode__) and not __force_no_embed__:
        print('Entering interacitve mode.')
        import IPython 
        IPython.embed()
        __cmd_run_mode__ == False
        __force_no_embed__ = True
        #sys.argv.remove('--runcmd')
    #count2rr_list_nogt = 
# NEED: 
# ALL TRUE POSITIVES - within a database
# First true negative - within db (nogt)
# Highest Interdatabase matches for each combination of db
# Smallest Gaussian window we can do
# Show top 5-10  scoring cross-database matches give them rest

# ---
# Handle a lot of images
# Multiple views of every animals (they come in all views)
# Handle partial views of an animal (know when there is not enough information)
# Too much data to check every result
# There are wide varieties and incomplete views of animals 

# Matching vs ground
# How to score (choice of k w/ dynamic database)
# What do fins feature-matches score? 
# 
# Do finns still match after you take top 1 out of the scores? 

