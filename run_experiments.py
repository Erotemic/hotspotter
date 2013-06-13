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
from pylab import *
    
        
font = {'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

# --- PARAMETERS ---
dbid_list   = ['Lionfish/LF_Bajo_bonito',
               'Lionfish/LF_OPTIMIZADAS_NI_V_E',
               'Lionfish/LF_WEST_POINT_OPTIMIZADAS']

__readable_dbmap__ = {'LF_Bajo_bonito':'Bajo Bonito', 
                      'LF_OPTIMIZADAS_NI_V_E':'Optimizadas',
                      'LF_WEST_POINT_OPTIMIZADAS':'West Point'}

__abbrev_dbmap__ = {'Bajo Bonito':'BB',
                    'Optimizadas':'OP',
                    'West Point':'WP'}
                    
#dbid_list = ['NAUT_Dan', 'WS_sharks']

workdir = '/media/SSD_Extra/'
if sys.platform == 'win32':
    workdir = 'D:/data/work/'

global __SANS_GT__

__cmd_mode__               = False
__cmd_run_mode__           = False
__method_2_matchthresh__   = {'LNRAT':12, 'COUNT':20}

__FIGSIZE__                = (19.2,10.8)
__K__                      = 5
__NUM_RANKS_CSPDF__        = 1
__METHOD__                 = 'LNRAT'

__SANS_GT__                = False
__THRESHOLD_MATCHINGS__    = False

__CHIPSCORE_PROBAILITIES__ = True
__INDIVIDUAL_CHIPSCORES__  = True
__AGGREGATE_CHIPSCORES__   = True
__ENSURE_MODEL__           = True

__FEATSCORE_STATISTICS__   = False
__SYMETRIC_MATCHINGS__     = False

__METHOD_AND_K_IN_TITLE__ = False

__all_exceptions__ = []
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
    for key, val in __abbrev_dbmap__.iteritems():
        sanatized_fname = sanatized_fname.replace(key, val)
    sanatized_fname = sanatized_fname.replace(' ','')
    sanatized_fname = sanatized_fname.replace('_','-')
    sanatized_fpath = join(full_path, sanatized_fname)
    fig.tight_layout()
    #fig.show()
    try: 
        fig.savefig(sanatized_fpath, format=format)
    except Exception as ex:
        __all_exceptions__ += [ex]
        print(str(ex))

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

def myfigure(fignum, doclf=False):
    fig = figure(fignum, figsize=__FIGSIZE__)
    if doclf:
        fig.clf()
    return fig
# ----- 
# DRIVERS
def visualize_all_results(dbvs_list, count2rr_list, symx_list, result_dir):
    global __SANS_GT__

    within_lbl = ['within-db','within-db-sans-gt'][__SANS_GT__]
    cross_lbl  = 'cross-db'
    within_label = within_lbl.replace('db','database')
    cross_label  = cross_lbl.replace('db','database')

    _CHIPSCORE_DIR_ = join_mkdir(result_dir, 'chipscore_frequencies')
    _COMBO_LEGEND_SIZE_ = 18
    
    # ALL TRUE POSITIVES - within a database
    # First true negative - within db (sansgt)
    # Highest Interdatabase matches for each combination of db
    # Smallest Gaussian window we can do
    # Show top 5-10  scoring cross-database matches give them rest
    i = 0
    within_ALL_TP_fignum   = 100
    within_FIRST_TN_fignum = 200
    cross_fignum           = 300

    myfigure(within_ALL_TP_fignum, doclf=True)
    myfigure(within_FIRST_TN_fignum, doclf=True)
    myfigure(cross_fignum, doclf=True)

    for i in range(len(dbvs_list)):
        # Database handles.
        hsA, hsB = dbvs_list[i]
        is_cross_database = not hsA is hsB
        results_name = get_results_name(hsA, hsB) 
        if results_name != 'Optimizadas vs self': continue
        print('--- DATABASE VIZUALIZATIONS: '+results_name+' ---')
        count2rr_AB = count2rr_list[i]

        if __CHIPSCORE_PROBAILITIES__:
            # Visualize the frequency of a chip score.
            print('  * Visualizing chip score frequencies '+results_name)
            chipscore_fname = join(_CHIPSCORE_DIR_, results_name+'-chipscore')

            chipscore_data, ischipscore_TP = get_chipscores(hsA, hsB, count2rr_AB)

            if __SANS_GT__ and not is_cross_database:
                # First true negative - within db (sansgt)
                fig = viz_chipscores(results_name, chipscore_data, ischipscore_TP,
                                        restype='', fignum=within_ALL_TP_fignum,
                                        holdon=True)
            elif not is_cross_database:
                # ALL TRUE POSITIVES - within a database
                fig = viz_chipscores(results_name, chipscore_data, ischipscore_TP,
                                        restype='TP', fignum=within_FIRST_TN_fignum,
                                        holdon=True)
            elif is_cross_database:
                # Highest Interdatabase matches for each combination of db
                fig = viz_chipscores(results_name, chipscore_data, ischipscore_TP,
                                        restype='', fignum=cross_fignum,
                                        holdon=True)
        if __THRESHOLD_MATCHINGS__:
            # Visualize chips which have a results with a high score
            thresh_dir     = join_mkdir(result_dir, 'threshold_matches')
            expt_lbl       = [within_lbl, cross_lbl][hsA is hsB]
            thresh_out_dir = join_mkdir(thresh_dir, expt_lbl)
            viz_threshold_matchings(hsA, hsB, count2rr_AB, thresh_out_dir)

# --- Visualizations ---

# Visualization of images which match above a threshold 
def viz_threshold_matchings(hsA, hsB, count2rr_AB, thresh_out_dir='threshold_matches'):
    'returns database, cx, database cx'
    import numpy as np
    valid_cxsB = hsB.cm.get_valid_cxs()
    num_matching = 0
    
    MATCH_THRESHOLD = __method_2_matchthresh__.get(__METHOD__, 10)
    results_name = get_results_name(hsA, hsB)
    print('  * Visualizing threshold matchings '+results_name)
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
            safe_savefig(fig, fig_fpath)
    print('  * Visualized %d above thresh: %f from expt: %s ' % (num_matching,
                                                                 match_threshold,
                                                                 results_name))

def viz_chipscores(results_name,
                   chipscore_data,
                   ischipscore_TP,
                   restype='',
                   fignum=0,
                   holdon=False,
                   releaseaxis=None,
                   releasetitle=None,
                   color=None,
                   labelaug='', 
                   **kwargs):
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
    title_str = 'Frequency of '+typestr+'chip scores \n'+results_name
    #print('  !! Viz - '+title_str+' fignum='+str(fignum)+' holdon='+str(holdon))
    if __METHOD_AND_K_IN_TITLE__:
        title_str += '\nscored with: '+__METHOD__+' k='+str(__K__)
    fig = myfigure(fignum, doclf=not holdon)
    if not holdon or releaseaxis is None or releaseaxis:
        xlabel('chip score')
        ylabel('frequency')
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

        print results_name
        
        try:
            print sort(scores)[0:5]
        except:
            pass
        chipscore_pdf = gaussian_kde(scores)
        #print chipscore_pdf.__dict__
        #print chipscore_pdf.covariance_factor
        #chipscore_pdf.covariance_factor = scipy.stats.kde.gaussian_kde.scotts_factor
        chipscore_pdf.covariance_factor = scipy.stats.kde.gaussian_kde.silverman_factor
        #chipscore_pdf.covariance_factor = lambda: .0001


        chipscore_domain = linspace(0, max_score, 200)
        extra_given = '' if restype == '' else ', '+restype
        rank_label = labelaug+'P(chip score | chip-rank = #'+str(rank)+extra_given+') '+\
                '#examples='+str(scores.size)
        if color is None:
            line_color = get_cmap('gist_rainbow')(tx/float(num_results))
        else: 
            line_color = color
        # --- plot agg
        histscale = np.logspace(0,3,100)
        histscale = histscale[histscale - 10 > 5] 
        histscale = [10, 20, 40, 60, 80, 100, 120, 160,
                     200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900,
                     1000]
        #print histscale
        highpdf = chipscore_pdf(scores).max()
        lowpdf = chipscore_pdf(scores).min()
        #print highpdf
        perb = (np.random.randn(len(scores))) * (highpdf - lowpdf)/40
        y_data = chipscore_pdf(scores)+perb
        y_data = [(highpdf - lowpdf)/2 for _ in scores]
        plot(scores, y_data, 'o', color=line_color)
        #hist(scores, normed=1, range=(0,max_score), alpha=.1, log=True,
             #bins=np.logspace(0,3,100),  histtype='stepfilled') # , bins=max_score/2 
        plot(chipscore_domain, chipscore_pdf(chipscore_domain), color=line_color, label=rank_label)
    if not holdon:
        legend()
    return fig

# --- Visualization Data ---
def get_chipscores(hsA, hsB, count2rr_AB):
    '''
    Input: Two database handles and queries from A to B
    Output: Matrix of chip scores. As well as 
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

    try: 
        __force_no_embed__
    except: 
        __force_no_embed__ = False
  
    if (__cmd_mode__ or __cmd_run_mode__) and not __force_no_embed__:
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
        __cmd_run_mode__ == False
        __force_no_embed__ = True
        #sys.argv.remove('--runcmd')


# NEED: 

# ALL TRUE POSITIVES - within a database
# First true negative - within db (sansgt)
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

