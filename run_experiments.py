from hotspotter.HotSpotterAPI import HotSpotterAPI 
from hotspotter.algo.spatial_functions import ransac
from hotspotter.helpers import alloc_lists, Timer
from hotspotter.other.AbstractPrintable import AbstractManager, AbstractPrintable
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.other.logger import logdbg, logerr, hsl, logmsg, logwarn
from numpy import spacing as eps
from os.path import join
import numpy as np
import os, sys
import cPickle
import pylab

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
        for fname in os.listdir(hsdb.db_dpath+'/.hs_internals/computed/query_results'):
            print '  '+fname
    sys.exit(0)

for hsdb in hsdb_list:
    hsdb.ensure_model()

def query_db_vs_db(hsA, hsB):
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

dbvslist = []
for hsdbA in hsdb_list:
    for hsdbB in hsdb_list:
        if not hsdbA is hsdbB:
            dbtup1 = (hsdbA, hsdbB)
            dbtup2 = (hsdbA, hsdbB)
            if not dbtup1 in dbvslist:
                assert not dbtup2 in dbvslist
                dbvslist.append(dbtup1)
                dbvslist.append(dbtup2)

cx2rr_list = [query_db_vs_db(hsA, hsB) for hsA, hsB in dbvslist]

def dostuff():
    for i in range(len(dbvslist))[::2]:
        hsA, hsB = dbvslist[i]
        cxA2rrB = cx2rr_list[i]
        cxB2rrA = cx2rr_list[i+1]
        symetric_matchings(hsA, hsB, rrlistA, rrlistB)

from hotspotter.QueryManager import QueryResult
def visualize_top_scores(hsA, hsB, cxA2rrB):
    ''' displays a pdf of how likely matching scores are '''
    cx = 1
    num_top = 5 #ensure this many num top
    num_q = len(cxA2rrB)
    ts_list = -np.ones((num_q, num_top))
    for qx in xrange(num_q):
        rr = cxA2rrB[qx]
        res = QueryResult(hsB, rr, hsA)
        res.force_num_top(num_top)
        top_scores = res.top_scores()
        ts_list[qx, 0:len(top_scores)] = top_scores
    from scipy.stats.kde import gaussian_kde
    from numpy import linspace,hstack
    from pylab import *
    min_score = 0
    max_score = round(ts_list.max()+1)
    score_range = (min_score, max_score)
    fig = plb.figure(0)
    fig.clf()
    xlabel('chip-score')
    ylabel('probability')
    titlestr = 'Probability of chip-scores \n' + \
            'queries from: '+hsA.get_dbid() + '\n' + \
            'results from: '+hsB.get_dbid()  + '\n' + \
            'scored with: '+hsB.am.algo_prefs.query.method
    title(titlestr)
    fig.canvas.set_window_title(titlestr)
    for tx in xrange(num_top):
        # --- plot info
        rank = tx+1
        scores = ts_list[:,tx] 
        my_pdf = gaussian_kde(scores)
        pdf_x = linspace(min_score, max_score, 100)
        titlestr = 'P(chip-score | chip-rank = #'+str(rank)+')'
        line_color = plb.get_cmap('gist_rainbow')(tx/float(num_top))
        # --- plot individual
        if 0:
            fig = plb.figure(rank)
            fig.clf()
            xlabel('score of rank #'+str(rank))
            ylabel('probability')
            title(titlestr)
            fig.canvas.set_window_title(titlestr)
            plot(pdf_x, my_pdf(pdf_x), color=line_color) 
            hist(scores, normed=1, alpha=.3) 
        # --- plot agg
        fig = plb.figure(0)
        #hist(scores, normed=1, range=score_range, bins=max_score/2, alpha=.3, label=titlestr) 
        plot(pdf_x, my_pdf(pdf_x), color=line_color, label=titlestr) 
    legend()
    show()

def symetric_matchings(hsA, hsB, cxA2rrB, cxB2rrA):
    sym_match_thresh = 10
    ''' Visualizes each query against its top matches which symetrically match
    across databases '''
    cx = 1
    for cx in xrange(len(cxA2rrB)):
        rr = cxA2rrB[cx]
        res = QueryResult(hsB, rr, hsA)
        top_scores = res.top_scores()
    pass

def high_score_matchings():
    'Visualizes each query against its top matches'
    pass


def scoring_metric_comparisons():
    'Plots a histogram of match scores of differnt types'
    pass


def spatial_consistent_match_comparisons():
    'Plots a histogram of spatially consistent match scores vs inconsistent'
    pass

#print hsdb1.query(1)
#res = hsdb2.query(1, hsdb1)
#res.visualize()

#pylab.show() # keep things on screen
