from __future__ import division
# Hotspotter imports
import draw_func2 as df2
import helpers
import load_data2 as ld2
import match_chips2 as mc2
import oxsty_results
import params
import vizualizations as viz
from Printable import DynStruct
# Scientific imports
import numpy as np
# Standard library imports
import datetime
import os
import subprocess
import sys
import textwrap
import fnmatch
import warnings
from itertools import izip
from os.path import realpath, join, normpath
import re

__DUMP__ = True # or __BROWSE__

REPORT_TOP5    = True
REPORT_MATRIX  = False
REPORT_STEM    = True
REPORT_RANKRES = True

REPORT_MATRIX_VIZ = True

def reload_module():
    import imp, sys
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

# ========================================================
# Report result initialization 
# ========================================================
class AllResults(DynStruct):
    'Data container for all compiled results'
    def __init__(self, hs, qcx2_res, SV):
        super(DynStruct, self).__init__()
        self.hs       = hs
        self.qcx2_res = qcx2_res
        self.SV = SV
        self.rankres_str         = None
        self.title_suffix        = None
        self.scalar_mAP_str      = '# mAP score = NA\n'
        self.scalar_summary      = None
        self.problem_false_pairs = None
        self.problem_true_pairs  = None
        self.greater1_cxs = None
        self.greater5_cxs = None
        self.matrix_str          = None

    def verbose():
        toret = ''


    def __str__(allres):
        #print = tores.append
        toret=('+======================\n')
        toret+=('| All Results \n')
        toret+=('| title_suffix=%s\n' % str(allres.title_suffix))
        toret+=('| scalar_summary=\n%s\n' % helpers.indent(str(allres.scalar_summary).strip(), '|   '))
        toret+=('| '+str(allres.scalar_mAP_str))
        toret+=('|---\n')
        toret+=('| greater5_cxs = %r \n' % (allres.greater5_cxs,))
        toret+=('|---\n')
        toret+=('| greater1_cxs = %r \n' % (allres.greater1_cxs,))
        toret+=('|---\n')
        toret+=('+======================.\n')
        #toret+=('| problem_false_pairs=\n%r' % allres.problem_false_pairs)
        #toret+=('| problem_true_pairs=\n%r' % allres.problem_true_pairs)
        return toret

class OrganizedResult(DynStruct):
    def __init__(self):
        super(DynStruct, self).__init__()
        self.qcxs   = []
        self.cxs    = []
        self.scores = []
        self.ranks  = []
    def append(self, qcx, cx, rank, score):
        self.qcxs.append(qcx)
        self.cxs.append(cx)
        self.scores.append(score)
        self.ranks.append(rank)
    def __len__(self):
        num_qcxs   = len(self.qcxs)
        num_cxs    = len(self.cxs)
        num_scores = len(self.scores)
        num_ranks  = len(self.ranks)
        assert num_qcxs == num_cxs
        assert num_cxs == num_scores
        assert num_scores == num_ranks
        return num_qcxs
    def iter(self):
        'useful for plotting'
        result_iter = izip(self.qcxs, self.cxs, self.scores, self.ranks)
        for qcx, cx, score, rank in result_iter:
            yield qcx, cx, score, rank
    def qcx_arrays(self, hs):
        'useful for reportres_str'
        cx2_cid     = hs.tables.cx2_cid
        qcx2_rank   = np.zeros(len(cx2_cid)) - 2
        qcx2_score  = np.zeros(len(cx2_cid)) - 2
        qcx2_cx     = np.arange(len(cx2_cid)) * -1
        #---
        for (qcx, cx, score, rank) in self.iter():
            qcx2_rank[qcx] = rank
            qcx2_score[qcx] = score
            qcx2_cx[qcx] = cx
        return qcx2_rank, qcx2_score, qcx2_cx
    def printme3(self):
        for qcx, cx, score, rank in self.iter():
            print('%4d %4d %6.1f %4d' % (qcx, cx, score, rank))


def res2_true_and_false(hs, res, SV):
    'Organizes results into true positive and false positive sets'
    if not 'SV' in vars(): 
        SV = True
    if not 'res' in vars():
        res = qcx2_res[qcx]
    indx_sample_cx = hs.indexed_sample_cx
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    unfilt_top_cx = np.argsort(cx2_score)[::-1]
    # Get top chip indexes and scores
    top_cx    = np.array(helpers.intersect_ordered(unfilt_top_cx, indx_sample_cx))
    top_score = cx2_score[top_cx]
    # Get the true and false ground truth ranks
    qnx         = hs.tables.cx2_nx[qcx]
    top_nx      = hs.tables.cx2_nx[top_cx]
    true_ranks  = np.where(np.logical_and(top_nx == qnx, top_cx != qcx))[0]
    false_ranks = np.where(np.logical_and(top_nx != qnx, top_cx != qcx))[0]
    # Construct the true positive tuple
    true_scores  = top_score[true_ranks]
    true_cxs     = top_cx[true_ranks]
    true_tup     = (true_cxs, true_scores, true_ranks)
    # Construct the false positive tuple
    false_scores = top_score[false_ranks]
    false_cxs    = top_cx[false_ranks]
    false_tup    = (false_cxs, false_scores, false_ranks)
    # Return tuples
    return true_tup, false_tup

def init_organized_results(allres):
    print('[rr2] Initialize organized results')
    hs = allres.hs
    SV = allres.SV
    qcx2_res = allres.qcx2_res
    allres.true          = OrganizedResult()
    allres.false         = OrganizedResult()
    allres.top_true      = OrganizedResult()
    allres.top_false     = OrganizedResult()
    allres.bot_true      = OrganizedResult()
    allres.problem_true  = OrganizedResult()
    allres.problem_false = OrganizedResult()
    # -----------------
    # Query result loop
    for qcx in hs.test_sample_cx:
        res = qcx2_res[qcx]
        # Use ground truth to sort into true/false
        true_tup, false_tup = res2_true_and_false(hs, res, SV)
        last_rank     = -1
        skipped_ranks = set([])
        # Record: all_true, missed_true, top_true, bot_true
        topx = 0
        for cx, score, rank in zip(*true_tup):
            allres.true.append(qcx, cx, rank, score)
            if rank - last_rank > 1:
                skipped_ranks.add(rank-1)
                allres.problem_true.append(qcx, cx, rank, score)
            if topx == 0:
                allres.top_true.append(qcx, cx, rank, score)
            last_rank = rank
            topx += 1
        if topx > 1: 
            allres.bot_true.append(qcx, cx, rank, score)
        # Record the all_false, false_positive, top_false
        topx = 0
        for cx, score, rank in zip(*false_tup):
            allres.false.append(qcx, cx, rank, score)
            if rank in skipped_ranks:
                allres.problem_false.append(qcx, cx, rank, score)
            if topx == 0:
                allres.top_false.append(qcx, cx, rank, score)
            topx += 1
    print('[rr2] len(allres.true)          = %r' % len(allres.true))
    print('[rr2] len(allres.false)         = %r' % len(allres.false))
    print('[rr2] len(allres.top_true)      = %r' % len(allres.top_true))
    print('[rr2] len(allres.top_false)     = %r' % len(allres.top_false))
    print('[rr2] len(allres.bot_true)      = %r' % len(allres.bot_true))
    print('[rr2] len(allres.problem_true)  = %r' % len(allres.problem_true))
    print('[rr2] len(allres.problem_false) = %r' % len(allres.problem_false))
    # qcx arrays for ttbttf
    allres.top_true_qcx_arrays  = allres.top_true.qcx_arrays(hs)
    allres.bot_true_qcx_arrays  = allres.bot_true.qcx_arrays(hs)
    allres.top_false_qcx_arrays = allres.top_false.qcx_arrays(hs)

def init_score_matrix(allres):
    print(' * init score matrix')
    hs = allres.hs
    SV = allres.SV
    qcx2_res = allres.qcx2_res
    cx2_nx = hs.tables.cx2_nx
    # Build name-to-chips dict
    nx2_cxs = {}
    for cx, nx in enumerate(cx2_nx):
        if not nx in nx2_cxs.keys():
            nx2_cxs[nx] = []
        nx2_cxs[nx].append(cx)
    # Sort names by number of chips
    nx_list = nx2_cxs.keys()
    nx_size = [len(nx2_cxs[nx]) for nx in nx_list]
    nx_sorted = [x for (y,x) in sorted(zip(nx_size, nx_list))]
    # Build sorted chip list
    cx_sorted = []
    test_cx_set = set(hs.test_sample_cx)
    for nx in iter(nx_sorted):
        cxs = nx2_cxs[nx]
        cx_sorted.extend(sorted(cxs))
    # get matrix data rows
    row_label_cx = []
    row_scores = []
    for qcx in iter(cx_sorted):
        if not qcx in test_cx_set: continue
        res = qcx2_res[qcx]
        cx2_score = res.cx2_score_V if SV else res.cx2_score
        row_label_cx.append(qcx)
        row_scores.append(cx2_score[cx_sorted])
    col_label_cx = cx_sorted
    # convert to numpy matrix array
    score_matrix = np.array(row_scores, dtype=np.float64)
    # Fill diagonal with -1's
    np.fill_diagonal(score_matrix, -np.ones(len(row_label_cx)))
    # Add score matrix to allres
    allres.score_matrix = score_matrix
    allres.col_label_cx = col_label_cx
    allres.row_label_cx = row_label_cx

def init_allres(hs, qcx2_res, SV=True,
                matrix=(REPORT_MATRIX or REPORT_MATRIX_VIZ),
                oxford=False,
                **kwargs):
    'Organizes results into a visualizable data structure'
    # Make AllResults data containter
    allres = AllResults(hs, qcx2_res, SV)
    SV_aug = ['_SVOFF','_SVon'][allres.SV]
    allres.title_suffix = params.get_query_uid() + SV_aug
    allres.summary_dir = join(hs.dirs.result_dir, 'summary_plots')
    helpers.ensurepath(allres.summary_dir)
    print('\n======================')
    print(' * Initializing all results')
    print(' * Title suffix: '+allres.title_suffix)
    #---
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    cx2_cid  = hs.tables.cx2_cid
    # Initialize
    if matrix: 
        init_score_matrix(allres)
    init_organized_results(allres)
    # Build
    build_rankres_str(allres)
    if matrix: 
        build_matrix_str(allres)
    if oxford is True:
        oxsty_map_csv, scalar_mAP_str = oxsty_results.oxsty_mAP_results(allres)
        allres.scalar_mAP_str = scalar_mAP_str
        allres.oxsty_map_csv = oxsty_map_csv
    print(allres)
    return allres

# ========================================================
# Build textfile result strings
# ========================================================
def build_matrix_str(allres):
    hs = allres.hs
    cx2_gx = hs.tables.cx2_gx
    gx2_gname = hs.tables.gx2_gname
    def cx2_gname(cx):
        return [os.path.splitext(gname)[0] for gname in gx2_gname[cx2_gx]]
    col_label_gname = cx2_gname(allres.col_label_cx)
    row_label_gname = cx2_gname(allres.row_label_cx)
    timestamp =  helpers.get_timestamp(format='comment')+'\n'
    header = '\n'.join(
        ['# Result score matrix',
         '# Generated on: '+timestamp,
         '# Format: rows separated by newlines, cols separated by commas',
         '# num_queries  / rows = '+repr(len(row_label_gname)),
         '# num_indexed  / cols = '+repr(len(col_label_gname)),
         '# row_labels = '+repr(row_label_gname),
         '# col_labels = '+repr(col_label_gname)])
    row_strings = []
    for row in allres.score_matrix:
        row_str = map(lambda x: '%5.2f' % x, row)
        row_strings.append(', '.join(row_str))
    body = '\n'.join(row_strings)
    matrix_str = '\n'.join([header,body])
    allres.matrix_str = matrix_str

def build_rankres_str(allres):
    'Builds csv files showing the cxs/scores/ranks of the query results'
    hs = allres.hs
    SV = allres.SV
    qcx2_res = allres.qcx2_res
    cx2_cid  = hs.tables.cx2_cid
    test_sample_cx = hs.test_sample_cx
    indx_sample_cx = hs.indexed_sample_cx
    # Get organized data for csv file
    (qcx2_top_true_rank,
    qcx2_top_true_score,
    qcx2_top_true_cx)  = allres.top_true_qcx_arrays

    (qcx2_bot_true_rank,
    qcx2_bot_true_score,
    qcx2_bot_true_cx)  = allres.bot_true_qcx_arrays 

    (qcx2_top_false_rank, 
    qcx2_top_false_score,
    qcx2_top_false_cx) = allres.top_false_qcx_arrays
    # Number of groundtruth per query 
    qcx2_numgt = np.zeros(len(cx2_cid)) - 2
    for qcx in test_sample_cx:
        qcx2_numgt[qcx] = len(hs.get_other_indexed_cxs(qcx))
    # Easy to digest results
    num_chips = len(test_sample_cx)
    num_nonquery = len(np.setdiff1d(indx_sample_cx, test_sample_cx))

    test_sample_cx_with_gt = np.array(test_sample_cx)[qcx2_numgt[test_sample_cx] > 0]
    num_with_gtruth = len(test_sample_cx_with_gt)
    if num_with_gtruth == 0:
        warnings.warn('[rr2] there were no queries with ground truth')
    def ranks_less_than_(thresh):
        if num_with_gtruth == 0:
            return [], ('NoGT','NoGT', -1, 'NoGT')
        testcx2_ttr = qcx2_top_true_rank[test_sample_cx_with_gt]
        greater_cxs = test_sample_cx_with_gt[np.where(testcx2_ttr > thresh)[0]]
        num_greater = len(greater_cxs)
        num_less = num_with_gtruth - num_greater
        num_greater = num_with_gtruth - num_less
        frac_less = 100.0 * num_less / num_with_gtruth
        fmt_tup = (num_less, num_with_gtruth, frac_less, num_greater)
        return greater_cxs, fmt_tup
    greater5_cxs, fmt5_tup = ranks_less_than_(5)
    greater1_cxs, fmt1_tup = ranks_less_than_(1)
    allres.greater1_cxs = greater1_cxs
    allres.greater5_cxs = greater5_cxs
    #print('greater5_cxs = %r ' % (allres.greater5_cxs,))
    #print('greater1_cxs = %r ' % (allres.greater1_cxs,))
    # CSV Metadata 
    header = '# Experiment allres.title_suffix = '+allres.title_suffix+'\n'
    header +=  helpers.get_timestamp(format='comment')+'\n'
    # Scalar summary
    scalar_summary  = '# Num Query Chips: %d \n' % num_chips
    scalar_summary += '# Num Query Chips with at least one match: %d \n' % num_with_gtruth
    scalar_summary += '# Num NonQuery Chips: %d \n' % num_nonquery
    scalar_summary += '# Ranks <= 5: %r/%r = %.1f%% (missed %r)\n' % (fmt5_tup)
    scalar_summary += '# Ranks <= 1: %r/%r = %.1f%% (missed %r)\n\n' % (fmt1_tup)
    header += scalar_summary
    # Experiment parameters
    header += '# Full Parameters: \n' + helpers.indent(params.param_string(),'#') + '\n\n'
    # More Metadata
    header += textwrap.dedent('''
    # Rank Result Metadata:
    #   QCX  = Query chip-index
    # QGNAME = Query images name
    # NUMGT  = Num ground truth matches
    #    TT  = top true  
    #    BT  = bottom true
    #    TF  = top false''').strip()
    # Build the CSV table
    test_sample_gx = hs.tables.cx2_gx[test_sample_cx]
    test_sample_gname = hs.tables.gx2_gname[test_sample_gx]
    test_sample_gname = [g.replace('.jpg','') for g in test_sample_gname]
    column_labels = ['QCX', 'NUM GT',
                     'TT CX', 'BT CX', 'TF CX',
                     'TT SCORE', 'BT SCORE', 'TF SCORE', 
                     'TT RANK', 'BT RANK', 'TF RANK',
                     'QGNAME', ]
    column_list = [
        test_sample_cx, qcx2_numgt[test_sample_cx],
        qcx2_top_true_cx[test_sample_cx], qcx2_bot_true_cx[test_sample_cx],
        qcx2_top_false_cx[test_sample_cx], qcx2_top_true_score[test_sample_cx],
        qcx2_bot_true_score[test_sample_cx], qcx2_top_false_score[test_sample_cx],
        qcx2_top_true_rank[test_sample_cx], qcx2_bot_true_rank[test_sample_cx],
        qcx2_top_false_rank[test_sample_cx], test_sample_gname, ]
    column_type = [int, int, int, int, int, 
                   float, float, float, int, int, int, str,]
    rankres_str = ld2.make_csv_table(column_labels, column_list, header, column_type)
    # Put some more data at the end
    problem_true_pairs = zip(allres.problem_true.qcxs, allres.problem_true.cxs)
    problem_false_pairs = zip(allres.problem_false.qcxs, allres.problem_false.cxs)
    problem_str = '\n'.join( [
        '#Problem Cases: ',
        '# problem_true_pairs = '+repr(problem_true_pairs),
        '# problem_false_pairs = '+repr(problem_false_pairs)])
    rankres_str += '\n'+problem_str
    # Attach results to allres structure
    allres.rankres_str = rankres_str
    allres.scalar_summary = scalar_summary
    allres.problem_false_pairs = problem_false_pairs
    allres.problem_true_pairs = problem_true_pairs
    allres.problem_false_pairs = problem_false_pairs
    allres.problem_true_pairs = problem_true_pairs

# ===========================
# Helper Functions
# ===========================
def __dump_text_report(allres, report_type):
    if not 'report_type' in vars():
        report_type = 'rankres_str'
    print('[rr2] Dumping textfile: '+report_type)
    report_str = allres.__dict__[report_type]
    # Get directories
    result_dir    = allres.hs.dirs.result_dir
    timestamp_dir = join(result_dir, 'timestamped_results')
    helpers.ensurepath(timestamp_dir)
    helpers.ensurepath(result_dir)
    # Write to timestamp and result dir
    timestamp = helpers.get_timestamp()
    csv_timestamp_fname = report_type+allres.title_suffix+timestamp+'.csv'
    csv_timestamp_fpath = join(timestamp_dir, csv_timestamp_fname)
    csv_fname  = report_type+allres.title_suffix+'.csv'
    csv_fpath = join(result_dir, csv_fname)
    helpers.write_to(csv_fpath, report_str)
    helpers.write_to(csv_timestamp_fpath, report_str)

# ===========================
# Driver functions
# ===========================
def dump_all(allres,
             missed_top5=REPORT_TOP5, 
             rankres=REPORT_RANKRES,
             stem=REPORT_STEM, 
             matrix=REPORT_MATRIX,#
             matrix_viz=REPORT_MATRIX_VIZ,#
             pdf=False, 
             hist=False,
             ttbttf=False,
             problems=False,
             gtmatches=False,
             oxford=False,
             no_viz=False):

    print('\n======================')
    print('[rr2] DUMP ALL')
    print('======================')
    viz.BROWSE = False
    viz.DUMP = True
        
    if rankres:
        dump_rankres_str_results(allres)
    if matrix:
        dump_matrix_str_results(allres)
    if oxford:
        dump_oxsty_mAP_results(allres)
    if no_viz:
        print('\n --- (NO VIZ) END DUMP ALL ---\n')
        return
    #
    if stem:
        dump_rank_stems(allres)
    if matrix_viz:
        dump_score_matrixes(allres)
    if hist:
        dump_rank_hists(allres)
    if pdf:
        dump_score_pdfs(allres)
    #
    if ttbttf: 
        dump_ttbttf_matches(allres)
    if problems: 
        dump_problem_matches(allres)
    if gtmatches: 
        dump_gt_matches(allres)
    if missed_top5:
        dump_missed_top5(allres)
    print('\n --- END DUMP ALL ---\n')

def dump_oxsty_mAP_results(allres):
    #print('\n---DUMPING OXSTYLE RESULTS---')
    __dump_text_report(allres, 'oxsty_map_csv')

def dump_rankres_str_results(allres):
    #print('\n---DUMPING RANKRES RESULTS---')
    __dump_text_report(allres, 'rankres_str')

def dump_matrix_str_results(allres):
    #print('\n---DUMPING MATRIX STRING RESULTS---')
    __dump_text_report(allres, 'matrix_str')
    
def dump_problem_matches(allres):
    #print('\n---DUMPING PROBLEM MATCHES---')
    dump_orgres_matches(allres, 'problem_false')
    dump_orgres_matches(allres, 'problem_true')

def dump_score_matrixes(allres):
    #print('\n---DUMPING SCORE MATRIX---')
    viz.plot_score_matrix(allres)

def dump_rank_stems(allres):
    #print('\n---DUMPING RANK STEMS---')
    viz.plot_rank_stem(allres, 'true')

def dump_rank_hists(allres):
    #print('\n---DUMPING RANK HISTS---')
    viz.plot_rank_histogram(allres, 'true')

def dump_score_pdfs(allres):
    #print('\n---DUMPING SCORE PDF ---')
    viz.plot_score_pdf(allres, 'true',      colorx=0.0, variation_truncate=True)
    viz.plot_score_pdf(allres, 'false',     colorx=0.2)
    viz.plot_score_pdf(allres, 'top_true',  colorx=0.4, variation_truncate=True)
    viz.plot_score_pdf(allres, 'bot_true',  colorx=0.6)
    viz.plot_score_pdf(allres, 'top_false', colorx=0.9)

def dump_gt_matches(allres):
    #print('\n---DUMPING GT MATCHES ---')
    'Displays the matches to ground truth for all queries'
    qcx2_res = allres.qcx2_res
    for qcx in xrange(0, len(qcx2_res)):
        viz.plot_cx(allres, qcx, 'gt_matches')

def dump_missed_top5(allres):
    #print('\n---DUMPING MISSED TOP 5---')
    'Displays the top5 matches for all queries'
    qcx2_res = allres.qcx2_res
    greater5_cxs = allres.greater5_cxs
    #qcx = greater5_cxs[0]
    for qcx in greater5_cxs:
        viz.plot_cx(allres, qcx, 'top5', 'missed_top5')
        viz.plot_cx(allres, qcx, 'gt_matches', 'missed_top5')

def dump_orgres_matches(allres, orgres_type):
    orgres = allres.__dict__[orgres_type]
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    # loop over each query / result of interest
    for qcx, cx, score, rank in orgres.iter():
        query_gname, _  = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[qcx]])
        result_gname, _ = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[cx]])
        res = qcx2_res[qcx]
        df2.figure(fignum=1, plotnum=121)
        df2.show_matches3(res, hs, cx, SV=False, fignum=1, plotnum=121)
        df2.show_matches3(res, hs, cx, SV=True,  fignum=1, plotnum=122)
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % \
                (score, rank, query_gname, result_gname)
        df2.set_figtitle(big_title)
        viz.__dump_or_browse(allres, orgres_type+'_matches'+allres.title_suffix)

#===============================
# MAIN SCRIPT
#===============================
def dinspect(qcx, cx=None, SV=True, reset=True):
    df2.reload_module()
    fignum=2
    res = qcx2_res[qcx]
    print('dinspect matches from qcx=%r' % qcx)
    if reset:
        print('reseting')
        df2.reset()
    if cx is None:
        df2.show_all_matches(hs, res, fignum)
    else: 
        df2.show_matches3(res, hs, cx, fignum, SV=SV)
    df2.present(wh=(900,600))

def report_all(hs, qcx2_res, SV=True, **kwargs):
    allres = init_allres(hs, qcx2_res, SV=SV, **kwargs)
    if not 'kwargs' in vars():
        kwargs = dict(rankres=True, stem=False, matrix=False, pdf=False,
                      hist=False, oxford=False, ttbttf=False, problems=False,
                      gtmatches=False)
    try: 
        dump_all(allres, **kwargs)
    except Exception as ex:
        import sys
        import traceback
        print('\n\n-----------------')
        print('report_all(hs, qcx2_res, SV=%r, **kwargs=%r' % (SV, kwargs))
        print('Caught Error in rr2.dump_all')
        print(repr(ex))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "*** print_tb:"
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print('Caught Error in rr2.dump_all')
        print('-----------------\n')
        raise
        return allres, ex
    return allres

def read_until(file, target):
    curent_line = file.readline()
    while not curent_line is None:
        if curent_line.find(target) > -1:
            return curent_line
        curent_line = file.readline()

def print_result_summaries_list():
    print('\n<(^_^<)\n')
    # Print out some summary of all results you have
    hs = ld2.HotSpotter()
    hs.load_tables(ld2.DEFAULT)
    result_file_list = os.listdir(hs.dirs.result_dir)

    sorted_rankres = []
    for result_fname in iter(result_file_list):  
        if fnmatch.fnmatch(result_fname, 'rankres_str*.csv'):
            print result_fname
            with open(join(hs.dirs.result_dir, result_fname), 'r') as file:

                metaline = file.readline()
                toprint = metaline
                # skip 4 metalines
                [file.readline() for _ in xrange(4)]
                top5line = file.readline()
                top1line = file.readline()
                toprint += top5line+top1line
                line = read_until(file, '# NumData')
                num_data = int(line.replace('# NumData',''))
                file.readline() # header
                res_data_lines = [file.readline() for _ in xrange(num_data)]
                res_data_str = np.array([line.split(',') for line in res_data_lines])
                tt_scores = np.array(res_data_str[:, 5], dtype=np.float)
                bt_scores = np.array(res_data_str[:, 6], dtype=np.float)
                tf_scores = np.array(res_data_str[:, 7], dtype=np.float)

                tt_score_sum = sum([score for score in tt_scores if score > 0])
                bt_score_sum = sum([score for score in bt_scores if score > 0])
                tf_score_sum = sum([score for score in tf_scores if score > 0])

                toprint += ('tt_scores = %r; ' % tt_score_sum)
                toprint += ('bt_scores = %r; ' % bt_score_sum)
                toprint += ('tf_scores = %r; ' % tf_score_sum)
                sorted_rankres.append(top5line+metaline)
                print toprint+'\n'

    print('\n(>^_^)>\n')

    sorted_mapscore = []
    for result_fname in iter(result_file_list):  
        if fnmatch.fnmatch(result_fname, 'oxsty_map_csv*.csv'):
            print result_fname
            with open(join(hs.dirs.result_dir, result_fname), 'r') as file:
                metaline = file.readline()
                scoreline = file.readline()
                toprint = metaline+scoreline

                sorted_mapscore.append(scoreline+metaline)
                print(toprint)

    print('\n'.join(sorted(sorted_rankres)))
    print('\n'.join(sorted(sorted_mapscore)))

    print('\n^(^_^)^\n')

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # Params
    print('[rr2] __main__ = report_results2.py')

    if '--list' in sys.argv:
        print_result_summaries_list()
        sys.exit(1)

    allres = helpers.search_stack_for_localvar('allres')
    #if allres is None:
        #hs = ld2.HotSpotter(ld2.DEFAULT)
        #qcx2_res = mc2.run_matching(hs)
        #SV = True
        ## Initialize Results
        #allres = init_allres(hs, qcx2_res, SV)
    # Do something
    import vizualizations as viz
    hs = ld2.HotSpotter(ld2.DEFAULT)
    qcx2_res = mc2.run_matching(hs)
    SV = True
    # Initialize Results
    oxford = ld2.DEFAULT == ld2.OXFORD
    allres = init_allres(hs, qcx2_res, SV, oxford=oxford)
    greater5_cxs = allres.greater5_cxs

    #Helper drawing functions
    gt_matches = lambda cx: viz.plot_cx(allres, cx, 'gt_matches')
    top5 = lambda cx: viz.plot_cx(allres, cx, 'top5')
    selc = lambda cx: viz.plot_cx(allres, cx, 'kpts')

    try:
        __IPYTHON__
    except Exception:
        dump_all(allres)
        pass
        #exec(nonipython_exec)
    print(allres)
