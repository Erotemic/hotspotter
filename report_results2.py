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
from itertools import izip
from os.path import realpath, join, normpath
import re

__DUMP__ = True # or __BROWSE__

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
        self.scalar_summary      = None
        self.problem_false_pairs = None
        self.problem_true_pairs  = None
        self.matrix_str          = None

    def __str__(allres):
        toret=('+======================')
        toret+=('| All Results ')
        toret+=('| title_suffix=%s' % str(allres.title_suffix))
        toret+=('| scalar_summary=\n%s' % helpers.indent(str(allres.scalar_summary), '|   '))
        toret+=('| problem_false_pairs=\n%r' % allres.problem_false_pairs)
        toret+=('| problem_true_pairs=\n%r' % allres.problem_true_pairs)
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
    db_sample_cx = hs.database_sample_cx
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    unfilt_top_cx = np.argsort(cx2_score)[::-1]
    # Get top chip indexes and scores
    top_cx    = np.array(helpers.intersect_ordered(unfilt_top_cx, db_sample_cx))
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
    print(' * init organized results')
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
    score_matrix = np.array(row_scores)
    # Fill diagonal with -1's
    np.fill_diagonal(score_matrix, -np.ones(len(row_label_cx)))
    # Add score matrix to allres
    allres.score_matrix = score_matrix
    allres.col_label_cx = col_label_cx
    allres.row_label_cx = row_label_cx

def init_allres(hs, qcx2_res, SV=True):
    'Organizes results into a visualizable data structure'
    # Make AllResults data containter
    allres = AllResults(hs, qcx2_res, SV)
    SV_aug = ['_SVOFF','_SVon'][allres.SV]
    allres.title_suffix = params.get_query_uid() + SV_aug
    allres.summary_dir = join(hs.dirs.result_dir, 'summary_plots')
    helpers.ensurepath(allres.summary_dir)
    print('+======================')
    print('| Initializing all results')
    print('| Title suffix: '+allres.title_suffix)
    #---
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    cx2_cid  = hs.tables.cx2_cid
    # Initialize
    init_score_matrix(allres)
    init_organized_results(allres)
    # Build
    build_rankres_str(allres)
    build_matrix_str(allres)
    print allres
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
         '# num_database / cols = '+repr(len(col_label_gname)),
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
    db_sample_cx = hs.database_sample_cx
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
        qcx2_numgt[qcx] = len(hs.get_other_cxs(qcx))
    # Easy to digest results
    num_chips = len(test_sample_cx)
    num_nonquery = len(np.setdiff1d(db_sample_cx, test_sample_cx))
    num_with_gtruth = (1 - np.isnan(qcx2_top_true_rank[test_sample_cx])).sum()
    num_rank_less5 = (qcx2_top_true_rank[test_sample_cx] < 5).sum()
    num_rank_less1 = (qcx2_top_true_rank[test_sample_cx] < 1).sum()
    # CSV Metadata 
    header = '# Experiment allres.title_suffix = '+allres.title_suffix+'\n'
    header +=  helpers.get_timestamp(format='comment')+'\n'
    # Scalar summary
    scalar_summary  = '# Num Query Chips: %d \n' % num_chips
    scalar_summary += '# Num Query Chips with at least one match: %d \n' % num_with_gtruth
    scalar_summary += '# Num NonQuery Chips: %d \n' % num_nonquery
    scalar_summary += '# Ranks <= 5: %d / %d\n' % (num_rank_less5, num_with_gtruth)
    scalar_summary += '# Ranks <= 1: %d / %d\n\n' % (num_rank_less1, num_with_gtruth)
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

# ===========================
# Helper Functions
# ===========================
def __dump_text_report(allres, report_type):
    if not 'report_type' in vars():
        report_type = 'rankres_str'
    print('report_results> Dumping textfile: '+report_type)
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
        __dump_or_browse(allres, orgres_type+'_matches'+allres.title_suffix)

def __dump_or_browse(allres, subdir=None):
    if __DUMP__:
        print('report_results> Dumping Image')
        fpath = allres.hs.dirs.result_dir
        if not subdir is None: 
            fpath = join(fpath, subdir)
            helpers.ensurepath(fpath)
        df2.save_figure(fpath=fpath, usetitle=True)
    else: # __BROWSE__
        print('report_results> Browsing Image')
        df2.show()
    df2.reset()

def plot_tt_bt_tf_matches(allres, qcx):
    #print('Visualizing result: ')
    #res.printme()
    res = allres.qcx2_res[qcx]

    ranks = (allres.top_true_qcx_arrays[0][qcx],
             allres.bot_true_qcx_arrays[0][qcx],
             allres.top_false_qcx_arrays[0][qcx])

    scores = (allres.top_true_qcx_arrays[1][qcx],
             allres.bot_true_qcx_arrays[1][qcx],
             allres.top_false_qcx_arrays[1][qcx])

    cxs = (allres.top_true_qcx_arrays[2][qcx],
           allres.bot_true_qcx_arrays[2][qcx],
           allres.top_false_qcx_arrays[2][qcx])

    titles = ('best True rank='+str(ranks[0])+' ',
              'worst True rank='+str(ranks[1])+' ',
              'best False rank='+str(ranks[2])+' ')

    df2.figure(fignum=1, plotnum=231)
    df2.show_matches3(res, hs, cxs[0], False, fignum=1, plotnum=131, title_aug=titles[0])
    df2.show_matches3(res, hs, cxs[1], False, fignum=1, plotnum=132, title_aug=titles[1])
    df2.show_matches3(res, hs, cxs[2], False, fignum=1, plotnum=133, title_aug=titles[2])
    fig_title = 'fig qcx='+str(qcx)+' TT BT TF -- ' + allres.title_suffix
    df2.set_figtitle(fig_title)
    #df2.set_figsize(_fn, 1200,675)

# ===========================
# Driver functions
# ===========================
def dump_all(allres, 
             text=True,
             stem=True, 
             matrix=True,
             pdf=False, 
             hist=False,
             oxford=False, 
             ttbttf=False,
             problems=False,
             gtmatches=False):
    if text:
        dump_text_results(allres)
    if oxford:
        dump_oxsty_mAP_results(allres)
    #
    if stem:
        dump_rank_stems(allres)
    if matrix:
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

def dump_score_matrixes(allres):
    plot_score_matrix(allres)


def dump_oxsty_mAP_results(allres):
    oxsty_map_csv = oxsty_results.oxsty_mAP_results(allres)
    __dump_report(hs, oxsty_map_csv, 'oxsty-mAP')

def dump_text_results(allres):
    __dump_text_report(allres, 'rankres_str')
    __dump_text_report(allres, 'matrix_str')
    
def dump_problem_matches(allres):
    dump_orgres_matches(allres, 'problem_false')
    dump_orgres_matches(allres, 'problem_true')

def dump_rank_stems(allres):
    plot_rank_stem(allres, 'true')

def dump_rank_hists(allres):
    plot_rank_histogram(allres, 'true')

def dump_score_pdfs(allres):
    plot_score_pdf(allres, 'true', colorx=0.0, variation_truncate=True)
    plot_score_pdf(allres, 'false', colorx=.2)
    plot_score_pdf(allres, 'top_true', colorx=.4, variation_truncate=True)
    plot_score_pdf(allres, 'bot_true', colorx=.6)
    plot_score_pdf(allres, 'top_false', colorx=.9)

def dump_gt_matches(allres):
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    'Displays the matches to ground truth for all queries'
    for qcx in xrange(0, len(qcx2_res)):
        res = qcx2_res[qcx]
        df2.show_all_matches(hs, res, fignum=1)
        __dump_or_browse(allres, 'gt_matches'+allres.title_suffix)

# ===========================
# Result Plotting
# ===========================

def plot_rank_stem(allres, orgres_type):
    print(' * plotting rank stem')
    # Visualize rankings with the stem plot
    hs = allres.hs
    df2.reload_module()
    title = orgres_type+'rankings stem plot\n'+allres.title_suffix
    orgres = allres.__dict__[orgres_type]
    fig = df2.figure(fignum=1, doclf=True, title=title)
    x_data = orgres.qcxs
    y_data = orgres.ranks
    df2.draw_stems(x_data, y_data)
    slice_num = int(np.ceil(np.log10(len(orgres.qcxs))))
    df2.set_xticks(hs.test_sample_cx[::slice_num])
    df2.set_xlabel('query chip indeX (qcx)')
    df2.set_ylabel('groundtruth chip ranks')
    #df2.set_yticks(list(seen_ranks))
    __dump_or_browse(allres, 'rankviz')

def plot_rank_histogram(allres, orgres_type): 
    print(' * plotting rank histogram')
    ranks = allres.__dict__[orgres_type].ranks
    label = 'P(rank | '+orgres_type+' match)'
    title = orgres_type+' match rankings histogram\n'+allres.title_suffix
    df2.figure(fignum=1, doclf=True, title=title)
    df2.draw_histpdf(ranks, label=label) # FIXME
    df2.set_xlabel('ground truth ranks')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres, 'rankviz')
    
def plot_score_pdf(allres, orgres_type, colorx=0.0, variation_truncate=False): 
    print(' * plotting score pdf')
    title  = orgres_type+' match score frequencies\n'+allres.title_suffix
    scores = allres.__dict__[orgres_type].scores
    label  = 'P(score | '+orgres_type+')'
    df2.figure(fignum=1, doclf=True, title=title)
    df2.draw_pdf(scores, label=label, colorx=colorx)
    if variation_truncate:
        df2.variation_trunctate(scores)
    #df2.variation_trunctate(false.scores)
    df2.set_xlabel('score')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres, 'scoreviz')

def plot_score_matrix(allres):
    print(' * plotting score matrix')
    score_matrix = allres.score_matrix
    title = 'Score Matrix\n'+allres.title_suffix
    # Find inliers
    #inliers = helpers.find_std_inliers(score_matrix)
    #max_inlier = score_matrix[inliers].max()
    # Trunate above 255
    score_img = np.copy(score_matrix)
    score_img[score_img < 0] = 0
    score_img[score_img > 255] = 255
    dim = 0
    #score_img = helpers.norm_zero_one(score_img, dim=dim)
    df2.figure(fignum=1, doclf=True, title=title)
    df2.imshow(score_img, fignum=1)
    df2.set_xlabel('database')
    df2.set_ylabel('queries')
    __dump_or_browse(allres, 'scoreviz')

def print_top_res_scores(hs, res, view_top=10, SV=False):
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    lbl = ['(assigned)', '(assigned+V)'][SV]
    cx2_nx     = hs.tables.cx2_nx
    nx2_name   = hs.tables.nx2_name
    qnx        = cx2_nx[qcx]
    other_cx   = hs.get_other_cxs(qcx)
    top_cx     = cx2_score.argsort()[::-1]
    top_scores = cx2_score[top_cx] 
    top_nx     = cx2_nx[top_cx]
    view_top   = min(len(top_scores), np.uint32(view_top))
    print('---------------------------------------')
    print('Inspecting matches of qcx=%d name=%s' % (qcx, nx2_name[qnx]))
    print(' * Matched against %d other chips' % len(cx2_score))
    print(' * Ground truth chip indexes:\n   other_cx=%r' % other_cx)
    print('The ground truth scores '+lbl+' are: ')
    for cx in iter(other_cx):
        score = cx2_score[cx]
        print('--> cx=%4d, score=%6.2f' % (cx, score))
    print('---------------------------------------')
    print(('The top %d chips and scores '+lbl+' are: ') % view_top)
    for topx in xrange(view_top):
        tscore = top_scores[topx]
        tcx    = top_cx[topx]
        if tcx == qcx: continue
        tnx    = cx2_nx[tcx]
        _mark = '-->' if tnx == qnx else '  -'
        print(_mark+' cx=%4d, score=%6.2f' % (tcx, tscore))
    print('---------------------------------------')
    print('---------------------------------------')

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
    allres = init_allres(hs, qcx2_res, SV=SV)
    if not 'kwargs' in vars():
        kwargs = dict(text=True,
             stem=False, 
             matrix=False,
             pdf=False, 
             hist=False,
             oxford=False, 
             ttbttf=False,
             problems=False,
             gtmatches=False)
    try: 
        dump_all(allres, **kwargs)
    except Exception as ex:
        print('\n\n-----------------')
        print('Caught Error in rr2.dump_all')
        print(repr(ex))
        print('Caught Error in rr2.dump_all')
        print('-----------------\n')
        return allres, ex
    return allres

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # Params
    print('Main: report_results')
    allres = helpers.search_stack_for_localvar('allres')
    #if allres is None:
        #hs = ld2.HotSpotter(ld2.DEFAULT)
        #qcx2_res = mc2.run_matching(hs)
        #SV = True
        ## Initialize Results
        #allres = init_allres(hs, qcx2_res, SV)
    # Do something
    from fnmatch import fnmatch

    help_ = textwrap.dedent(r'''
    Enter a command.
        q (or space) : quit 
        h            : help
        cx [cx]    : shows a chip
    ''')
    print(help_)
    firstcmd = 'cx 0'
    ans = None
    while True:
        ans = raw_input('>') if not ans is None else firstcmd
        if ans == 'q' or ans == ' ':
            break
        if allres is None:
            hs = ld2.HotSpotter(ld2.DEFAULT)
            qcx2_res = mc2.run_matching(hs)
            SV = True
            # Initialize Results
            allres = init_allres(hs, qcx2_res, SV)
        if ans == 'h':
            print help_
        elif re.match('cx [0-9][0-9]*', ans):
            cx = int(ans.replace('cx ',''))
            viz.plot_cx(allres, cx)
        elif re.match('[0-9][0-9]*', ans):
            cx = int(ans)
            viz.plot_cx(allres, cx)
        else:
            exec(ans)
        df2.update()

    #browse='--browse' in sys.argv
    #stem='--stem' in sys.argv
    #hist='--hist' in sys.argv
    #pdf='--pdf'   in sys.argv
    dump_all(allres)
    if '--vrd' in sys.argv:
        helpers.vd(allres.hs.dirs.result_dir)
    #dinspect(18)
    print allres
    exec(df2.present(wh=(900,600)))
