# Hotspotter imports
import drawing_functions2 as df2
import helpers
import load_data2 as ld2
import match_chips2 as mc2
import oxsty_results
import params
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
from os.path import realpath, join

__DUMP__ = False # or __BROWSE__

def reload_module():
    import imp, sys
    imp.reload(sys.modules[__name__])

# ===========================
# Driver functions
# ===========================
def dump_all(hs, qcx2_res, matrix=True, summary=True, 
             problems=True, ranks=True, oxford=False):
    if matrix:
        dump_matrix_results(hs, qcx2_res, all_results, SV=SV)
        viz_score_matrix(hs, qcx2_res, all_results, SV=SV)
    if ranks:
        dump_rank_results(hs, qcx2_res, all_results, SV=SV)
    if oxford:
        oxsty_results.dump_mAP_results(hs, qcx2_res, SV=SV)
    if summary:
        plot_summary_visualizations(hs, qcx2_res)
    if problems: 
        dump_problems(hs, qcx2_res)

def dump_matrix_results(hs, qcx2_res, all_results, SV=True):
    matrix_str = matrix_results(hs, qcx2_res, all_results, SV)
    __dump_report(hs, matrix_str, 'score_matrix', SV)

def dump_rank_results(hs, qcx2_res, all_results, SV=True):
    rankres_str = get_rankres_str(hs, qcx2_res, all_results, SV)
    __dump_report(hs, rankres_str, 'rank', SV)

def dump_summary_visualizations(hs, qcx2_res, all_results, SV=True):
    '''Plots (and outputs data): 
        rank stem plot
        rank histogram '''
    SV = SV if 'SV' in vars() else True
    SV_aug = ['_SVOFF','_SVon'][SV] #TODO: SV should go into params
    query_uid = params.get_query_uid().strip('_')+SV_aug
    result_dir = hs.dirs.result_dir
    summary_dir = join(result_dir, 'summary_vizualizations')
    helpers.ensurepath(summary_dir)
    # Visualize rankings with the stem plot
    draw_and_save(plot_stem, all_results.true, 'Rankings Stem Plot\n'+query_uid,
                  outdir=summary_dir)
    draw_and_save(plot_histogram, all_results.true.ranks,
                  'True Match Rankings Histogram\n'+query_uid,
                  outdir=summary_dir)
    draw_and_save(plot_pdf, all_results.true.scores,
                  'True Match Score Frequencies\n'+query_uid,
                  outdir=summary_dir,
                  label='P(score | true match)', colorx=0.0)
    draw_and_save(plot_pdf, all_results.false.scores,
                  'False Match Score Frequencies\n'+query_uid,
                  outdir=summary_dir,
                  label='P(score | false match)', colorx=.2)
    draw_and_save(plot_pdf, all_results.top_true.scores,
                  'Top True Match Score Frequencies\n'+query_uid,
                  outdir=summary_dir,
                  label='P(score | top true match)', colorx=.4)
    draw_and_save(plot_pdf, all_results.bot_true.scores,
                  'Top True Match Score Frequencies\n'+query_uid,
                  outdir=summary_dir,
                  label='P(score | bot true match)', colorx=.6)
    draw_and_save(plot_pdf, all_results.top_false.scores,
                  'Top False Match Score Frequencies\n'+query_uid,
                  outdir=summary_dir,
                  label='P(score | top false match)', colorx=.9)
    draw_and_save(plot_score_matrix, all_results.score_matrix,
                  'Score Matrix\n'+query_uid, 
                  outdir=summary_dir)

# ===========================
# Helper Functions
# ===========================
def __dump_report(hs, report_str, report_type, SV):
    result_dir    = hs.dirs.result_dir
    timestamp_dir = join(result_dir, 'timestamped_results')
    helpers.ensurepath(timestamp_dir)
    helpers.ensurepath(result_dir)
    timestamp = helpers.get_timestamp()
    query_uid = params.get_query_uid()
    SV_aug = ['_SVOFF_','_SVon_'][SV] #TODO: SV should go into params
    csv_timestamp_fname = report_type+query_uid+SV_aug+timestamp+'.csv'
    csv_timestamp_fpath = join(timestamp_dir, csv_timestamp_fname)
    csv_fname  = report_type+query_uid+SV_aug+'.csv'
    csv_fpath = join(result_dir, csv_fname)
    #if __DUMP__: 
    helpers.write_to(csv_fpath, report_str)
    helpers.write_to(csv_timestamp_fpath, report_str)
    if '--gvim' in sys.argv:
        helpers.gvim(csv_fpath)

def __dump_figure(outdir, title):
    if __DUMP__:
        fpath = join(outdir, title)
        df2.save_figure(fpath=fpath)
    else: # __BROWSE__
        df2.show()

def draw_and_save(func, data, title, outdir='.', **kwargs):
    df2.reset()
    func(data, title=title, **kwargs)
    __dump_figure(outdir, title)

# ===========================
# Drawing stuff
# ===========================
def dump_gt_matches(hs, qcx2_res):
    'Displays the matches to ground truth for all queries'
    for qcx in xrange(0, len(qcx2_res)):
        df2.reset()
        df2.show_all_matches(hs, res, fignum)
        __dump_figure(outdir, title)

def plot_stem(true, title, fignum=1, **kwargs):
    # Visualize rankings with the stem plot
    #title = 'Rankings Stem Plot\n'+query_uid
    df2.figure(fignum=fignum, doclf=True, title=title)
    df2.draw_stems(true.qcxs, true.ranks)
    slice_num = int(np.ceil(np.log10(len(true.qcxs))))
    df2.set_xticks(hs.test_sample_cx[::slice_num])
    df2.set_xlabel('query chip indeX (qcx)')
    df2.set_ylabel('groundtruth chip ranks')
    #df2.set_yticks(list(seen_ranks))

def plot_histogram(ranks, title):
    df2.figure(fignum=1, doclf=True, title=title)
    df2.draw_histpdf(ranks, label=('P(rank | true match)'))
    df2.set_xlabel('ground truth ranks')
    df2.set_ylabel('frequency')
    df2.legend()

def plot_pdf(scores, title, label, colorx):
    df2.figure(fignum=1, doclf=True, title=title)
    df2.draw_pdf(scores, label=label, colorx=colorx)
    #df2.variation_trunctate(true.scores)
    #df2.variation_trunctate(false.scores)
    df2.set_xlabel('score')
    df2.set_ylabel('frequency')
    df2.legend()

def plot_score_matrix(hs, score_matrix, title, **kwargs):
    df2.figure(fignum=1, doclf=True, title=title)
    inliers = find_std_inliers(score_matrix)
    max_inlier = score_matrix[inliers].max()
    # Truncate outliers
    score_matrix[score_matrix > max_inlier] = max_inlier
    dim = 0 # None
    score_img = helpers.norm_zero_one(score_matrix, dim=dim)
    df2.set_xlabel('database')
    df2.set_ylabel('queries')
    df2.imshow(score_img)

def dump_problems(hs, qcx2_res, all_results):
    top_true = all_results.top_true
    top_false = all_results.top_false
    bot_true = all_results.bot_true
    problem_false = all_results.top_false
    problem_true = all_results.bot_true
    SV = True
    SV_aug = ['_SVOFF','_SVon'][SV] #TODO: SV should go into params
    query_uid = params.get_query_uid().strip('_')+SV_aug
    result_dir = hs.dirs.result_dir
    # Dump problem cases
    problem_true_dump_dir  = join(result_dir, 'problem_true'+query_uid)
    problem_false_dump_dir = join(result_dir, 'problem_false'+query_uid)
    top_true_dump_dir      = join(result_dir, 'top_true'+query_uid)
    bot_true_dump_dir      = join(result_dir, 'bot_true'+query_uid)
    top_false_dump_dir     = join(result_dir, 'top_false'+query_uid)
    dump_matches(hs, problem_true_dump_dir, problem_true, qcx2_res, SV)
    dump_matches(hs, problem_false_dump_dir, problem_false, qcx2_res, SV)
    dump_matches(hs, top_true_dump_dir, top_true, qcx2_res, SV)
    dump_matches(hs, bot_true_dump_dir, bot_true, qcx2_res, SV)
    dump_matches(hs, top_false_dump_dir, top_false, qcx2_res, SV)

def dump_matches(hs, dump_dir, org_res, qcx2_res, SV):
    helpers.ensurepath(dump_dir)
    cx2_gx = hs.tables.cx2_gx
    gx2_gname = hs.tables.gx2_gname
    # loop over each query / result of interest
    for qcx, cx, score, rank in org_res.iter():
        query_gname, _  = os.path.splitext(gx2_gname[cx2_gx[qcx]])
        result_gname, _ = os.path.splitext(gx2_gname[cx2_gx[cx]])
        df2.reset()
        res = qcx2_res[qcx]
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % \
                (score, rank, query_gname, result_gname)
        df2.show_matches3(res, hs, cx, False, fignum=1, plotnum=121)
        df2.show_matches3(res, hs, cx,  True, fignum=1, plotnum=122)
        df2.set_figtitle(big_title)
        fig_fpath = join(dump_dir, big_title)
        df2.save_figure(qcx, fig_fpath+'.png')

def get_rankres_str(hs, qcx2_res, all_results, SV=True):
    'Builds csv files showing the cxs/scores/ranks of the query results'
    #if not 'all_results' in vars() or all_results is None:
    if not 'SV' in vars(): SV = True
    #---
    cx2_cid  = hs.tables.cx2_cid
    test_sample_cx = hs.test_sample_cx
    db_sample_cx = hs.database_sample_cx
    #---
    qcx2_top_true_rank, qcx2_top_true_score, qcx2_top_true_cx  =\
            all_results.top_true.qcx_arrays(hs)
    qcx2_bot_true_rank, qcx2_bot_true_score, qcx2_bot_true_cx  =\
            all_results.bot_true.qcx_arrays(hs)
    qcx2_top_false_rank, qcx2_top_false_score, qcx2_top_false_cx =\
            all_results.top_false.qcx_arrays(hs)
    #---
    qcx2_numgt = np.zeros(len(cx2_cid)) - 2
    for qcx in test_sample_cx:
        qcx2_numgt[qcx] = len(hs.get_other_cxs(qcx))
    #---
    # Easy to digest results
    num_chips = len(test_sample_cx)
    num_nonquery = len(np.setdiff1d(db_sample_cx, test_sample_cx))
    num_with_gtruth = (1 - np.isnan(qcx2_top_true_rank[test_sample_cx])).sum()
    num_rank_less5 = (qcx2_top_true_rank[test_sample_cx] < 5).sum()
    num_rank_less1 = (qcx2_top_true_rank[test_sample_cx] < 1).sum()
    # Output ranking results
    # TODO: mAP score
    # Build the experiment csv metadata
    SV_aug = ['_SVOFF_','_SVon_'][SV] #TODO: SV should go into params
    query_uid = params.get_query_uid()+SV_aug
    header = '# Experiment Settings (params.query_uid):'+query_uid+'\n'
    header +=  helpers.get_timestamp(format='comment')+'\n'
    scalar_summary  = '# Num Query Chips: %d \n' % num_chips
    scalar_summary += '# Num Query Chips with at least one match: %d \n' % num_with_gtruth
    scalar_summary += '# Num NonQuery Chips: %d \n' % num_nonquery
    scalar_summary += '# Ranks <= 5: %d / %d\n' % (num_rank_less5, num_with_gtruth)
    scalar_summary += '# Ranks <= 1: %d / %d\n\n' % (num_rank_less1, num_with_gtruth)
    header += scalar_summary
    print scalar_summary
    #---
    header += '# Full Parameters: \n' + helpers.indent(params.param_string(),'#') + '\n\n'
    #---
    header += textwrap.dedent('''
    # Rank Result Metadata:
    #   QCX  = Query chip-index
    # QGNAME = Query images name
    # NUMGT  = Num ground truth matches
    #    TT  = top true  
    #    BT  = bottom true
    #    TF  = top false''').strip()
    # Build the experiemnt csv header
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
    problem_true_pairs = zip(all_results.problem_true.qcxs, all_results.problem_true.cxs)
    problem_false_pairs = zip(all_results.problem_false.qcxs, all_results.problem_false.cxs)
    problem_str = '\n'.join( [
        '#Problem Cases: ',
        '# problem_true_pairs = '+repr(problem_true_pairs),
        '# problem_false_pairs = '+repr(problem_false_pairs)])
    print(problem_str+'\n')
    rankres_str += '\n'+problem_str
    return rankres_str

# ========================================================
# Organize results functions
# ========================================================
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
            qcx2_cx[qcx] = score
        return qcx2_rank, qcx2_score, qcx2_cx

def compile_results(hs, qcx2_res, SV=True):
    'Organizes results into a visualizable data structure'
    if not 'SV' in vars():
        SV = True
    cx2_cid  = hs.tables.cx2_cid
    # ---
    true          = OrganizedResult()
    false         = OrganizedResult()
    top_true      = OrganizedResult()
    top_false     = OrganizedResult()
    bot_true      = OrganizedResult()
    problem_true  = OrganizedResult()
    problem_false = OrganizedResult()
    # -----------------
    # Query result loop
    for qcx in hs.test_sample_cx:
        res = qcx2_res[qcx]
        # Use ground truth to sort into true/false
        true_tup, false_tup = res2_true_and_false(hs, res, SV)
        last_rank     = -1
        skipped_ranks = set([])
        # True matches loop
        # Record: all_true, missed_true, top_true, bot_true
        topx = 0
        for cx, score, rank in zip(*true_tup):
            true.append(qcx, cx, rank+1, score)
            if rank - last_rank > 1:
                skipped_ranks.add(rank-1)
                problem_true.append(qcx, cx, rank+1, score)
            if topx == 0:
                top_true.append(qcx, cx, rank+1, score)
            last_rank = rank
            topx += 1
        if topx > 1: 
            bot_true.append(qcx, cx, rank+1, score)
        # False matches loop
        # Record the all_false, false_positive, top_false
        topx = 0
        for cx, score, rank in zip(*false_tup):
            false.append(qcx, cx, rank+1, score)
            if rank in skipped_ranks:
                problem_false.append(qcx, cx, rank+1, score)
            if topx == 0:
                top_false.append(qcx, cx, rank+1, score)
            topx += 1
    # End Query Loop
    # -----------------
    score_matrix = build_score_matrix(hs, qcx2_res, SV)
    all_results = DynStruct()
    all_results.true          = true
    all_results.false         = false
    all_results.top_true      = top_true
    all_results.top_false     = top_false
    all_results.bot_true      = bot_true
    all_results.problem_true  = problem_true
    all_results.problem_false = problem_false
    all_results.score_matrix = score_matrix
    return all_results

def build_score_matrix(hs, qcx2_res, SV=True):
    SV = SV  if 'SV' in vars() else True
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
    return score_matrix, col_label_cx, row_label_cx

def res2_true_and_false(hs, res, SV):
    'Organizes results into true positive and false positive sets'
    if not 'SV' in vars(): 
        SV = True
    if not 'res' in vars():
        qcx2res[qcx]
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
#===============================


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

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # Params
    SV = True
    hs = ld2.HotSpotter(ld2.DEFAULT)
    qcx2_res = mc2.run_matching(hs)
    all_results = compile_results(hs, qcx2_res, SV)
    # Labels
    SV_aug = ['_SVOFF_','_SVon_'][SV] #TODO: SV should go into params
    query_uid = params.get_query_uid()+SV_aug
    #dump_rank_results(hs, qcx2_res, all_results, SV)
    if '--browse' in sys.argv:
        __DUMP__ = False
    if '--summary' in sys.argv:
        dump_summary_visualizations(hs, qcx2_res, all_results, SV)
    if '--stem' in sys.argv:
        plot_stem(all_results.true, 'Rankings Stem Plot\n'+query_uid)
    if '--dump' in sys.argv :
        dump_all(hs, qcx2_res)
    #if '--dump-problems' in sys.argv:
        #dump_problems(hs, qcx2_res)
    #dinspect(18)
    problem_true_pairs = zip(all_results.problem_true.qcxs, all_results.problem_true.cxs)
    problem_false_pairs = zip(all_results.problem_false.qcxs, all_results.problem_false.cxs)
    exec(df2.present(wh=(900,600)))
