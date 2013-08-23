import drawing_functions2 as df2
import load_data2
from Printable import DynStruct
import subprocess
import params
import helpers
import numpy as np
import datetime
import textwrap
import os
import sys
from os.path import realpath, join
from itertools import izip
# reloads this module when I mess with it
def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

# ========================================================
# Driver functions
# ========================================================

def write_matrix_results(hs, qcx2_res, SV=True):
    matrix_str = matrix_results(hs, qcx2_res, SV)
    __write_report(hs, matrix_str, 'score_matrix', SV)

def write_rank_results(hs, qcx2_res, SV=True):
    rankres_str = rank_results(hs, qcx2_res, SV)
    __write_report(hs, rankres_str, 'rank', SV)

def write_oxsty_mAP_results(hs, qcx2_res, SV=True):
    oxsty_map_csv = oxsty_mAP_results(hs, qcx2_res, SV)
    __write_report(hs, oxsty_map_csv, 'oxsty-mAP', SV)

# ========================================================
# Result processing functions
# ========================================================

def build_score_matrix(hs, qcx2_res, SV=True):
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
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    db_sample_cx = range(len(cx2_desc)) if hs.database_sample_cx is None \
                               else hs.database_sample_cx
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
    def iter():
        result_iter = izip(self.qcxs, self.cxs, self.scores, self.ranks)
        for qcx, cx, score, rank in result_iter:
            yield qcx, cx, score, rank

def compile_results(hs, qcx2_res, SV=None):
    'Organizes results into a visualizable data structure'
    SV = True if SV is None else SV
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
        # Record the TRUE match
        # Record that there was a FALSE-POSITIVE and missed true match
        # Record the TOP-RANKED true match
        # Record the BOTTOM-RANKED true match
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
        # Record the FALSE match
        # Record previously seen FALSE-POSITIVE
        # Record the TOP-RANKED false match
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
    #---
    qcx2_num_groundtruth = np.zeros(len(cx2_cid)) - 100
    #---
    qcx2_top_true_rank   = np.zeros(len(cx2_cid)) - 100
    qcx2_top_true_score  = np.zeros(len(cx2_cid)) - 100
    qcx2_top_true_cx     = np.zeros(len(cx2_cid)) - 100
    #---
    qcx2_bot_true_rank   = np.zeros(len(cx2_cid)) - 100
    qcx2_bot_true_score  = np.zeros(len(cx2_cid)) - 100
    qcx2_bot_true_cx     = np.zeros(len(cx2_cid)) - 100
    #---
    qcx2_top_false_rank   = np.zeros(len(cx2_cid)) - 100
    qcx2_top_false_score  = np.zeros(len(cx2_cid)) - 100
    qcx2_top_false_cx     = np.zeros(len(cx2_cid)) - 100
    #---
    all_results = DynStruct()
    all_results.true          = true
    all_results.false         = false
    all_results.top_true      = top_true
    all_results.top_false     = top_false
    all_results.bot_true      = bot_true
    all_results.problem_true  = problem_true
    all_results.problem_false = problem_false
    return all_results

