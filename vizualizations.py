import draw_func2 as df2
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
from os.path import realpath, join, normpath
import re

BROWSE = True
DUMP = False
FIGNUM = 1

def reload_module():
    import imp, sys
    imp.reload(sys.modules[__name__])

def rrr():
    reload_module()

def cx_info(allres, cx, SV=True):
    hs = allres.hs
    res = allres.qcx2_res[cx]
    print_top_res_scores(hs, res, view_top=10)
    gt_cxs = hs.get_other_cxs(cx)
    print('Ground truth cxs: '+repr(gt_cxs))
    print('num groundtruth = '+str(len(gt_cxs)))
    print_top_res_scores(hs, res, view_top=10, SV=True)

def print_top_res_scores(hs, res, view_top=10, SV=True):
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
    print('[viz]Inspecting matches of qcx=%d name=%s' % (qcx, nx2_name[qnx]))
    print('[viz] * Matched against %d other chips' % len(cx2_score))
    print('[viz] * Ground truth chip indexes:\n   other_cx=%r' % other_cx)
    print('[viz]The ground truth scores '+lbl+' are: ')
    for cx in iter(other_cx):
        score = cx2_score[cx]
        print('[viz]--> cx=%4d, score=%6.2f' % (cx, score))
    print('---------------------------------------')
    print(('The top %d chips and scores '+lbl+' are: ') % view_top)
    for topx in xrange(view_top):
        tscore = top_scores[topx]
        tcx    = top_cx[topx]
        if tcx == qcx: continue
        tnx    = cx2_nx[tcx]
        _mark = '-->' if tnx == qnx else '  -'
        print('[viz]'+_mark+' cx=%4d, score=%6.2f' % (tcx, tscore))
    print('---------------------------------------')
    print('---------------------------------------')

def plot_cx(allres, cx, style='kpts', subdir=None):
    hs    = allres.hs
    qcx2_res = allres.qcx2_res
    #cx_info(allres, cx)
    if 'kpts' == style:
        subdir = 'plot_cx' if subdir is None else subdir
        rchip = hs.get_chip(cx)
        kpts  = hs.feats.cx2_kpts[cx]
        title = 'cx: %d\n%s' % (cx, allres.title_suffix)
        print('[viz] Plotting'+title)
        fig = df2.imshow(rchip, fignum=FIGNUM, title=title, doclf=True)
        df2.draw_kpts2(kpts)
    if 'gt_matches'  == style: 
        subdir = 'gt_matches' if subdir is None else subdir
        res = qcx2_res[cx]
        df2.show_gt_matches(hs, res, fignum=FIGNUM)
    if 'top5' == style:
        subdir = 'top5' if subdir is None else subdir
        res = qcx2_res[cx]
        df2.show_top5_matches(hs, res, fignum=FIGNUM)
    subdir += allres.title_suffix
    __dump_or_browse(allres, subdir)

def plot_rank_stem(allres, orgres_type='true'):
    print(' * plotting rank stem')
    # Visualize rankings with the stem plot
    hs = allres.hs
    df2.reload_module()
    title = orgres_type+'rankings stem plot\n'+allres.title_suffix
    orgres = allres.__dict__[orgres_type]
    fig = df2.figure(fignum=FIGNUM, doclf=True, title=title)
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
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
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
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
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
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    df2.imshow(score_img, fignum=FIGNUM)
    df2.set_xlabel('database')
    df2.set_ylabel('queries')
    __dump_or_browse(allres, 'scoreviz')

# Dump logic
def __dump_or_browse(allres, subdir=None):
    fig = df2.plt.gcf()
    fig.tight_layout()
    if BROWSE:
        print('[viz] Browsing Image')
        df2.show()
    if DUMP:
        print('[viz] Dumping Image')
        fpath = allres.hs.dirs.result_dir
        if not subdir is None: 
            fpath = join(fpath, subdir)
            helpers.ensurepath(fpath)
        df2.save_figure(fpath=fpath, usetitle=True)
        df2.reset()

def dump_score_matrixes(allres):
    plot_score_matrix(allres)

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
        df2.show_all_matches(hs, res, fignum=FIGNUM)
        __dump_or_browse(allres, 'gt_matches'+allres.title_suffix)

def dump_orgres_matches(allres, orgres_type):
    orgres = allres.__dict__[orgres_type]
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    # loop over each query / result of interest
    for qcx, cx, score, rank in orgres.iter():
        query_gname, _  = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[qcx]])
        result_gname, _ = os.path.splitext(hs.tables.gx2_gname[hs.tables.cx2_gx[cx]])
        res = qcx2_res[qcx]
        df2.figure(fignum=FIGNUM, plotnum=121)
        df2.show_matches3(res, hs, cx, SV=False, fignum=FIGNUM, plotnum=121)
        df2.show_matches3(res, hs, cx, SV=True,  fignum=FIGNUM, plotnum=122)
        big_title = 'score=%.2f_rank=%d_q=%s_r=%s' % \
                (score, rank, query_gname, result_gname)
        df2.set_figtitle(big_title)
        __dump_or_browse(allres, orgres_type+'_matches'+allres.title_suffix)
