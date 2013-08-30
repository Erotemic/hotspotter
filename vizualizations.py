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

DUMP = True # or __BROWSE__
FIGNUM = 1

def cx_info(allres, cx):
    pass

def plot_cx(allres, cx):
    title = 'cx: %d\n%s' % (cx, allres.title_suffix)
    print('Plotting'+title)
    fig = df2.figure(figure=FIGNUM, doclf=True, title=title)
    hs = allres.hs
    rchip = hs.get_chip(cx)
    kpts = hs.feats.cx2_kpts[cx]
    df2.imshow(rchip, fignum=FIGNUM,  doclf=True, title='cx=%d' % cx)
    df2.draw_kpts2(kpts)

def plot_rank_stem(allres, orgres_type):
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
    if DUMP:
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
