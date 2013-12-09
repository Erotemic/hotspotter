from __future__ import division, print_function
import __builtin__
import sys
# Standard library imports
import datetime
import os
import subprocess
import textwrap
from itertools import izip
from os.path import realpath, join, normpath
import re
# Hotspotter imports
import draw_func2 as df2
import helpers
import load_data2 as ld2
import oxsty_results
import params
from Printable import DynStruct
# Scientific imports
import numpy as np
import matplotlib.gridspec as gridspec

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write
def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write
def print_off():
    global print, print_
    def print(*args, **kwargs): pass
    def print_(*args, **kwargs): pass

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[viz] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

BROWSE = True
DUMP = False
FIGNUM = 1

def plot_name_of_cx(hs, cx, **kwargs):
    nx = hs.tables.cx2_nx[cx]
    plot_name(hs, nx, hl_cxs=[cx], **kwargs)

def plot_name(hs, nx, nx2_cxs=None, fignum=0, hl_cxs=[], subtitle='',
              annote=True, **kwargs):
    print('[viz] plot_name nx=%r' % nx)
    nx2_name = hs.tables.nx2_name
    cx2_nx   = hs.tables.cx2_nx
    name = nx2_name[nx]
    if not nx2_cxs is None:
        cxs = nx2_cxs[nx]
    else: 
        cxs = np.where(cx2_nx == nx)[0]
    print('[viz] plot_name %r' % hs.cxstr(cxs))
    ncxs  = len(cxs)
    #nCols = int(min(np.ceil(np.sqrt(ncxs)), 5))
    nCols = int(min(ncxs, 5))
    nRows = int(np.ceil(ncxs / nCols))
    print('[viz*] r=%r, c=%r' % (nRows, nCols))
    #gs2 = gridspec.GridSpec(nRows, nCols)
    pnum = lambda px: (nRows, nCols, px+1)
    fig = df2.figure(fignum=fignum, plotnum=pnum(0), **kwargs)
    fig.clf()
    for px, cx in enumerate(cxs):
        df2.show_chip(hs, cx=cx, plotnum=pnum(px), draw_kpts=annote, kpts_alpha=.2)
        if cx in hl_cxs:
            ax = df2.gca()
            df2.draw_border(ax, df2.GREEN, 4)
        #plot_cx3(hs, cx)
    title = 'nx=%r -- name=%r' % (nx, name)
    if not annote:
        title += ' noannote'
    #gs2.tight_layout(fig)
    #gs2.update(top=df2.TOP_SUBPLOT_ADJUST)
    df2.set_figtitle(title, subtitle)

def plot_cx3(hs, cx):
    ax = df2.gca()
    rchip = hs.get_chip(cx)
    ax.imshow(rchip, interpolation='nearest')
    df2.plt.set_cmap('gray')
    df2.set_ticks([],[])
    gname = hs.cx2_gname(cx)
    cid = hs.tables.cx2_cid[cx]
    ax.set_xlabel(gname)
    ax.set_title(hs.cxstr(cx))

def cx_info(allres, cx, SV=True):
    hs = allres.hs
    res = allres.qcx2_res[cx]
    print_top_res_scores(hs, res, view_top=10)
    #gt_cxs = hs.get_other_cxs(cx)
    gt_cxs = hs.get_other_indexed_cxs(cx)
    print('[viz] Ground truth '+hs.cx_liststr(gt_cxs))
    print('[viz] num groundtruth = '+str(len(gt_cxs)))
    print_top_res_scores(hs, res, view_top=10, SV=True)

def print_top_res_scores(hs, res, view_top=10, SV=True):
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    lbl = ['(assigned)', '(assigned+V)'][SV]
    cx2_nx     = hs.tables.cx2_nx
    nx2_name   = hs.tables.nx2_name
    qnx        = cx2_nx[qcx]
    #other_cx   = hs.get_other_cxs(qcx)
    other_cx   = hs.get_other_indexed_cxs(qcx)
    top_cx     = cx2_score.argsort()[::-1]
    top_scores = cx2_score[top_cx] 
    top_nx     = cx2_nx[top_cx]
    view_top   = min(len(top_scores), np.uint32(view_top))
    print('---------------------------------------')
    print('[viz]Inspecting matches of q%s name=%s' % (hs.cxstr(qcx), hs.cx2_name(qcx)))
    print('[viz] * Matched against %d other chips' % len(cx2_score))
    print('[viz] * Ground truth chip indexes:\n   gt_%s' % hs.cx_liststr(other_cx))
    print('[viz]The ground truth scores '+lbl+' are: ')
    for cx in iter(other_cx):
        score = cx2_score[cx]
        print('[viz]--> %s, score=%6.2f' % (hs.cxstr(cx, 4), score))
    print('---------------------------------------')
    print(('The top %d chips and scores '+lbl+' are: ') % view_top)
    for topx in xrange(view_top):
        tscore = top_scores[topx]
        tcx    = top_cx[topx]
        if tcx == qcx: continue
        tnx    = cx2_nx[tcx]
        _mark = '-->' if tnx == qnx else '  -'
        print('[viz]'+_mark+' %s, score=%6.2f' % (hs.cxstr(cx), tscore))
    print('---------------------------------------')
    print('---------------------------------------')

def plot_cx(allres, cx, style='kpts', subdir=None, annotations=True, title_aug=''):
    hs    = allres.hs
    qcx2_res = allres.qcx2_res
    res = qcx2_res[cx]
    plot_cx2(hs, res, style=style, subdir=subdir, annotations=annotations, title_aug=title_aug)

def plot_cx2(hs, res, style='kpts', subdir=None, annotations=True, title_aug=''):
    #cx_info(allres, cx)
    cx = res.qcx
    raise Exception("fix no rr2")
    #title_suffix = rr2.get_title_suffix()
    if 'kpts' == style:
        subdir = 'plot_cx' if subdir is None else subdir
        rchip = hs.get_chip(cx)
        kpts  = hs.feats.cx2_kpts[cx]
        title = '%s\n%s' % (hs.cxstr(cx), title_suffix)
        print('[viz] Plotting'+title)
        df2.imshow(rchip, fignum=FIGNUM, title=title, doclf=True)
        df2.draw_kpts2(kpts)
    if 'gt_matches' == style: 
        subdir = 'gt_matches' if subdir is None else subdir
        df2.show_gt_matches(hs, res, fignum=FIGNUM)
    if 'top5' == style:
        subdir = 'top5' if subdir is None else subdir
        df2.show_topN_matches(hs, res, N=5, fignum=FIGNUM)
    if 'analysis' == style:
        subdir = 'analysis' if subdir is None else subdir
        df2.show_match_analysis(hs, res, N=5, fignum=FIGNUM,
                                annotations=annotations, figtitle=title_aug)
    subdir += title_suffix
    __dump_or_browse(hs, subdir)

def plot_rank_stem(allres, orgres_type='true'):
    print('[viz] plotting rank stem')
    # Visualize rankings with the stem plot
    hs = allres.hs
    title = orgres_type+'rankings stem plot\n'+allres.title_suffix
    orgres = allres.__dict__[orgres_type]
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    x_data = orgres.qcxs
    y_data = orgres.ranks
    df2.draw_stems(x_data, y_data)
    slice_num = int(np.ceil(np.log10(len(orgres.qcxs))))
    df2.set_xticks(hs.test_sample_cx[::slice_num])
    df2.set_xlabel('query chip indeX (qcx)')
    df2.set_ylabel('groundtruth chip ranks')
    #df2.set_yticks(list(seen_ranks))
    __dump_or_browse(allres.hs, 'rankviz')

def plot_rank_histogram(allres, orgres_type): 
    print('[viz] plotting '+orgres_type+' rank histogram')
    ranks = allres.__dict__[orgres_type].ranks
    label = 'P(rank | '+orgres_type+' match)'
    title = orgres_type+' match rankings histogram\n'+allres.title_suffix
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    df2.draw_histpdf(ranks, label=label) # FIXME
    df2.set_xlabel('ground truth ranks')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.hs, 'rankviz')
    
def plot_score_pdf(allres, orgres_type, colorx=0.0, variation_truncate=False): 
    print('[viz] plotting '+orgres_type+' score pdf')
    title  = orgres_type+' match score frequencies\n'+allres.title_suffix
    scores = allres.__dict__[orgres_type].scores
    print('[viz] len(scores) = %r ' % (len(scores),)) 
    label  = 'P(score | '+orgres_type+')'
    df2.figure(fignum=FIGNUM, doclf=True, title=title)
    df2.draw_pdf(scores, label=label, colorx=colorx)
    if variation_truncate:
        df2.variation_trunctate(scores)
    #df2.variation_trunctate(false.scores)
    df2.set_xlabel('score')
    df2.set_ylabel('frequency')
    df2.legend()
    __dump_or_browse(allres.hs, 'scoreviz')

def plot_score_matrix(allres):
    print('[viz] plotting score matrix')
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
    __dump_or_browse(allres.hs, 'scoreviz')

# Dump logic

def __browse():
    print('[viz] Browsing Image')
    df2.show()

def __dump(hs, subdir):
    #print('[viz] Dumping Image')
    fpath = hs.dirs.result_dir
    if not subdir is None: 
        fpath = join(fpath, subdir)
        helpers.ensurepath(fpath)
    df2.save_figure(fpath=fpath, usetitle=True)
    df2.reset()

def save_if_requested(hs, subdir):
    if not hs.args.save_figures:
        return
    #print('[viz] Dumping Image')
    fpath = hs.dirs.result_dir
    if not subdir is None: 
        subdir = helpers.sanatize_fname2(subdir)
        fpath = join(fpath, subdir)
        helpers.ensurepath(fpath)
    df2.save_figure(fpath=fpath, usetitle=True)
    df2.reset()

def __dump_or_browse(hs, subdir=None):
    #fig = df2.plt.gcf()
    #fig.tight_layout()
    if BROWSE:
        __browse()
    if DUMP:
        __dump(hs, subdir)

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
    df2.show_matches_annote_res(res, hs, cxs[0], False, fignum=1, plotnum=131, title_aug=titles[0])
    df2.show_matches_annote_res(res, hs, cxs[1], False, fignum=1, plotnum=132, title_aug=titles[1])
    df2.show_matches_annote_res(res, hs, cxs[2], False, fignum=1, plotnum=133, title_aug=titles[2])
    fig_title = 'fig q'+hs.cxstr(qcx)+' TT BT TF -- ' + allres.title_suffix
    df2.set_figtitle(fig_title)
    #df2.set_figsize(_fn, 1200,675)

def dump_gt_matches(allres):
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    'Displays the matches to ground truth for all queries'
    for qcx in xrange(0, len(qcx2_res)):
        res = qcx2_res[qcx]
        df2.show_gt_matches(hs, res, fignum=FIGNUM)
        __dump_or_browse(allres.hs, 'gt_matches'+allres.title_suffix)

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
        __dump_or_browse(allres.hs, orgres_type+'_matches'+allres.title_suffix)
#------------------------------

def show_image(hs, gx, annote=True):
    fig = df2.figure(doclf=True)
    img = hs.gx2_image(gx)
    df2.imshow(img)
    ax = df2.gca()
    if annote:
        # draw chips in the image
        gx2_cxs = hs.get_gx2_cxs()
        cxs = gx2_cxs[gx]
        for cx in cxs:
            roi = hs.get_roi(cx)
            df2.draw_roi(ax, roi, hs.cxstr(cx))
    df2.draw()
        
def show_splash(hs):
    fig = df2.figure(doclf=True)
    print('[viz] show_splash()')
    img = hs.splash_image()
    df2.imshow(img)
    df2.draw()
    #fig = self.win.plotWidget.figure
    #ax = fig.get_axes()[0]
    #ax.imshow(img)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #fig.canvas.draw()
    #print(fig)
    #print(fig is self.win.plotWidget.figure)

def show_chip(hs, cx, annote=True):
    #print(hs)
    fig = df2.figure(doclf=True)
    df2.show_chip(hs, cx=cx, draw_kpts=annote)
    df2.draw()
    

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
