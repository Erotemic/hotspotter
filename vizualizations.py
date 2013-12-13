from __future__ import division, print_function
import __builtin__
import matplotlib
matplotlib.use('Qt4Agg')
import draw_func2 as df2
# Python
from os.path import realpath, join
import multiprocessing
import os
import re
import sys
# Scientific
import numpy as np
# Hotspotter
import fileio as io
import helpers

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

def printDBG(msg):
    pass

def present(*args, **kwargs):
    return df2.present()

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
        draw_chip(hs, cx=cx, plotnum=pnum(px), draw_kpts=annote, kpts_alpha=.2)
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
    #dim = 0
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

def plot_tt_bt_tf_matches(hs, allres, qcx):
    #print('Visualizing result: ')
    #res.printme()
    res = allres.qcx2_res[qcx]
    ranks = (allres.top_true_qcx_arrays[0][qcx],
             allres.bot_true_qcx_arrays[0][qcx],
             allres.top_false_qcx_arrays[0][qcx])
    #scores = (allres.top_true_qcx_arrays[1][qcx],
             #allres.bot_true_qcx_arrays[1][qcx],
             #allres.top_false_qcx_arrays[1][qcx])
    cxs = (allres.top_true_qcx_arrays[2][qcx],
           allres.bot_true_qcx_arrays[2][qcx],
           allres.top_false_qcx_arrays[2][qcx])
    titles = ('best True rank='+str(ranks[0])+' ',
              'worst True rank='+str(ranks[1])+' ',
              'best False rank='+str(ranks[2])+' ')
    df2.figure(fignum=1, plotnum=231)
    res.plot_matches(res, hs, cxs[0], False, fignum=1, plotnum=131, title_aug=titles[0])
    res.plot_matches(res, hs, cxs[1], False, fignum=1, plotnum=132, title_aug=titles[1])
    res.plot_matches(res, hs, cxs[2], False, fignum=1, plotnum=133, title_aug=titles[2])
    fig_title = 'fig q'+hs.cxstr(qcx)+' TT BT TF -- ' + allres.title_suffix
    df2.set_figtitle(fig_title)
    #df2.set_figsize(_fn, 1200,675)

def dump_gt_matches(allres):
    hs = allres.hs
    qcx2_res = allres.qcx2_res
    'Displays the matches to ground truth for all queries'
    for qcx in xrange(0, len(qcx2_res)):
        res = qcx2_res[qcx]
        res.show_gt_matches(hs, fignum=FIGNUM)
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
#^^^ OLD 

callback_id = None
def _annotate_image(hs, fig, ax, gx, highlight_cxs, cx_clicked_func,
                    draw_roi=True, draw_roi_lbls=True, **kwargs):
    global callback_id
    # draw chips in the image
    cx_list = hs.gx2_cxs(gx)
    centers = []
    interact = cx_clicked_func is not None
    # Draw all chip indexes in the image
    for cx in cx_list:
        roi = hs.get_roi(cx)
        # Draw the ROI
        roi_lbl = hs.cxstr(cx)
        if cx in highlight_cxs:
            bbox_color = df2.ORANGE      * np.array([1,1,1,.95])
            lbl_color  = df2.BLACK       * np.array([1,1,1,.75])
        else:
            dark_alpha = np.array([1,1,1,.6])
            bbox_color = df2.DARK_ORANGE * dark_alpha
            lbl_color  = df2.BLACK       * dark_alpha
        df2.draw_roi(ax, roi, roi_lbl, bbox_color, lbl_color)
        # Index the roi centers (for interaction)
        (x,y,w,h) = roi
        xy_center = np.array([x+(w/2), y+(h/2)])
        centers.append(xy_center)
    # Put roi centers in numpy array
    centers = np.array(centers)
    # Create callback wrapper
    def _on_click(event):
        'Slot for matplotlib event'
        if event.xdata is None: return
        if len(centers) == 0: return
        #print('\n'.join(['%r=%r' % tup for tup in event.__dict__.iteritems()]))
        x,y = event.xdata, event.ydata
        # Find nearest neighbor
        dist = (centers.T[0] - x)**2 + (centers.T[1] - y)**2
        cx = cx_list[dist.argsort()[0]]
        cx_clicked_func(cx)
    if callback_id is not None: 
        fig.canvas.mpl_disconnect(callback_id)
        callback_id = None
    if interact:
        callback_id = fig.canvas.mpl_connect('button_press_event', _on_click)

#def start_image_interaction(hs, gx, cx_clicked_func):

def show_image(hs, gx,
               highlight_cxs=None,
               cx_clicked_func=None,
               draw_rois=True,
               fnum = 1, 
               **kwargs):
    '''Shows an image. cx_clicked_func(cx) is a callback function'''
    fig = df2.figure(fignum=1, doclf=True)
    gname = hs.tables.gx2_gname[gx]
    img = hs.gx2_image(gx)
    df2.imshow(img, title=gname)
    ax = df2.gca()
    if draw_rois:
        if highlight_cxs is None: highlight_cxs = []
        _annotate_image(hs, fig, ax, gx, highlight_cxs, cx_clicked_func,
                        draw_rois, **kwargs)
        
def show_splash(fnum=1):
    print('[viz] show_splash()')
    df2.figure(fignum=fnum, doclf=True)
    splash_fpath = realpath('_frontend/splash.png')
    img = io.imread(splash_fpath)
    df2.imshow(img, fignum=fnum)
    #fig = self.win.plotWidget.figure
    #ax = fig.get_axes()[0]
    #ax.imshow(img)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #fig.canvas.draw()
    #print(fig)
    #print(fig is self.win.plotWidget.figure)

def show_chip_interaction(hs, cx, notes, fnum=2, **kwargs):
    import dev
    from Printable import DynStruct
    import extract_patch

    rchip = hs.get_chip(cx) # this has to be first in case chips arnt loaded
    kpts = hs.get_kpts(cx)
    desc = hs.get_desc(cx)

    chip_info_locals = dev.chip_info(hs, cx)
    chip_title = chip_info_locals['cxstr']+' '+chip_info_locals['name']
    chip_xlabel = chip_info_locals['gname']
    class State(DynStruct):
        def __init__(state):
            super(State, state).__init__()
            state.reset()
        def reset(state):
            state.res = None
            state.scale_min = None
            state.scale_max = None
            state.fnum = 1
            state.fnum_offset = 1
    state = State()
    state.fnum = fnum
    fx_ptr = [0]
    hprint = helpers.horiz_print
    scale = np.sqrt(kpts.T[2]*kpts.T[4])
    # Start off keypoints with no filters
    is_valid = np.ones(len(kpts), dtype=bool)

    def select_ith_keypoint(fx):
        print('-------------------------------------------')
        print('[interact] viewing ith=%r keypoint' % fx)
        kp = kpts[fx]
        sift = desc[fx]
        np.set_printoptions(precision=5)
        df2.cla()
        fig1 = df2.figure(state.fnum, **kwargs)
        df2.imshow(rchip, plotnum=(2,1,1))
        #df2.imshow(rchip, plotnum=(1,2,1), title='inv(sqrtm(invE*)')
        #df2.imshow(rchip, plotnum=(1,2,2), title='inv(A)')
        ell_args = {'ell_alpha':.4, 'ell_linewidth':1.8, 'rect':False}
        df2.draw_kpts2(kpts[is_valid], ell_color=df2.ORANGE, **ell_args)
        df2.draw_kpts2(kpts[fx:fx+1], ell_color=df2.BLUE, **ell_args)
        ax = df2.gca()
        #ax.set_title(str(fx)+' old=b(inv(sqrtm(invE*)) and new=o(A=invA)')
        scale = np.sqrt(kp[2]*kp[4])
        printops = np.get_printoptions()
        np.set_printoptions(precision=1)
        ax.set_title(chip_title)
        ax.set_xlabel(chip_xlabel)
        
        extract_patch.draw_keypoint_patch(rchip, kp, sift, plotnum=(2,2,3))
        ax = df2.gca()
        ax.set_title('affine feature\nfx=%r scale=%.1f' % (fx, scale))
        extract_patch.draw_keypoint_patch(rchip, kp, sift, warped=True, plotnum=(2,2,4))
        ax = df2.gca()
        ax.set_title('warped feature\ninvA=%r ' % str(kp))
        golden_wh = lambda x:map(int,map(round,(x*.618 , x*.312)))
        Ooo_50_50 = {'num_rc':(1,1), 'wh':golden_wh(1400*2)}
        np.set_printoptions(**printops)
        #df2.present(**Ooo_50_50)
        #df2.update()
        fig1.show()
        fig1.canvas.draw()
        #df2.show()

    fig = df2.figure(state.fnum )
    xy = kpts.T[0:2].T
    # Flann doesn't help here at all
    flann_ptr = [None]
    def on_click(event):
        if event.xdata is None: return
        print('[interact] button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata))
        x,y = event.xdata, event.ydata
        dist = (kpts.T[0] - x)**2 + (kpts.T[1] - y)**2
        fx_ptr[0] = dist.argsort()[0]
        select_ith_keypoint(fx_ptr[0])
        print('>>>')
    select_ith_keypoint(fx_ptr[0])
    callback_id = fig.canvas.mpl_connect('button_press_event', on_click)
    select_ith_keypoint(fx_ptr[0])

def show_chip(hs, cx=None, res=None, fnum=2, **kwargs):
    df2.figure(fignum=fnum, doclf=True)
    draw_chip(hs, cx=cx, res=res, **kwargs)
    df2.draw()

def draw_chip(hs, cx=None, allres=None, res=None, info=True, draw_kpts=True,
              nRandKpts=None, kpts_alpha=None, prefix='', **kwargs):
    if not res is None:
        cx = res.qcx
    if not allres is None:
        res = allres.qcx2_res[cx]
    rchip1    = hs.get_chip(cx)
    title_str = prefix
    # Add info to title
    if info: 
        gname = hs.cx2_gname(cx)
        name = hs.cx2_name(cx)
        ngt_str = hs.num_indexed_gt_str(cx)
        title_str += ', '.join([name, hs.cxstr(cx), ngt_str])
    fig, ax = df2.imshow(rchip1, title=title_str, **kwargs)
    #if not res is None: 
    if info:
        ax.set_xlabel(gname, fontproperties=df2.FONTS.xlabel)
    if not draw_kpts:
        return
    kpts1 = hs.get_kpts(cx)
    kpts_args = dict(offset=(0,0), ell_linewidth=1.5, ell=True, pts=False)
    # Draw keypoints with groundtruth information
    if not res is None:
        gt_cxs = hs.get_other_indexed_cxs(cx)
        # Get keypoint indexes
        def stack_unique(fx_list):
            try:
                if len(fx_list) == 0:
                    return np.array([], dtype=int)
                stack_list = np.hstack(fx_list)
                stack_ints = np.array(stack_list, dtype=int)
                unique_ints = np.unique(stack_ints)
                return unique_ints
            except Exception as ex:
                 # debug in case of exception (seem to be happening)
                 print('==============')
                 print('Ex: %r' %ex)
                 print('----')
                 print('fx_list = %r ' % fx_list)
                 print('----')
                 print('stack_insts = %r' % stack_ints)
                 print('----')
                 print('unique_ints = %r' % unique_ints)
                 print('==============')
                 print(unique_ints)
                 raise
        all_fx = np.arange(len(kpts1))
        cx2_fm = res.get_cx2_fm()
        fx_list1 = [fm[:,0] for fm in cx2_fm]
        fx_list2 = [fm[:,0] for fm in cx2_fm[gt_cxs]] if len(gt_cxs) > 0 else np.array([])
        matched_fx = stack_unique(fx_list1)
        true_matched_fx = stack_unique(fx_list2)
        noise_fx = np.setdiff1d(all_fx, matched_fx)
        # Print info
        print('[df2] %s has %d keypoints. %d true-matching. %d matching. %d noisy.' %
             (hs.cxstr(cx), len(all_fx), len(true_matched_fx), len(matched_fx), len(noise_fx)))
        # Get keypoints
        kpts_true  = kpts1[true_matched_fx]
        kpts_match = kpts1[matched_fx, :]
        kpts_noise = kpts1[noise_fx, :]
        # Draw keypoints
        legend_tups = []
        # helper function taking into acount phantom labels
        def _kpts_helper(kpts_, color, alpha, label):
            df2.draw_kpts2(kpts_, ell_color=color, ell_alpha=alpha, **kpts_args)
            phant_ = df2.Circle((0, 0), 1, fc=color)
            legend_tups.append((phant_, label))
        _kpts_helper(kpts_noise,  df2.RED, .1, 'Unverified')
        _kpts_helper(kpts_match, df2.BLUE, .4, 'Verified')
        _kpts_helper(kpts_true, df2.GREEN, .6, 'True Matches')
        #plt.legend(*zip(*legend_tups), framealpha=.2)
    # Just draw boring keypoints
    else:
        if kpts_alpha is None: 
            kpts_alpha = .4
        if not nRandKpts is None: 
            nkpts1 = len(kpts1)
            fxs1 = np.arange(nkpts1)
            size = nRandKpts
            replace = False
            p = np.ones(nkpts1)
            p = p / p.sum()
            fxs_randsamp = np.random.choice(fxs1, size, replace, p)
            kpts1 = kpts1[fxs_randsamp]
            df2.set_xlabel('displaying %r/%r keypoints' % (nRandKpts, nkpts1))
            # show a random sample of kpts
        df2.draw_kpts2(kpts1, ell_alpha=kpts_alpha, ell_color=df2.RED, **kpts_args)

def show_keypoints(rchip,kpts,fignum=0,title=None, **kwargs):
    df2.imshow(rchip,fignum=fignum,title=title,**kwargs)
    df2.draw_kpts2(kpts)

def show_top(res, hs, N=5, fnum=3, figtitle='', **kwargs):
    figtitle += ('q%s -- TOP %r' % (hs.cxstr(res.qcx), N))
    topN_cxs = res.topN_cxs(N)
    return _show_chip_matches(hs, res, topN_cxs=topN_cxs, fignum=fnum, figtitle=figtitle,
                        all_kpts=False, **kwargs)

def res_show_analysis(res, hs, N=5, fignum=3, figtitle='', show_query=None,
                      annotations=True, compare_cxs=None, q_cfg=None, **kwargs):
        print('[res] show_analysis()')
        if show_query is None: 
            show_query = not hs.args.noshow_query
        if not compare_cxs is None:
            topN_cxs = compare_cxs
            figtitle = 'comparing to '+hs.cxstr(topN_cxs) + figtitle
        else:
            topN_cxs = res.topN_cxs(N, q_cfg)
            if len(topN_cxs) == 0: 
                warnings.warn('len(topN_cxs) == 0')
                figtitle = 'WARNING: no top scores!' + hs.cxstr(res.qcx)
            else:
                topscore = res.get_cx2_score()[topN_cxs][0]
                figtitle = ('topscore=%r -- q%s' % (topscore, hs.cxstr(res.qcx))) + figtitle
        all_gt_cxs = hs.get_other_indexed_cxs(res.qcx)
        missed_gt_cxs = np.setdiff1d(all_gt_cxs, topN_cxs)
        if hs.args.noshow_gt:
            missed_gt_cxs = []
        max_nCols = min(5,N)
        return _show_chip_matches(hs, res, gt_cxs=missed_gt_cxs, topN_cxs=topN_cxs,
                                  figtitle=figtitle, max_nCols=max_nCols,
                                  show_query=show_query, fignum=fignum,
                                  annotations=annotations, q_cfg=q_cfg, **kwargs)


def show_matches_annote_res(res, hs, cx,
                            fignum=None, 
                            plotnum=None,
                            title_aug=None, 
                            **kwargs):
    '''
    Wrapper for show_matches_annote
    '''
    qcx = res.qcx
    cx2_score = res.get_cx2_score()
    cx2_fm    = res.get_cx2_fm()
    cx2_fs    = res.get_cx2_fs()
    title_suff = None
    return show_matches_annote(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx,
                         fignum, plotnum, title_aug, title_suff, **kwargs)

def show_matches_annote(hs, qcx, cx2_score, 
                        cx2_fm, cx2_fs, cx,
                        fignum=None, plotnum=None, 
                        title_pref=None, 
                        title_suff=None,
                        show_cx=False,
                        show_cid=True,
                        show_gname=True,
                        showTF=True,
                        showScore=True,
                        **kwargs):
    ' Shows matches with annotations '
    printDBG('[df2] Showing matches from %s in fignum=%r' % (hs.vs_str(cx, qcx), fignum))
    if np.isnan(cx):
        nan_img = np.zeros((100,100), dtype=np.uint8)
        title='(qx%r v NAN)' % (qcx)
        df2.imshow(nan_img, fignum=fignum, plotnum=plotnum, title=title)
        return 
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    kpts1, kpts2   = hs.get_kpts([qcx, cx])
    score = cx2_score[cx]
    fm = cx2_fm[cx]; fs = cx2_fs[cx]
    # Build the title string
    isgt_str  = hs.is_true_match_str(qcx, cx)
    title = ''
    if showTF:
        title += '*' + isgt_str  + '*'
    if showScore:
        score_str = (' score='+helpers.num_fmt(score)) % (score)
        title += score_str
    if not title_pref is None: title = title_pref + title
    if not title_suff is None: title = title + title_suff
    # Draw the matches
    qcx_str = 'q'+hs.cxstr(qcx)
    cx_str = hs.cxstr(cx)
    fig, ax,  woff, hoff = df2.show_matches2(rchip1, rchip2, kpts1, kpts2, fm, fs, 
                            fignum=fignum, plotnum=plotnum, lbl1=qcx_str,
                                             lbl2=cx_str,
                            title=title, **kwargs)
    offset = (woff, hoff)
    #df2.upperright_text(qcx_str)
    #df2.upperright_text(cx_str, offset=offset)
    #df2.lowerright_text(cx_str)
    # Finish annotations
    if   isgt_str == hs.UNKNOWN_STR: df2.draw_border(ax, df2.DARK_PURP, 4, offset=offset)
    elif isgt_str == hs.TRUE_STR:    df2.draw_border(ax, df2.GREEN, 4, offset=offset)
    elif isgt_str == hs.FALSE_STR:   df2.draw_border(ax, df2.ORANGE, 4, offset=offset)
    if show_gname:
        ax.set_xlabel(hs.cx2_gname(cx), fontproperties=df2.FONTS.xlabel)
    return ax

def _show_chip_matches(hs, res, figtitle='', max_nCols=5,
                       topN_cxs=None, gt_cxs=None, show_query=False,
                       all_kpts=False, fignum=3, annotations=True, q_cfg=None,
                       split_plots=False, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    #print('========================')
    #print('[viz] Show chip matches:')
    if topN_cxs is None: topN_cxs = []
    if gt_cxs is None: gt_cxs = []
    print('[viz]----------------')
    print('[viz] #top=%r #missed_gts=%r' % (len(topN_cxs),len(gt_cxs)))
    print('[viz] * max_nCols=%r' % (max_nCols,))
    print('[viz] * show_query=%r' % (show_query,))
    ranked_cxs = res.topN_cxs('all', q_cfg=q_cfg)
    annote = annotations
    # Build a subplot grid
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts = nQuerySubplts + (0 if gt_cxs is None else len(gt_cxs))
    nTopNSubplts  = 0 if topN_cxs is None else len(topN_cxs)
    nTopNCols = min(max_nCols, nTopNSubplts)
    nGTCols   = min(max_nCols, nGtSubplts)
    if not split_plots:
        nGTCols = max(nGTCols, nTopNCols)
        nTopNCols = nGTCols
    nGtRows   = int(np.ceil(nGtSubplts / nGTCols))
    nTopNRows = int(np.ceil(nTopNSubplts / nTopNCols))
    nGtCells = nGtRows * nGTCols
    if split_plots:
        nRows = nGtRows
    else:
        nRows = nTopNRows+nGtRows
    # Helper function for drawing matches to one cx
    def show_matches_(cx, orank, plotnum):
        aug = 'rank=%r\n' % orank
        printDBG('[viz] plotting: %r'  % (plotnum,))
        kwshow  = dict(draw_ell=annote, draw_pts=annote, draw_lines=annote,
                       ell_alpha=.5, all_kpts=all_kpts, **kwargs)
        show_matches_annote_res(res, hs, cx, title_aug=aug, plotnum=plotnum, **kwshow)
    def plot_query(plotx_shift, rowcols):
        printDBG('Plotting Query:')
        plotx = plotx_shift + 1
        plotnum = (rowcols[0], rowcols[1], plotx)
        printDBG('[viz] plotting: %r' % (plotnum,))
        show_chip(hs, res=res, plotnum=plotnum, draw_kpts=annote, prefix='query ')
    # Helper to draw many cxs
    def plot_matches_cxs(cx_list, plotx_shift, rowcols):
        if cx_list is None: return
        for ox, cx in enumerate(cx_list):
            plotx = ox + plotx_shift + 1
            plotnum = (rowcols[0], rowcols[1], plotx)
            oranks = np.where(ranked_cxs == cx)[0]
            if len(oranks) == 0:
                orank = -1
                continue
            orank = oranks[0] + 1
            show_matches_(cx, orank, plotnum)

    query_uid = res.query_uid
    query_uid = re.sub(r'_trainID\([0-9]*,........\)','', query_uid)
    query_uid = re.sub(r'_indxID\([0-9]*,........\)','', query_uid)
    query_uid = re.sub(r'_dcxs\(........\)','', query_uid)

    fig = df2.figure(fignum=fignum); fig.clf()
    df2.plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query: 
        plot_query(0, (nRows, nGTCols))
    # Plot Ground Truth
    plot_matches_cxs(gt_cxs, nQuerySubplts, (nRows, nGTCols)) 
    # Plot TopN in a new figure
    if split_plots:
        df2.set_figtitle(figtitle+'GT', query_uid)
        nRows = nTopNRows
        fig = df2.figure(fignum=fignum+9000); fig.clf()
        df2.plt.subplot(nRows, nTopNCols, 1)
        shift_topN = 0
    else:
        shift_topN = nGtCells
    plot_matches_cxs(topN_cxs, shift_topN, (nRows, nTopNCols))
    if split_plots:
        df2.set_figtitle(figtitle+'topN', query_uid)
    else:
        df2.set_figtitle(figtitle, query_uid)
    print('-----------------')
    return fig

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print('=================================')
    print('[viz] __main__ = vizualizations.py')
    print('=================================')
    import main
    import HotSpotter
    args = main.parse_arguments(db='NAUTS')
    hs = HotSpotter.HotSpotter(args)
    hs.load(load_all=True)
    cx = helpers.get_arg_after('--cx', type_=int)
    qcx = hs.get_valid_cxs()[0]
    if cx is not None: qcx = cx
    res = hs.query(qcx)
    N=5
    res.show_top(hs, N)
    df2.update()
    exec(df2.present())
