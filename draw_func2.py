''' Lots of functions for drawing and plotting visiony things '''
from __future__ import division, print_function
import matplotlib
from _localhelpers.draw_func2_helpers import *
from matplotlib import gridspec
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.transforms import Affine2D
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
import scipy.stats
import sys
import types
import warnings
import itertools
import helpers
import params
import os
#print('LOAD_MODULE: draw_func2.py')

def reload_module():
    import imp
    import sys
    print('[df2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def execstr_global():
    execstr = ['global' +key for key in globals().keys()]
    return execstr

# ---- IMAGE CREATION FUNCTIONS ---- 
def draw_sift(desc, kp=None):
    '''
    desc = np.random.rand(128)
    desc = desc / np.sqrt((desc**2).sum())
    desc = np.round(desc * 255)
    '''
    ax = plt.gca()
    tau = np.float64(np.pi * 2)
    DSCALE = .25
    XYSCALE = .5
    XYSHIFT = -.75
    THETA_SHIFT = 1/8 * tau
    # SIFT CONSTANTS
    NORIENTS = 8; NX = 4; NY = 4; NBINS = NX * NY
    def cirlce_rad2xy(radians, mag):
        return np.cos(radians)*mag, np.sin(radians)*mag
    discrete_theta = (np.arange(0,NORIENTS)*(tau/NORIENTS) + THETA_SHIFT)[::-1]
    # Build list of plot positions
    dim_mag   = desc / 255.0
    dim_theta = np.tile(discrete_theta, (NBINS, 1)).flatten()
    dim_xy = np.array(zip(*cirlce_rad2xy(dim_theta, dim_mag))) 
    yxt_gen = itertools.product(xrange(NY),xrange(NX),xrange(NORIENTS))
    yx_gen  = itertools.product(xrange(NY),xrange(NX))

    # Transforms
    axTrans = ax.transData
    kpTrans = None
    if kp is None:
        kp = [0, 0, 1, 0, 1]
    kp = np.array(kp)   
    kpT = kp.T
    x, y, a, c, d = kpT[:,0]
    a_ = 1/np.sqrt(a) 
    b_ = c/(-np.sqrt(a)*d - a*np.sqrt(d))
    d_ = 1/np.sqrt(d)
    transMat = [( a_, b_, x),
                ( 0,  d_, y),
                ( 0,  0, 1)]
    kpTrans = Affine2D(transMat)
    axTrans = ax.transData
    #print('\ntranform=%r ' % transform)
    # Draw Arms
    arrow_patches = []
    arrow_patches2 = []
    for y,x,t in yxt_gen:
        #print((x, y, t))
        #index = 127 - ((NY - 1 - y)*(NX*NORIENTS) + (NX - 1 - x)*(NORIENTS) + (NORIENTS - 1 - t))
        #index = ((y)*(NX*NORIENTS) + (x)*(NORIENTS) + (t))
        index = ((NY - 1 - y)*(NX*NORIENTS) + (NX - 1 - x)*(NORIENTS) + (t))
        #print(index)
        (dx, dy) = dim_xy[index]
        arw_x  = ( x*XYSCALE) + XYSHIFT
        arw_y  = ( y*XYSCALE) + XYSHIFT
        arw_dy = (dy*DSCALE) * 1.5 # scale for viz Hack
        arw_dx = (dx*DSCALE) * 1.5
        posA = (arw_x, arw_y)
        posB = (arw_x+arw_dx, arw_y+arw_dy)
        arw_patch = FancyArrow(arw_x, arw_y, arw_dx, arw_dy, head_width=.0001,
                               transform=kpTrans, length_includes_head=False)
        arw_patch2 = FancyArrow(arw_x, arw_y, arw_dx, arw_dy, head_width=.0001,
                                transform=kpTrans, length_includes_head=False)
        arrow_patches.append(arw_patch)
        arrow_patches2.append(arw_patch2)
    # Draw Circles
    circle_patches = []
    for y,x in yx_gen:
        circ_xy = ((x*XYSCALE)+XYSHIFT, (y*XYSCALE)+XYSHIFT)
        circ_radius = DSCALE
        circ_patch = Circle(circ_xy, circ_radius,
                               transform=kpTrans)
        circle_patches.append(circ_patch)
        
    circ_collection = matplotlib.collections.PatchCollection(circle_patches)
    circ_collection.set_facecolor('none')
    circ_collection.set_transform(axTrans)
    circ_collection.set_edgecolor(BLACK)
    circ_collection.set_alpha(.5)

    # Body of arrows
    arw_collection = matplotlib.collections.PatchCollection(arrow_patches)
    arw_collection.set_transform(axTrans)
    arw_collection.set_linewidth(.5)
    arw_collection.set_color(RED)
    arw_collection.set_alpha(1)

    #Border of arrows
    arw_collection2 = matplotlib.collections.PatchCollection(arrow_patches2)
    arw_collection2.set_transform(axTrans)
    arw_collection2.set_linewidth(1)
    arw_collection2.set_color(BLACK)
    arw_collection2.set_alpha(1)

    ax.add_collection(circ_collection)
    ax.add_collection(arw_collection2)
    ax.add_collection(arw_collection)
def draw_matches2(kpts1, kpts2, fm=None, fs=None, kpts2_offset=(0,0)):
    global LINE_ALPHA
    global SHOW_LINE
    global LINE_CMAP
    global LINE_COLOR
    global LINE_WIDTH
    # input data
    if not SHOW_LINES:
        return 
    if fm is None: # assume kpts are in director correspondence
        assert kpts1.shape == kpts2.shape
    if len(fm) == 0: 
        return
    ax = plt.gca()
    woff, hoff = kpts2_offset
    # Draw line collection
    kpts1_m = kpts1[fm[:,0]].T
    kpts2_m = kpts2[fm[:,1]].T
    xxyy_iter = iter(zip(kpts1_m[0],
                         kpts2_m[0]+woff,
                         kpts1_m[1],
                         kpts2_m[1]+hoff))
    if fs is None: # Draw with solid color
        segments  = [ ((x1, y1), (x2,y2)) for (x1,x2,y1,y2) in xxyy_iter ] 
        colors    = [ LINE_COLOR for fx in xrange(len(fm)) ] 
        linewidth = [ LINE_WIDTH for fx in xrange(len(fm)) ] 
    else: # Draw with colors proportional to score difference
        assert len(fs.shape) == 1, 'score must be 1d'
        cmap = plt.get_cmap(LINE_CMAP)
        mins = fs.min()
        rnge = fs.max() - mins
        segments  = [((x1, y1), (x2,y2)) for (x1,x2,y1,y2) in xxyy_iter] 
        score2_01 = lambda score: .1+.9*(score-mins)/(rnge)
        colors    = [cmap(score2_01(fs[fx])) for fx in xrange(len(fm))]
        linewidth = [LINE_WIDTH for fx in xrange(len(fm)) ] 
    line_group = LineCollection(segments, linewidth, colors, alpha=LINE_ALPHA)
    ax.add_collection(line_group)

def draw_kpts2(kpts, offset=(0,0),
               ell=SHOW_ELLS, 
               pts=False, 
               pts_color=ORANGE, 
               pts_size=POINT_SIZE, 
               ell_alpha=ELL_ALPHA,
               ell_linewidth=ELL_LINEWIDTH,
               ell_color=ELL_COLOR):
    printDBG('drawkpts2: Drawing Keypoints! ell=%r pts=%r' % (ell, pts))
    # get matplotlib info
    ax = plt.gca()
    pltTrans = ax.transData
    ell_actors = []
    eps = 1E-9
    # data
    kpts = np.array(kpts)   
    kptsT = kpts.T
    x = kptsT[0,:] + offset[0]
    y = kptsT[1,:] + offset[1]
    printDBG('[df2] draw_kpts()----------')
    printDBG('[df2] draw_kpts() ell=%r pts=%r' % (ell, pts))
    printDBG('[df2] draw_kpts() drawing kpts.shape=%r' % (kpts.shape,))
    if pts:
        printDBG('[df2] draw_kpts() drawing pts x.shape=%r y.shape=%r' % (x.shape, y.shape))
        ax.plot(x, y, linestyle='None', 
                marker='o',
                markerfacecolor=pts_color,
                markersize=pts_size, 
                markeredgewidth=0)
    if ell:
        printDBG('[df2] draw_kpts() drawing ell kptsT.shape=%r' % (kptsT.shape,))
        a = kptsT[2]
        c = kptsT[3]
        d = kptsT[4]
        # Sympy Calculated sqrtm(inv(A) for A in kpts)
        # inv(sqrtm([(a, 0), (c, d)]) = 
        #  [1/sqrt(a), c/(-sqrt(a)*d - a*sqrt(d))]
        #  [        0,                  1/sqrt(d)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aIS = 1/np.sqrt(a) 
            bIS = c/(-np.sqrt(a)*d - a*np.sqrt(d))
            dIS = 1/np.sqrt(d)
            #cIS = (c/np.sqrt(d) - c/np.sqrt(d)) / (a-d+eps)
        kpts_iter = iter(zip(x,y,aIS,bIS,dIS))
        # This has to be the sexiest piece of code I've ever written
        ell_actors = [ Circle( (0,0), 1, 
                            transform=Affine2D([( a_, b_, x_),
                                                ( 0 , d_, y_),
                                                ( 0 , 0 , 1)]) )
                    for (x_,y_,a_,b_,d_) in kpts_iter ]
        ellipse_collection = matplotlib.collections.PatchCollection(ell_actors)
        ellipse_collection.set_facecolor('none')
        ellipse_collection.set_transform(pltTrans)
        ellipse_collection.set_alpha(ell_alpha)
        ellipse_collection.set_linewidth(ell_linewidth)
        ellipse_collection.set_edgecolor(ell_color)
        ax.add_collection(ellipse_collection)

# ---- CHIP DISPLAY COMMANDS ----
def imshow(img, 
           fignum=None,
           title=None, 
           figtitle=None, 
           plotnum=None,
           interpolation='nearest', 
           **kwargs):
    printDBG('[df2] *** imshow in fig=%r title=%r *** ' % (fignum, title))
    printDBG('[df2] *** fignum = %r, plotnum = %r ' % (fignum, plotnum))
    fig = figure(fignum=fignum, plotnum=plotnum, title=title, figtitle=figtitle, **kwargs)
    ax = plt.gca()
    plt.imshow(img, interpolation=interpolation)
    plt.set_cmap('gray')
    ax = fig.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    #try:
        #if plotnum == 111:
            #fig.tight_layout()
    #except Exception as ex:
        #print('[df2] !! Exception durring fig.tight_layout: '+repr(ex))
        #raise
    return fig, ax


def show_matches2(rchip1, rchip2, kpts1, kpts2,
                  fm=None, fs=None, fignum=None, plotnum=None,
                  title=None, vert=None, all_kpts=True, 
                  draw_lines=True,
                  draw_ell=True, 
                  draw_pts=True,
                  ell_alpha=None, **kwargs):
    '''Draws feature matches 
    kpts1 and kpts2 use the (x,y,a,c,d)
    '''
    if fm is None:
        assert kpts1.shape == kpts2.shape
        fm = np.tile(np.arange(0, len(kpts1)), (2,1)).T
    (h1, w1) = rchip1.shape[0:2] # get chip dimensions 
    (h2, w2) = rchip2.shape[0:2]
    woff = 0; hoff = 0 
    if vert is None: # Display match up/down or side/side
        vert = False if h1 > w1 and h2 > w2 else True
    if vert: wB=max(w1,w2); hB=h1+h2; hoff=h1
    else:    hB=max(h1,h2); wB=w1+w2; woff=w1
    # concatentate images
    match_img = np.zeros((hB, wB, 3), np.uint8)
    match_img[0:h1, 0:w1, :] = rchip1
    match_img[hoff:(hoff+h2), woff:(woff+w2), :] = rchip2
    # get matching keypoints + offset
    fig, ax = imshow(match_img, fignum=fignum,
                plotnum=plotnum, title=title,
                **kwargs)
    upperleft_text('#match=%d' % len(fm))
    if all_kpts:
        # Draw all keypoints as simple points
        all_args = dict(ell=False, pts=draw_pts, pts_color=GREEN, pts_size=2, ell_alpha=ell_alpha)
        draw_kpts2(kpts1, **all_args)
        draw_kpts2(kpts2, offset=(woff,hoff), **all_args) 
    if len(fm) == 0:
        printDBG('[df2] There are no feature matches to plot!')
    else:
        # Draw matching ellipses
        ell_args = dict(ell=draw_ell, pts=draw_pts, pts_color=ORANGE, pts_size=4, ell_alpha=ell_alpha)
        draw_kpts2(kpts1[fm[:,0]], **ell_args)
        draw_kpts2(kpts2[fm[:,1]], offset=(woff,hoff), **ell_args)
        # Draw matching lines
        if draw_lines:
            draw_matches2(kpts1, kpts2, fm, fs, kpts2_offset=(woff,hoff))
    return fig, ax

def show_matches_annote_res(res, hs, cx,
                  SV=True, 
                  fignum=None, 
                  plotnum=None,
                  title_aug=None, 
                  **kwargs):
    '''
    Wrapper for show_matches_annote
    '''
    qcx = res.qcx
    cx2_score = res.get_cx2_score(SV)
    cx2_fm    = res.get_cx2_fm(SV)
    cx2_fs    = res.get_cx2_fs(SV)
    title_suff = '(+V)' if SV else None
    return show_matches_annote(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx,
                         fignum, plotnum, title_aug, title_suff, **kwargs)

# TODO: This should go in viz
def show_matches_annote(hs, qcx, cx2_score, 
                        cx2_fm, cx2_fs, cx,
                        fignum=None, plotnum=None, 
                        title_pref=None, 
                        title_suff=None,
                        show_cx=False,
                        show_cid=True,
                        **kwargs):
    ' Shows matches with annotations '
    printDBG('[df2] Showing matches from %s in fignum=%r' % (hs.vs_str(cx, qcx), fignum))
    if np.isnan(cx):
        nan_img = np.zeros((100,100), dtype=np.uint8)
        title='(qx%r v NAN)' % (qcx)
        imshow(nan_img,fignum=fignum,plotnum=plotnum,title=title)
        return 
    # Read query and result info (chips, names, ...)
    rchip1, rchip2 = hs.get_chip([qcx, cx])
    kpts1, kpts2   = hs.get_kpts([qcx, cx])
    score = cx2_score[cx]
    fm = cx2_fm[cx]; fs = cx2_fs[cx]
    # Build the title string
    score_str = (' score='+helpers.num_fmt(score)) % (score)
    isgt_str  = hs.is_true_match_str(qcx, cx) 
    title     = '*' + isgt_str + '*' + '\n' + score_str
    if not title_pref is None: title = title_pref + title
    if not title_suff is None: title = title + title_suff
    # Draw the matches
    fig, ax = show_matches2(rchip1, rchip2, kpts1, kpts2, fm, fs, 
                            fignum=fignum, plotnum=plotnum,
                            title=title, **kwargs)
    upperright_text(hs.vs_str(qcx, cx))
    # Finish annotations
    if   isgt_str == hs.UNKNOWN_STR: draw_border(ax, WHITE, 4)
    elif isgt_str == hs.TRUE_STR:    draw_border(ax, GREEN, 4)
    elif isgt_str == hs.FALSE_STR:   draw_border(ax, RED, 4)
    ax.set_xlabel(hs.get_gname(cx))
    return ax

def show_img(hs, cx, **kwargs):
    # Get the chip roi
    roi = hs.get_roi(roi)
    (rx,ry,rw,rh) = roi
    rxy = (rx,ry)
    # Get the image
    img = hs.get_image(gx)
    # Draw image
    imshow(img, **kwargs)
    # Draw ROI
    ax = plt.gca()
    bbox = matplotlib.patches.Rectangle(rxy,rw,rh) 
    bbox_color = [1, 0, 0]
    bbox.set_fill(False)
    bbox.set_edgecolor(bbox_color)
    ax.add_patch(bbox)

def show_keypoints(rchip,kpts,fignum=0,title=None, **kwargs):
    imshow(rchip,fignum=fignum,title=title,**kwargs)
    draw_kpts2(kpts)

def show_chip(hs, cx=None, allres=None, res=None, info=True, draw_kpts=True,
              nRandKpts=None, SV=True, **kwargs):
    if not res is None:
        cx = res.qcx
    if not allres is None:
        res = allres.qcx2_res[cx]
    rchip1    = hs.get_chip(cx)
    title_str = hs.cxstr(cx)
    # Add info to title
    if info: 
        title_str += ', '+hs.num_indexed_gt_str(cx)
    fig, ax = imshow(rchip1, title=title_str, **kwargs)
    if not res is None: 
        gname = hs.get_gname(cx)
        ax.set_xlabel(gname)
    if not draw_kpts:
        return
    kpts1  = hs.get_kpts(cx)
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
        import match_chips2 as mc2
        cx2_fm = res.get_cx2_fm(SV)
        #mc2.debug_cx2_fm_shape(cx2_fm)
        #cx2_fm = mc2.fix_cx2_fm_shape(cx2_fm)
        #mc2.debug_cx2_fm_shape(cx2_fm)
        #print('>>>>')
        #print('In show_chip')
        #helpers.printvar(locals(), 'cx2_fm')
        #print('<<<')
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
            draw_kpts2(kpts_, ell_color=color, ell_alpha=alpha, **kpts_args)
            phant_ = Circle((0, 0), 1, fc=color)
            legend_tups.append((phant_, label))
        _kpts_helper(kpts_noise,   RED, .1, 'Unverified')
        _kpts_helper(kpts_match,  BLUE, .4, 'Verified')
        _kpts_helper(kpts_true,  GREEN, .6, 'True Matches')
        #plt.legend(*zip(*legend_tups), framealpha=.2)
    # Just draw boring keypoints
    else:
        if not nRandKpts is None: 
            nkpts1 = len(kpts1)
            fxs1 = np.arange(nkpts1)
            size = nRandKpts
            replace = False
            p = np.ones(nkpts1)
            p = p / p.sum()
            fxs_randsamp = np.random.choice(fxs1, size, replace, p)
            kpts1 = kpts1[fxs_randsamp]
            ax = plt.gca()
            ax.set_xlabel('displaying %r/%r keypoints' % (nRandKpts, nkpts1))
            # show a random sample of kpts
        draw_kpts2(kpts1, ell_alpha=.7, ell_color=(.9,.1,.1), **kpts_args)

def show_topN_matches(hs, res, N=5, SV=True, fignum=4): 
    figtitle = ('q%s -- TOP %r' % (hs.cxstr(res.qcx), N))
    topN_cxs = res.topN_cxs(N)
    max_nCols = max(5,N)
    _show_chip_matches(hs, res, topN_cxs=topN_cxs, figtitle=figtitle, 
                       fignum=fignum, all_kpts=False)

def show_gt_matches(hs, res, SV=True, fignum=3): 
    figtitle = ('q%s -- GroundTruth' % (hs.cxstr(res.qcx)))
    gt_cxs = hs.get_other_indexed_cxs(res.qcx)
    max_nCols = max(5,len(gt_cxs))
    _show_chip_matches(hs, res, gt_cxs=gt_cxs, figtitle=figtitle, 
                       fignum=fignum, all_kpts=True)

def show_match_analysis(hs, res, N=5, fignum=3, figtitle='',
                        show_query=True, annotations=True, SV=True,
                        compare_SV=False, **kwargs):
    topN_cxs = res.topN_cxs(N, SV)
    topscore = res.get_cx2_score(SV)[topN_cxs][0]
    figtitle = ('topscore=%r -- q%s' % (topscore, hs.cxstr(res.qcx))) + figtitle
    all_gt_cxs = hs.get_other_indexed_cxs(res.qcx)
    missed_gt_cxs = np.setdiff1d(all_gt_cxs, topN_cxs)
    max_nCols = min(5,N)
    return _show_chip_matches(hs, res,
                              gt_cxs=missed_gt_cxs, 
                              topN_cxs=topN_cxs,
                              figtitle=figtitle,
                              max_nCols=max_nCols,
                              show_query=show_query,
                              fignum=fignum,
                              annotations=annotations,
                              compare_SV=compare_SV,
                              SV=SV, **kwargs)

def _show_chip_matches(hs, res, figtitle='', max_nCols=5,
                       topN_cxs=None, gt_cxs=None, show_query=True,
                       all_kpts=False, fignum=3, annotations=True, 
                       SV=True, compare_SV=False, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    print('========================')
    print('[df2] Show chip matches:')
    print('[df2] #top=%r #missed_gts=%r' % (len(topN_cxs),len(gt_cxs)))
    printDBG('[df2] * max_nCols=%r' % (max_nCols,))
    printDBG('[df2] * show_query=%r' % (show_query,))
    fig = figure(fignum=fignum); fig.clf()
    ranked_cxs = res.get_cx2_score(SV).argsort()[::-1]
    annote = annotations
    # Build a subplot grid
    # * Get top N informatino
    nQuerySubplts = 1 if show_query else 0
    nGtSubplts = nQuerySubplts + (0 if gt_cxs is None else len(gt_cxs))
    nTopNSubplts  = 0 if topN_cxs is None else len(topN_cxs)
    nCols     = min(max_nCols, max(nGtSubplts, nTopNSubplts))
    nGtRows   = int(np.ceil(nGtSubplts / nCols))
    nTopNRows = int(np.ceil(nTopNSubplts / nCols))
    nGtCells = nGtRows * nCols
    nTopNCells = nTopNRows * nCols
    nRows = nTopNRows+nGtRows
    if compare_SV:
        nRows += nTopNRows

    # Helper function for drawing matches to one cx
    def show_matches_(cx, orank, plotx, SV):
        title_aug = 'rank=%r ' % orank
        plotnum=(nRows, nCols, plotx)
        kwshow  = dict(draw_ell=annote, draw_pts=annote, draw_lines=annote,
                       ell_alpha=.5, all_kpts=all_kpts, SV=SV, **kwargs)
        show_matches_annote_res(res, hs, cx, title_aug=title_aug,
                                plotnum=plotnum, **kwshow)
    # Helper to draw many cxs
    def plot_matches_cxs(cx_list, plotx_shift, SV):
        if gt_cxs is None:
            return
        for ox, cx in enumerate(cx_list):
            plotx = ox + plotx_shift + 1
            orank = np.where(ranked_cxs == cx)[0][0] + 1
            show_matches_(cx, orank, plotx, SV)

    # Plot Query
    plt.subplot(nRows, nCols, 1)
    if show_query: 
        printDBG('Plotting Query:')
        plotnum=(nRows, nCols, 1)
        show_chip(hs, res=res, plotnum=plotnum, draw_kpts=annote, SV=SV)
    # Plot Ground Truth
    plot_matches_cxs(gt_cxs, nQuerySubplts, SV) 
    # Plot Top N
    plot_matches_cxs(topN_cxs, nGtCells, SV)    
    if compare_SV:
        offset = nGtCells + nTopNCells
        # Plot Ground Truth
        # plot_matches_cxs(gt_cxs, offset + nQuerySubplts, not SV) 
        # Plot Top N
        plot_matches_cxs(topN_cxs, offset, not SV)    
        #plotx_shift = 1 + nGtCells#num_cells - num_subplots + 1
        #for ox, cx in enumerate(topN_cxs):
            #plotx = ox + plotx_shift
            #orank = np.where(ranked_cxs == cx)[0][0] + 1
            #show_matches_(cx, orank, plotx, SV)


    set_figtitle(figtitle, res.query_uid)
    printDBG('[df2] + nTopNRows=%r' % nTopNRows)
    printDBG('[df2] + nGtRows=%r' % nGtRows)
    printDBG('[df2] + nGtCells=%r' % nGtCells)
    printDBG('[df2] + nCols=%r' % nCols)
    printDBG('[df2] + nRows=%r' % nRows)
    print('-----------------')
    return fig

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('=================================')
    print('[df2] __main__ = draw_func2.py')
    print('=================================')
    from __init__ import *
    qcx = 0
    hs = ld2.HotSpotter()
    hs.load_tables(ld2.DEFAULT)
    hs.load_chips()
    hs.load_features()
    hs.set_samples()
    res = mc2.build_result_qcx(hs, qcx)
    print('')
    print('''
    exec(open("draw_func2.py").read())
    ''')
    N=5
    df2.rrr()
    figtitle='q%s -- Analysis' % (hs.cxstr(res.qcx),)
    topN_cxs = res.topN_cxs(N)
    all_gt_cxs = hs.get_other_indexed_cxs(res.qcx)
    gt_cxs = np.setdiff1d(all_gt_cxs, topN_cxs)
    max_nCols = max(5,N)
    fignum=3
    show_query = True
    all_kpts = False
    #get_geometry(1)
    df2.show_match_analysis(hs, res, N)
    df2.update()
    exec(df2.present())
