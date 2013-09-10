''' Lots of functions for drawing and plotting visiony things '''
from __future__ import division
import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
    print('df2>matplotlib.use(Qt4Agg)')
    matplotlib.use('Qt4Agg', warn=True, force=True)
    matplotlib.rcParams['toolbar'] = 'None'
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from PyQt4.QtCore import Qt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import types
import warnings
import helpers
import textwrap
import os
import sys
import pylab
import types
#print('LOAD_MODULE: draw_func2.py')

def execstr_global():
    execstr = ['global' +key for key in globals().keys()]
    return execstr

def my_prefs():
    global LINE_COLOR
    global ELL_COLOR
    global ELL_LINEWIDTH
    global ELL_ALPHA
    LINE_COLOR = (1, 0, 0)
    ELL_COLOR = (0, 0, 1)
    ELL_LINEWIDTH = 2
    ELL_ALPHA = .5

try:
    if sys.platform == 'win32':
        compname = os.environ['COMPUTER_NAME']
        if compname == 'Ooo':
            TILE_WITHIN = (-1912, 30, -969, 1071)
except KeyError:
    TILE_WITHIN = (0, 30, 969, 1041)

DPI = 80
#FIGSIZE = (24) # default windows fullscreen
FIGSIZE = (20,10) 

LINE_ALPHA = .4
ELL_ALPHA  = .3
ELL_LINEWIDTH = 1
ELL_COLOR  = (0, 0, 1)

LINE_COLOR = (1, 0, 0)
LINE_CMAP  = 'hot'
LINE_WIDTH = 1.4

SHOW_LINES = True #True
SHOW_ELLS  = True

POINT_SIZE = 2

def reload_module():
    import imp
    import sys
    print 'reloading '+__name__
    imp.reload(sys.modules[__name__])
def rrr():
    reload_module()

def printDBG(msg):
    #print(msg)
    pass

# ---- GENERAL FIGURE COMMANDS ----
def sanatize_img_fpath(fpath):
    [dpath, fname_clean] = os.path.split(fpath)
    search_replace_list = [(' ', '_'), ('\n', '--'), ('\\', ''), ('/','')]
    for old, new in search_replace_list:
        fname_clean = fname_clean.replace(old, new)
    fpath_clean = os.path.join(dpath, fname_clean)
    root, ext = os.path.splitext(fpath_clean)
    # Check for correct extensions
    if not ext.lower() in helpers.IMG_EXTENSIONS:
        fpath_clean += '.png'
    fpath_clean = os.path.normpath(fpath_clean)
    return fpath_clean

def save(fig, fpath=None):
    if fpath is None:
        # Find the title
        fpath = fig.canvas.get_window_title()
    if fpath is None: 
        fpath = 'Figure'+str(fig.number)+'.png'
    # Sanatize the filename
    fpath_clean = sanatize_img_fpath(fpath)
    print('Saving figure to: '+repr(fpath_clean))
    fig.savefig(fpath_clean, dpi=DPI)

def save_figure(fignum=None, fpath=None, usetitle=False):
    # Find the figure
    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum, figsize=FIGSIZE, dpi=DPI)
    fignum = fig.number
    if fpath is None:
        # Find the title
        fpath = fig.canvas.get_window_title()
    if usetitle:
        title = fig.canvas.get_window_title()
        fpath = os.path.join(fpath, title)
    # Sanatize the filename
    fpath_clean = sanatize_img_fpath(fpath)
    print('df2.save_figure> '+repr(os.path.split(fpath_clean)[1]))
    fig.savefig(fpath_clean, dpi=DPI)

def update_figure_size(fignum, width, height):
    fig = get_fig(fignum)
    set_geometry(fig, 40, 40, width, height)
    fig.canvas.draw()

def get_fig(fignum=None):
    if fignum is None: fig = plt.gcf()
    else: fig = plt.figure(fignum)
    return fig

def set_geometry(fignum, x, y, w, h):
    fig = get_fig(fignum)
    qtwin = fig.canvas.manager.window
    qtwin.setGeometry(x, y, w, h)

def get_geometry():
    fig = get_fig(fignum)
    qtwin = fig.canvas.manager.window
    (x1, y1, x2, y2) = qtwin.geometry().getCoords()
    (x, y, w, h) = (x1, y1, x2-x1, y2-y1)
    return (x,y,w,h)

def get_all_figures():
    all_figures_=[manager.canvas.figure for manager in
                 matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    all_figures = []
    # Make sure you dont show figures that this module closed
    for fig in iter(all_figures_):
        if not 'df2_closed' in fig.__dict__.keys() or not fig.df2_closed:
            all_figures.append(fig)
    # Return all the figures sorted by their number
    all_figures = sorted(all_figures, key=lambda fig:fig.number)
    return all_figures

def all_figures_show():
    for fig in iter(get_all_figures()):
        fig.show()
        fig.canvas.draw()

def all_figures_tile(num_rc=(4,4),
                     wh=(350,250),
                     xy_off=(0,0),
                     wh_off=(0,10),
                     row_first=True):
    num_rows,num_cols = num_rc
    w,h = wh
    x_off, y_off = xy_off
    w_off, h_off = wh_off
    x_pad, y_pad = (0, 0)
    if sys.platform == 'win32':
        x_off, yoff = (x_off+40, y_off+40)
        #x_off, yoff = (x_off-2000, y_off-1000)
        x_pad, y_pad = (0, 40)
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        if row_first:
            y = (i % num_rows)*(h+h_off) + 40
            x = (int(i/num_rows))*(w+w_off) + x_pad
        else:
            x = (i % num_cols)*(w+w_off) + 40
            y = (int(i/num_cols))*(h+h_off) + y_pad
        x+=x_off
        y+=y_off
        qtwin.setGeometry(x,y,w,h)

def all_figures_bring_to_front():
    all_figures = get_all_figures()
    for fig in iter(all_figures):
        bring_to_front(fig)

def close_all_figures():
    all_figures = get_all_figures()
    for fig in iter(all_figures):
        close_figure(fig)

def close_figure(fig):
    fig.clf()
    fig.df2_closed = True
    qtwin = fig.canvas.manager.window
    qtwin.close()

def bring_to_front(fig):
    #what is difference between show and show normal?
    qtwin = fig.canvas.manager.window
    qtwin.raise_()
    qtwin.activateWindow()
    qtwin.setWindowFlags(Qt.WindowStaysOnTopHint)
    qtwin.show()
    qtwin.setWindowFlags(Qt.WindowFlags(0))
    qtwin.show()

def show():
    all_figures_show()
    all_figures_bring_to_front()
    plt.show()

def reset():
    close_all_figures()

def draw():
    all_figures_show()

def update():
    draw()
    all_figures_bring_to_front()

def present(*args, **kwargs):
    'execing present should cause IPython magic'
    print('Presenting figures...')
    all_figures_tile(*args, **kwargs)
    all_figures_show()
    all_figures_bring_to_front()
    # Return an exec string
    return textwrap.dedent(r'''
    import helpers
    import matplotlib.pyplot as plt
    embedded = False
    if not helpers.inIPython():
        if '--cmd' in sys.argv:
            print('df2: Requested IPython shell with --cmd argument.')
            if helpers.haveIPython():
                print('df2: Found IPython')
                try: 
                    import IPython
                    print('df2: Presenting in new ipython shell.')
                    embedded = True
                    IPython.embed()
                except Exception as ex:
                    helpers.printWARN(repr(ex)+'\n!!!!!!!!')
                    embedded = False
            else:
                print('df2: IPython is not installed')
        if not embedded: 
            print('df2: Presenting in normal shell.')
            print('df2: ... plt.show()')
            plt.show()
    else: 
        print('Presenting in current ipython shell.')
    ''')

'''
import draw_func2 as df2
import matplotlib.pyplot as plt
img = df2.test_img()

import imp
imp.reload(df2)
'''

def test_img(index=0):
    import matplotlib.cbook as cbook
    from PIL import Image
    sample_fnames = ['grace_hopper.jpg',
                     'lena.png',
                     'ada.png']
    if index <= len(sample_fnames):
        test_file = cbook.get_sample_data(sample_fnames[index])
    else:
        import load_data2
        chip_dir  = load_data2.DEFAULT+load_data2.RDIR_CHIP
        test_file = chip_dir+'/CID_%d.png' % (1+index-len(sample_fnames))
    test_img = np.asarray(Image.open(test_file).convert('L'))
    return test_img

def set_xticks(tick_set):
    ax = plt.gca()
    ax.set_xticks(tick_set)

def set_yticks(tick_set):
    ax = plt.gca()
    ax.set_yticks(tick_set)

def set_xlabel(lbl):
    ax = plt.gca()
    ax.set_xlabel(lbl)

def set_ylabel(lbl):
    ax = plt.gca()
    ax.set_ylabel(lbl)

def plot(*args, **kwargs):
    return plt.plot(*args, **kwargs)

def set_figtitle(figtitle):
    fig = plt.gcf()
    fig.canvas.set_window_title(figtitle)

def figure(fignum=None,
           doclf=False,
           title=None,
           plotnum=111,
           figtitle=None,
           **kwargs):
    fig = plt.figure(num=fignum, figsize=FIGSIZE, dpi=DPI)
    axes_list = fig.get_axes()
    if not 'user_stat_list' in fig.__dict__.keys() or doclf:
        fig.user_stat_list = []
        fig.user_notes = []
    fig.df2_closed = False
    if doclf or len(axes_list) == 0:
        #if plotnum==111:
            #fig.clf()
        ax = plt.subplot(plotnum)
        ax.cla()
        printDBG('*** NEW FIGURE '+str(fignum)+'.'+str(plotnum)+' ***')
    else: 
        printDBG('*** OLD FIGURE '+str(fignum)+'.'+str(plotnum)+' ***')
        ax = plt.subplot(plotnum)
        #ax  = axes_list[0]
    if not title is None:
        ax.set_title(title)
        # Add title to figure
        if figtitle is None and plotnum == 111:
            figtitle = title
        if not figtitle is None:
            fig.canvas.set_window_title('fig '+repr(fignum)+' '+figtitle)
    return fig

    
# ---- IMAGE CREATION FUNCTIONS ---- 

# adapted from:
# http://jayrambhia.com/blog/sift-keypoint-matching-using-python-opencv/
def draw_matches(rchip1, rchip2, kpts1, kpts2, fm12, vert=False):
    global LINE_COLOR
    h1, w1 = rchip1.shape[0:2]
    h2, w2 = rchip2.shape[0:2]
    woff = 0; hoff = 0 # offsets 
    if vert:
        wB = max(w1, w2); hB = h1+h2; hoff = h1
    else: 
        hB = max(h1, h2); wB = w1+w2; woff = w1
    # Concat images
    match_img = np.zeros((hB, wB, 3), np.uint8)
    match_img[0:h1, 0:w1, :] = rchip1
    match_img[hoff:(hoff+h2), woff:(woff+w2), :] = rchip2
    # Draw lines
    for kx1, kx2 in iter(fm12):
        pt1 = (int(kpts1[kx1,0]),      int(kpts1[kx1,1]))
        pt2 = (int(kpts2[kx2,0])+woff, int(kpts2[kx2,1])+hoff)
        match_img = cv2.line(match_img, pt1, pt2, LINE_COLOR*255)
    return match_img

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
        cmap = plt.get_cmap(LINE_CMAP)
        mins = fs.min()
        maxs = fs.max()
        segments  = [ ((x1, y1), (x2,y2)) for (x1,x2,y1,y2) in xxyy_iter ] 
        colors    = [ cmap(.1+ .9*(fs[fx]-mins)/(maxs-mins)) for fx in xrange(len(fm)) ] 
        linewidth = [ LINE_WIDTH for fx in xrange(len(fm)) ] 

    line_collection = matplotlib.collections.LineCollection(segments,
                                                            linewidth, 
                                                            colors,
                                                            alpha=LINE_ALPHA)
    ax.add_collection(line_collection)

def draw_kpts2(kpts, offset=(0,0), ell=True, 
               pts=False, pts_color='r', pts_size=2, ell_alpha=None):
    printDBG('drawkpts2: Drawing Keypoints! ell=%r pts=%r' % (ell, pts))
    global SHOW_ELLS
    global ELL_COLOR
    global ELL_ALPHA
    global ELL_LINEWIDTH
    global POINT_SIZE
    if not SHOW_ELLS:
        return
    if ell_alpha is None:
        ell_alpha = ELL_ALPHA
    # get matplotlib info
    ax = plt.gca()
    pltTrans = ax.transData
    ell_actors = []
    eps = 1E-9
    # data
    kptsT = kpts.T
    x = kptsT[0,:] + offset[0]
    y = kptsT[1,:] + offset[1]
    printDBG('draw_kpts>----------')
    printDBG('draw_kpts> ell=%r pts=%r' % (ell, pts))
    printDBG('draw_kpts> drawing kpts.shape=%r' % (kpts.shape,))
    if pts:
        printDBG('draw_kpts> drawing pts x.shape=%r y.shape=%r' % (x.shape, y.shape))
        ax.plot(x, y, linestyle='None', 
                marker='o',
                markerfacecolor=pts_color,
                markersize=pts_size, 
                markeredgewidth=0)
    if ell:
        printDBG('draw_kpts> drawing ell kptsT.shape=%r' % (kptsT.shape,))
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
            bIS = -c/(np.sqrt(a)*d + a*np.sqrt(d))
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
        ellipse_collection.set_linewidth(ELL_LINEWIDTH)
        ellipse_collection.set_edgecolor(ELL_COLOR)
        ax.add_collection(ellipse_collection)

# ---- CHIP DISPLAY COMMANDS ----
def legend():
    ax = plt.gca()
    ax.legend(**{'fontsize':18})

def draw_histpdf(data, label=None):
    freq, _ = draw_hist(data)
    draw_pdf(data, draw_support=False, scale_to=freq.max(), label=label)

def draw_stems(x_data, y_data):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = x_data[y_data.argsort()[::-1]]
    y_data = y_data[y_data.argsort()[::-1]]

    markerline, stemlines, baseline = pylab.stem(x_data, y_data, linefmt='-')
    pylab.setp(markerline, 'markerfacecolor', 'b')
    pylab.setp(baseline, 'linewidth', 0)
    ax = plt.gca()
    ax.set_xlim(min(x_data)-1, max(x_data)+1)
    ax.set_ylim(min(y_data)-1, max(max(y_data), max(x_data))+1)

def draw_hist(data, bins=None):
    if type(data) == types.ListType:
        data = np.array(data)
    if bins is None:
        dmin = data.min()
        dmax = data.max()
        bins = dmax - dmin
    ax  = plt.gca()
    freq, bins_, patches = ax.hist(data, range=(dmin,dmax))
    return freq, bins_
    
def variation_trunctate(data):
    ax = plt.gca()
    data = np.array(data)
    trunc_max = data.mean() + data.std() * 2
    trunc_min = np.floor(data.min())
    ax.set_xlim(trunc_min,trunc_max)
    #trunc_xticks = np.linspace(0, int(trunc_max),11)
    #trunc_xticks = trunc_xticks[trunc_xticks >= trunc_min]
    #trunc_xticks = np.append([int(trunc_min)], trunc_xticks)
    #no_zero_yticks = ax.get_yticks()[ax.get_yticks() > 0]
    #ax.set_xticks(trunc_xticks)
    #ax.set_yticks(no_zero_yticks)
    
import scipy.stats
def draw_pdf(data, draw_support=True, scale_to=None, label=None, colorx=0):
    data = np.array(data)
    fig = plt.gcf()
    ax = plt.gca()
    bw_factor = .05
    line_color = plt.get_cmap('gist_rainbow')(colorx)
    # Estimate a pdf
    data_pdf = scipy.stats.gaussian_kde(data, bw_factor)
    data_pdf.covariance_factor = bw_factor
    # Get probability of seen data
    probability = data_pdf(data)
    # Get probability of unseen data data
    x_data = np.linspace(0, data.max(), 500)
    y_data = data_pdf(x_data)
    # Scale if requested
    if not scale_to is None:
        scale_factor = scale_to / y_data.max()
        y_data *= scale_factor
        probability *= scale_factor
    #Plot the actual datas on near the bottom perterbed in Y
    if draw_support:
        pdfrange = probability.max() - probability.min() 
        perb   = (np.random.randn(len(data))) * pdfrange/30.
        preb_y_data = np.abs([pdfrange/50. for _ in data]+perb)
        ax.plot(data, preb_y_data, 'o', color=line_color, figure=fig, alpha=.1)
    # Plot the pdf (unseen data)
    ax.plot(x_data, y_data, color=line_color, label=label)
    
def show_histogram(data, bins=None, **kwargs):
    if bins is None:
        dmin = data.min()
        dmax = data.max()
        bins = dmax - dmin
    fig = figure(**kwargs)
    ax  = plt.gca()
    ax.hist(data, bins=bins, range=(dmin,dmax))
    #help(np.bincount)
    fig.show()

def show_signature(sig, **kwargs):
    fig = figure(**kwargs)
    plt.plot(sig)
    fig.show()
    

def imshow(img, fignum=0, title=None, figtitle=None, plotnum=111,
           interpolation='nearest', **kwargs):
    printDBG('*** imshow in fig=%r title=%r *** ' % (fignum, title))
    printDBG('   * fignum = %r, plotnum = %r ' % (fignum, plotnum))
    fig = figure(fignum=fignum, plotnum=plotnum, title=title, figtitle=figtitle, **kwargs)
    plt.imshow(img, interpolation=interpolation)
    plt.set_cmap('gray')
    ax = fig.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        if plotnum == 111:
            fig.tight_layout()
    except Exception as ex:
        print('!! Exception durring fig.tight_layout: '+repr(ex))
        raise
    return fig

def show_top5_matches(hs, res, SV=True, fignum=4): 
    SV = True
    qcx = res.qcx
    title_aug=None
    top5_cxs = res.top5_cxs()
    others = top5_cxs
    num_others = len(others)
    plotnum= 100 + num_others*10 + 1 
    if num_others == 0:
        print('no known matches to qcx=%r' % qcx)
        return
    figtitle='qcx=%r -- TOP 5' % qcx
    figure(fignum=fignum, plotnum=plotnum)
    #print 'figure(plotnum)='+str(plotnum)
    for ox, cx in enumerate(others):
        plotnumcx = plotnum + ox
        #print 'plot_to: plotnum='+str(plotnumcx)
        show_matches3(res, hs, cx, fignum=fignum, plotnum=plotnumcx, all_kpts=False, ell_alpha=.5)
    set_figtitle(figtitle)

def show_all_matches(*args, **kwargs): 
    show_gt_matches(*args, **kwargs)

def show_gt_matches(hs, res, SV=True, fignum=3): 
    SV = True
    qcx = res.qcx
    title_aug=None
    others = hs.get_other_cxs(qcx)
    num_others = len(others)
    plotnum= 100 + num_others*10 + 1 
    if num_others == 0:
        print('no known ma`tches to qcx=%r' % qcx)
        return
    figtitle='qcx=%r -- GroundTruth' % qcx
    figure(fignum=fignum, plotnum=plotnum)
    #print 'figure(plotnum)='+str(plotnum)
    for ox, cx in enumerate(others):
        plotnumcx = plotnum + ox
        #print 'plot_to: plotnum='+str(plotnumcx)
        show_matches3(res, hs, cx, fignum=fignum, plotnum=plotnumcx)
    set_figtitle(figtitle)

def show_matches2(rchip1, rchip2, kpts1, kpts2,
                  fm=None, fs=None, fignum=0, plotnum=111,
                  title=None, vert=True, all_kpts=True, 
                  draw_lines=True, ell_alpha=None, **kwargs):
    '''Draws feature matches 
    kpts1 and kpts2 use the (x,y,a,c,d)
    '''
    if fm is None:
        assert kpts1.shape == kpts2.shape
        fm = np.tile(np.arange(0, len(kpts1)), (2,1)).T
    # get chip dimensions 
    (h1,w1) = rchip1.shape[0:2]
    (h2,w2) = rchip2.shape[0:2]
    woff = 0; hoff = 0 
    if vert: wB=max(w1,w2); hB=h1+h2; hoff=h1
    else:    hB=max(h1,h2); wB=w1+w2; woff=w1
    #vert = True
    #if vert: wB=max(w1,w2); hB=h1+h2; hoff=h1
    #else:    hB=max(h1,h2); wB=w1+w2; woff=w1
    # concatentate images
    match_img = np.zeros((hB, wB, 3), np.uint8)
    match_img[0:h1, 0:w1, :] = rchip1
    match_img[hoff:(hoff+h2), woff:(woff+w2), :] = rchip2
    # get matching keypoints + offset
    if len(fm) == 0:
        printDBG('There are no feature matches to plot!')
        imshow(match_img,fignum=fignum,plotnum=plotnum,title=title, **kwargs)
    else: 
        # matplotlib stuff
        imshow(match_img,fignum=fignum,plotnum=plotnum,title=title, **kwargs)
        # Draw all keypoints as simple points
        if all_kpts:
            draw_kpts2(kpts1, ell=False, pts=True, pts_color='g', pts_size=2, ell_alpha=ell_alpha)
            draw_kpts2(kpts2, offset=(woff,hoff), ell=False, pts=True,
                       pts_color='g', pts_size=2, ell_alpha=ell_alpha)
        # Draw matching ellipses
        orange=np.array((255, 127, 0, 255))/255.0
        draw_kpts2(kpts1[fm[:,0]],
                   pts=True, pts_color=orange, pts_size=4, ell_alpha=ell_alpha)
        draw_kpts2(kpts2[fm[:,1]], offset=(woff,hoff),
                   pts=True, pts_color=orange, pts_size=4, ell_alpha=ell_alpha)
        # Draw matching lines
        if draw_lines:
            draw_matches2(kpts1, kpts2, fm, fs, kpts2_offset=(woff,hoff))

def show_matches3(res, hs, cx, SV=True, fignum=3, plotnum=111, title_aug=None, **kwargs):
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    cx2_fm    = res.cx2_fm_V if SV else res.cx2_fm
    cx2_fs    = res.cx2_fs_V if SV else res.cx2_fs
    title_suff = '(+V)' if SV else None
    return show_matches4(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx, fignum,
                         plotnum, title_aug, title_suff, **kwargs)

def show_matches4(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx, fignum=0, plotnum=111,
                  title_pref=None, title_suff=None, **kwargs):
    printDBG('Showing matches from '+str(qcx)+' to '+str(cx)+' in fignum'+repr(fignum))
    if np.isnan(cx):
        nan_img = np.zeros((100,100), dtype=np.uint8)
        title='(qx%r v NAN)' % (qcx)
        imshow(nan_img,fignum=fignum,plotnum=plotnum,title=title)
        return 
    cx2_nx = hs.tables.cx2_nx
    qnx = cx2_nx[qcx]
    nx  = cx2_nx[cx]
    cx2_rchip_path = hs.cpaths.cx2_rchip_path
    cx2_kpts = hs.feats.cx2_kpts
    rchip1 = cv2.imread(cx2_rchip_path[qcx])
    rchip2 = cv2.imread(cx2_rchip_path[cx])
    kpts1  = cx2_kpts[qcx]
    kpts2  = cx2_kpts[cx]
    score = cx2_score[cx]
    fm    = cx2_fm[cx]
    fs    = cx2_fs[cx]

    cx_str = '(qx%r v cx%r)' % (qcx, cx)
    score_str = ' #match=%r score=%.2f' % (len(fm), score)
    isgt_str = '*true match*' if nx == qnx and qnx > 1 else ''
    title= cx_str + isgt_str + '\n' + score_str
    if not title_pref is None:
        title = title_pref + title
    if not title_suff is None:
        title = title + title_suff
    return show_matches2(rchip1, rchip2, kpts1,  kpts2, fm, fs, fignum=fignum,
                         plotnum=plotnum, title=title, **kwargs)


def show_keypoints(rchip,kpts,fignum=0,title=None, **kwargs):
    imshow(rchip,fignum=fignum,title=title,**kwargs)
    draw_kpts2(kpts)


def show_chip(hs, cx, **kwargs):
    pass

def show_img(hs, cx, **kwargs):
    cx2_roi = hs.tables.cx2_roi
    cx2_gx = hs.tables.cx2_gx
    gx2_gname = hs.tables.gx2_gname

    roi = cx2_roi[cx]
    gx  = cx2_gx[cx]
    img_fname = gx2_gname[gx]
    img_fpath = os.path.join(hs.dirs.img_dir, img_fname)

    img = cv2.imread(img_fpath)
    imshow(img, **kwargs)

    [rx,ry,rw,rh] = roi
    rxy = (rx,ry)
    bbox = matplotlib.patches.Rectangle(rxy,rw,rh) 

    ax = plt.gca()

    bbox_color = [1, 0, 0]

    bbox.set_fill(False)
    bbox.set_edgecolor(bbox_color)
    ax.add_patch(bbox)




