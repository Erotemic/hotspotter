'''
Lots of functions for drawing and plotting visiony things
'''
import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
    #print('Configuring matplotlib for Qt4Agg')
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
#print('LOAD_MODULE: drawing_functions2.py')

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


def save_figure(fignum=None, fpath=None):
    # Find the figure
    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum, figsize=FIGSIZE, dpi=DPI)
    fignum = fig.number
    if fpath is None:
        # Find the title
        fpath = fig.canvas.get_window_title()
    # Sanatize the filename
    fpath_clean = sanatize_img_fpath(fpath)
    print('Saving figure to: '+repr(fpath_clean))
    fig.savefig(fpath_clean, dpi=DPI)

def update_figure_size(fignum, width, height):
    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum, figsize=FIGSIZE, dpi=DPI)
    set_geometry(fig, 40, 40, width, height)
    fig.canvas.draw()

def set_geometry(fig, x, y, w, h):
    qtwin = fig.canvas.manager.window
    qtwin.setGeometry(40,40,width,height)

def get_geometry():
    fig = plt.gcf()
    qtwin = fig.canvas.manager.window
    (x,y,w,h) = qtwin.geometry().getCoords()
    return (x,y,w,h)

def save_figsize():
    fig = plt.gcf()
    (x,y,w,h) = get_geometry()
    fig.df2_geometry = (x,y,w,h)

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
import drawing_functions2 as df2
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


def __parse_fignum(fignum_, plotnum_=111):
    'Extendend fignum format = fignum.plotnum'
    # This entir function was a bad idea. needs to go
    if type(fignum_) == types.StringType:
        (fignum2, plotnum2) = (fignum_, plotnum_)
        #(fignum2, plotnum2) = map(int, fignum.split('.'))
    elif type(fignum_) == types.FloatType:
        raise Exception('Error. This is bad buisness')
        (fignum2, plotnum2) = (int(fignum_), int(round(fignum_*1000)) - int(fignum_)*1000)
    else:
        (fignum2, plotnum2) = (fignum_, plotnum_)
    return fignum2, plotnum2

import pylab
def draw_stems(x_data, y_data):
    markerline, stemlines, baseline = pylab.stem(x_data, y_data, '-.')
    pylab.setp(markerline, 'markerfacecolor', 'b')
    pylab.setp(baseline, 'color','r', 'linewidth', 2)

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

def figure(fignum=None, doclf=False, title=None, plotnum=111,
           figtitle=None, **kwargs):
    fignum, plotnum = __parse_fignum(fignum, plotnum)
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

def set_figtitle(figtitle):
    fig = plt.gcf()
    fig.canvas.set_window_title(figtitle)

    
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

def draw_kpts2(kpts, offset=(0,0), ell=True, pts=False):
    printDBG('drawkpts2: Drawing Keypoints! ell=%r pts=%r' % (ell, pts))
    global SHOW_ELLS
    global ELL_COLOR
    global ELL_ALPHA
    global ELL_LINEWIDTH
    global POINT_SIZE
    if not SHOW_ELLS:
        return
    # get matplotlib info
    ax = plt.gca()
    pltTrans = ax.transData
    ell_actors = []
    eps = 1E-9
    # data
    kptsT = kpts.T
    x = kptsT[0] + offset[0]
    y = kptsT[1] + offset[1]
    if ell:
        a = kptsT[2]
        c = kptsT[3]
        d = kptsT[4]

        # Manually Calculated sqrtm(inv(A) for A in kpts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aIS = 1/np.sqrt(a) 
            cIS = (c/np.sqrt(d) - c/np.sqrt(d)) / (a-d+eps)
            dIS = 1/np.sqrt(d)
        kpts_iter = iter(zip(x,y,aIS,cIS,dIS))
        # This has to be the sexiest piece of code I've ever written
        ell_actors = [ Circle( (0,0), 1, 
                            transform=Affine2D([( a_, 0 , x),
                                                ( c_, d_, y),
                                                ( 0 , 0 , 1)]) )
                    for (x,y,a_,c_,d_) in kpts_iter ]
        ellipse_collection = matplotlib.collections.PatchCollection(ell_actors)
        ellipse_collection.set_facecolor('none')
        ellipse_collection.set_transform(pltTrans)
        ellipse_collection.set_alpha(ELL_ALPHA)
        ellipse_collection.set_linewidth(ELL_LINEWIDTH)
        ellipse_collection.set_edgecolor(ELL_COLOR)
        ax.add_collection(ellipse_collection)

    if pts:
        ax.plot(x, y, linestyle='None', 
                marker='o',
                markerfacecolor='r',
                markersize=POINT_SIZE, 
                markeredgewidth=0)
        
def inv_sqrtm_acd(acd):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eps = 1e-9
        a = acd[0]
        c = acd[1]
        d = acd[2]
        _a = 1.0 / np.sqrt(a) 
        _c = (c / np.sqrt(d) - c / np.sqrt(d)) / (a - d + eps)
        _d = 1.0 / np.sqrt(d)
        return _a, _c, _d

def draw_kpts_scale_color(kpts, offset=(0,0), ell=True, pts=False):
    printDBG('drawkpts2: Drawing Keypoints! ell=%r pts=%r' % (ell, pts))
    # get matplotlib info
    ax = plt.gca()
    pltTrans = ax.transData
    ell_actors = []
    eps = 1E-9
    # data
    kptsT = kpts.T
    x = kptsT[0] + offset[0]
    y = kptsT[1] + offset[1]
    acd = kptsT[2:5]
    if ell:
        a = kptsT[2]
        c = kptsT[3]
        d = kptsT[4]
        with warnings.catch_warnings():
        # Manually Calculated sqrtm(inv(A) for A in kpts)
            aIS = 1/np.sqrt(a) 
            cIS = (c/np.sqrt(d) - c/np.sqrt(d)) / (a-d+eps)
            dIS = 1/np.sqrt(d)
        kpts_iter = iter(zip(x,y,aIS,cIS,dIS))
        # This has to be the sexiest piece of code I've ever written
        ell_actors = [ Circle( (0,0), 1, 
                            transform=Affine2D([( a_, 0 , x),
                                                ( c_, d_, y),
                                                ( 0 , 0 , 1)]) )
                    for (x,y,a_,c_,d_) in kpts_iter ]
        ellipse_collection = matplotlib.collections.PatchCollection(ell_actors)
        ellipse_collection.set_facecolor('none')
        ellipse_collection.set_transform(pltTrans)
        ellipse_collection.set_alpha(ELL_ALPHA)
        ellipse_collection.set_linewidth(ELL_LINEWIDTH)
        ellipse_collection.set_edgecolor(ELL_COLOR)
        ax.add_collection(ellipse_collection)

    if pts:
        ax.plot(x, y, linestyle='None', 
                marker='o',
                markerfacecolor='r',
                markersize=POINT_SIZE, 
                markeredgewidth=0)


# ---- CHIP DISPLAY COMMANDS ----

def imshow(img, fignum=0, title=None, figtitle=None, plotnum=111, **kwargs):
    printDBG('*** imshow in fig=%r title=%r *** ' % (fignum, title))
    fignum, plotnum = __parse_fignum(fignum,plotnum)
    printDBG('   * fignum = %r, plotnum = %r ' % (fignum, plotnum))
    fig = figure(fignum=fignum, plotnum=plotnum, title=title, figtitle=figtitle,
                **kwargs)
    plt.imshow(img)
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

def draw_histpdf(data, label=None):
    freq, _ = draw_hist(data)
    draw_pdf(data, draw_support=False, scale_to=freq.max(), label=label)

def legend():
    ax = plt.gca()
    ax.legend(**{'fontsize':18})
    
import types
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

def show_matches2(rchip1, rchip2, kpts1, kpts2,
                  fm=None, fs=None, fignum=0, plotnum=111,
                  title=None, vert=True, all_kpts=True, 
                  draw_lines=True, **kwargs):
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
        kpts1_m = kpts1[fm[:,0]]
        kpts2_m = kpts2[fm[:,1]]
        # matplotlib stuff
        imshow(match_img,fignum=fignum,plotnum=plotnum,title=title, **kwargs)
        # Draw all keypoints as simple points
        if all_kpts:
            draw_kpts2(kpts1, ell=False, pts=True)
            draw_kpts2(kpts2, offset=(woff,hoff), ell=False, pts=True)
        # Draw matching ellipses
        draw_kpts2(kpts1_m)
        draw_kpts2(kpts2_m, offset=(woff,hoff))
        # Draw matching lines
        if draw_lines:
            draw_matches2(kpts1, kpts2, fm, fs, kpts2_offset=(woff,hoff))

def show_matches3(res, hs, cx, SV=True, fignum=0, plotnum=111, title_aug=None):
    qcx = res.qcx
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    cx2_fm    = res.cx2_fm_V if SV else res.cx2_fm
    cx2_fs    = res.cx2_fs_V if SV else res.cx2_fs
    title_suff = '(+V)' if SV else None
    return show_matches4(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx, fignum, plotnum, title_aug, title_suff)

def show_all_matches(hs, res): 
    SV = True
    qcx = res.qcx
    fignum=0
    title_aug=None
    others = hs.get_other_cxs(qcx)
    num_others = len(others)
    plotnum=num_others*100 + 11
    for ox, cx in enumerate(others):
        show_matches3(res, hs, cx, plotnum=plotnum+cx)

def show_matches4(hs, qcx, cx2_score, cx2_fm, cx2_fs, cx, fignum=0, plotnum=111, title_pref=None, title_suff=None):
    printDBG('Showing matches from '+str(qcx)+' to '+str(cx)+' in fignum'+repr(fignum))
    if np.isnan(cx):
        nan_img = np.zeros((100,100), dtype=np.uint8)
        title='(qx%r v NAN)' % (qcx)
        imshow(nan_img,fignum=fignum,plotnum=plotnum,title=title)
        return 
    cx2_rchip_path = hs.cpaths.cx2_rchip_path
    cx2_kpts = hs.feats.cx2_kpts
    rchip1 = cv2.imread(cx2_rchip_path[qcx])
    rchip2 = cv2.imread(cx2_rchip_path[cx])
    kpts1  = cx2_kpts[qcx]
    kpts2  = cx2_kpts[cx]
    score = cx2_score[cx]
    fm    = cx2_fm[cx]
    fs    = cx2_fs[cx]
    title='(qx%r v cx%r)\n #match=%r score=%.2f' % (qcx, cx, len(fm), score)
    if not title_pref is None:
        title = title_pref + title
    if not title_suff is None:
        title = title + title_suff
    return show_matches2(rchip1, rchip2, kpts1,  kpts2, fm, fs, fignum=fignum, plotnum=plotnum, title=title)


def show_keypoints(rchip,kpts,fignum=0,title=None, **kwargs):
    imshow(rchip,fignum=fignum,title=title,**kwargs)
    draw_kpts2(kpts)
