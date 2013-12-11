''' Lots of functions for drawing and plotting visiony things '''
from __future__ import division, print_function
import __builtin__
import sys
import matplotlib
import multiprocessing
def printDBG(msg):
    #print(msg)
    pass
from guitools import configure_matplotlib
configure_matplotlib()
from matplotlib import gridspec
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from PyQt4.QtCore import Qt
import time
import scipy.stats
import types
import textwrap
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
import sys
import types
import warnings
import itertools
from itertools import izip
import helpers
import re
import params
import os
from Printable import DynStruct

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
    print('[df2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
    helpermodule = sys.modules['draw_func2_helpers']
    print('[df2] reloading '+__name__)
    imp.reload(sys.modules[__name__])
def rrr(): reload_module()


SMALL_FONTS = False
if SMALL_FONTS:
    SMALLER  = 7
    SMALL    = 7
    MED      = 8#14
    LARGE    = 8
    LARGER   = 18
else:
    SMALLER  = 8
    SMALL    = 12
    MED      = 12
    LARGE    = 14
#fpargs = dict(family=None, style=None, variant=None, stretch=None, fname=None)
FONTS = DynStruct()
FONTS.small     = FontProperties(weight='light', size=SMALL)
FONTS.smaller   = FontProperties(weight='light', size=SMALLER)
FONTS.med       = FontProperties(weight='light', size=MED)
FONTS.large     = FontProperties(weight='light', size=LARGE)
FONTS.medbold   = FontProperties(weight='bold', size=MED)
FONTS.largebold = FontProperties(weight='bold', size=LARGE)

FONTS.legend   = FONTS.large
FONTS.figtitle = FONTS.largebold
FONTS.axtitle  = FONTS.med
FONTS.subtitle = FONTS.med
FONTS.xlabel   = FONTS.small
FONTS.ylabel   = FONTS.small
FONTS.relative = FONTS.smaller

ORANGE = np.array((255, 127,   0, 255))/255.0
DARK_ORANGE = np.array((127, 63,   0, 255))/255.0
RED    = np.array((255,   0,   0, 255))/255.0
GREEN  = np.array((  0, 255,   0, 255))/255.0
BLUE   = np.array((  0,   0, 255, 255))/255.0
YELLOW = np.array((255, 255,   0, 255))/255.0
BLACK  = np.array((  0,   0,   0, 255))/255.0
WHITE  = np.array((255, 255, 255, 255))/255.0

DPI = 80
#FIGSIZE = (24) # default windows fullscreen
FIGSIZE_MED = (20,10) 
FIGSIZE_BIG = (24,12) 

FIGSIZE = FIGSIZE_BIG 

try:
    if sys.platform == 'win32':
        compname = os.environ['COMPUTER_NAME']
        if compname == 'Ooo':
            TILE_WITHIN = (-1912, 30, -969, 1071)
except KeyError:
    TILE_WITHIN = (0, 30, 969, 1041)

plotWidget = None
def register_matplotlib_widget(plotWidget_):
    'talks to PyQt4 guis'
    global plotWidget
    plotWidget = plotWidget_
    fig = plotWidget.figure
    axes_list = fig.get_axes()
    ax = axes_list[0]
    #plt.sca(ax)

def imread(img_fpath):
    _img = cv2.imread(img_fpath, flags=cv2.IMREAD_COLOR)
    return cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

def OooScreen2():
    nRows = 1
    nCols = 1
    x_off = 30*4
    y_off = 30*4
    x_0 = -1920
    y_0 = 30
    return dict(num_rc=(1,1),
            wh=((1912-x_off)/nRows, (1080-y_off)/nCols),
            xy_off=(x_0, y_0),
            wh_off=(0,10),
            row_first=True,
            no_tile=False)

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

def _axis_xy_width_height(ax, xaug=0, yaug=0, waug=0, haug=0):
    'gets geometry of a subplot'
    autoAxis = ax.axis()
    xy     = (autoAxis[0]+xaug, autoAxis[2]+yaug)
    width  = (autoAxis[1]-autoAxis[0])+waug
    height = (autoAxis[3]-autoAxis[2])+haug
    return xy, width, height
    
def draw_border(ax, color=GREEN, lw=2):
    'draws rectangle border around a subplot'
    xy, width, height = _axis_xy_width_height(ax, -.7, -.2, 1, .4)
    rect = matplotlib.patches.Rectangle(xy, width, height, lw=lw)
    rect = ax.add_patch(rect)
    rect.set_clip_on(False)
    rect.set_fill(False)
    rect.set_edgecolor(color)

# ---- GENERAL FIGURE COMMANDS ----
def sanatize_img_fname(fname):
    fname_clean = fname
    search_replace_list = [(' ', '_'), ('\n', '--'), ('\\', ''), ('/','')]
    for old, new in search_replace_list:
        fname_clean = fname_clean.replace(old, new)
    fname_noext, ext = splitext(fname_clean)
    fname_clean = fname_noext + ext.lower()
    # Check for correct extensions
    if not ext.lower() in helpers.IMG_EXTENSIONS:
        fname_clean += '.png'
    return fname_clean

def sanatize_img_fpath(fpath):
    [dpath, fname] = split(fpath)
    fname_clean = sanatize_img_fname(fname)
    fpath_clean = join(dpath, fname_clean)
    fpath_clean = normpath(fpath_clean)
    return fpath_clean


def set_geometry(fignum, x, y, w, h):
    fig = get_fig(fignum)
    qtwin = fig.canvas.manager.window
    qtwin.setGeometry(x, y, w, h)

def get_geometry(fignum):
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
    if plotWidget is not None:
        plotWidget.figure.show()
        plotWidget.figure.canvas.draw()
    for fig in iter(get_all_figures()):
        time.sleep(.1)
        fig.show()
        fig.canvas.draw()

def all_figures_tight_layout():
    for fig in iter(get_all_figures()):
        fig.tight_layout()
        #adjust_subplots()
        time.sleep(.1)

golden_wh = lambda x:map(int,map(round,(x*.618 , x*.312)))
WH = map(lambda(x,y): x+y, zip(golden_wh(1000), (0, 20)))
def all_figures_tile(num_rc=(3,4),
                     wh=WH,
                     xy_off=(0,0),
                     wh_off=(0,10),
                     row_first=True,
                     no_tile=False):
    if no_tile:
        return
    num_rows,num_cols = num_rc
    w,h = wh
    x_off, y_off = xy_off
    w_off, h_off = wh_off
    x_pad, y_pad = (0, 0)
    if sys.platform == 'win32':
        x_off, yoff = (x_off+40, y_off+40)
        #x_off, yoff = (x_off-2000, y_off-1000)
        x_pad, y_pad = (0, 100)
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
    print('[df2] Presenting figures...')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        all_figures_tile(*args, **kwargs)
        all_figures_show()
        all_figures_bring_to_front()
    # Return an exec string
    return textwrap.dedent(r'''
    import matplotlib.pyplot as plt
    embedded = False

    try:
        __IPYTHON__
        in_ipython = True
    except NameError as nex:
        in_ipython = False

    try:
        import IPython
        have_ipython = True
    except NameError as nex:
        have_ipython = False
    
    if not in_ipython:
        if '--cmd' in sys.argv:
            print('[df2] Requested IPython shell with --cmd argument.')
            if have_ipython:
                print('[df2] Found IPython')
                try: 
                    import IPython
                    print('df2: Presenting in new ipython shell.')
                    embedded = True
                    IPython.embed()
                except Exception as ex:
                    helpers.printWARN(repr(ex)+'\n!!!!!!!!')
                    embedded = False
            else:
                print('[df2] IPython is not installed')
        if not embedded: 
            print('[df2] Presenting in normal shell.')
            print('[df2] ... plt.show()')
            plt.show()
    else: 
        print('Presenting in current ipython shell.')
    ''')

def save_figure(fignum=None, fpath=None, usetitle=False):
    #import warnings
    #warnings.simplefilter("error")
    # Find the figure
    if fignum is None:
        fig = gcf()
    else:
        fig = plt.figure(fignum, figsize=FIGSIZE, dpi=DPI)
    fignum = fig.number
    if fpath is None:
        # Find the title
        fpath = sanatize_img_fname(fig.canvas.get_window_title())
    if usetitle:
        title = sanatize_img_fname(fig.canvas.get_window_title())
        fpath = join(fpath, title)
    # Sanatize the filename
    fpath_clean = sanatize_img_fpath(fpath)
    fname_clean = split(fpath_clean)[1]
    print('[df2] save_figure() %r' % (fpath_clean,))
    #adjust_subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        fig.savefig(fpath_clean, dpi=DPI)

def set_ticks(xticks, yticks):
    ax = gca()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

def set_xticks(tick_set):
    ax = gca()
    ax.set_xticks(tick_set)

def set_yticks(tick_set):
    ax = gca()
    ax.set_yticks(tick_set)

def set_xlabel(lbl):
    ax = gca()
    ax.set_xlabel(lbl, fontproperties=FONTS.xlabel)

def set_ylabel(lbl):
    ax = gca()
    ax.set_ylabel(lbl, fontproperties=FONTS.xlabel)

def plot(*args, **kwargs):
    return plt.plot(*args, **kwargs)

def plot2(x_data, 
          y_data, 
          marker,
          x_label,
          y_label,
          title_pref,
          *args,
          **kwargs):
    do_plot = True
    ax = gca()
    if len(x_data) != len(y_data):
        warnstr = '[df2] ! Warning:  len(x_data) != len(y_data). Cannot plot2'
        warnings.warn(warnstr)
        draw_text(warnstr)
        do_plot = False
    if len(x_data) == 0:
        warnstr = '[df2] ! Warning:  len(x_data) == 0. Cannot plot2'
        warnings.warn(warnstr)
        draw_text(warnstr)
        do_plot = False
    if do_plot:
        ax.plot(x_data, y_data, marker, *args, **kwargs)
    ax.set_xlabel(x_label, fontproperties=FONTS.xlabel)
    ax.set_ylabel(y_label, fontproperties=FONTS.xlabel)
    ax.set_title(title_pref + ' ' + x_label+' vs '+y_label,
                 fontproperties=FONTS.axtitle)


def adjust_subplots_xylabels():
    adjust_subplots(left=.03, right=1, bottom=.1, top=.9, hspace=.15)
    
def adjust_subplots(left=0.02,  bottom=0.02,
                   right=0.98,     top=0.90, 
                   wspace=0.1,   hspace=0.15):
    '''
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2
    '''
    #print('[df2] adjust_subplots(%r)' % locals())
    plt.subplots_adjust(left,   bottom, right,  top, wspace, hspace)

def upperleft_text(txt):
    txtargs = dict(horizontalalignment='left',
                   verticalalignment='top',
                   #fontsize='smaller',
                   #fontweight='ultralight', 
                   backgroundcolor=(0,0,0,.5),
                   color=ORANGE)
    ax_relative_text(.02, .02, txt, **txtargs)

def upperright_text(txt):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   #fontsize='smaller',
                   #fontweight='ultralight', 
                   backgroundcolor=(0,0,0,.5),
                   color=ORANGE)
    ax_relative_text(.98, .02, txt, **txtargs)

def ax_relative_text(x, y, txt, ax=None, **kwargs):
    if ax is None: ax = gca()
    xy, width, height = _axis_xy_width_height(ax)
    x_, y_ = ((xy[0])+x*width, (xy[1]+height)-y*height)
    ax_absolute_text(x_, y_, txt, ax=ax, **kwargs)

def ax_absolute_text(x_, y_, txt, ax=None, **kwargs):
    if ax is None: ax = gca()
    if not kwargs.has_key('fontproperties'):
        kwargs['fontproperties'] = FONTS.relative
    ax.text(x_, y_, txt, **kwargs)

def fig_relative_text(x, y, txt, **kwargs):
    kwargs['horizontalalignment'] = 'center'
    kwargs['verticalalignment'] = 'center'
    fig = gcf()
    #xy, width, height = _axis_xy_width_height(ax)
    #x_, y_ = ((xy[0]+width)+x*width, (xy[1]+height)-y*height)
    fig.text(x, y, txt, **kwargs)

def set_figtitle(figtitle, subtitle=''):
    fig = gcf()
    if subtitle != '':
        subtitle = '\n'+subtitle
    fig.suptitle(figtitle+subtitle, fontsize=14, fontweight='bold')
    fig.suptitle(figtitle, x=.5, y=.98, fontproperties=FONTS.figtitle)
    fig_relative_text(.5, .95, subtitle, fontproperties=FONTS.subtitle)
    fig.canvas.set_window_title(figtitle)
    adjust_subplots()

def customize_figure(fig, doclf):
    if not 'user_stat_list' in fig.__dict__.keys() or doclf:
        fig.user_stat_list = []
        fig.user_notes = []
    fig.df2_closed = False

def gcf():
    if plotWidget is not None:
        #print('is plotwidget visible = %r' % plotWidget.isVisible())
        fig = plotWidget.figure
        return fig
    return plt.gcf()

def gca():
    if plotWidget is not None:
        #print('is plotwidget visible = %r' % plotWidget.isVisible())
        axes_list = plotWidget.figure.get_axes()
        current = 0
        ax = axes_list[current]
        return ax
    return plt.gca()

def cla():
    return plt.cla()

def clf():
    return plt.clf()

def get_fig(fignum=None):
    printDBG('[df2] get_fig(fignum=%r)' % fignum)
    fig_kwargs = dict(figsize=FIGSIZE, dpi=DPI)
    if plotWidget is not None:
        return gcf()
    if fignum is None:
        try: 
            fig = gcf()
        except Exception as ex:
            printDBG('[df2] get_fig(): ex=%r' % ex)
            fig = plt.figure(**fig_kwargs)
        fignum = fig.number
    else:
        try:
            fig = plt.figure(fignum, **fig_kwargs)
        except Exception as ex:
            print(repr(ex))
            warnings.warn(repr(ex))
            fig = gcf()
    return fig

def get_ax(fignum=None, plotnum=None):
    figure(fignum=fignum, plotnum=plotnum)
    ax = gca()
    return ax

def figure(fignum=None,
           doclf=False,
           title=None,
           plotnum=(1,1,1),
           figtitle=None,
           **kwargs):
    #matplotlib.pyplot.xkcd() 
    fig = get_fig(fignum)
    axes_list = fig.get_axes()
    # Ensure my customized settings
    customize_figure(fig, doclf)
    # Convert plotnum to tuple format
    if type(plotnum) == types.IntType:
        nr = plotnum // 100
        nc = plotnum // 10 - (nr * 10)
        px = plotnum - (nr * 100) - (nc * 10)
        plotnum = (nr, nc, px)
    # Get the subplot
    if doclf or len(axes_list) == 0:
        printDBG('[df2] *** NEW FIGURE '+str(fignum)+'.'+str(plotnum)+' ***')
        if not plotnum is None:
            #ax = plt.subplot(*plotnum)
            ax = fig.add_subplot(*plotnum)
            ax.cla()
        else:
            ax = gca()
    else: 
        printDBG('[df2] *** OLD FIGURE '+str(fignum)+'.'+str(plotnum)+' ***')
        if not plotnum is None:
            ax = plt.subplot(*plotnum) # fig.add_subplot fails here
            #ax = fig.add_subplot(*plotnum)
        else:
            ax = gca()
        #ax  = axes_list[0]
    # Set the title
    if not title is None:
        ax = gca()
        ax.set_title(title, fontproperties=FONTS.axtitle)
        # Add title to figure
        if figtitle is None and plotnum == (1,1,1):
            figtitle = title
        if not figtitle is None:
            fig.canvas.set_window_title('fig '+repr(fignum)+' '+figtitle)
    return fig

# GRAVEYARD
def update_figure_size(fignum, width, height):
    fig = get_fig(fignum)
    set_geometry(fig, 40, 40, width, height)
    fig.canvas.draw()

def draw_pdf(data, draw_support=True, scale_to=None, label=None, colorx=0):
    fig = gcf()
    ax = gca()
    data = np.array(data)
    if len(data) == 0:
        warnstr = '[df2] ! Warning: len(data) = 0. Cannot visualize pdf'
        warnings.warn(warnstr)
        draw_text(warnstr)
        return
    bw_factor = .05
    line_color = plt.get_cmap('gist_rainbow')(colorx)
    # Estimate a pdf
    data_pdf = estimate_pdf(data, bw_factor)
    # Get probability of seen data
    prob_x = data_pdf(data)
    # Get probability of unseen data data
    x_data = np.linspace(0, data.max(), 500)
    y_data = data_pdf(x_data)
    # Scale if requested
    if not scale_to is None:
        scale_factor = scale_to / y_data.max()
        y_data *= scale_factor
        prob_x *= scale_factor
    #Plot the actual datas on near the bottom perterbed in Y
    if draw_support:
        pdfrange = prob_x.max() - prob_x.min() 
        perb   = (np.random.randn(len(data))) * pdfrange/30.
        preb_y_data = np.abs([pdfrange/50. for _ in data]+perb)
        ax.plot(data, preb_y_data, 'o', color=line_color, figure=fig, alpha=.1)
    # Plot the pdf (unseen data)
    ax.plot(x_data, y_data, color=line_color, label=label)
    
def estimate_pdf(data, bw_factor):
    try:
        data_pdf = scipy.stats.gaussian_kde(data, bw_factor)
        data_pdf.covariance_factor = bw_factor
    except Exception as ex:
        print('[df2] ! Exception while estimating kernel density')
        print('[df2] data=%r' % (data,))
        print('[df2] ex=%r' % (ex,))
        raise
    return data_pdf
    
def show_histogram(data, bins=None, **kwargs):
    sys.stdout.write('[df2] show_histogram()\n')
    dmin = int(np.floor(data.min()))
    dmax = int(np.ceil(data.max()))
    if bins is None:
        bins = dmax - dmin
    fig = figure(**kwargs)
    ax  = gca()
    ax.hist(data, bins=bins, range=(dmin,dmax))
    #help(np.bincount)
    fig.show()

def show_signature(sig, **kwargs):
    fig = figure(**kwargs)
    plt.plot(sig)
    fig.show()

def draw_stems(x_data, y_data):
    if len(x_data) != len(y_data):
        print('[df2] WARNING draw_stems(): len(x_data)!=len(y_data)')
    if len(x_data) == 0: 
        print('[df2] WARNING draw_stems(): len(x_data)=len(y_data)=0')
    x_data_ = np.array(x_data)
    y_data_ = np.array(y_data)
    x_data_sort = x_data_[y_data_.argsort()[::-1]]
    y_data_sort = y_data_[y_data_.argsort()[::-1]]

    markerline, stemlines, baseline = pylab.stem(x_data_sort, y_data_sort, linefmt='-')
    pylab.setp(markerline, 'markerfacecolor', 'b')
    pylab.setp(baseline, 'linewidth', 0)
    ax = gca()
    ax.set_xlim(min(x_data)-1, max(x_data)+1)
    ax.set_ylim(min(y_data)-1, max(max(y_data), max(x_data))+1)

def legend():
    ax = gca()
    ax.legend(prop=FONTS.legend)

def draw_histpdf(data, label=None, draw_support=False, nbins=10):
    freq, _ = draw_hist(data, nbins=nbins)
    draw_pdf(data, draw_support=draw_support, scale_to=freq.max(), label=label)

def draw_hist(data, bins=None, nbins=10, weights=None):
    if type(data) == types.ListType:
        data = np.array(data)
    if bins is None:
        dmin = data.min()
        dmax = data.max()
        bins = dmax - dmin
    ax  = gca()
    freq, bins_, patches = ax.hist(data, bins=nbins, weights=weights, range=(dmin,dmax))
    return freq, bins_
    
def variation_trunctate(data):
    ax = gca()
    data = np.array(data)
    if len(data) == 0: 
        warnstr = '[df2] ! Warning: len(data) = 0. Cannot variation_truncate'
        warnings.warn(warnstr)
        return
    trunc_max = data.mean() + data.std() * 2
    trunc_min = np.floor(data.min())
    ax.set_xlim(trunc_min,trunc_max)
    #trunc_xticks = np.linspace(0, int(trunc_max),11)
    #trunc_xticks = trunc_xticks[trunc_xticks >= trunc_min]
    #trunc_xticks = np.append([int(trunc_min)], trunc_xticks)
    #no_zero_yticks = ax.get_yticks()[ax.get_yticks() > 0]
    #ax.set_xticks(trunc_xticks)
    #ax.set_yticks(no_zero_yticks)

def draw_text(text_str, rgb_textFG=(0,0,0), rgb_textBG=(1,1,1)):
    ax = gca()
    xy, width, height = _axis_xy_width_height(ax)
    text_x = xy[0] + (width / 2)
    text_y = xy[1] + (height / 2)
    ax.text(text_x, text_y, text_str,
            horizontalalignment ='center',
            verticalalignment   ='center',
            color               =rgb_textFG,
            backgroundcolor     =rgb_textBG)

#_----------------- HELPERS ^^^ ---------

DISTINCT_COLORS = True #and False
DARKEN = None
ELL_LINEWIDTH = 1.5
if DISTINCT_COLORS: 
    ELL_ALPHA  = .6
    LINE_ALPHA = .35
else:
    ELL_ALPHA  = .4
    LINE_ALPHA = .4
ELL_COLOR  = BLUE

LINE_COLOR = RED
LINE_WIDTH = 1.4

SHOW_LINES = True #True
SHOW_ELLS  = True

POINT_SIZE = 2

def my_prefs():
    global LINE_COLOR
    global ELL_COLOR
    global ELL_LINEWIDTH
    global ELL_ALPHA
    LINE_COLOR = (1, 0, 0)
    ELL_COLOR = (0, 0, 1)
    ELL_LINEWIDTH = 2
    ELL_ALPHA = .5

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
    ax = gca()
    tau = 2*np.pi
    DSCALE = .25
    XYSCALE = .5
    XYSHIFT = -.75
    ORI_SHIFT = 0 # -tau #1/8 * tau
    # SIFT CONSTANTS
    NORIENTS = 8; NX = 4; NY = 4; NBINS = NX * NY
    def cirlce_rad2xy(radians, mag):
        return np.cos(radians)*mag, np.sin(radians)*mag
    discrete_ori = (np.arange(0,NORIENTS)*(tau/NORIENTS) + ORI_SHIFT)
    # Build list of plot positions
    # Build an "arm" for each sift measurement
    arm_mag   = desc / 255.0
    arm_ori = np.tile(discrete_ori, (NBINS, 1)).flatten()
    # The offset x,y's for each sift measurment
    arm_dxy = np.array(zip(*cirlce_rad2xy(arm_ori, arm_mag))) 
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
    #a_ = 1/a
    #b_ = (-c)/(a*d)
    #d_ = 1/d
    #a_ = 1/np.sqrt(a) 
    #b_ = c/(-np.sqrt(a)*d - a*np.sqrt(d))
    #d_ = 1/np.sqrt(d)
    transMat = [( a, 0, x),
                ( c, d, y),
                ( 0, 0, 1)]
    kpTrans = Affine2D(transMat)
    axTrans = ax.transData
    #print('\ntranform=%r ' % transform)
    # Draw Arms
    arrow_patches = []
    arrow_patches2 = []
    #print(index)
    #print((x, y, t))
    #index = 127 - ((NY - 1 - y)*(NX*NORIENTS) + (NX - 1 - x)*(NORIENTS) + (NORIENTS - 1 - t))
    #index = ((NY - 1 - y)*(NX*NORIENTS) + (NX - 1 - x)*(NORIENTS) + (t))
    #index = ((x)*(NY*NORIENTS) + (y)*(NORIENTS) + (t))
    for y,x,t in yxt_gen:
        index = y*NX*NORIENTS + x*NORIENTS + t
        (dx, dy) = arm_dxy[index]
        arw_x  = x*XYSCALE + XYSHIFT
        arw_y  = y*XYSCALE + XYSHIFT
        arw_dy = dy*DSCALE * 1.5 # scale for viz Hack
        arw_dx = dx*DSCALE * 1.5
        posA = (arw_x, arw_y)
        posB = (arw_x+arw_dx, arw_y+arw_dy)
        _args = [arw_x, arw_y, arw_dx, arw_dy]
        _kwargs = dict(head_width=.0001, transform=kpTrans, length_includes_head=False)
        arrow_patches  += [FancyArrow(*_args, **_kwargs)]
        arrow_patches2 += [FancyArrow(*_args, **_kwargs)]
    # Draw Circles
    circle_patches = []
    for y,x in yx_gen:
        circ_xy = (x*XYSCALE + XYSHIFT, y*XYSCALE + XYSHIFT)
        circ_radius = DSCALE
        circle_patches += [Circle(circ_xy, circ_radius, transform=kpTrans)]
        
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

def feat_scores_to_color(fs, cmap_='hot'):
    assert len(fs.shape) == 1, 'score must be 1d'
    cmap = plt.get_cmap(cmap_)
    mins = fs.min()
    rnge = fs.max() - mins
    if rnge == 0:
        return [cmap(.5) for fx in xrange(len(fs))]
    score2_01 = lambda score: .1+.9*(float(score)-mins)/(rnge)
    colors    = [cmap(score2_01(fs[fx])) for fx in xrange(len(fs))]
    return colors

def draw_matches2(kpts1, kpts2, fm=None, fs=None, kpts2_offset=(0,0),
                  color_list=None):
    if not DISTINCT_COLORS:
        color_list = None
    # input data
    if not SHOW_LINES:
        return 
    if fm is None: # assume kpts are in director correspondence
        assert kpts1.shape == kpts2.shape
    if len(fm) == 0: 
        return
    ax = gca()
    woff, hoff = kpts2_offset
    # Draw line collection
    kpts1_m = kpts1[fm[:,0]].T
    kpts2_m = kpts2[fm[:,1]].T
    xxyy_iter = iter(zip(kpts1_m[0],
                         kpts2_m[0]+woff,
                         kpts1_m[1],
                         kpts2_m[1]+hoff))
    if color_list is None:
        if fs is None: # Draw with solid color
            color_list    = [ LINE_COLOR for fx in xrange(len(fm)) ] 
        else: # Draw with colors proportional to score difference
            color_list = feat_scores_to_color(fs)
    segments  = [((x1, y1), (x2,y2)) for (x1,x2,y1,y2) in xxyy_iter] 
    linewidth = [LINE_WIDTH for fx in xrange(len(fm)) ] 
    line_group = LineCollection(segments, linewidth, color_list, alpha=LINE_ALPHA)
    ax.add_collection(line_group)

def draw_kpts2(kpts, offset=(0,0),
               ell=SHOW_ELLS, 
               pts=False, 
               pts_color=ORANGE, 
               pts_size=POINT_SIZE, 
               ell_alpha=ELL_ALPHA,
               ell_linewidth=ELL_LINEWIDTH,
               ell_color=ELL_COLOR,
               color_list=None,
               wrong_way=False,
               rect=None):
    if not DISTINCT_COLORS:
        color_list = None
    printDBG('drawkpts2: Drawing Keypoints! ell=%r pts=%r' % (ell, pts))
    # get matplotlib info
    ax = gca()
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
    if rect is None:
        rect = ell
        rect = False
        if pts is True:
            rect = False
    if ell or rect:
        printDBG('[df2] draw_kpts() drawing ell kptsT.shape=%r' % (kptsT.shape,))
        a = kptsT[2]
        b = np.zeros(len(a))
        c = kptsT[3]
        d = kptsT[4]
        # Sympy Calculated sqrtm(inv(A) for A in kpts)
        # inv(sqrtm([(a, 0), (c, d)]) = 
        #  [1/sqrt(a), c/(-sqrt(a)*d - a*sqrt(d))]
        #  [        0,                  1/sqrt(d)]
        if wrong_way:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aIS = 1/np.sqrt(a) 
                bIS = c/(-np.sqrt(a)*d - a*np.sqrt(d))
                dIS = 1/np.sqrt(d)
                cIS = b
                #cIS = (c/np.sqrt(d) - c/np.sqrt(d)) / (a-d+eps)
        else:
            aIS = a
            bIS = b
            cIS = c
            dIS = d
            # Just inverse
            #aIS = 1/a 
            #bIS = -c/(a*d)
            #dIS = 1/d

        kpts_iter = izip(x,y,aIS,bIS,cIS,dIS)
        aff_list = [Affine2D([( a_, b_, x_),
                              ( c_, d_, y_),
                              ( 0 , 0 , 1)])
                    for (x_,y_,a_,b_,c_,d_) in kpts_iter]
        patch_list = []
        ell_actors = [Circle( (0,0), 1, transform=aff) for aff in aff_list]
        if ell:
            patch_list += ell_actors
        if rect:
            rect_actors = [Rectangle( (-1,-1), 2, 2, transform=aff) for aff in aff_list]
            patch_list += rect_actors
        ellipse_collection = matplotlib.collections.PatchCollection(patch_list)
        ellipse_collection.set_facecolor('none')
        ellipse_collection.set_transform(pltTrans)
        ellipse_collection.set_alpha(ell_alpha)
        ellipse_collection.set_linewidth(ell_linewidth)
        if not color_list is None: 
            ell_color = color_list
        ellipse_collection.set_edgecolor(ell_color)
        ax.add_collection(ellipse_collection)
    if pts:
        printDBG('[df2] draw_kpts() drawing pts x.shape=%r y.shape=%r' % (x.shape, y.shape))
        if color_list is None:
            color_list = [pts_color for _ in xrange(len(x))]
        ax.autoscale(enable=False)
        ax.scatter(x, y, c=color_list, s=2*pts_size, marker='o', edgecolor='none')
        #ax.autoscale(enable=False)
        #ax.plot(x, y, linestyle='None', marker='o', markerfacecolor=pts_color, markersize=pts_size, markeredgewidth=0)

# ---- CHIP DISPLAY COMMANDS ----
def imshow(img, 
           fignum=None,
           title=None, 
           figtitle=None, 
           plotnum=None,
           interpolation='nearest', 
           **kwargs):
    'other interpolations = nearest, bicubic, bilinear'
    #printDBG('[df2] ----- IMSHOW ------ ')
    #printDBG('[df2] *** imshow in fig=%r title=%r *** ' % (fignum, title))
    #printDBG('[df2] *** fignum = %r, plotnum = %r ' % (fignum, plotnum))
    #printDBG('[df2] *** img.shape = %r ' % (img.shape,))
    #printDBG('[df2] *** img.stats = %r ' % (helpers.printable_mystats(img),))
    fig = figure(fignum=fignum, plotnum=plotnum, title=title, figtitle=figtitle, **kwargs)
    ax = gca()
    if not DARKEN is None:
        imgdtype = img.dtype
        img = np.array(img, dtype=float) * DARKEN
        img = np.array(img, dtype=imgdtype) 
    ax.imshow(img, interpolation=interpolation)
    #plt.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_autoscale(False)
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
    nMatches = len(fm)
    upperleft_text('#match=%d' % nMatches)
    if all_kpts:
        # Draw all keypoints as simple points
        all_args = dict(ell=False, pts=draw_pts, pts_color=GREEN, pts_size=2, ell_alpha=ell_alpha)
        draw_kpts2(kpts1, **all_args)
        draw_kpts2(kpts2, offset=(woff,hoff), **all_args) 
    if nMatches == 0:
        printDBG('[df2] There are no feature matches to plot!')
    else:
        #color_list = [((x)/nMatches,1-((x)/nMatches),0) for x in xrange(nMatches)]
        #cmap = lambda x: (x, 1-x, 0)
        cmap = plt.get_cmap('prism')
        #color_list = [cmap(mx/nMatches) for mx in xrange(nMatches)]
        colors = distinct_colors(nMatches)
        pt2_args = dict(pts=draw_pts, ell=False, pts_color=BLACK, pts_size=8)
        pts_args = dict(pts=draw_pts, ell=False, pts_color=ORANGE, pts_size=6,
                        color_list=add_alpha(colors))
        ell_args = dict(ell=draw_ell, pts=False, color_list=colors)
        # Draw matching ellipses
        offset=(woff,hoff)
        def _drawkpts(**kwargs):
            draw_kpts2(kpts1[fm[:,0]], **kwargs)
            draw_kpts2(kpts2[fm[:,1]], offset=offset, **kwargs)
        def _drawlines(**kwargs):
            draw_matches2(kpts1, kpts2, fm, fs, kpts2_offset=offset, **kwargs)
        # Draw matching lines
        if draw_ell:
            _drawkpts(**ell_args)
        if draw_lines:
            _drawlines(color_list=colors)
        if draw_pts: 
            #_drawkpts(**pts_args)
            acolors = add_alpha(colors)
            pts_args.update(dict(pts_size=6, color_list=acolors))
            _drawkpts(**pt2_args)
            _drawkpts(**pts_args)

    return fig, ax

def deterministic_shuffle(list_):
    randS = int(np.random.rand()*np.uint(0-2)/2)
    np.random.seed(len(list_))
    np.random.shuffle(list_)
    np.random.seed(randS)

def distinct_colors(N):
    # http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html
    import colorsys
    sat = .878
    val = .878
    HSV_tuples = [(x*1.0/N, sat, val) for x in xrange(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    deterministic_shuffle(RGB_tuples)
    return RGB_tuples

def add_alpha(colors):
    return [list(color)+[1] for color in colors]

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

# TODO: This should go in viz
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
        imshow(nan_img, fignum=fignum, plotnum=plotnum, title=title)
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
    fig, ax = show_matches2(rchip1, rchip2, kpts1, kpts2, fm, fs, 
                            fignum=fignum, plotnum=plotnum,
                            title=title, **kwargs)
    upperright_text(hs.vs_str(qcx, cx))
    # Finish annotations
    if   isgt_str == hs.UNKNOWN_STR: draw_border(ax, WHITE, 4)
    elif isgt_str == hs.TRUE_STR:    draw_border(ax, GREEN, 4)
    elif isgt_str == hs.FALSE_STR:   draw_border(ax, RED, 4)
    if show_gname:
        ax.set_xlabel(hs.cx2_gname(cx), fontproperties=FONTS.xlabel)
    return ax

def show_img(hs, cx, **kwargs):
    # Get the chip roi
    roi = hs.get_roi(cx)
    # Get the image
    img = hs.get_image(cx=cx)
    # Draw image
    imshow(img, **kwargs)
    # Draw ROI
    ax = gca()
    draw_roi(ax, roi)

def draw_roi(ax, roi, label=None, bbox_color=(1,0,0)):
    (rx,ry,rw,rh) = roi
    rxy = (rx,ry)
    bbox = matplotlib.patches.Rectangle(rxy,rw,rh) 
    bbox.set_fill(False)
    bbox.set_edgecolor(bbox_color)
    ax.add_patch(bbox)
    if label is not None:
        ax_absolute_text(rx, ry, label, ax=ax,
                horizontalalignment ='center',
                verticalalignment   ='center',
                color               =(1,1,1),
                backgroundcolor     =(0,0,0))

def show_keypoints(rchip,kpts,fignum=0,title=None, **kwargs):
    imshow(rchip,fignum=fignum,title=title,**kwargs)
    draw_kpts2(kpts)

def show_chip(hs, cx=None, allres=None, res=None, info=True, draw_kpts=True,
              nRandKpts=None, kpts_alpha=None, prefix='', **kwargs):
    if not res is None:
        cx = res.qcx
    if not allres is None:
        res = allres.qcx2_res[cx]
    rchip1    = hs.get_chip(cx)
    title_str = prefix + hs.cxstr(cx)
    # Add info to title
    if info: 
        title_str += ', '+hs.num_indexed_gt_str(cx)
    fig, ax = imshow(rchip1, title=title_str, **kwargs)
    if not res is None: 
        gname = hs.cx2_gname(cx)
        ax.set_xlabel(gname, fontproperties=FONTS.xlabel)
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
            draw_kpts2(kpts_, ell_color=color, ell_alpha=alpha, **kpts_args)
            phant_ = Circle((0, 0), 1, fc=color)
            legend_tups.append((phant_, label))
        _kpts_helper(kpts_noise,   RED, .1, 'Unverified')
        _kpts_helper(kpts_match,  BLUE, .4, 'Verified')
        _kpts_helper(kpts_true,  GREEN, .6, 'True Matches')
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
            ax = gca()
            ax.set_xlabel('displaying %r/%r keypoints' % (nRandKpts, nkpts1), fontproperties=FONTS.xlabel)
            # show a random sample of kpts
        draw_kpts2(kpts1, ell_alpha=kpts_alpha, ell_color=RED, **kpts_args)

def show_topN_matches(hs, res, N=5, fignum=4): 
    figtitle = ('q%s -- TOP %r' % (hs.cxstr(res.qcx), N))
    topN_cxs = res.topN_cxs(N)
    max_nCols = max(5,N)
    _show_chip_matches(hs, res, topN_cxs=topN_cxs, figtitle=figtitle, 
                       fignum=fignum, all_kpts=False)

def show_gt_matches(hs, res, fignum=3): 
    figtitle = ('q%s -- GroundTruth' % (hs.cxstr(res.qcx)))
    gt_cxs = hs.get_other_indexed_cxs(res.qcx)
    max_nCols = max(5,len(gt_cxs))
    _show_chip_matches(hs, res, gt_cxs=gt_cxs, figtitle=figtitle, 
                       fignum=fignum, all_kpts=True)

def show_match_analysis(hs, res, N=5, fignum=3, figtitle='', show_query=None,
                        annotations=True, compare_cxs=None, q_cfg=None, **kwargs):
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
    return _show_chip_matches(hs, res,
                              gt_cxs=missed_gt_cxs, 
                              topN_cxs=topN_cxs,
                              figtitle=figtitle,
                              max_nCols=max_nCols,
                              show_query=show_query,
                              fignum=fignum,
                              annotations=annotations,
                              q_cfg=q_cfg,
                              **kwargs)

def _show_chip_matches(hs, res, figtitle='', max_nCols=5,
                       topN_cxs=None, gt_cxs=None, show_query=False,
                       all_kpts=False, fignum=3, annotations=True, q_cfg=None,
                       split_plots=False, **kwargs):
    ''' Displays query chip, groundtruth matches, and top 5 matches'''
    #print('========================')
    #print('[df2] Show chip matches:')
    if topN_cxs is None: topN_cxs = []
    if gt_cxs is None: gt_cxs = []
    print('[df2]----------------')
    print('[df2] #top=%r #missed_gts=%r' % (len(topN_cxs),len(gt_cxs)))
    print('[df2] * max_nCols=%r' % (max_nCols,))
    print('[df2] * show_query=%r' % (show_query,))
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
    nTopNCells = nTopNRows * nTopNCols
    if split_plots:
        nRows = nGtRows
    else:
        nRows = nTopNRows+nGtRows
    # Helper function for drawing matches to one cx
    def show_matches_(cx, orank, plotnum):
        aug = 'rank=%r\n' % orank
        printDBG('[df2] plotting: %r'  % (plotnum,))
        kwshow  = dict(draw_ell=annote, draw_pts=annote, draw_lines=annote,
                       ell_alpha=.5, all_kpts=all_kpts, **kwargs)
        show_matches_annote_res(res, hs, cx, title_aug=aug, plotnum=plotnum, **kwshow)
    def plot_query(plotx_shift, rowcols):
        printDBG('Plotting Query:')
        plotx = plotx_shift + 1
        plotnum = (rowcols[0], rowcols[1], plotx)
        printDBG('[df2] plotting: %r' % (plotnum,))
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

    fig = figure(fignum=fignum); fig.clf()
    plt.subplot(nRows, nGTCols, 1)
    # Plot Query
    if show_query: 
        plot_query(0, (nRows, nGTCols))
    # Plot Ground Truth
    plot_matches_cxs(gt_cxs, nQuerySubplts, (nRows, nGTCols)) 
    # Plot TopN in a new figure
    if split_plots:
        set_figtitle(figtitle+'GT', query_uid)
        nRows = nTopNRows
        fig = figure(fignum=fignum+9000); fig.clf()
        plt.subplot(nRows, nTopNCols, 1)
        shift_topN = 0
    else:
        shift_topN = nGtCells
    plot_matches_cxs(topN_cxs, shift_topN, (nRows, nTopNCols))
    if split_plots:
        set_figtitle(figtitle+'topN', query_uid)
    else:
        set_figtitle(figtitle, query_uid)
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
