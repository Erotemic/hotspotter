from __future__ import division, print_function
DEBUG = False
def printDBG(msg):
    if DEBUG:
        print(msg)
import matplotlib
import multiprocessing
MPL_BACKEND = matplotlib.get_backend()
matplotlib.rcParams['toolbar'] = 'toolbar2'
if MPL_BACKEND != 'Qt4Agg':
    if multiprocessing.current_process().name == 'MainProcess':
        printDBG('[df2] current backend is: %r' % MPL_BACKEND)
        printDBG('[df2] matplotlib.use(Qt4Agg)')
    matplotlib.use('Qt4Agg', warn=True, force=True)
    MPL_BACKEND = matplotlib.get_backend()
    if multiprocessing.current_process().name == 'MainProcess':
        printDBG('[df2] current backend is: %r' % MPL_BACKEND)
    #matplotlib.rc('text', usetex=True)
    #matplotlib.rcParams['toolbar'] = 'None'
    #matplotlib.rcParams['interactive'] = True
import matplotlib.pyplot as plt
from PyQt4.QtCore import Qt
import numpy as np
import sys
import os
import time
import types
import textwrap

ORANGE = np.array((255, 127,   0, 255))/255.0
RED    = np.array((255,   0,   0, 255))/255.0
GREEN  = np.array((  0, 255,   0, 255))/255.0
BLUE   = np.array((  0,   0, 255, 255))/255.0
BLACK  = np.array((  0,   0,  0, 255))/255.0
WHITE  = np.array((255,   255, 255, 255))/255.0

DPI = 80
#FIGSIZE = (24) # default windows fullscreen
FIGSIZE_MED = (20,10) 
FIGSIZE_BIG = (24,12) 

FIGSIZE = FIGSIZE_BIG 

ELL_LINEWIDTH = 1.5
ELL_ALPHA  = .4
ELL_COLOR  = BLUE

LINE_ALPHA = .4
LINE_COLOR = RED
LINE_CMAP  = 'hot'
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

try:
    if sys.platform == 'win32':
        compname = os.environ['COMPUTER_NAME']
        if compname == 'Ooo':
            TILE_WITHIN = (-1912, 30, -969, 1071)
except KeyError:
    TILE_WITHIN = (0, 30, 969, 1041)

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
    for fig in iter(get_all_figures()):
        time.sleep(.1)
        fig.show()
        fig.canvas.draw()

def all_figures_tight_layout():
    for fig in iter(get_all_figures()):
        fig.tight_layout()
        #adjust_subplots()
        time.sleep(.1)

def all_figures_tile(num_rc=(4,4),
                     wh=(350,250),
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
    print('[df2] Presenting figures...')
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
            print('[df2] Requested IPython shell with --cmd argument.')
            if helpers.haveIPython():
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
    fname_clean = os.path.split(fpath_clean)[1]
    print('[df2] save_figure() %r' % (fpath_clean,))
    #adjust_subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        fig.savefig(fpath_clean, dpi=DPI)

def set_ticks(xticks, yticks):
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

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

def plot2(x_data, 
          y_data, 
          marker,
          x_label,
          y_label,
          title_pref,
          *args,
          **kwargs):
    do_plot = True
    ax = plt.gca()
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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_pref + ' ' + x_label+' vs '+y_label)


def adjust_subplots_xylabels():
    adjust_subplots(left=.03, right=1, bottom=.1, top=.9, hspace=.01)
    
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
    printDBG('[df2] adjust_subplots(%r)' % locals())
    plt.subplots_adjust(left,   bottom, right,  top, wspace, hspace)

def upperleft_text(txt):
    txtargs = dict(horizontalalignment='left',
                   verticalalignment='top',
                   fontsize='smaller',
                   fontweight='ultralight', 
                   backgroundcolor=(0,0,0,.5),
                   color=ORANGE)
    ax_relative_text(.02, .02, txt, **txtargs)

def upperright_text(txt):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   fontsize='smaller',
                   fontweight='ultralight', 
                   backgroundcolor=(0,0,0,.5),
                   color=ORANGE)
    ax_relative_text(.98, .02, txt, **txtargs)

def ax_relative_text(x, y, txt, **kwargs):
    ax = plt.gca()
    xy, width, height = _axis_xy_width_height(ax)
    x_, y_ = ((xy[0])+x*width, (xy[1]+height)-y*height)
    ax.text(x_, y_, txt, **kwargs)

def fig_relative_text(x, y, txt, **kwargs):
    kwargs['horizontalalignment'] = 'center'
    kwargs['verticalalignment'] = 'center'
    fig = plt.gcf()
    #xy, width, height = _axis_xy_width_height(ax)
    #x_, y_ = ((xy[0]+width)+x*width, (xy[1]+height)-y*height)
    fig.text(x, y, txt, **kwargs)

def set_figtitle(figtitle, subtitle=''):
    fig = plt.gcf()
    if subtitle != '':
        subtitle = '\n'+subtitle
    #fig.suptitle(figtitle+subtitle, fontsize=14, fontweight='bold')
    fig.suptitle(figtitle, x=.5, y=.98, fontsize=14, fontweight='bold')
    fig_relative_text(.5, .95, subtitle, fontsize=12)
    fig.canvas.set_window_title(figtitle)
    adjust_subplots()

def customize_figure(fig, doclf):
    if not 'user_stat_list' in fig.__dict__.keys() or doclf:
        fig.user_stat_list = []
        fig.user_notes = []
    fig.df2_closed = False

def get_fig(fignum=None):
    printDBG('[df2] get_fig(fignum=%r)' % fignum)
    fig_kwargs = dict(figsize=FIGSIZE, dpi=DPI)
    if fignum is None:
        try: 
            fig = plt.gcf()
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
            fig = plt.gcf()
    return fig

def get_ax(fignum=None, plotnum=None):
    figure(fignum=fignum, plotnum=plotnum)
    ax = plt.gca()
    return ax

def figure(fignum=None,
           doclf=False,
           title=None,
           plotnum=(1,1,1),
           figtitle=None,
           **kwargs):
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
            ax = plt.subplot(*plotnum)
            ax.cla()
        else:
            ax = plt.gca()
    else: 
        printDBG('[df2] *** OLD FIGURE '+str(fignum)+'.'+str(plotnum)+' ***')
        if not plotnum is None:
            ax = plt.subplot(*plotnum)
        else:
            ax = plt.gca()
        #ax  = axes_list[0]
    # Set the title
    if not title is None:
        ax = plt.gca()
        ax.set_title(title)
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
    fig = plt.gcf()
    ax = plt.gca()
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

def legend():
    ax = plt.gca()
    ax.legend(**{'fontsize':18})

def draw_histpdf(data, label=None):
    freq, _ = draw_hist(data)
    draw_pdf(data, draw_support=False, scale_to=freq.max(), label=label)

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
    ax = plt.gca()
    xy, width, height = _axis_xy_width_height(ax)
    text_x = xy[0] + (width / 2)
    text_y = xy[1] + (height / 2)
    ax.text(text_x, text_y, text_str,
            horizontalalignment ='center',
            verticalalignment   ='center',
            color               =rgb_textFG,
            backgroundcolor     =rgb_textBG)
