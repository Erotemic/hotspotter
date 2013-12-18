''' Lots of functions for drawing and plotting visiony things '''
from __future__ import division, print_function
import __builtin__
# Python
from itertools import izip
from os.path import splitext, split, join, normpath
import colorsys
import itertools
import pylab
import sys
import textwrap
import time
import warnings
import multiprocessing
# Matplotlib / Qt
import matplotlib
backend = matplotlib.get_backend()
if multiprocessing.current_process().name == 'MainProcess':
    print('[df2] current backend is: %r' % backend)
    print('[df2] matplotlib.use(Qt4Agg)')
    matplotlib.rcParams['toolbar'] = 'toolbar2'
    matplotlib.rc('text', usetex=False)
    #matplotlib.rcParams['text'].usetex = False
    if backend != 'Qt4Agg':
        matplotlib.use('Qt4Agg', warn=True, force=True)
        backend = matplotlib.get_backend()
        if multiprocessing.current_process().name == 'MainProcess':
            print('[*guitools] current backend is: %r' % backend)
        #matplotlib.rcParams['toolbar'] = 'None'
        #matplotlib.rcParams['interactive'] = True
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
# Qt
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
# Scientific
import numpy as np
import scipy.stats
# HotSpotter
import tools
from Printable import DynStruct
import helpers


def printDBG(msg):
    #print(msg)
    pass

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write


def print_on():
    global print, print_
    print  = __builtin__.print
    print_ = sys.stdout.write


def print_off():
    global print, print_

    def print(*args, **kwargs):
        pass

    def print_(*args, **kwargs):
        pass


def rrr():
    'Dynamic module reloading'
    import imp
    import sys
    print('[df2] reloading ' + __name__)
    imp.reload(sys.modules[__name__])


QT4_WINS = []
plotWidget = None


SMALL_FONTS = False
if SMALL_FONTS:
    SMALLER  = 7
    SMALL    = 7
    MED      = 8  # 14
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
FONTS.figtitle = FONTS.med
FONTS.axtitle  = FONTS.med
FONTS.subtitle = FONTS.med
FONTS.xlabel   = FONTS.small
FONTS.ylabel   = FONTS.small
FONTS.relative = FONTS.smaller

ORANGE = np.array((255, 127,   0, 255)) / 255.0
DARK_ORANGE = np.array((127, 63,   0, 255)) / 255.0
RED    = np.array((255,   0,   0, 255)) / 255.0
GREEN  = np.array((  0, 255,   0, 255)) / 255.0
BLUE   = np.array((  0,   0, 255, 255)) / 255.0
YELLOW = np.array((255, 255,   0, 255)) / 255.0
BLACK  = np.array((  0,   0,   0, 255)) / 255.0
WHITE  = np.array((255, 255, 255, 255)) / 255.0
GRAY   = np.array((127, 127, 127, 255)) / 255.0
DARK_PURP = np.array((102,  0, 153, 255)) / 255.0


DPI = 80
#FIGSIZE = (24) # default windows fullscreen
FIGSIZE_MED = (12, 6)
FIGSIZE_BIG = (24, 12)

FIGSIZE = FIGSIZE_MED

tile_within = (-1, 30, 969, 1041)
if helpers.get_computer_name() == 'Ooo':
    TILE_WITHIN = (-1912, 30, -969, 1071)

DISTINCT_COLORS = True  # and False
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

SHOW_LINES = True  # True
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
    execstr = ['global' + key for key in globals().keys()]
    return execstr


def register_matplotlib_widget(plotWidget_):
    'talks to PyQt4 guis'
    global plotWidget
    plotWidget = plotWidget_
    #fig = plotWidget.figure
    #axes_list = fig.get_axes()
    #ax = axes_list[0]
    #plt.sca(ax)


def register_qt4_win(win):
    global QT4_WINS
    QT4_WINS.append(win)


def OooScreen2():
    nRows = 1
    nCols = 1
    x_off = 30 * 4
    y_off = 30 * 4
    x_0 = -1920
    y_0 = 30
    w = (1912 - x_off) / nRows
    h = (1080 - y_off) / nCols
    return dict(num_rc=(1, 1), wh=(w, h), xy_off=(x_0, y_0), wh_off=(0, 10),
                row_first=True, no_tile=False)


def deterministic_shuffle(list_):
    randS = int(np.random.rand() * np.uint(0 - 2) / 2)
    np.random.seed(len(list_))
    np.random.shuffle(list_)
    np.random.seed(randS)


def distinct_colors(N, brightness=.878):
    # http://blog.jianhuashao.com/2011/09/generate-n-distinct-colors.html
    sat = brightness
    val = brightness
    HSV_tuples = [(x * 1.0 / N, sat, val) for x in xrange(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    deterministic_shuffle(RGB_tuples)
    return RGB_tuples


def add_alpha(colors):
    return [list(color) + [1] for color in colors]


def _axis_xy_width_height(ax, xaug=0, yaug=0, waug=0, haug=0):
    'gets geometry of a subplot'
    autoAxis = ax.axis()
    xy     = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width  = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height


def draw_border(ax, color=GREEN, lw=2, offset=None):
    'draws rectangle border around a subplot'
    xy, width, height = _axis_xy_width_height(ax, -.7, -.2, 1, .4)
    if offset is not None:
        xoff, yoff = offset
        xy = [xoff, yoff]
        height = - height - yoff
        width = width - xoff
    rect = matplotlib.patches.Rectangle(xy, width, height, lw=lw)
    rect = ax.add_patch(rect)
    rect.set_clip_on(False)
    rect.set_fill(False)
    rect.set_edgecolor(color)


def draw_roi(ax, roi, label=None, bbox_color=(1, 0, 0),
             lbl_bgcolor=(0, 0, 0), lbl_txtcolor=(1, 1, 1)):
    (rx, ry, rw, rh) = roi
    rxy = (rx, ry)
    bbox = matplotlib.patches.Rectangle(rxy, rw, rh, lw=2)
    bbox.set_fill(False)
    bbox.set_edgecolor(bbox_color)
    ax.add_patch(bbox)
    if label is not None:
        ax_absolute_text(rx, ry, label, ax=ax,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=lbl_txtcolor,
                         backgroundcolor=lbl_bgcolor)


# ---- GENERAL FIGURE COMMANDS ----
def sanatize_img_fname(fname):
    fname_clean = fname
    search_replace_list = [(' ', '_'), ('\n', '--'), ('\\', ''), ('/', '')]
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
    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
    return (x, y, w, h)


def get_all_figures():
    all_figures_ = [manager.canvas.figure for manager in
                    matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    all_figures = []
    # Make sure you dont show figures that this module closed
    for fig in iter(all_figures_):
        if not 'df2_closed' in fig.__dict__.keys() or not fig.df2_closed:
            all_figures.append(fig)
    # Return all the figures sorted by their number
    all_figures = sorted(all_figures, key=lambda fig: fig.number)
    return all_figures


def get_all_qt4_wins():
    return QT4_WINS


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


def golden_wh(x):
    'returns a width / height with a golden aspect ratio'
    return map(int, map(round, (x * .618, x * .312)))


def all_figures_tile(num_rc=(3, 4), wh=1000, xy_off=(0, 0), wh_off=(0, 10),
                     row_first=True, no_tile=False):
    'Lays out all figures in a grid. if wh is a scalar, a golden ratio is used'
    from matplotlib.backends import backend_qt4

    if no_tile:
        return

    if not np.iterable(wh):
        wh = golden_wh(wh)

    num_rows, num_cols = num_rc
    w, h = wh
    x_off, y_off = xy_off
    w_off, h_off = wh_off
    x_pad, y_pad = (0, 0)

    printDBG('[df2] Tile all figures: ')
    printDBG('[df2]     wh = %r' % ((w, h),))
    printDBG('[df2]     xy_offsets = %r' % ((x_off, y_off),))
    printDBG('[df2]     wh_offsets = %r' % ((w_off, h_off),))
    printDBG('[df2]     xy_pads = %r' % ((x_pad, y_pad),))

    if sys.platform == 'win32':
        h_off +=   0
        w_off +=  40
        x_off +=  40
        y_off +=  40
        x_pad +=   0
        y_pad += 100

    all_figures = get_all_figures()
    all_qt4wins = get_all_qt4_wins()

    def position_window(i, win):
        isqt4_mpl = isinstance(win, backend_qt4.MainWindow)
        isqt4_back = isinstance(win, QtGui.QMainWindow)
        if not isqt4_mpl and not isqt4_back:
            raise NotImplementedError('%r-th Backend %r is not a Qt Window' % (i, win))
        if row_first:
            y = (i % num_rows) * (h + h_off) + 40
            x = (int(i / num_rows)) * (w + w_off) + x_pad
        else:
            x = (i % num_cols) * (w + w_off) + 40
            y = (int(i / num_cols)) * (h + h_off) + y_pad
        x += x_off
        y += y_off
        win.setGeometry(x, y, w, h)
    ioff = 0
    for i, win in enumerate(all_qt4wins):
        position_window(i, win)
        ioff += 1
    for i, fig in enumerate(all_figures):
        win = fig.canvas.manager.window
        position_window(i + ioff, win)


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
        warnings.simplefilter("ignore")
        all_figures_tile(*args, **kwargs)
        all_figures_show()
        all_figures_bring_to_front()
    # Return an exec string
    execstr = helpers.ipython_execstr()
    execstr += textwrap.dedent('''
    if not embedded:
        print('[df2] Presenting in normal shell.')
        print('[df2] ... plt.show()')
        plt.show()
    ''')
    return execstr


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
    #fname_clean = split(fpath_clean)[1]
    print('[df2] save_figure() %r' % (fpath_clean,))
    #adjust_subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
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


def plot2(x_data, y_data, marker, x_label, y_label, title_pref, *args,
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
    ax.set_title(title_pref + ' ' + x_label + ' vs ' + y_label,
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
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)


# TEXT FUNCTIONS
# TODO: I have too many of these. Need to consolidate

def upperleft_text(txt):
    txtargs = dict(horizontalalignment='left',
                   verticalalignment='top',
                   #fontsize='smaller',
                   #fontweight='ultralight',
                   backgroundcolor=(0, 0, 0, .5),
                   color=ORANGE)
    ax_relative_text(.02, .02, txt, **txtargs)


def upperright_text(txt, offset=None):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   #fontsize='smaller',
                   #fontweight='ultralight',
                   backgroundcolor=(0, 0, 0, .5),
                   color=ORANGE,
                   offset=offset)
    ax_relative_text(.98, .02, txt, **txtargs)


def lowerright_text(txt):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   #fontsize='smaller',
                   #fontweight='ultralight',
                   backgroundcolor=(0, 0, 0, .5),
                   color=ORANGE)
    ax_relative_text(.98, .92, txt, **txtargs)


def absolute_lbl(x_, y_, txt, **kwargs):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   backgroundcolor=(0, 0, 0, .5),
                   color=ORANGE,
                   **kwargs)
    ax_absolute_text(x_, y_, txt, **txtargs)


def ax_relative_text(x, y, txt, ax=None, offset=None, **kwargs):
    if ax is None:
        ax = gca()
    xy, width, height = _axis_xy_width_height(ax)
    x_, y_ = ((xy[0]) + x * width, (xy[1] + height) - y * height)
    if offset is not None:
        xoff, yoff = offset
        x_ += xoff
        y_ += yoff
    ax_absolute_text(x_, y_, txt, ax=ax, **kwargs)


def ax_absolute_text(x_, y_, txt, ax=None, **kwargs):
    if ax is None:
        ax = gca()
    if 'fontproperties' in kwargs:
        kwargs['fontproperties'] = FONTS.relative
    ax.text(x_, y_, txt, **kwargs)


def fig_relative_text(x, y, txt, **kwargs):
    kwargs['horizontalalignment'] = 'center'
    kwargs['verticalalignment'] = 'center'
    fig = gcf()
    #xy, width, height = _axis_xy_width_height(ax)
    #x_, y_ = ((xy[0]+width)+x*width, (xy[1]+height)-y*height)
    fig.text(x, y, txt, **kwargs)


def draw_text(text_str, rgb_textFG=(0, 0, 0), rgb_textBG=(1, 1, 1)):
    ax = gca()
    xy, width, height = _axis_xy_width_height(ax)
    text_x = xy[0] + (width / 2)
    text_y = xy[1] + (height / 2)
    ax.text(text_x, text_y, text_str,
            horizontalalignment='center',
            verticalalignment='center',
            color=rgb_textFG,
            backgroundcolor=rgb_textBG)


def set_figtitle(figtitle, subtitle=''):
    fig = gcf()
    if subtitle != '':
        subtitle = '\n' + subtitle
    fig.suptitle(figtitle + subtitle, fontsize=14, fontweight='bold')
    fig.suptitle(figtitle, x=.5, y=.98, fontproperties=FONTS.figtitle)
    #fig_relative_text(.5, .96, subtitle, fontproperties=FONTS.subtitle)
    fig.canvas.set_window_title(figtitle)
    adjust_subplots()


TMP_mevent = None


def convert_keypress_event_mpl_to_qt4(mevent):
    global TMP_mevent
    TMP_mevent = mevent
    # Grab the key from the mpl.KeyPressEvent
    key = mevent.key
    print('[df2] convert event mpl -> qt4')
    print('[df2] key=%r' % key)
    # dicts modified from backend_qt4.py
    mpl2qtkey = {'control': Qt.Key_Control, 'shift': Qt.Key_Shift,
                 'alt': Qt.Key_Alt, 'super': Qt.Key_Meta,
                 'enter': Qt.Key_Return, 'left': Qt.Key_Left, 'up': Qt.Key_Up,
                 'right': Qt.Key_Right, 'down': Qt.Key_Down,
                 'escape': Qt.Key_Escape, 'f1': Qt.Key_F1, 'f2': Qt.Key_F2,
                 'f3': Qt.Key_F3, 'f4': Qt.Key_F4, 'f5': Qt.Key_F5,
                 'f6': Qt.Key_F6, 'f7': Qt.Key_F7, 'f8': Qt.Key_F8,
                 'f9': Qt.Key_F9, 'f10': Qt.Key_F10, 'f11': Qt.Key_F11,
                 'f12': Qt.Key_F12, 'home': Qt.Key_Home, 'end': Qt.Key_End,
                 'pageup': Qt.Key_PageUp, 'pagedown': Qt.Key_PageDown}
    # Reverse the control and super (aka cmd/apple) keys on OSX
    if sys.platform == 'darwin':
        mpl2qtkey.update({'super': Qt.Key_Control, 'control': Qt.Key_Meta, })

    # Try to reconstruct QtGui.KeyEvent
    type_ = QtCore.QEvent.Type(QtCore.QEvent.KeyPress)  # The type should always be KeyPress
    text = ''
    # Try to extract the original modifiers
    modifiers = QtCore.Qt.NoModifier  # initialize to no modifiers
    if key.find(u'ctrl+') >= 0:
        modifiers = modifiers | QtCore.Qt.ControlModifier
        key = key.replace(u'ctrl+', u'')
        print('[df2] has ctrl modifier')
        text += 'Ctrl+'
    if key.find(u'alt+') >= 0:
        modifiers = modifiers | QtCore.Qt.AltModifier
        key = key.replace(u'alt+', u'')
        print('[df2] has alt modifier')
        text += 'Alt+'
    if key.find(u'super+') >= 0:
        modifiers = modifiers | QtCore.Qt.MetaModifier
        key = key.replace(u'super+', u'')
        print('[df2] has super modifier')
        text += 'Super+'
    if key.isupper():
        modifiers = modifiers | QtCore.Qt.ShiftModifier
        print('[df2] has shift modifier')
        text += 'Shift+'
    # Try to extract the original key
    try:
        if key in mpl2qtkey:
            key_ = mpl2qtkey[key]
        else:
            key_ = ord(key.upper())  # Qt works with uppercase keys
            text += key.upper()
    except Exception as ex:
        print('[df2] ERROR key=%r' % key)
        print('[df2] ERROR %r' % ex)
        raise
    autorep = False  # default false
    count   = 1  # default 1
    text = QtCore.QString(text)  # The text is somewhat arbitrary
    # Create the QEvent
    print('----------------')
    print('[df2] Create event')
    print('[df2] type_ = %r' % type_)
    print('[df2] text = %r' % text)
    print('[df2] modifiers = %r' % modifiers)
    print('[df2] autorep = %r' % autorep)
    print('[df2] count = %r ' % count)
    print('----------------')
    qevent = QtGui.QKeyEvent(type_, key_, modifiers, text, autorep, count)
    return qevent


def test_build_qkeyevent():
    import draw_func2 as df2
    qtwin = df2.QT4_WINS[0]
    # This reconstructs an test mplevent
    canvas = df2.figure(1).canvas
    mevent = matplotlib.backend_bases.KeyEvent('key_press_event', canvas, u'ctrl+p', x=672, y=230.0)
    qevent = df2.convert_keypress_event_mpl_to_qt4(mevent)
    app = qtwin.backend.app
    app.sendEvent(qtwin.ui, mevent)
    #type_ = QtCore.QEvent.Type(QtCore.QEvent.KeyPress)  # The type should always be KeyPress
    #text = QtCore.QString('A')  # The text is somewhat arbitrary
    #modifiers = QtCore.Qt.NoModifier  # initialize to no modifiers
    #modifiers = modifiers | QtCore.Qt.ControlModifier
    #modifiers = modifiers | QtCore.Qt.AltModifier
    #key_ = ord('A')  # Qt works with uppercase keys
    #autorep = False  # default false
    #count   = 1  # default 1
    #qevent = QtGui.QKeyEvent(type_, key_, modifiers, text, autorep, count)
    return qevent


# This actually doesn't matter
def on_key_press_event(event):
    'redirects keypress events to main window'
    global QT4_WINS
    print('[df2] %r' % event)
    print('[df2] %r' % str(event.__dict__))
    for qtwin in QT4_WINS:
        qevent = convert_keypress_event_mpl_to_qt4(event)
        app = qtwin.backend.app
        print('[df2] attempting to send qevent to qtwin')
        app.sendEvent(qtwin, qevent)
        # TODO: FINISH ME
        #PyQt4.QtGui.QKeyEvent
        #qtwin.keyPressEvent(event)
        #fig.canvas.manager.window.keyPressEvent()


def customize_figure(fig, doclf):
    if not 'user_stat_list' in fig.__dict__.keys() or doclf:
        fig.user_stat_list = []
        fig.user_notes = []
    # We dont need to catch keypress events because you just need to set it as
    # an application level shortcut
    # Catch key press events
    #key_event_cbid = fig.__dict__.get('key_event_cbid', None)
    #if key_event_cbid is not None:
        #fig.canvas.mpl_disconnect(key_event_cbid)
    #fig.key_event_cbid = fig.canvas.mpl_connect('key_press_event', on_key_press_event)
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
           plotnum=(1, 1, 1),
           figtitle=None,
           **kwargs):
    #matplotlib.pyplot.xkcd()
    fig = get_fig(fignum)
    axes_list = fig.get_axes()
    # Ensure my customized settings
    customize_figure(fig, doclf)
    # Convert plotnum to tuple format
    if tools.is_int(plotnum):
        nr = plotnum // 100
        nc = plotnum // 10 - (nr * 10)
        px = plotnum - (nr * 100) - (nc * 10)
        plotnum = (nr, nc, px)
    # Get the subplot
    if doclf or len(axes_list) == 0:
        printDBG('[df2] *** NEW FIGURE %r.%r ***' % (fignum, plotnum))
        if not plotnum is None:
            #ax = plt.subplot(*plotnum)
            ax = fig.add_subplot(*plotnum)
            ax.cla()
        else:
            ax = gca()
    else:
        printDBG('[df2] *** OLD FIGURE %r.%r ***' % (fignum, plotnum))
        if not plotnum is None:
            ax = plt.subplot(*plotnum)  # fig.add_subplot fails here
            #ax = fig.add_subplot(*plotnum)
        else:
            ax = gca()
        #ax  = axes_list[0]
    # Set the title
    if not title is None:
        ax = gca()
        ax.set_title(title, fontproperties=FONTS.axtitle)
        # Add title to figure
        if figtitle is None and plotnum == (1, 1, 1):
            figtitle = title
        if not figtitle is None:
            fig.canvas.set_window_title('fig %r %s' % (fignum, figtitle))
    return fig


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
        perb   = (np.random.randn(len(data))) * pdfrange / 30.
        preb_y_data = np.abs([pdfrange / 50. for _ in data] + perb)
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
    ax.hist(data, bins=bins, range=(dmin, dmax))
    #help(np.bincount)
    fig.show()


def show_signature(sig, **kwargs):
    fig = figure(**kwargs)
    plt.plot(sig)
    fig.show()


def draw_stems(x_data=None, y_data=None):
    if y_data is not None and x_data is None:
        x_data = np.arange(len(y_data))
        pass
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
    ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
    ax.set_ylim(min(y_data) - 1, max(max(y_data), max(x_data)) + 1)


def draw_sift_signature(sift, title=''):
    ax = gca()
    draw_bars(sift, 16)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 256)
    xy, width, height = _axis_xy_width_height(ax)

    rect = matplotlib.patches.Rectangle(xy, width, height, lw=0, zorder=0)
    rect = ax.add_patch(rect)
    rect.set_clip_on(False)
    rect.set_fill(True)
    rect.set_color(BLACK * .9)
    ax.set_xticks(np.arange(9) * 16)
    ax.set_yticks(np.arange(9) * 32)
    ax.set_title(title)


def draw_bars(y_data, nColorSplits=1):
    width = 1
    nDims = len(y_data)
    nGroup = nDims // nColorSplits
    ori_colors = distinct_colors(nColorSplits)
    x_data = np.arange(nDims)
    ax = gca()
    for ix in xrange(nColorSplits):
        xs = np.arange(nGroup) + (nGroup * ix)
        color = ori_colors[ix]
        x_dat = x_data[xs]
        y_dat = y_data[xs]
        ax.bar(x_dat, y_dat, width, color=color, edgecolor=np.array(color) * .8)


def legend():
    ax = gca()
    ax.legend(prop=FONTS.legend)


def draw_histpdf(data, label=None, draw_support=False, nbins=10):
    freq, _ = draw_hist(data, nbins=nbins)
    draw_pdf(data, draw_support=draw_support, scale_to=freq.max(), label=label)


def draw_hist(data, bins=None, nbins=10, weights=None):
    if isinstance(data, list):
        data = np.array(data)
    if bins is None:
        dmin = data.min()
        dmax = data.max()
        bins = dmax - dmin
    ax  = gca()
    freq, bins_, patches = ax.hist(data, bins=nbins, weights=weights, range=(dmin, dmax))
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
    ax.set_xlim(trunc_min, trunc_max)
    #trunc_xticks = np.linspace(0, int(trunc_max),11)
    #trunc_xticks = trunc_xticks[trunc_xticks >= trunc_min]
    #trunc_xticks = np.append([int(trunc_min)], trunc_xticks)
    #no_zero_yticks = ax.get_yticks()[ax.get_yticks() > 0]
    #ax.set_xticks(trunc_xticks)
    #ax.set_yticks(no_zero_yticks)
#_----------------- HELPERS ^^^ ---------


# ---- IMAGE CREATION FUNCTIONS ----
@tools.debug_exception
def draw_sift(desc, kp=None):
    ''' desc = np.random.rand(128)
        desc = desc / np.sqrt((desc**2).sum())
        desc = np.round(desc * 255) '''
    ax = gca()
    tau = 2 * np.pi
    DSCALE = .25
    XYSCALE = .5
    XYSHIFT = -.75
    ORI_SHIFT = 0  # -tau #1/8 * tau
    # SIFT CONSTANTS
    NORIENTS = 8
    NX = 4
    NY = 4
    NBINS = NX * NY

    def cirlce_rad2xy(radians, mag):
        return np.cos(radians) * mag, np.sin(radians) * mag
    discrete_ori = (np.arange(0, NORIENTS) * (tau / NORIENTS) + ORI_SHIFT)
    # Build list of plot positions
    # Build an "arm" for each sift measurement
    arm_mag   = desc / 255.0
    arm_ori = np.tile(discrete_ori, (NBINS, 1)).flatten()
    # The offset x,y's for each sift measurment
    arm_dxy = np.array(zip(*cirlce_rad2xy(arm_ori, arm_mag)))
    yxt_gen = itertools.product(xrange(NY), xrange(NX), xrange(NORIENTS))
    yx_gen  = itertools.product(xrange(NY), xrange(NX))
    # Transform the drawing of the SIFT descriptor to the its elliptical patch
    axTrans = ax.transData
    kpTrans = None
    if kp is None:
        kp = [0, 0, 1, 0, 1]
    kp = np.array(kp)
    kpT = kp.T
    x, y, a, c, d = kpT[:, 0]
    transMat = [( a, 0, x),
                ( c, d, y),
                ( 0, 0, 1)]
    kpTrans = Affine2D(transMat)
    axTrans = ax.transData
    # Draw 8 directional arms in each of the 4x4 grid cells
    arrow_patches = []
    arrow_patches2 = []
    for y, x, t in yxt_gen:
        index = y * NX * NORIENTS + x * NORIENTS + t
        (dx, dy) = arm_dxy[index]
        arw_x  = x * XYSCALE + XYSHIFT
        arw_y  = y * XYSCALE + XYSHIFT
        arw_dy = dy * DSCALE * 1.5  # scale for viz Hack
        arw_dx = dx * DSCALE * 1.5
        #posA = (arw_x, arw_y)
        #posB = (arw_x+arw_dx, arw_y+arw_dy)
        _args = [arw_x, arw_y, arw_dx, arw_dy]
        _kwargs = dict(head_width=.0001, transform=kpTrans, length_includes_head=False)
        arrow_patches  += [FancyArrow(*_args, **_kwargs)]
        arrow_patches2 += [FancyArrow(*_args, **_kwargs)]
    # Draw circles around each of the 4x4 grid cells
    circle_patches = []
    for y, x in yx_gen:
        circ_xy = (x * XYSCALE + XYSHIFT, y * XYSCALE + XYSHIFT)
        circ_radius = DSCALE
        circle_patches += [Circle(circ_xy, circ_radius, transform=kpTrans)]
    # Efficiently draw many patches with PatchCollections
    circ_collection = PatchCollection(circle_patches)
    circ_collection.set_facecolor('none')
    circ_collection.set_transform(axTrans)
    circ_collection.set_edgecolor(BLACK)
    circ_collection.set_alpha(.5)
    # Body of arrows
    arw_collection = PatchCollection(arrow_patches)
    arw_collection.set_transform(axTrans)
    arw_collection.set_linewidth(.5)
    arw_collection.set_color(RED)
    arw_collection.set_alpha(1)
    # Border of arrows
    arw_collection2 = matplotlib.collections.PatchCollection(arrow_patches2)
    arw_collection2.set_transform(axTrans)
    arw_collection2.set_linewidth(1)
    arw_collection2.set_color(BLACK)
    arw_collection2.set_alpha(1)
    # Add artists to axes
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
    score2_01 = lambda score: .1 + .9 * (float(score) - mins) / (rnge)
    colors    = [cmap(score2_01(fs[fx])) for fx in xrange(len(fs))]
    return colors


def draw_matches2(kpts1, kpts2, fm=None, fs=None, kpts2_offset=(0, 0),
                  color_list=None):
    if not DISTINCT_COLORS:
        color_list = None
    # input data
    if not SHOW_LINES:
        return
    if fm is None:  # assume kpts are in director correspondence
        assert kpts1.shape == kpts2.shape
    if len(fm) == 0:
        return
    ax = gca()
    woff, hoff = kpts2_offset
    # Draw line collection
    kpts1_m = kpts1[fm[:, 0]].T
    kpts2_m = kpts2[fm[:, 1]].T
    xxyy_iter = iter(zip(kpts1_m[0],
                         kpts2_m[0] + woff,
                         kpts1_m[1],
                         kpts2_m[1] + hoff))
    if color_list is None:
        if fs is None:  # Draw with solid color
            color_list    = [ LINE_COLOR for fx in xrange(len(fm))]
        else:  # Draw with colors proportional to score difference
            color_list = feat_scores_to_color(fs)
    segments  = [((x1, y1), (x2, y2)) for (x1, x2, y1, y2) in xxyy_iter]
    linewidth = [LINE_WIDTH for fx in xrange(len(fm))]
    line_group = LineCollection(segments, linewidth, color_list, alpha=LINE_ALPHA)
    ax.add_collection(line_group)


def draw_kpts2(kpts, offset=(0, 0),
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
    # data
    kpts = np.array(kpts)
    kptsT = kpts.T
    x = kptsT[0, :] + offset[0]
    y = kptsT[1, :] + offset[1]
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
                aIS = 1 / np.sqrt(a)
                bIS = c / (-np.sqrt(a) * d - a * np.sqrt(d))
                dIS = 1 / np.sqrt(d)
                cIS = b
                #cIS = (c/np.sqrt(d) - c/np.sqrt(d)) / (a-d+1E-9)
        else:
            aIS = a
            bIS = b
            cIS = c
            dIS = d
            # Just inverse
            #aIS = 1/a
            #bIS = -c/(a*d)
            #dIS = 1/d

        kpts_iter = izip(x, y, aIS, bIS, cIS, dIS)
        aff_list = [Affine2D([( a_, b_, x_),
                              ( c_, d_, y_),
                              (  0,  0,  1)])
                    for (x_, y_, a_, b_, c_, d_) in kpts_iter]
        patch_list = []
        ell_actors = [Circle( (0, 0), 1, transform=aff) for aff in aff_list]
        if ell:
            patch_list += ell_actors
        if rect:
            rect_actors = [Rectangle( (-1, -1), 2, 2, transform=aff) for aff in aff_list]
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
        ax.scatter(x, y, c=color_list, s=2 * pts_size, marker='o', edgecolor='none')
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

    plt_imshow_kwargs = {
        'interpolation': interpolation,
        'cmap': plt.get_cmap('gray'),
        'vmin': 0,
        'vmax': 255,
    }
    try:
        ax.imshow(img, **plt_imshow_kwargs)
    except TypeError as te:
        print('ERROR %r' % te)
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


def get_num_channels(img):
    ndims = len(img.shape)
    if ndims == 2:
        nChannels = 1
    elif ndims == 3 and img.shape[2] == 3:
        nChannels = 3
    elif ndims == 3 and img.shape[2] == 1:
        nChannels = 1
    else:
        raise Exception('Cannot determine number of channels')
    return nChannels


def stack_images(img1, img2, vert=None):
    nChannels = get_num_channels(img1)
    nChannels2 = get_num_channels(img2)
    assert nChannels == nChannels2
    (h1, w1) = img1.shape[0: 2]  # get chip dimensions
    (h2, w2) = img2.shape[0: 2]
    woff = 0
    hoff = 0
    if vert is None:  # Display match up/down or side/side
        vert = False if h1 > w1 and h2 > w2 else True
    if vert:
        wB = max(w1, w2)
        hB = h1 + h2
        hoff = h1
    else:
        hB = max(h1, h2)
        wB = w1 + w2
        woff = w1
    # concatentate images
    if nChannels == 3:
        imgB = np.zeros((hB, wB, 3), np.uint8)
        imgB[0:h1, 0:w1, :] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2), :] = img2
    elif nChannels == 1:
        imgB = np.zeros((hB, wB), np.uint8)
        imgB[0:h1, 0:w1] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2
    return imgB, woff, hoff


def show_matches2(rchip1, rchip2, kpts1, kpts2,
                  fm=None, fs=None, fignum=None, plotnum=None,
                  title=None, vert=None, all_kpts=True,
                  draw_lines=True,
                  draw_ell=True,
                  draw_pts=True,
                  ell_alpha=None,
                  lbl1=None,
                  lbl2=None, **kwargs):
    '''Draws feature matches
    kpts1 and kpts2 use the (x,y,a,c,d)
    '''
    if fm is None:
        assert kpts1.shape == kpts2.shape
        fm = np.tile(np.arange(0, len(kpts1)), (2, 1)).T
    # get matching keypoints + offset
    (h1, w1) = rchip1.shape[0:2]  # get chip dimensions
    (h2, w2) = rchip2.shape[0:2]
    match_img, woff, hoff = stack_images(rchip1, rchip2, vert)
    fig, ax = imshow(match_img, fignum=fignum,
                     plotnum=plotnum, title=title,
                     **kwargs)
    nMatches = len(fm)
    #upperleft_text('#match=%d' % nMatches)
    if lbl1 is not None:
        absolute_lbl(w1, 0, lbl1)
    if lbl2 is not None:
        absolute_lbl(w2 + woff, 0 + hoff, lbl2)
    if all_kpts:
        # Draw all keypoints as simple points
        all_args = dict(ell=False, pts=draw_pts, pts_color=GREEN, pts_size=2, ell_alpha=ell_alpha)
        draw_kpts2(kpts1, **all_args)
        draw_kpts2(kpts2, offset=(woff, hoff), **all_args)
    if nMatches == 0:
        printDBG('[df2] There are no feature matches to plot!')
    else:
        #color_list = [((x)/nMatches,1-((x)/nMatches),0) for x in xrange(nMatches)]
        #cmap = lambda x: (x, 1-x, 0)
        #cmap = plt.get_cmap('prism')
        #color_list = [cmap(mx/nMatches) for mx in xrange(nMatches)]
        colors = distinct_colors(nMatches)
        pt2_args = dict(pts=draw_pts, ell=False, pts_color=BLACK, pts_size=8)
        pts_args = dict(pts=draw_pts, ell=False, pts_color=ORANGE, pts_size=6,
                        color_list=add_alpha(colors))
        ell_args = dict(ell=draw_ell, pts=False, color_list=colors)
        # Draw matching ellipses
        offset = (woff, hoff)

        def _drawkpts(**kwargs):
            draw_kpts2(kpts1[fm[:, 0]], **kwargs)
            draw_kpts2(kpts2[fm[:, 1]], offset=offset, **kwargs)

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
    return fig, ax, woff, hoff
