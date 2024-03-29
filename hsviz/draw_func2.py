''' Lots of functions for drawing and plotting visiony things '''
# TODO: New naming scheme
# viz_<func_name> will clear everything. The current axes and fig: clf, cla.  # Will add annotations
# interact_<func_name> will clear everything and start user interactions.
# show_<func_name> will always clear the current axes, but not fig: cla # Might # add annotates?
# plot_<func_name> will not clear the axes or figure. More useful for graphs
# draw_<func_name> same as plot for now. More useful for images
from __future__ import division, print_function
from hscom import __common__
(print, print_, print_on, print_off, rrr, profile,
 printDBG) = __common__.init(__name__, '[df2]', DEBUG=False, initmpl=True)
# Python
# from itertools import izip
izip = zip
from os.path import splitext, split, join, normpath, exists
import colorsys
import itertools
import pylab
import sys
import textwrap
import time
import warnings
# Matplotlib / Qt
import matplotlib
import matplotlib as mpl  # NOQA
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
# Qt
if 0:
    from matplotlib.backends import backend_qt4 as backend_qt
    from PyQt4 import QtCore, QtGui
    from PyQt4.QtCore import Qt
    QtGui = QtWidgets
else:
    from matplotlib.backends import backend_qt5 as backend_qt
    from PyQt5 import QtCore
    from PyQt5 import QtGui
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5 import QtWidgets

# Scientific
import numpy as np
import scipy.stats
import cv2
# HotSpotter
from hscom import helpers as util
from hscom import tools
from hscom.Printable import DynStruct

#================
# GLOBALS
#================

TMP_mevent = None
QT4_WINS = []
plotWidget = None

# GENERAL FONTS

SMALLER  = 8
SMALL    = 10
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

# SPECIFIC FONTS

FONTS.legend   = FONTS.small
FONTS.figtitle = FONTS.med
FONTS.axtitle  = FONTS.small
FONTS.subtitle = FONTS.med
#FONTS.xlabel   = FONTS.smaller
FONTS.xlabel   = FONTS.small
FONTS.ylabel   = FONTS.small
FONTS.relative = FONTS.smaller

# COLORS

ORANGE = np.array((255, 127,   0, 255)) / 255.0
RED    = np.array((255,   0,   0, 255)) / 255.0
GREEN  = np.array((  0, 255,   0, 255)) / 255.0
BLUE   = np.array((  0,   0, 255, 255)) / 255.0
YELLOW = np.array((255, 255,   0, 255)) / 255.0
BLACK  = np.array((  0,   0,   0, 255)) / 255.0
WHITE  = np.array((255, 255, 255, 255)) / 255.0
GRAY   = np.array((127, 127, 127, 255)) / 255.0
DEEP_PINK    = np.array((255,  20, 147, 255)) / 255.0
PINK         = np.array((255,  100, 100, 255)) / 255.0
FALSE_RED    = np.array((255,  51,   0, 255)) / 255.0
TRUE_GREEN   = np.array((  0, 255,   0, 255)) / 255.0
DARK_ORANGE  = np.array((127,  63,   0, 255)) / 255.0
DARK_YELLOW  = np.array((127,  127,   0, 255)) / 255.0
PURPLE = np.array((102,   0, 153, 255)) / 255.0
UNKNOWN_PURP = PURPLE


# GOLDEN RATIOS
PHI_numer = 1 + np.sqrt(5)
PHI_denom = 2.0
PHI = PHI_numer / PHI_denom


def golden_wh2(sz):
    return (PHI * sz, sz)


def golden_wh(x):
    'returns a width / height with a golden aspect ratio'
    return map(int, map(round, (x * .618, x * .312)))


# FIGURE GEOMETRY
DPI = 80
#DPI = 160
#FIGSIZE = (24) # default windows fullscreen
FIGSIZE_MED = (12, 6)
FIGSIZE_SQUARE = (12, 12)
FIGSIZE_GOLD = golden_wh2(8)
FIGSIZE_BIGGER = (24, 12)
FIGSIZE_HUGE = (32, 16)

FIGSIZE = FIGSIZE_MED
# Quality drawings
#FIGSIZE = FIGSIZE_SQUARE
#DPI = 120

# DEFAULTS. (TODO: Can these be cleaned up?)

DISTINCT_COLORS = True  # and False
DARKEN = None
ELL_LINEWIDTH = 1.5
if DISTINCT_COLORS:
    ELL_ALPHA  = .6
    LINE_ALPHA = .35
else:
    ELL_ALPHA  = .4
    LINE_ALPHA = .4

LINE_ALPHA_OVERRIDE = util.get_arg('--line-alpha-override', type_=float, default=None)
ELL_ALPHA_OVERRIDE = util.get_arg('--ell-alpha-override', type_=float, default=None)
#LINE_ALPHA_OVERRIDE = None
#ELL_ALPHA_OVERRIDE = None
ELL_COLOR  = BLUE

LINE_COLOR = RED
LINE_WIDTH = 1.4

SHOW_LINES = True  # True
SHOW_ELLS  = True

POINT_SIZE = 2


base_fnum = 9001


def next_fnum():
    global base_fnum
    base_fnum += 1
    return base_fnum


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


def unregister_qt4_win(win):
    global QT4_WINS
    if win == 'all':
        QT4_WINS = []


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


def _axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    if ax is None:
        ax = gca()
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


def draw_roi(roi, label=None, bbox_color=(1, 0, 0),
             lbl_bgcolor=(0, 0, 0), lbl_txtcolor=(1, 1, 1), theta=0, ax=None):
    if ax is None:
        ax = gca()
    (rx, ry, rw, rh) = roi
    #cos_ = np.cos(theta)
    #sin_ = np.sin(theta)
    #rot_t = Affine2D([( cos_, -sin_, 0),
                      #( sin_,  cos_, 0),
                      #(  0,       0, 1)])
    #scale_t = Affine2D([( rw,  0, 0),
                        #( 0,  rh, 0),
                        #( 0,   0, 1)])
    #trans_t = Affine2D([( 1,  0, rx + rw / 2),
                        #( 0,  1, ry + rh / 2),
                        #( 0,  0, 1)])
    #t_end = scale_t + rot_t + trans_t + t_start
    # Transformations are specified in backwards order.
    trans_roi = Affine2D()
    trans_roi.scale(rw, rh)
    trans_roi.rotate(theta)
    trans_roi.translate(rx + rw / 2, ry + rh / 2)
    t_end = trans_roi + ax.transData
    bbox = matplotlib.patches.Rectangle((-.5, -.5), 1, 1, lw=2, transform=t_end)
    arw_x, arw_y, arw_dx, arw_dy   = (-0.5, -0.5, 1.0, 0.0)
    arrowargs = dict(head_width=.1, transform=t_end, length_includes_head=True)
    arrow = FancyArrow(arw_x, arw_y, arw_dx, arw_dy, **arrowargs)

    bbox.set_fill(False)
    #bbox.set_transform(trans)
    bbox.set_edgecolor(bbox_color)
    arrow.set_edgecolor(bbox_color)
    arrow.set_facecolor(bbox_color)

    ax.add_patch(bbox)
    ax.add_patch(arrow)
    #ax.add_patch(arrow2)
    if label is not None:
        ax_absolute_text(rx, ry, label, ax=ax,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=lbl_txtcolor,
                         backgroundcolor=lbl_bgcolor)


# ---- GENERAL FIGURE COMMANDS ----


def set_geometry(fnum, x, y, w, h):
    fig = get_fig(fnum)
    qtwin = fig.canvas.manager.window
    qtwin.setGeometry(x, y, w, h)


def get_geometry(fnum):
    fig = get_fig(fnum)
    qtwin = fig.canvas.manager.window
    (x1, y1, x2, y2) = qtwin.geometry().getCoords()
    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
    return (x, y, w, h)


def get_screen_info():
    desktop = QtWidgets.QDesktopWidget()
    mask = desktop.mask()  # NOQA
    layout_direction = desktop.layoutDirection()  # NOQA
    screen_number = desktop.screenNumber()  # NOQA
    normal_geometry = desktop.normalGeometry()  # NOQA
    num_screens = desktop.screenCount()  # NOQA
    avail_rect = desktop.availableGeometry()  # NOQA
    screen_rect = desktop.screenGeometry()  # NOQA
    QtWidgets.QDesktopWidget().availableGeometry().center()  # NOQA
    normal_geometry = desktop.normalGeometry()  # NOQA


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


def ensure_app_is_running():
    from hsgui import guitools
    app, is_root = guitools.init_qtapp()


def get_monitor_geom(monitor_num=0):
    ensure_app_is_running()
    desktop = QtWidgets.QDesktopWidget()
    rect = desktop.availableGeometry(screen=monitor_num)
    geom = (rect.x(), rect.y(), rect.width(), rect.height())
    return geom


def get_monitor_geometries():
    ensure_app_is_running()
    monitor_geometries = {}
    desktop = QtWidgets.QDesktopWidget()
    for screenx in xrange(desktop.numScreens()):
        rect = desktop.availableGeometry(screen=screenx)
        geom = (rect.x(), rect.y(), rect.width(), rect.height())
        monitor_geometries[screenx] = geom
    return monitor_geometries


def all_figures_tile(num_rc=None, wh=400, xy_off=(0, 0), wh_off=(0, 0),
                     row_first=True, no_tile=False, override1=False):
    'Lays out all figures in a grid. if wh is a scalar, a golden ratio is used'
    print('[df2] all_figures_tile()')
    # RCOS TODO:
    # I want this function to layout all the figures and qt windows within the
    # bounds of a rectangle. (taken from the get_monitor_geom, or specified by
    # the user i.e. left half of monitor 0). It should lay them out
    # rectangularly and choose figure sizes such that all of them will fit.
    if no_tile:
        return

    if not np.iterable(wh):
        wh = golden_wh(wh)

    all_figures = get_all_figures()
    all_qt4wins = get_all_qt4_wins()

    if override1:
        if len(all_figures) == 1:
            fig = all_figures[0]
            win = fig.canvas.manager.window
            win.setGeometry(0, 0, 900, 900)
            update()
            return

    #nFigs = len(all_figures) + len(all_qt4_wins)

    # Win7 Areo
    win7_sizes = {
        'os_border_x':   20,
        'os_border_y':   35,
        'os_border_h':   30,
        'win_border_x':  17,
        'win_border_y':  10,
        'mpl_toolbar_y': 10,
    }

    # Ubuntu (Medeterrainian Dark)
    gnome3_sizes = {
        'os_border_x':    0,
        'os_border_y':   35,  # for gnome3 title bar
        'os_border_h':    0,
        'win_border_x':   5,
        'win_border_y':  30,
        'mpl_toolbar_y':  0,
    }

    w, h = wh
    x_off, y_off = xy_off
    w_off, h_off = wh_off
    x_pad, y_pad = (0, 0)
    # Good offset measurements for...
    #Windows 7
    if sys.platform.startswith('win32'):
        stdpxls = win7_sizes
    elif sys.platform.startswith('linux'):
        stdpxls = gnome3_sizes
    else:
        stdpxls = gnome3_sizes
    x_off +=  0
    y_off +=  0
    w_off +=  stdpxls['win_border_x']
    h_off +=  stdpxls['win_border_y'] + stdpxls['mpl_toolbar_y']
    # Pads are applied to all windows
    x_pad +=  stdpxls['os_border_x']
    y_pad +=  stdpxls['os_border_y']

    effective_w = w + w_off
    effective_h = h + h_off
    startx = 0
    starty = 0

    if num_rc is None:
        monitor_geometries = get_monitor_geometries()
        printDBG('[df2] monitor_geometries = %r' % (monitor_geometries,))
        geom = monitor_geometries[0]
        # Use all of monitor 0
        available_geom = (geom[0], geom[1], geom[2] - stdpxls['os_border_h'], geom[3])
        startx = available_geom[0]
        starty = available_geom[1]
        avail_width = available_geom[2] - available_geom[0]
        avail_height = available_geom[3] - available_geom[1]
        printDBG('[df2] available_geom = %r' % (available_geom,))
        printDBG('[df2] avail_width = %r' % (avail_width,))
        printDBG('[df2] avail_height = %r' % (avail_height,))

        nRows = int(avail_height // (effective_h))
        nCols = int(avail_width // (effective_w))
    else:
        nRows, nCols = num_rc

    printDBG('[df2] Tile all figures: ')
    printDBG('[df2]     wh = %r' % ((w, h),))
    printDBG('[df2]     xy_offsets = %r' % ((x_off, y_off),))
    printDBG('[df2]     wh_offsets = %r' % ((w_off, h_off),))
    printDBG('[df2]     wh_effective = %r' % ((effective_w, effective_h),))
    printDBG('[df2]     xy_pads = %r' % ((x_pad, y_pad),))
    printDBG('[df2]     nRows, nCols = %r' % ((nRows, nCols),))

    def position_window(ix, win):
        isqt4_mpl = isinstance(win, backend_qt.MainWindow)
        isqt4_back = isinstance(win, QtWidgets.QMainWindow)
        if not isqt4_mpl and not isqt4_back:
            raise NotImplementedError('%r-th Backend %r is not a Qt Window' %
                                      (ix, win))
        if row_first:
            rowx = ix % nRows
            colx = int(ix // nRows)
        else:
            colx = (ix % nCols)
            rowx = int(ix // nCols)
        x = startx + colx * (effective_w)
        y = starty + rowx * (effective_h)
        printDBG('ix=%r) rowx=%r colx=%r, x=%r y=%r, w=%r, h=%r' %
                 (ix, rowx, colx, x, y, w, h))
        try:
            #(x, y, w1, h1) = win.getGeometry()
            win.setGeometry(x + x_pad, y + y_pad, w, h)
        except Exception as ex:
            print(ex)
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
    execstr = util.ipython_execstr()
    execstr += textwrap.dedent('''
    if not embedded:
        if not '--quiet' in sys.argv:
            print('[df2] Presenting in normal shell.')
            print('[df2] ... plt.show()')
        plt.show()
    ''')
    return execstr


def prepare_figure_for_save(fnum):
    # Resizes the figure for quality saving
    if fnum is None:
        fig = gcf()
    else:
        fig = plt.figure(fnum, figsize=FIGSIZE, dpi=DPI)
    # Enforce inches and DPI
    fig.set_size_inches(FIGSIZE[0], FIGSIZE[1])
    fnum = fig.number
    return fig, fnum


def sanatize_img_fname(fname, defaultext='.jpg'):
    fname_clean = fname
    search_replace_list = [(' ', '_'), ('\n', '--'), ('\\', ''), ('/', '')]
    for old, new in search_replace_list:
        fname_clean = fname_clean.replace(old, new)
    fname_noext, ext = splitext(fname_clean)
    fname_clean = fname_noext + ext.lower()
    # Check for correct extensions
    if not ext.lower() in util.IMG_EXTENSIONS:
        fname_clean += defaultext
    return fname_clean


def sanatize_img_fpath(fpath, defaultext):
    [dpath, fname] = split(fpath)
    fname_clean = sanatize_img_fname(fname, defaultext)
    fpath_clean = join(dpath, fname_clean)
    fpath_clean = normpath(fpath_clean)
    return fpath_clean


def prepare_figure_fpath(fig, fpath, fnum, usetitle, defaultext):
    if fpath is None:
        # Find the title
        fpath = sanatize_img_fname(fig.canvas.get_window_title())
    if usetitle:
        title = sanatize_img_fname(fig.canvas.get_window_title())
        fpath = join(fpath, title)
    # Add in DPI information
    fpath_noext, ext = splitext(fpath)
    size_suffix = '_DPI=%r_FIGSIZE=%d,%d' % (DPI, FIGSIZE[0], FIGSIZE[1])
    fpath = fpath_noext + size_suffix + ext
    # Sanatize the filename
    fpath_clean = sanatize_img_fpath(fpath, defaultext)
    return fpath_clean


def save_figure(fnum=None, fpath=None, usetitle=False, overwrite=True,
                defaultext='.jpg'):
    fig, fnum = prepare_figure_for_save(fnum)
    fpath_clean = prepare_figure_fpath(fig, fpath, fnum, usetitle, defaultext)
    #fname_clean = split(fpath_clean)[1]
    #adjust_subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        if overwrite or not exists(fpath_clean):
            print('[df2] save_figure() %r' % (fpath_clean,))
            fig.savefig(fpath_clean, dpi=DPI)
        else:
            print('[df2] not overwriteing')


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


def set_xlabel(lbl, ax=None):
    if ax is None:
        ax = gca()
    ax.set_xlabel(lbl, fontproperties=FONTS.xlabel)


def set_title(title='', ax=None):
    if ax is None:
        ax = gca()
    ax.set_title(title, fontproperties=FONTS.axtitle)


def set_ylabel(lbl):
    ax = gca()
    ax.set_ylabel(lbl, fontproperties=FONTS.xlabel)


def get_good_logyscale_kwargs(y_data):
    # Attempts to detect knee points by looking for
    # log derivatives way past the normal standard deviations
    # The input data is assumed to be sorted and y_data
    basey = 10
    nStdDevs_thresh = 10
    # Take the log of the data
    logy = np.log(y_data)
    logy[np.isnan(logy)] = 0
    logy[np.isinf(logy)] = 0
    # Find the derivative of data
    dy = np.diff(logy)
    dy_sortx = dy.argsort()
    # Get mean and standard deviation
    dy_stats = util.mystats(dy)
    dy_sorted = dy[dy_sortx]
    # Find the number of standard deveations past the mean each datapoint is
    nStdDevs = np.abs(dy_sorted - dy_stats['mean']) / dy_stats['std']
    # Mark any above a threshold as knee points
    knee_indexes = np.where(nStdDevs > nStdDevs_thresh)[0]
    knee_mag = nStdDevs[knee_indexes]
    knee_points = dy_sortx[knee_indexes]
    print('[df2] knee_points = %r' % (knee_points,))
    # Check to see that we have found a knee
    if len(knee_points) > 0:
        # Use linear scaling up the the knee points and
        # scale it by the magnitude of the knee
        kneex = knee_points.argmin()
        linthreshx = knee_points[kneex] + 1
        linthreshy = y_data[linthreshx] * basey
        linscaley = min(2, max(1, (knee_mag[kneex] / (basey * 2))))
    else:
        linthreshx = 1E-15
        linthreshy = 1E-15
        linscaley = 1
    logscale_kwargs = {
        'basey': basey,
        'nonposx': 'clip',
        'nonposy': 'clip',
        'linthreshy': linthreshy,
        'linthreshx': linthreshx,
        'linscalex': 1,
        'linscaley': linscaley,
    }
    #print(logscale_kwargs)
    return logscale_kwargs


def set_logyscale_from_data(y_data):
    logscale_kwargs = get_good_logyscale_kwargs(y_data)
    ax = gca()
    ax.set_yscale('symlog', **logscale_kwargs)


def plot(*args, **kwargs):
    yscale = kwargs.pop('yscale', 'linear')
    xscale = kwargs.pop('xscale', 'linear')
    logscale_kwargs = kwargs.pop('logscale_kwargs', {'nonposx': 'clip'})
    plot = plt.plot(*args, **kwargs)
    ax = plt.gca()

    yscale_kwargs = logscale_kwargs if yscale in ['log', 'symlog'] else {}
    xscale_kwargs = logscale_kwargs if xscale in ['log', 'symlog'] else {}

    ax.set_yscale(yscale, **yscale_kwargs)
    ax.set_xscale(xscale, **xscale_kwargs)
    return plot


def plot2(x_data, y_data, marker='o', title_pref='', x_label='x', y_label='y', *args,
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

    min_ = min(x_data.min(), y_data.min())
    max_ = max(x_data.max(), y_data.max())
    # Equal aspect ratio
    ax.set_xlim(min_, max_)
    ax.set_ylim(min_, max_)
    ax.set_aspect('equal')
    ax.set_xlabel(x_label, fontproperties=FONTS.xlabel)
    ax.set_ylabel(y_label, fontproperties=FONTS.xlabel)
    set_title(title_pref + ' ' + x_label + ' vs ' + y_label, ax=None)


def adjust_subplots_xlabels():
    adjust_subplots(left=.03, right=.97, bottom=.2, top=.9, hspace=.15)


def adjust_subplots_xylabels():
    adjust_subplots(left=.03, right=1, bottom=.1, top=.9, hspace=.15)


SAFE_POS = {
    'left': .1,
    'right': .9,
    'top': .9,
    'bottom': .1,
    'wspace': .3,
    'hspace': .5,
}


def adjust_subplots_safe(**kwargs):
    for key in SAFE_POS.iterkeys():
        if not key in kwargs:
            kwargs[key] = SAFE_POS[key]
    adjust_subplots(**kwargs)


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


#=======================
# TEXT FUNCTIONS
# TODO: I have too many of these. Need to consolidate
#=======================


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


def absolute_lbl(x_, y_, txt, roffset=(-.02, -.02), **kwargs):
    txtargs = dict(horizontalalignment='right',
                   verticalalignment='top',
                   backgroundcolor=(0, 0, 0, .5),
                   color=ORANGE,
                   **kwargs)
    ax_absolute_text(x_, y_, txt, roffset=roffset, **txtargs)


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


def ax_absolute_text(x_, y_, txt, ax=None, roffset=None, **kwargs):
    if ax is None:
        ax = gca()
    if 'fontproperties' in kwargs:
        kwargs['fontproperties'] = FONTS.relative
    if roffset is not None:
        xroff, yroff = roffset
        xy, width, height = _axis_xy_width_height(ax)
        x_ += xroff * width
        y_ += yroff * height

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


def set_figtitle(figtitle, subtitle='', forcefignum=True, incanvas=True):
    if figtitle is None:
        figtitle = ''
    fig = gcf()
    if incanvas:
        if subtitle != '':
            subtitle = '\n' + subtitle
        #fig.suptitle(figtitle + subtitle, fontsize=14, fontweight='bold')
        fig.suptitle(figtitle + subtitle, fontproperties=FONTS.figtitle)
        #fig_relative_text(.5, .96, subtitle, fontproperties=FONTS.subtitle)
    else:
        fig.suptitle('')
    window_figtitle = ('fig(%d) ' % fig.number) + figtitle
    window_figtitle = window_figtitle.replace('\n', ' ')
    fig.canvas.set_window_title(window_figtitle)


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

    # Try to reconstruct QtWidgets.KeyEvent
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
    qevent = QtWidgets.QKeyEvent(type_, key_, modifiers, text, autorep, count)
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
    #qevent = QtWidgets.QKeyEvent(type_, key_, modifiers, text, autorep, count)
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
        #PyQt4.QtWidgets.QKeyEvent
        #qtwin.keyPressEvent(event)
        #fig.canvas.manager.window.keyPressEvent()


def customize_figure(fig, docla):
    if not 'user_stat_list' in fig.__dict__.keys() or docla:
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


def get_fig(fnum=None):
    printDBG('[df2] get_fig(fnum=%r)' % fnum)
    fig_kwargs = dict(figsize=FIGSIZE, dpi=DPI)
    if plotWidget is not None:
        return gcf()
    if fnum is None:
        try:
            fig = gcf()
        except Exception as ex:
            printDBG('[df2] get_fig(): ex=%r' % ex)
            fig = plt.figure(**fig_kwargs)
        fnum = fig.number
    else:
        try:
            fig = plt.figure(fnum, **fig_kwargs)
        except Exception as ex:
            print(repr(ex))
            warnings.warn(repr(ex))
            fig = gcf()
    return fig


def get_ax(fnum=None, pnum=None):
    figure(fnum=fnum, pnum=pnum)
    ax = gca()
    return ax


def figure(fnum=None, docla=False, title=None, pnum=(1, 1, 1), figtitle=None,
           doclf=False, **kwargs):
    '''
    fnum = fignum = figure number
    pnum = plotnum = plot tuple
    '''
    #matplotlib.pyplot.xkcd()
    fig = get_fig(fnum)
    axes_list = fig.get_axes()
    # Ensure my customized settings
    customize_figure(fig, docla)
    # Convert pnum to tuple format
    if tools.is_int(pnum):
        nr = pnum // 100
        nc = pnum // 10 - (nr * 10)
        px = pnum - (nr * 100) - (nc * 10)
        pnum = (nr, nc, px)
    if doclf:  # a bit hacky. Need to rectify docla and doclf
        fig.clf()
    # Get the subplot
    if docla or len(axes_list) == 0:
        printDBG('[df2] *** NEW FIGURE %r.%r ***' % (fnum, pnum))
        if not pnum is None:
            #ax = plt.subplot(*pnum)
            ax = fig.add_subplot(*pnum)
            ax.cla()
        else:
            ax = gca()
    else:
        printDBG('[df2] *** OLD FIGURE %r.%r ***' % (fnum, pnum))
        if not pnum is None:
            ax = plt.subplot(*pnum)  # fig.add_subplot fails here
            #ax = fig.add_subplot(*pnum)
        else:
            ax = gca()
        #ax  = axes_list[0]
    # Set the title
    if not title is None:
        ax = gca()
        set_title(title, ax=ax)
        # Add title to figure
        if figtitle is None and pnum == (1, 1, 1):
            figtitle = title
        if not figtitle is None:
            set_figtitle(figtitle, incanvas=False)
    return fig


def plot_pdf(data, draw_support=True, scale_to=None, label=None, color=0,
             nYTicks=3):
    fig = gcf()
    ax = gca()
    data = np.array(data)
    if len(data) == 0:
        warnstr = '[df2] ! Warning: len(data) = 0. Cannot visualize pdf'
        warnings.warn(warnstr)
        draw_text(warnstr)
        return
    if len(data) == 1:
        warnstr = '[df2] ! Warning: len(data) = 1. Cannot visualize pdf'
        warnings.warn(warnstr)
        draw_text(warnstr)
        return
    bw_factor = .05
    if isinstance(color, (int, float)):
        colorx = color
        line_color = plt.get_cmap('gist_rainbow')(colorx)
    else:
        line_color = color

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
    if nYTicks is not None:
        yticks = np.linspace(min(y_data), max(y_data), nYTicks)
        ax.set_yticks(yticks)


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
    print('[df2] show_histogram()')
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


def plot_stems(x_data=None, y_data=None):
    if y_data is not None and x_data is None:
        x_data = np.arange(len(y_data))
        pass
    if len(x_data) != len(y_data):
        print('[df2] WARNING plot_stems(): len(x_data)!=len(y_data)')
    if len(x_data) == 0:
        print('[df2] WARNING plot_stems(): len(x_data)=len(y_data)=0')
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


def plot_sift_signature(sift, title='', fnum=None, pnum=None):
    figure(fnum=fnum, pnum=pnum)
    ax = gca()
    plot_bars(sift, 16)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 256)
    space_xticks(9, 16)
    space_yticks(5, 64)
    set_title(title, ax=ax)
    dark_background(ax)
    return ax


def dark_background(ax=None, doubleit=False):
    if ax is None:
        ax = gca()
    xy, width, height = _axis_xy_width_height(ax)
    if doubleit:
        halfw = (doubleit) * (width / 2)
        halfh = (doubleit) * (height / 2)
        xy = (xy[0] - halfw, xy[1] - halfh)
        width *= (doubleit + 1)
        height *= (doubleit + 1)
    rect = matplotlib.patches.Rectangle(xy, width, height, lw=0, zorder=0)
    rect.set_clip_on(True)
    rect.set_fill(True)
    rect.set_color(BLACK * .9)
    rect = ax.add_patch(rect)


def space_xticks(nTicks=9, spacing=16, ax=None):
    if ax is None:
        ax = gca()
    ax.set_xticks(np.arange(nTicks) * spacing)
    small_xticks(ax)


def space_yticks(nTicks=9, spacing=32, ax=None):
    if ax is None:
        ax = gca()
    ax.set_yticks(np.arange(nTicks) * spacing)
    small_yticks(ax)


def small_xticks(ax=None):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)


def small_yticks(ax=None):
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)


def plot_bars(y_data, nColorSplits=1):
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


def phantom_legend_label(label, color, loc='upper right'):
    'adds a legend label without displaying an actor'
    pass
    #phantom_actor = plt.Circle((0, 0), 1, fc=color, prop=FONTS.legend, loc=loc)
    #plt.legend(phant_actor, label, framealpha=.2)
    #plt.legend(*zip(*legend_tups), framealpha=.2)
    #legend_tups = []
    #legend_tups.append((phantom_actor, label))


LEGEND_LOCATION = {
    'upper right':  1,
    'upper left':   2,
    'lower left':   3,
    'lower right':  4,
    'right':        5,
    'center left':  6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center':      10,
}


def legend(loc='upper right'):
    ax = gca()
    ax.legend(prop=FONTS.legend, loc=loc)


def plot_histpdf(data, label=None, draw_support=False, nbins=10):
    freq, _ = plot_hist(data, nbins=nbins)
    plot_pdf(data, draw_support=draw_support, scale_to=freq.max(), label=label)


def plot_hist(data, bins=None, nbins=10, weights=None):
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
    # TODO: There might be a divide by zero warning in here.
    ''' desc = np.random.rand(128)
        desc = desc / np.sqrt((desc**2).sum())
        desc = np.round(desc * 255) '''
    # This is draw, because it is an overlay
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
    kpTrans = Affine2D([( a, 0, x),
                        ( c, d, y),
                        ( 0, 0, 1)])
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


def scores_to_color(score_list, cmap_='hot', logscale=False):
    assert len(score_list.shape) == 1, 'score must be 1d'
    if logscale:
        score_list = np.log2(np.log2(score_list + 2) + 1)
    cmap = plt.get_cmap(cmap_)
    mins = score_list.min()
    rnge = score_list.max() - mins
    if rnge == 0:
        return [cmap(.5) for fx in xrange(len(score_list))]
    else:
        if logscale:
            score2_01 = lambda score: np.log2(1.1 + .9 * (float(score) - mins) / (rnge))
            score_list = np.array(score_list)
            #rank_multiplier = score_list.argsort() / len(score_list)
            #normscore = np.array(map(score2_01, score_list)) * rank_multiplier
            normscore = np.array(map(score2_01, score_list))
            colors =  map(cmap, normscore)
        else:
            score2_01 = lambda score: .1 + .9 * (float(score) - mins) / (rnge)
        colors    = [cmap(score2_01(score)) for score in score_list]
        return colors


def scores_to_cmap(scores, colors=None, cmap_='hot'):
    if colors is None:
        colors = scores_to_color(scores, cmap_=cmap_)
    sorted_colors = [x for (y, x) in sorted(zip(scores, colors))]
    # Make a listed colormap and mappable object
    listed_cmap = mpl.colors.ListedColormap(sorted_colors)
    return listed_cmap


def colorbar(scalars, colors):
    'adds a color bar next to the axes'
    # Parameters
    xy, width, height = _axis_xy_width_height()
    orientation = ['vertical', 'horizontal'][0]
    TICK_FONTSIZE = 8
    #
    listed_cmap = scores_to_cmap(scalars, colors)
    # Create scalar mappable with cmap
    sorted_scalars = sorted(scalars)
    sm = plt.cm.ScalarMappable(cmap=listed_cmap)
    sm.set_array(sorted_scalars)
    # Use mapable object to create the colorbar
    COLORBAR_SHRINK = .42  # 1
    COLORBAR_PAD = .01  # 1
    COLORBAR_ASPECT = np.abs(20 * height / (width))  # 1
    printDBG('[df] COLORBAR_ASPECT = %r' % COLORBAR_ASPECT)

    cb = plt.colorbar(sm, orientation=orientation, shrink=COLORBAR_SHRINK,
                      pad=COLORBAR_PAD, aspect=COLORBAR_ASPECT)
    # Add the colorbar to the correct label
    axis = cb.ax.xaxis if orientation == 'horizontal' else cb.ax.yaxis
    position = 'bottom' if orientation == 'horizontal' else 'right'
    axis.set_ticks_position(position)
    axis.set_ticks([0, .5, 1])
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)


def draw_lines2(kpts1, kpts2, fm=None, fs=None, kpts2_offset=(0, 0),
                color_list=None, scale_factor=1, **kwargs):
    printDBG('-------------')
    printDBG('draw_lines2()')
    printDBG(' * len(fm) = %r' % len(fm))
    printDBG(' * scale_factor = %r' % scale_factor)
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
    xxyy_iter = iter(zip(kpts1_m[0] * scale_factor,
                         kpts2_m[0] * scale_factor + woff,
                         kpts1_m[1] * scale_factor,
                         kpts2_m[1] * scale_factor + hoff))
    if color_list is None:
        if fs is None:  # Draw with solid color
            color_list    = [ LINE_COLOR for fx in xrange(len(fm))]
        else:  # Draw with colors proportional to score difference
            color_list = scores_to_color(fs)
    segments  = [((x1, y1), (x2, y2)) for (x1, x2, y1, y2) in xxyy_iter]
    linewidth = [LINE_WIDTH for fx in xrange(len(fm))]
    line_alpha = LINE_ALPHA
    if LINE_ALPHA_OVERRIDE is not None:
        line_alpha = LINE_ALPHA_OVERRIDE
    line_group = LineCollection(segments, linewidth, color_list, alpha=line_alpha)
    #plt.colorbar(line_group, ax=ax)
    ax.add_collection(line_group)
    #figure(100)
    #plt.hexbin(x,y, cmap=plt.cm.YlOrRd_r)


def draw_kpts(kpts, *args, **kwargs):
    draw_kpts2(kpts, *args, **kwargs)


def draw_kpts2(kpts, offset=(0, 0), ell=SHOW_ELLS, pts=False, pts_color=ORANGE,
               pts_size=POINT_SIZE, ell_alpha=ELL_ALPHA,
               ell_linewidth=ELL_LINEWIDTH, ell_color=ELL_COLOR,
               color_list=None, rect=None, arrow=False, scale_factor=1, **kwargs):
    if not DISTINCT_COLORS:
        color_list = None
    printDBG('-------------')
    printDBG('draw_kpts2():')
    printDBG(' * ell=%r pts=%r' % (ell, pts))
    printDBG(' * scale_factor=%r' % (scale_factor,))
    printDBG(' * offset=%r' % (offset,))
    # get matplotlib info
    ax = gca()
    pltTrans = ax.transData
    ell_actors = []
    # data
    kpts = np.array(kpts)
    kptsT = kpts.T
    x = kptsT[0, :] * scale_factor + offset[0]
    y = kptsT[1, :] * scale_factor + offset[1]
    printDBG(' * drawing kpts.shape=%r' % (kpts.shape,))
    if rect is None:
        rect = ell
        rect = False
        if pts is True:
            rect = False
    if ell or rect:
        # We have the transformation from unit circle to ellipse here. (inv(A))
        a = kptsT[2] * scale_factor
        b = np.zeros(len(a))
        c = kptsT[3] * scale_factor
        d = kptsT[4] * scale_factor

        kpts_iter = izip(x, y, a, b, c, d)
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
        if arrow:
            _kwargs = dict(head_width=.01, length_includes_head=False)
            arrow_actors1 = [FancyArrow(0, 0, 0, 1, transform=aff, **_kwargs) for aff in aff_list]
            arrow_actors2 = [FancyArrow(0, 0, 1, 0, transform=aff, **_kwargs) for aff in aff_list]
            patch_list += arrow_actors1
            patch_list += arrow_actors2
        ellipse_collection = matplotlib.collections.PatchCollection(patch_list)
        ellipse_collection.set_facecolor('none')
        ellipse_collection.set_transform(pltTrans)
        if ELL_ALPHA_OVERRIDE is not None:
            ell_alpha = ELL_ALPHA_OVERRIDE
        ellipse_collection.set_alpha(ell_alpha)
        ellipse_collection.set_linewidth(ell_linewidth)
        if not color_list is None:
            ell_color = color_list
        if ell_color == 'distinct':
            ell_color = distinct_colors(len(kpts))
        ellipse_collection.set_edgecolor(ell_color)
        ax.add_collection(ellipse_collection)
    if pts:
        if color_list is None:
            color_list = [pts_color for _ in xrange(len(x))]
        ax.autoscale(enable=False)
        ax.scatter(x, y, c=color_list, s=2 * pts_size, marker='o', edgecolor='none')
        #ax.autoscale(enable=False)
        #ax.plot(x, y, linestyle='None', marker='o', markerfacecolor=pts_color, markersize=pts_size, markeredgewidth=0)


# ---- CHIP DISPLAY COMMANDS ----
def imshow(img, fnum=None, title=None, figtitle=None, pnum=None,
           interpolation='nearest', cmap=None, heatmap=False, **kwargs):
    'other interpolations = nearest, bicubic, bilinear'
    #printDBG('[df2] ----- IMSHOW ------ ')
    #printDBG('[***df2.imshow] fnum=%r pnum=%r title=%r *** ' % (fnum, pnum, title))
    #printDBG('[***df2.imshow] img.shape = %r ' % (img.shape,))
    #printDBG('[***df2.imshow] img.stats = %r ' % (util.printable_mystats(img),))
    fig = figure(fnum=fnum, pnum=pnum, title=title, figtitle=figtitle, **kwargs)
    ax = gca()
    if not DARKEN is None:
        imgdtype = img.dtype
        img = np.array(img, dtype=float) * DARKEN
        img = np.array(img, dtype=imgdtype)

    plt_imshow_kwargs = {
        'interpolation': interpolation,
        #'cmap': plt.get_cmap('gray'),
    }
    if cmap is None and not heatmap:
        plt_imshow_kwargs['vmin'] = 0
        plt_imshow_kwargs['vmax'] = 255
    if heatmap:
        cmap = 'hot'
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            # img is in a color format
            imgBGR = img
            if imgBGR.dtype == np.float64:
                if imgBGR.max() <= 1:
                    printDBG('Drawing Float Color Image < 1')
                    imgBGR = np.array(imgBGR, dtype=np.float32)
                else:
                    printDBG('Drawing Float Color Image > 1')
                    imgBGR = np.array(imgBGR, dtype=np.uint8)
            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
            ax.imshow(imgRGB, **plt_imshow_kwargs)
        elif len(img.shape) == 2:
            # img is in grayscale
            imgGRAY = img
            if cmap is None:
                cmap = plt.get_cmap('gray')
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            if imgGRAY.dtype == np.float32 and False:
                if imgGRAY.max() <= 1:
                    printDBG('Drawing Float Grey Image < 1')
                    imgGRAY = np.array(np.round(imgGRAY * 255), dtype=np.uint8)
                else:
                    printDBG('Drawing Float Grey Image > 1')
                    imgGRAY = np.array(np.round(imgGRAY), dtype=np.uint8)
            ax.imshow(imgGRAY, cmap=cmap, **plt_imshow_kwargs)
        else:
            raise Exception('unknown image format')
    except TypeError as te:
        print('[df2] imshow ERROR %r' % (te,))
        raise
    except Exception as ex:
        print('!!!!!!!!!!!!!!WARNING!!!!!!!!!!!')
        print('[df2] type(img) = %r' % type(img))
        if not isinstance(img, np.ndarray):
            print('!!!!!!!!!!!!!!ERRROR!!!!!!!!!!!')
            pass
            #print('img = %r' % (img,))
        print('[df2] img.dtype = %r' % (img.dtype,))
        print('[df2] type(img) = %r' % (type(img),))
        print('[df2] img.shape = %r' % (img.shape,))
        print('[df2] imshow ERROR %r' % ex)
        raise
    #plt.set_cmap('gray')
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_autoscale(False)
    #try:
        #if pnum == 111:
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


def stack_image_list(img_list, **kwargs):
    if len(img_list) == 0:
        return None
    imgB = img_list[0]
    offset_list = []
    for count, img2 in enumerate(img_list):
        if count == 0:
            continue
        imgB, woff, hoff = stack_images(imgB, img2, **kwargs)
        offset_list.append((woff, hoff))
    return imgB


def stack_image_recurse(img_list1, img_list2=None, vert=True):
    if img_list2 is None:
        # Initialization and error checking
        if len(img_list1) == 0:
            return None
        if len(img_list1) == 1:
            return img_list1[0]
        return stack_image_recurse(img_list1[0::2], img_list1[1::2], vert=vert)
    if len(img_list1) == 1:
        # Left base case
        img1 = img_list1[0]
    else:
        # Left recurse
        img1 = stack_image_recurse(img_list1[0::2], img_list1[1::2], vert=not vert)
    if len(img_list2) == 1:
        # Right base case
        img2 = img_list2[0]
    else:
        # Right Recurse
        img2 = stack_image_recurse(img_list2[0::2], img_list2[1::2], vert=not vert)
    imgB, woff, hoff = stack_images(img1, img2, vert=vert)
    return imgB


def stack_images(img1, img2, vert=None):
    nChannels = get_num_channels(img1)
    nChannels2 = get_num_channels(img2)
    assert nChannels == nChannels2
    (h1, w1) = img1.shape[0: 2]  # get chip dimensions
    (h2, w2) = img2.shape[0: 2]
    woff, hoff = 0, 0
    vert_wh  = max(w1, w2), h1 + h2
    horiz_wh = w1 + w2, max(h1, h2)
    if vert is None:
        # Display the orientation with the better (closer to 1) aspect ratio
        vert_ar  = max(vert_wh) / min(vert_wh)
        horiz_ar = max(horiz_wh) / min(horiz_wh)
        vert = vert_ar < horiz_ar
    if vert:
        wB, hB = vert_wh
        hoff = h1
    else:
        wB, hB = horiz_wh
        woff = w1
    # concatentate images
    dtype = img1.dtype
    assert img1.dtype == img2.dtype
    if nChannels == 3:
        imgB = np.zeros((hB, wB, 3), dtype)
        imgB[0:h1, 0:w1, :] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2), :] = img2
    elif nChannels == 1:
        imgB = np.zeros((hB, wB), dtype)
        imgB[0:h1, 0:w1] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2
    return imgB, woff, hoff


def show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=None, fs=None, title=None,
                    vert=None, fnum=None, pnum=None, heatmap=False, **kwargs):
    '''Draws two chips and the feature matches between them. feature matches
    kpts1 and kpts2 use the (x,y,a,c,d)
    '''
    printDBG('[df2] draw_matches2() fnum=%r, pnum=%r' % (fnum, pnum))
    # get matching keypoints + offset
    (h1, w1) = rchip1.shape[0:2]  # get chip (h, w) dimensions
    (h2, w2) = rchip2.shape[0:2]
    # Stack the compared chips
    match_img, woff, hoff = stack_images(rchip1, rchip2, vert)
    xywh1 = (0, 0, w1, h1)
    xywh2 = (woff, hoff, w2, h2)
    # Show the stacked chips
    fig, ax = imshow(match_img, title=title, fnum=fnum, pnum=pnum, heatmap=heatmap)
    # Overlay feature match nnotations
    draw_fmatch(xywh1, xywh2, kpts1, kpts2, fm, fs, **kwargs)
    return ax, xywh1, xywh2


# draw feature match
def draw_fmatch(xywh1, xywh2, kpts1, kpts2, fm, fs=None, lbl1=None, lbl2=None,
                fnum=None, pnum=None, rect=False, colorbar_=True, **kwargs):
    '''Draws the matching features. This is draw because it is an overlay
    xywh1 - location of rchip1 in the axes
    xywh2 - location or rchip2 in the axes
    '''
    if fm is None:
        assert kpts1.shape == kpts2.shape, 'shapes different or fm not none'
        fm = np.tile(np.arange(0, len(kpts1)), (2, 1)).T
    pts       = kwargs.get('draw_pts', False)
    ell       = kwargs.get('draw_ell', True)
    lines     = kwargs.get('draw_lines', True)
    ell_alpha = kwargs.get('ell_alpha', .4)
    nMatch = len(fm)
    #printDBG('[df2.draw_fnmatch] nMatch=%r' % nMatch)
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    offset2 = (x2, y2)
    # Custom user label for chips 1 and 2
    if lbl1 is not None:
        absolute_lbl(x1 + w1, y1, lbl1)
    if lbl2 is not None:
        absolute_lbl(x2 + w2, y2, lbl2)
    # Plot the number of matches
    if kwargs.get('show_nMatches', False):
        upperleft_text('#match=%d' % nMatch)
    # Draw all keypoints in both chips as points
    if kwargs.get('all_kpts', False):
        all_args = dict(ell=False, pts=pts, pts_color=GREEN, pts_size=2,
                        ell_alpha=ell_alpha, rect=rect)
        all_args.update(kwargs)
        draw_kpts2(kpts1, **all_args)
        draw_kpts2(kpts2, offset=offset2, **all_args)
    # Draw Lines and Ellipses and Points oh my
    if nMatch > 0:
        colors = [kwargs['colors']] * nMatch if 'colors' in kwargs else distinct_colors(nMatch)
        if fs is not None:
            colors = scores_to_color(fs, 'hot')

        acols = add_alpha(colors)

        # Helper functions
        def _drawkpts(**_kwargs):
            _kwargs.update(kwargs)
            fxs1 = fm[:, 0]
            fxs2 = fm[:, 1]
            draw_kpts2(kpts1[fxs1], rect=rect, **_kwargs)
            draw_kpts2(kpts2[fxs2], offset=offset2, rect=rect, **_kwargs)

        def _drawlines(**_kwargs):
            _kwargs.update(kwargs)
            draw_lines2(kpts1, kpts2, fm, fs, kpts2_offset=offset2, **_kwargs)

        # User util
        if ell:
            _drawkpts(pts=False, ell=True, color_list=colors)
        if pts:
            _drawkpts(pts_size=8, pts=True, ell=False, pts_color=BLACK)
            _drawkpts(pts_size=6, pts=True, ell=False, color_list=acols)
        if lines:
            _drawlines(color_list=colors)
    else:
        draw_boxedX(xywh2)
    if fs is not None and colorbar_ and 'colors' in vars() and colors is not None:
        colorbar(fs, colors)
    #legend()
    return None


def draw_boxedX(xywh, color=RED, lw=2, alpha=.5, theta=0):
    'draws a big red x. redx'
    ax = gca()
    x1, y1, w, h = xywh
    x2, y2 = x1 + w, y1 + h
    segments = [((x1, y1), (x2, y2)),
                ((x1, y2), (x2, y1))]
    trans = Affine2D()
    trans.rotate(theta)
    trans = trans + ax.transData
    width_list = [lw] * len(segments)
    color_list = [color] * len(segments)
    line_group = LineCollection(segments, width_list, color_list, alpha=alpha,
                                transOffset=trans)
    ax.add_collection(line_group)


def disconnect_callback(fig, callback_type, **kwargs):
    #print('[df2] disconnect %r callback' % callback_type)
    axes = kwargs.get('axes', [])
    for ax in axes:
        ax._hs_viewtype = ''
    cbid_type = callback_type + '_cbid'
    cbfn_type = callback_type + '_func'
    cbid = fig.__dict__.get(cbid_type, None)
    cbfn = fig.__dict__.get(cbfn_type, None)
    if cbid is not None:
        fig.canvas.mpl_disconnect(cbid)
    else:
        cbfn = None
    fig.__dict__[cbid_type] = None
    return cbid, cbfn


def connect_callback(fig, callback_type, callback_fn):
    #print('[df2] register %r callback' % callback_type)
    if callback_fn is None:
        return
    cbid_type = callback_type + '_cbid'
    cbfn_type = callback_type + '_func'
    fig.__dict__[cbid_type] = fig.canvas.mpl_connect(callback_type, callback_fn)
    fig.__dict__[cbfn_type] = callback_fn
