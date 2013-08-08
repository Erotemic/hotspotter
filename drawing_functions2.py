'''
Lots of functions for drawing and plotting visiony things
'''
import matplotlib
if matplotlib.get_backend() != 'Qt4Agg':
    print('Configuring matplotlib for Qt4Agg')
    matplotlib.use('Qt4Agg', warn=True, force=True)
    matplotlib.rcParams['toolbar'] = 'None'
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
import cv2
import matplotlib.pyplot as plt
import numpy as np
import types
import warnings
import helpers
import textwrap
print('LOAD_MODULE: drawing_functions2.py')

DPI = 80
#FIGSIZE = (24) # default windows fullscreen
FIGSIZE = (20,10) 


def printDBG(msg):
    #print(msg)
    pass

# ---- GENERAL FIGURE COMMANDS ----
def save_figure(fignum=None, fpath=None):
    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum, figsize=FIGSIZE, dpi=DPI)
    fignum = fig.number
    if fpath is None:
       fpath = fig.title
    print('Saving figure to: '+repr(fpath))
    fig.savefig(fpath, dpi=DPI)

def set_figsize(fignum, width, height):
    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum, figsize=FIGSIZE, dpi=DPI)
    qtwin = fig.canvas.manager.window
    qtwin.setGeometry(40,40,width,height)
    fig.canvas.draw()

def get_all_figures():
    all_figures_=[manager.canvas.figure for manager in
                 matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    all_figures = []
    # Make sure you dont show figures that this module closed
    for fig in iter(all_figures_):
        if not 'df2_closed' in fig.__dict__.keys() or not fig.df2_closed:
            all_figures.append(fig)
    # Return all the figures sorted by their number
    all_figures = sorted(all_figures, key=lambda fig: fig.number)
    return all_figures

def show_all_figures():
    for fig in iter(get_all_figures()):
        fig.show()
        fig.canvas.draw()
import sys
def tile_all_figures(num_rc=(4,4),
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
#myprint(fig.canvas.manager.window) 
# the manager should be a qt window

def bring_to_front_all_figures():
    from PyQt4.QtCore import Qt
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        qtwin.raise_()
        qtwin.activateWindow()
        qtwin.setWindowFlags(Qt.WindowStaysOnTopHint)
        qtwin.show()
        qtwin.setWindowFlags(Qt.WindowFlags(0))
        qtwin.show()
        #what is difference between show and show normal?

def close_all_figures():
    from PyQt4.QtCore import Qt
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        fig.clf()
        fig.df2_closed = True
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        qtwin.close()

def reset():
    close_all_figures()

def present(*args, **kwargs):
    'execing present should cause IPython magic'
    print('Presenting figures...')
    tile_all_figures(*args, **kwargs)
    show_all_figures()
    bring_to_front_all_figures()
    # Return an exec string
    return textwrap.dedent(r'''
    import helpers
    import matplotlib.pyplot as plt
    embedded = False
    if not helpers.inIPython():
        if '--cmd' in sys.argv:
            print('Requested IPython shell with --cmd argument.')
            if helpers.haveIPython():
                print('Found IPython')
                try: 
                    import IPython
                    print('Presenting in new ipython shell.')
                    embedded = True
                    IPython.embed()
                except Exception as ex:
                    printWARN(repr(ex)+'\n!!!!!!!!')
                    embedded = False
            else:
                print('IPython is not installed')
        if not embedded: 
            print('Presenting in normal shell.')
            print('... plt.show()')
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
    if type(fignum_) == types.StringType:
        (fignum2, plotnum2) = map(int, fignum.split('.'))
    elif type(fignum_) == types.FloatType:
        raise Exception('Error. This is bad buisness')
        (fignum2, plotnum2) = (int(fignum_), int(round(fignum_*1000)) - int(fignum_)*1000)
    else:
        (fignum2, plotnum2) = (fignum_, plotnum_)
    return fignum2, plotnum2

def figure(fignum=None, doclf=False, title=None, plotnum=111, figtitle=None):
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
            fig.canvas.set_window_title('fig '+str(fignum)+' '+figtitle)
    return fig

def set_figtitle(figtitle):
    fig = plt.gcf()
    fig.canvas.set_window_title(figtitle)

    
# ---- IMAGE CREATION FUNCTIONS ---- 

# adapted from:
# http://jayrambhia.com/blog/sift-keypoint-matching-using-python-opencv/
def draw_matches(rchip1, rchip2, kpts1, kpts2, fm12, vert=False, color=(255,0,0)):
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
        match_img = cv2.line(match_img, pt1, pt2, color)
    return match_img

def draw_matches2(kpts1, kpts2, fm, fs=None, kpts2_offset=(0,0), color=(1.,0.,0.), alpha=.4):
    # input data
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
    ''' 
    OLD WAY: 
    line_actors = [ plt.Line2D((x1,x2), 
                               (y1,y2))
                           for (x1,x2,y1,y2) in xxyy_iter ]
    # add lines
    line_collection = matplotlib.collections.PatchCollection(line_actors,
                                                             color=color,
                                                             alpha=alpha)
    '''
    if fs is None:
        segments  = [ ((x1, y1), (x2,y2)) for (x1,x2,y1,y2) in xxyy_iter ] 
        colors    = [ color for fx in xrange(len(fm)) ] 
        linewidth = [ 1.5 for fx in xrange(len(fm)) ] 
    else:
        cmap = plt.get_cmap('hot')
        mins = fs.min()
        maxs = fs.max()
        segments  = [ ((x1, y1), (x2,y2)) for (x1,x2,y1,y2) in xxyy_iter ] 
        colors    = [ cmap(.1+ .9*(fs[fx]-mins)/(maxs-mins)) for fx in xrange(len(fm)) ] 
        linewidth = [ 1.5 for fx in xrange(len(fm)) ] 

    line_collection = matplotlib.collections.LineCollection(segments, linewidth, colors,
                                          alpha=alpha)
    ax.add_collection(line_collection)

def draw_kpts2(kpts, color=(0.,0.,1.), alpha=.5, offset=(0,0)):
    # get matplotlib info
    ax = plt.gca()
    pltTrans = ax.transData
    ell_actors = []
    eps = 1E-9
    # data
    kptsT = kpts.T
    x = kptsT[0] + offset[0]
    y = kptsT[1] + offset[1]
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
    ellipse_collection.set_alpha(alpha)
    ellipse_collection.set_edgecolor(color)
    ax.add_collection(ellipse_collection)
    
def draw_kpts(_rchip, _kpts, color=(0,0,255)):
    kpts_img = np.copy(_rchip)
    # Draw circles
    for (x,y,a,d,c) in iter(_kpts):
        center = (int(x), int(y))
        radius = int(3*np.sqrt(1/a))
        kpts_img = cv2.circle(kpts_img, center, radius, color)
    return kpts_img

def cv2_draw_kpts(img, cvkpts):
    cv_flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    kpts_img = cv2.drawKeypoints(img, cvkpts, flags=cv_flags)
    return kpts_img


# ---- OLD CHIP DISPLAY COMMANDS ----

def show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12, fignum=0, title=None):
    printDBG('*** Showing %d matches between cxs=(%d,%d) ***' % (len(fm12), qcx, cx))
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    rchip1 = cv2.imread(cx2_rchip_path[qcx])
    rchip2 = cv2.imread(cx2_rchip_path[cx])
    kpts1  = cx2_kpts[qcx]
    kpts2  = cx2_kpts[cx]
    assert kpts1.shape[1] == 5, 'kpts1 must be Nx5 ellipse'
    assert kpts2.shape[1] == 5, 'kpts2 must be Nx5 ellipse'
    assert  fm12.shape[1] == 2, 'fm must be Mx2'
    rchipkpts1 = draw_kpts(rchip1, kpts1[fm12[:,0],:])
    rchipkpts2 = draw_kpts(rchip2, kpts2[fm12[:,1],:])
    chipkptsmatches = draw_matches(rchipkpts1, rchipkpts2,
                                   kpts1, kpts2,
                                   fm12, vert=True)
    imshow(chipkptsmatches, fignum=fignum, title=title)

# ---- CHIP DISPLAY COMMANDS ----

def imshow(img, fignum=0, title=None, figtitle=None, plotnum=111):
    printDBG('*** imshow in fig=%r title=%r *** ' % (fignum, title))
    fignum, plotnum = __parse_fignum(fignum,plotnum)
    printDBG('   * fignum = %r, plotnum = %r ' % (fignum, plotnum))
    fig = figure(fignum=fignum, plotnum=plotnum, title=title, figtitle=figtitle)
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
                  fm, fs=None, fignum=0, plotnum=111,
                  title=None, vert=True):
    '''Draws feature matches 
    kpts1 and kpts2 use the (x,y,a,c,d)
    '''
    # get chip dimensions 
    (h1,w1) = rchip1.shape[0:2]
    (h2,w2) = rchip2.shape[0:2]
    woff = 0; hoff = 0 
    if vert: wB=max(w1,w2); hB=h1+h2; hoff=h1
    else:    hB=max(h1,h2); wB=w1+w2; woff=w1
    vert = True
    if vert: wB=max(w1,w2); hB=h1+h2; hoff=h1
    else:    hB=max(h1,h2); wB=w1+w2; woff=w1
    # concatentate images
    match_img = np.zeros((hB, wB, 3), np.uint8)
    match_img[0:h1, 0:w1, :] = rchip1
    match_img[hoff:(hoff+h2), woff:(woff+w2), :] = rchip2
    # get matching keypoints + offset
    if len(fm) == 0:
        imshow(match_img,fignum=fignum,plotnum=plotnum,title=title)
    else: 
        kpts1_m = kpts1[fm[:,0]]
        kpts2_m = kpts2[fm[:,1]]
        # matplotlib stuff
        imshow(match_img,fignum=fignum,plotnum=plotnum,title=title)
        draw_kpts2(kpts1_m)
        draw_kpts2(kpts2_m,offset=(woff,hoff))
        draw_matches2(kpts1,kpts2,fm,fs,kpts2_offset=(woff,hoff))

def show_matches3(res, hs, cx, SV=True, fignum=0, plotnum=111, title_aug=None):
    print('Showing matches from '+str(res.qcx)+' to '+str(cx)+' in fignum'+repr(fignum))
    if np.isnan(cx):
        nan_img = np.zeros((100,100), dtype=np.uint8)
        title='(qx%r v NAN)' % (res.qcx)
        imshow(nan_img,fignum=fignum,plotnum=plotnum,title=title)
        return 
    cx2_rchip_path = hs.cpaths.cx2_rchip_path
    cx2_kpts = hs.feats.cx2_kpts
    qcx = res.qcx
    rchip1 = cv2.imread(cx2_rchip_path[qcx])
    rchip2 = cv2.imread(cx2_rchip_path[cx])
    kpts1  = cx2_kpts[qcx]
    kpts2  = cx2_kpts[cx]
    cx2_score = res.cx2_score_V if SV else res.cx2_score
    cx2_fm    = res.cx2_fm_V if SV else res.cx2_fm
    cx2_fs    = res.cx2_fs_V if SV else res.cx2_fs
    score = cx2_score[cx]
    fm    = cx2_fm[cx]
    fs    = cx2_fs[cx]
    nMatches = len(fm)
    title='(qx%r v cx%r)\n #match=%r score=%.2f' % (qcx, cx, nMatches, score)
    if not title_aug is None:
        title = title_aug + title
    if SV:
        title += '(+V)'
    show_matches2(rchip1, rchip2, kpts1,  kpts2, 
                  fm, fs, fignum=fignum, plotnum=plotnum, title=title)


def show_keypoints(rchip,kpts,fignum=0,title=None):
    imshow(rchip,fignum=fignum,title=title)
    draw_kpts2(kpts)


