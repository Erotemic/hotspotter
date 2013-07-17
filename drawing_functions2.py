import matplotlib
print('Configuring matplotlib for Qt4')
matplotlib.use('Qt4Agg')
matplotlib.rcParams['toolbar'] = 'None'

#import pylab
#pylab.set_cmap('gray')
import numpy as np
import matplotlib.pyplot as plt
import hotspotter.tpl.cv2 as cv2

def printDBG(msg):
    #print(msg)
    pass


# ---- GENERAL FIGURE COMMANDS ----
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

def tile_all_figures():
    row_first = False
    num_rows=3
    num_cols=4
    hpad = 0 #75
    wpad = 0
    h = 250
    w = 350
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        if row_first:
            y = (i % num_rows)*(h+hpad)
            x = (int(i/num_rows))*w
        else:
            x = (i % num_cols)*w
            y = (int(i/num_cols))*(h+hpad)
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

def present():
    tile_all_figures()
    show_all_figures()
    bring_to_front_all_figures()

def figure(fignum, doclf=False, title=None):
    fig = plt.figure(fignum)
    axes_list = fig.get_axes()
    if not 'user_stat_list' in fig.__dict__.keys() or doclf:
        fig.user_stat_list = []
        fig.user_notes = []
    fig.df2_closed = False
    if doclf or len(axes_list) == 0:
        fig.clf()
        ax = plt.subplot(111)
        printDBG('*** NEW FIGURE '+str(fignum)+' ***')
    else: 
        ax  = axes_list[0]
    if not title is None:
        ax.set_title(title)
        fig.canvas.set_window_title('fig '+str(fignum)+' '+title)
    return fig
    
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
    
def draw_kpts(_rchip, _kpts, color=(0,0,255)):
    kpts_img = np.copy(_rchip)
    # Draw circles
    for (x,y,a,d,c) in iter(_kpts):
        center = (int(x), int(y))
        radius = int(3*np.sqrt(1/a))
        kpts_img = cv2.circle(kpts_img, center, radius, color)
    return kpts_img

# ---- CHIP DISPLAY COMMANDS ----

def imshow(img, fignum=0, title=None):
    printDBG('*** imshow in fig=%d title=%r *** ' % (fignum, title))
    fig = figure(fignum, doclf=True, title=title)
    plt.imshow(img)
    ax = fig.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        fig.tight_layout()
    except Exception as ex:
        print('!! Exception durring fig.tight_layout: '+repr(ex))

def show_matches(qcx, cx, hs_cpaths, cx2_kpts, fm12, fignum=0, title=None):
    printDBG('*** Showing %d matches between cxs=(%d,%d) ***' % (len(fm12), qcx, cx))
    cx2_rchip_path = hs_cpaths.cx2_rchip_path
    rchip1 = cv2.imread(cx2_rchip_path[qcx])
    rchip2 = cv2.imread(cx2_rchip_path[cx])
    kpts1  = cx2_kpts[qcx]
    kpts2  = cx2_kpts[cx]
    assert kpts1.shape[1] == 5, 'kpts1 must be Nx5 ellipse'
    assert kpts2.shape[1] == 5, 'kpts2 must be Nx5 ellipse'
    assert fm12.shape[1] == 2, 'fm must be Mx2'
    rchipkpts1 = draw_kpts(rchip1, kpts1[fm12[:,0],:])
    rchipkpts2 = draw_kpts(rchip2, kpts2[fm12[:,1],:])
    chipkptsmatches = draw_matches(rchipkpts1, rchipkpts2,
                                   kpts1, kpts2,
                                   fm12, vert=True)
    imshow(chipkptsmatches, fignum=fignum, title=title)

def show_matches2(rchip1, rchip2, kpts1, kpts2, fm12, fignum=0, title=None):
    img_rchip_kpts1 = draw_kpts(rchip1, kpts1[fm12[:,0],:])
    img_rchip_kpts2 = draw_kpts(rchip2, kpts2[fm12[:,1],:])
    img_matches = draw_matches(img_rchip_kpts1, img_rchip_kpts2,
                               kpts1, kpts2, fm12, vert=True)
    imshow(img_matches, fignum=fignum, title=title)

def show_keypoints(rchip, kpts, fignum=0, title=None):
    img_rchip_kpts = draw_kpts(rchip, kpts)
    imshow(img_rchip_kpts, fignum=fignum, title=title)
