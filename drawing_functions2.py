#import pylab
#pylab.set_cmap('gray')
import numpy as np
import matplotlib
import hotspotter.tpl.cv2  as cv2

# adapted from:
# http://jayrambhia.com/blog/sift-keypoint-matching-using-python-opencv/
def draw_matches(rchip1, rchip2, kpts1, kpts2, matches12, vert=False, color=(255,0,0)):
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
    for kx1, kx2 in iter(matches12):
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

# --------
def get_all_figures():
    all_figures=[manager.canvas.figure for manager in
                 matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    return all_figures

def show_all_figures():
    for fig in iter(get_all_figures()):
        fig.show()
        fig.canvas.draw()

def tile_all_figures(num_rows=3):
    all_figures = get_all_figures()
    for i, fig in enumerate(all_figures):
        #myprint(fig.canvas.manager.window) 
        # the manager should be a qt window
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        h = 300
        w = 300
        y = (i%num_rows)*w
        x = (int(i/num_rows))*h
        qtwin.setGeometry(x,y,w,h)

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
        qtwin = fig.canvas.manager.window
        if not isinstance(qtwin, matplotlib.backends.backend_qt4.MainWindow):
            raise NotImplemented('need to add more window manager handlers')
        qtwin.close()
