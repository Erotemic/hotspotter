import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.pyplot import draw, figure, get_cmap, gray
from matplotlib.transforms import Affine2D
from numpy import array, uint32, round, sqrt, ceil, asarray, append
from numpy import spacing as eps
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.other.ConcretePrintable import Pref
from hotspotter.other.logger import logmsg, logdbg, logwarn
from warnings import catch_warnings, simplefilter
import colorsys
import os.path
import sys

class DrawManager(AbstractManager):
    def init_preferences(dm, default_bit=False):
        iom = dm.hs.iom
        if dm.draw_prefs == None:
            dm.draw_prefs = Pref(fpath=iom.get_prefs_fpath('draw_prefs'))
        dm.draw_prefs.bbox_bit       = True
        dm.draw_prefs.ellipse_bit    = False
        dm.draw_prefs.ellipse_alpha  = .6
        dm.draw_prefs.points_bit     = False
        dm.draw_prefs.result_view  = Pref(1, choices=['in_image', 'in_chip'])
        dm.draw_prefs.fignum         = 0
        dm.draw_prefs.num_result_cols = 3
        dm.draw_prefs.figsize        = (5,5)
        dm.draw_prefs.colormap       = Pref('hsv', hidden=True)
        dm.draw_prefs.in_qtc_bit     = Pref(False, hidden=True) #Draw in the Qt Console
        dm.draw_prefs.use_thumbnails = Pref(False, hidden=True)
        dm.draw_prefs.thumbnail_size = Pref(128, hidden=True)
        if not default_bit:
            dm.draw_prefs.load()
    # ---
    def show_splash(dm):
        splash_fname = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'splash.png')
        if not os.path.exists(splash_fname):
            root_dir = os.path.realpath(os.path.dirname(__file__))
            while root_dir!=None:
                splash_fname = os.path.join(root_dir, "hotspotter", "front", 'splash.png')
                logdbg(splash_fname)
                exists_test = os.path.exists(splash_fname)
                logdbg('Exists:'+str(exists_test))
                if exists_test:
                    break
                tmp = os.path.dirname(root_dir)
                if tmp == root_dir:
                    root_dir = None
                else:
                    root_dir = tmp
        logdbg('Splash Fname: %r '% splash_fname)
        splash_img = asarray(Image.open(splash_fname))
        dm.add_images([splash_img],['Welcome to Hotspotter'])
        dm.end_draw()
    # ---
    def show_image(dm, gx):
        gm, cm = dm.hs.get_managers('gm','cm')
        gid        = gm.gx2_gid[gx]
        img_list   = gm.gx2_img_list(gx)
        title_list = ['gid='+str(gid)+'   gname='+gm.gx2_gname[gx]]
        dm.add_images(img_list, title_list)
        cx_list    = gm.gx2_cx_list[gx]
        if dm.draw_prefs.use_thumbnails is True:
            pass
        for cx in iter(cx_list):
            dm.draw_chiprep2(cx, axi=0, in_image_bit=True)
        dm.end_draw()
    # ---
    def show_chip(dm, cx, in_raw_chip=False, **kwargs):
        cm = dm.hs.cm
        cid, gname, chip = cm.cx2_(cx, 'cid', 'gname', 'chip')
        if in_raw_chip:
            chip = np.asarray(cm.cx2_pil_chip(cx, scaled=True,
                                              preprocessed=False, rotated=True,
                                             colored=True))
        if dm.draw_prefs.use_thumbnails is True:
            pass
        dm.add_images([chip], [gname])
        # Draw chiprep and return fsel incase rand is good
        fsel_ret = dm.draw_chiprep2(cx, axi=0, **kwargs)
        dm.end_draw()
        return fsel_ret
    # ---
    def show_query(dm, res, titleargs=None, enddraw=True):
        # Make sure draw is valid
        if res is None: dm.show_splash(); return
        # Get Chip Properties

        cm  = res.hs.cm
        qcm = res.qhs.cm

        titleargs =\
        ('cx', 'cid', 'nid', 'name', 'gname')
        ( qcx, qcid , qnid , qname , qgname ) = res.qcid2_(*titleargs)
        ( tcx, tcid , tnid , tname , tgname ) = res.tcid2_(*titleargs)

        (tcx, tscore, ) = res.tcid2_('cx','score')
        # Titles of the Plot

        #qtitle = 'gname: %s\nQUERY cid=%d, nid=%d' % (qgname, qcid, qnid)
        #ttile = ['cid=%d\n gname: %s\nrank/score=%d,%.2f' % tup for tup in zip(tcid, tgname, range(1,len(tscore)+1), tscore)]
        qtitle = 'gname: %s\nQUERY nid=%d' % (qgname, qnid)
        ttile = ['gname: %s\nrank/score=%d/%.2f' % tup for tup in zip(tgname, range(1,len(tscore)+1), tscore)]
        title_list = [qtitle] + ttile
        if dm.draw_prefs.use_thumbnails is True:
            pass

        # Add the images to draw
        in_image_bit = dm.draw_prefs.result_view == 'in_image'
        if in_image_bit:
            qimg = qcm.cx2_img(qcx)
            timg = cm.cx2_img_list(tcx)
            dm.add_images([qimg] + timg, title_list)
        elif dm.draw_prefs.result_view == 'in_chip':
            qchip = qcm.cx2_chip_list(qcx)
            tchip = cm.cx2_chip_list(tcx)
            dm.add_images(qchip + tchip, title_list)

        # Draw the Query Chiprep
        qaxi       = 0; qfsel      = []
        dm.draw_chiprep2(qcx, axi=qaxi, fsel=qfsel, qcm=qcm)
        # Draw the Top Result Chipreps
        for (tx, cx) in enumerate(tcx):
            fm    = res.rr.cx2_fm[cx]
            fs    = res.rr.cx2_fs[cx]
            axi   = tx+1
            if len(fs) == 0:
                qfsel = np.array([], np.uint32)
                fsel  = np.array([], np.uint32)
            else:
                qfsel = fm[fs > 0][:,0]
                fsel  = fm[fs > 0][:,1]
            dm.draw_chiprep2(cx,
                             axi=axi,
                             axi_color=axi,
                             fsel=fsel,
                             in_image_bit=in_image_bit)
            dm.draw_chiprep2(qcx,
                             axi=qaxi,
                             axi_color=axi,
                             fsel=qfsel,
                             in_image_bit=in_image_bit,
                             qcm=qcm)
        if enddraw:
            dm.end_draw()

    # ---

    def __init__(dm, hs):
        super( DrawManager, dm ).__init__( hs )        
        dm.hs      =   hs
        dm.fignum  =    0
        dm.dpi     = 100 #72
        dm.draw_prefs = None
        dm.ax_list =   []
        dm.init_preferences()
    # ---
    def update_figsize(dm):
        fig = dm.get_figure()
        dm.draw_prefs.figsize = (fig.get_figheight(), fig.get_figwidth())
    # ---
    def get_figure(dm):
        guifig = dm.hs.uim.get_gui_figure()
        if guifig != None and dm.fignum == 0: # Check to see if we have access to the gui
            return guifig
        fig = figure(num=dm.fignum, figsize=dm.draw_prefs.figsize, dpi=dm.dpi, facecolor='w', edgecolor='k')
        return fig
    # ---
    def annotate_orientation(dm):
        logmsg('Please select an orientation of the torso (Click Two Points on the Image)')
        try:
            # Compute an angle from user interaction
            sys.stdout.flush()
            fig = dm.get_figure()
            pts = np.array(fig.ginput(2))
            logdbg('GInput Points are: '+str(pts))
            # Get reference point to origin 
            refpt = pts[0] - pts[1] 
            #theta = np.math.atan2(refpt[1], refpt[0])
            theta = np.math.atan(refpt[1]/refpt[0])
            logmsg('The angle in radians is: '+str(theta))
            return theta
        except Exception as ex: 
            logmsg('Annotate Orientation Failed'+str(ex))
            return None
    def annotate_roi(dm):
        logmsg('Please Select a Rectangular Region of Interest (Click Two Points on the Image)')
        try:
            sys.stdout.flush()
            fig = dm.get_figure()
            pts = fig.ginput(2)
            logdbg('GInput Points are: '+str(pts))
            (x1, y1, x2, y2) = (pts[0][0], pts[0][1], pts[1][0], pts[1][1]) 
            xm = min(x1,x2)
            xM = max(x1,x2)
            ym = min(y1,y2)
            yM = max(y1,y2)
            (x, y, w, h) = (xm, ym, xM-xm, yM-ym) 
            roi = array(round([x,y,w,h]),dtype=uint32)
            logmsg('The new ROI is: '+str(roi))
            return roi
        except Exception as ex:
            logmsg('Annotate ROI Failed'+str(ex))
            return None
    # ---
    def end_draw(dm):
        #gray()
        logdbg('Finalizing Draw with '+str(len(dm.ax_list))+' axes')
        fig = dm.get_figure()
        #fig.subplots_adjust(hspace=0.2, wspace=0.2)
        #fig.tight_layout(pad=.3, h_pad=None, w_pad=None)
        #fig.tight_layout()
        if dm.draw_prefs.in_qtc_bit:
            try:
                from IPython.back.display import display
                display(fig)
            except:
                logwarn('Cannot Draw in QTConsole')
        fig.show()
        dm.hs.uim.redraw_gui()
        fig.canvas.draw()
        #draw() 
    # ---
    def save_fig(dm, save_file):
        dm.end_draw()
        fig = dm.get_figure()
        fig.savefig(save_file, format='png')
    # ---
    def add_images(dm, img_list, title_list=[]):
        fig = dm.get_figure(); fig.clf()
        num_images = len(img_list)
        #
        dm.ax_list     = [None]*num_images
        title_list     = title_list + ['NoTitle']*(num_images-len(title_list))
        # Fit Images into a grid
        max_columns = min(num_images, max(1,dm.draw_prefs.num_result_cols))
        if max_columns == 0: max_columns = 1
        nr = int( ceil( float(num_images)/max_columns) )
        nc = max_columns if num_images >= max_columns else 1
        #
        gs = gridspec.GridSpec( nr, nc )
        for i in xrange(num_images):
            #logdbg(' Adding the '+str(i)+'th Image')
            #logdbg('   * type(img_list[i]): %s'+str(type(img_list[i])))
            #logdbg('   * img_list[i].shape: %s'+str(img_list[i].shape))
            dm.ax_list[i] = fig.add_subplot(gs[i])
            imgplot = dm.ax_list[i].imshow(img_list[i])
            imgplot.set_cmap('gray')
            dm.ax_list[i].get_xaxis().set_ticks([])
            dm.ax_list[i].get_yaxis().set_ticks([])
            dm.ax_list[i].set_title(title_list[i])
            # transData: data coordinates -> display coordinates
            # transAxes: axes coordinates -> display coordinates
            # transLimits: data - > axes
        #
        #gs.tight_layout(fig)
        logdbg('Added '+str(num_images)+' images/axes')
    # ---
    def _get_fpt_ell_collection(dm, fpts, T_data, alpha, edgecolor):
        ell_patches = []
        for (x,y,a,c,d) in fpts: # Manually Calculated sqrtm(inv(A))
            with catch_warnings():
                simplefilter("ignore")
                aIS = 1/sqrt(a)
                cIS = (c/sqrt(a) - c/sqrt(d))/(a - d + eps(1))
                dIS = 1/sqrt(d)
            transEll = Affine2D([\
                    ( aIS,   0,   x),\
                    ( cIS, dIS,   y),\
                    (   0,   0,   1)])
            unitCirc1 = Circle((0,0),1,transform=transEll)
            ell_patches = [unitCirc1] + ell_patches
        ellipse_collection = PatchCollection(ell_patches)
        ellipse_collection.set_facecolor('none')
        ellipse_collection.set_transform(T_data)
        ellipse_collection.set_alpha(alpha)
        ellipse_collection.set_edgecolor(edgecolor)
        return ellipse_collection

    
    def draw_chiprep2(dm, cx, axi=0, fsel=None, in_image_bit=False, axi_color=0,
                      bbox_bit=None, 
                      ell_alpha=None,
                      ell_bit=None,
                      xy_bit=None,
                      color=None,
                      qcm=None,
                      **kwargs):
        '''
        Draws a chip representation over an already drawn chip
        cx           - the chiprep to draw. Managed by the chip manager
        axi          - the axis index to draw it in
        #TODO: in_image_bit becomes data_coordinates
        in_image_bit - are the data coordinates image or rotated chip? 
                       raw the chip by itself or in its original image
        axi_color    - use the color associated with axis index 
                       (used for ploting queries)
        ---
        Others are preference overloads
        bbox_bit - 
        ell_alpha
        ell_bit 
        xy_bit
        ell_color
        '''
        # Allows display of cross database queries
        cm = dm.hs.cm if qcm is None else qcm
        # Grab Preferences
        xy_bit  = dm.draw_prefs.points_bit    if xy_bit    is None else xy_bit
        ell_bit = dm.draw_prefs.ellipse_bit   if ell_bit   is None else ell_bit
        bbox_bit  = dm.draw_prefs.bbox_bit      if bbox_bit  is None else bbox_bit
        ell_alpha = dm.draw_prefs.ellipse_alpha if ell_alpha is None else ell_alpha 

        # Make sure alpha in range [0,1]
        if ell_alpha > 1: ell_alpha = 1.0
        if ell_alpha < 0: ell_alpha = 0.0

        # Get color from colormap or overloaded parameter
        if color is None:
            color   = plt.get_cmap('hsv')(float(axi_color)/len(dm.ax_list))[0:3]
            if axi_color == 0: 
                color = [color[0], color[1]+.5, color[2]]
        
        # Axis We are drawing to.
        ax     = dm.ax_list[axi]
        T_data = ax.transData # data coordinates -> display coordinates
        # Data coordinates are chip coords

        if xy_bit or ell_bit or fsel != None:
            T_fpts = T_data if not in_image_bit else\
                    Affine2D(cm.cx2_T_chip2img(cx) ) + T_data
            fpts = cm.get_fpts(cx)
            if fsel is None: fsel = range(len(fpts))
            # ---DEVHACK---
            # Randomly sample the keypoints. (Except be sneaky)
            elif fsel == 'rand': 
                # Get Relative Position
                minxy = fpts.min(0)[0:2]
                maxxy = fpts.max(0)[0:2] 
                rel_pos = (fpts[:,0]-minxy[0])/(maxxy[0]-minxy[0])
                to_zero = 1 - np.abs(rel_pos - .5)/.5
                pdf = (to_zero / to_zero.sum())
                # Transform Relative Position to Probabilities
                # making it more likely to pick a centerpoint
                fsel = np.random.choice(xrange(len(fpts)), size=88, replace=False, p=pdf)
            # ---/DEVHACK---
            # Plot ellipses
            if ell_bit and len(fpts) > 0 and len(fsel) > 0: 
                ells = dm._get_fpt_ell_collection(fpts[fsel,:],
                                                  T_fpts,
                                                  ell_alpha,
                                                  color)
                ax.add_collection(ells)
            # Plot xy points
            if xy_bit and len(fpts) > 0 and len(fsel) > 0: 
                ax.plot(fpts[fsel,0], fpts[fsel,1], 'o',\
                        markeredgecolor=color,\
                        markerfacecolor=color,\
                        transform=T_fpts,\
                        markersize=2)
        # === 
        if bbox_bit:
            # Draw Bounding Rectangle in Image Coords
            [rx,ry,rw,rh] = cm.cx2_roi[cx]
            rxy = (rx,ry)
            # Convert to Chip Coords if needbe
            T_bbox = T_data if in_image_bit else\
                Affine2D( np.linalg.inv(cm.cx2_T_chip2img(cx)) ) + T_data
            bbox = Rectangle(rxy,rw,rh,transform=T_bbox) 
            # Visual Properties
            bbox.set_fill(False)
            bbox.set_edgecolor(color)
            ax.add_patch(bbox)

            # Draw Text Annotation
            cid   = cm.cx2_cid[cx]
            name  = cm.cx2_name(cx)
            # Lower the value to .2 for the background color and set alpha=.7 
            rgb_textFG = [1,1,1]
            hsv_textBG = colorsys.rgb_to_hsv(*color)[0:2]+(.2,)
            rgb_textBG = colorsys.hsv_to_rgb(*hsv_textBG)+(.7,)
            # Draw Orientation Backwards
            degrees = 0 if not in_image_bit else -cm.cx2_theta[cx]*180/np.pi
            txy = (0,rh) if not in_image_bit else (rx, ry+rh)

            chip_text =  'name='+name+'\n'+'cid='+str(cid)
            ax.text(txy[0]+1, txy[1]+1, chip_text,
                    horizontalalignment ='left',
                    verticalalignment   ='top',
                    transform           =T_data,
                    rotation            =degrees,
                    color               =rgb_textFG,
                    backgroundcolor     =rgb_textBG)
        return fsel
