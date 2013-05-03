import numpy as np
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
        dm.draw_prefs.ellipse_alpha  = .5
        dm.draw_prefs.points_bit     = False
        dm.draw_prefs.result_view  = Pref(1, choices=['in_image', 'in_chip'])
        dm.draw_prefs.fignum         = 0
        dm.draw_prefs.colormap       = Pref('hsv', hidden=True)
        dm.draw_prefs.in_qtc_bit     = Pref(False, hidden=True) #Draw in the Qt Console
        dm.draw_prefs.use_thumbnails = Pref(False, hidden=True)
        dm.draw_prefs.thumbnail_size = Pref(128, hidden=True)
        if not default_bit:
            dm.draw_prefs.load()
    # ---
    def show_splash(dm):
        splash_fname = os.path.join(dm.hs.get_source_fpath(), 'front', 'splash.tif')
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
    def show_chip(dm, cx):
        cm = dm.hs.cm
        cid, name, chip = cm.cx2_(cx, 'cid', 'name', 'chip')
        if dm.draw_prefs.use_thumbnails is True:
            pass
        dm.add_images([chip], [name])
        dm.draw_chiprep2(cx, axi=0)
        dm.end_draw()
    # ---
    def show_query(dm, res):
        cm = dm.hs.cm
        # Make sure draw is valid
        if res is None: dm.show_splash(); return
        # Get Chip Properties
        dynargs =\
        ('cx', 'cid', 'nid', 'name')
        (qcx , qcid , qnid , qname ) =  res.qcid2_(*dynargs)
        (tcx , tcid , tnid , tname , tscore ) = res.tcid2_(*dynargs+('score',))
        # Titles of the Plot
        qtitle = 'name: %s\nQuery cid=%d, nid=%d' % (qname, qcid, qnid)
        ttile = ['name: %s\nscore=%.2f' % (name_, score_) for name_, score_ in zip(tname, tscore)]
        title_list = [qtitle] + ttile
        if dm.draw_prefs.use_thumbnails is True:
            pass

        # Add the images to draw
        if dm.draw_prefs.result_view == 'in_image':
            qimg = cm.cx2_img(qcx)
            timg = cm.cx2_img_list(tcx)
            dm.add_images([qimg] + timg, title_list)
        elif dm.draw_prefs.result_view == 'in_chip':
            qchip = cm.cx2_chip_list(qcx)
            tchip = cm.cx2_chip_list(tcx)

        # Draw the Query Chiprep
        qaxi       = 0; qfsel      = []
        dm.draw_chiprep2(qcx, axi=qaxi, fsel=qfsel)
        # Draw the Top Result Chipreps
        for (tx, cx) in enumerate(tcx):
            fm    = res.rr.cx2_fm[cx]
            fs    = res.rr.cx2_fs[cx]
            axi   = tx+1
            if len(fs) == 0:
                qfsel = array([], uint32)
                fsel = array([], uint32)
            else:
                qfsel = fm[fs > 0][:,0]
                fsel  = fm[fs > 0][:,1]
            dm.draw_chiprep2(cx,  axi=axi,  fsel=fsel)
            dm.draw_chiprep2(qcx, axi=qaxi, axi_color=axi, fsel=qfsel, bbox_bit = False)
        dm.end_draw()

    # ---

    def __init__(dm, hs):
        super( DrawManager, dm ).__init__( hs )        
        dm.hs      =   hs
        dm.fignum =    0
        dm.draw_prefs = None
        dm.ax_list =   []
        dm.init_preferences()
    # ---
    def get_figure(dm):
        guifig = dm.hs.uim.get_gui_figure()
        if guifig != None and dm.fignum == 0: # Check to see if we have access to the gui
            return guifig
        fig = figure(num=dm.fignum, figsize=(5,5), dpi=72, facecolor='w', edgecolor='k')
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
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
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
        nr = int( ceil( float(num_images)/2 ) )
        nc = 2 if num_images >= 2 else 1
        #
        gs = gridspec.GridSpec( nr, nc )
        for i in xrange(num_images):
            #logdbg(' Adding the '+str(i)+'th Image')
            #logdbg('   * type(img_list[i]): %s'+str(type(img_list[i])))
            #logdbg('   * img_list[i].shape: %s'+str(img_list[i].shape))
            dm.ax_list[i] = fig.add_subplot(gs[i])
            imgplot = dm.ax_list[i].imshow( img_list[i])
            imgplot.set_cmap('gray')
            dm.ax_list[i].get_xaxis().set_ticks([])
            dm.ax_list[i].get_yaxis().set_ticks([])
            dm.ax_list[i].set_title(title_list[i])
            # transData: data coordinates -> display coordinates
            # transAxes: axes coordinates -> display coordinates
            # transLimits: data - > axes
        #
        logdbg('Added '+str(num_images)+' images/axes')
    # ---
    def _get_fpt_ell_collection(dm, fpts, transData, alpha, edgecolor):
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
        ellipse_collection.set_transform(transData)
        ellipse_collection.set_alpha(alpha)
        ellipse_collection.set_edgecolor(edgecolor)
        return ellipse_collection
    # ---
    def draw_chiprep(dm, cx, transData, axi=0, fsel=None,\
                     qcx=None, qtransData=None, qaxi=None, qfsel=None):
        ''' draws the instance in chip coordinates'''
        #TODO: This is pretty ugly. It should be fixed
        #logdbg('Drawing Chip Representation:')
        #dbgstr = 'cx=%s\ntransData=%s\naxi=%s\nfsel=%s\nqcx=%s\nqtransData=%s\nqaxi=%s\nqfsel=%s'\
        #        %(str(cx), str(transData), str(axi), printableVal(fsel),\
        #          str(qcx), str(qtransData), str(qaxi), printableVal(qfsel))
        #logdbg(dbgstr)

        # transData - affine transformation to the data you are drawing
        # axi - the index of the image you are drawing to. (see: add_images)
        # fsel - select the feature indexes to show
        # q[things] - relate to the query
        logdbg('Drawing Chip CX='+str(cx))

        transImg = Affine2D( cm.cx2_transImg(cx) ) 
        theta = cm.cx2_theta[cx]
        rot = Affine2D(np.array(([np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [            0,              0, 1]), dtype=np.float32))
        trans_kpts = transImg + rot + transData
        trans_bbox = transImg + transData

        hs = dm.hs
        feat_xy_bit  = dm.draw_prefs.points_bit 
        fpts_ell_bit = dm.draw_prefs.ellipse_bit 
        bbox_bit     = dm.draw_prefs.bbox_bit 
        ell_alpha    = dm.draw_prefs.ellipse_alpha
        if ell_alpha > 1: 
            ell_alpha = 1.0
        if ell_alpha < 0:
            ell_alpha = 0.0
        colormap     = dm.draw_prefs.colormap
        try: 
            map_color   = get_cmap(colormap)(float(axi)/len(dm.ax_list))
        except Exception:
            map_color   = get_cmap('hsv')(float(axi)/len(dm.ax_list))
        if axi == 0:
            map_color = [map_color[0], map_color[1]+.5, map_color[2], map_color[3]]

        textcolor  = map_color

        cm        = dm.hs.cm
        ax        = dm.ax_list[axi]
        # ===
        force_recomp=False
        if feat_xy_bit or fpts_ell_bit or qfsel != None:
            fpts = cm.get_fpts(cx, force_recomp=force_recomp)

            if fsel is None:
                fsel = range(len(fpts))
            if fpts_ell_bit and len(fpts) > 0:
                ells = dm._get_fpt_ell_collection(fpts[fsel,:], transData, ell_alpha, map_color)
                ax.add_collection(ells)
            if feat_xy_bit and len(fpts) > 0: 
                ax.plot(fpts[fsel,0], fpts[fsel,1], 'o',\
                        markeredgecolor=map_color,\
                        markerfacecolor=map_color,\
                        transform=transData,\
                        markersize=2)
            if qfsel != None and len(fpts) > 0:
                # Draw Feature Matches
                qcolor = get_cmap(colormap)(float(qaxi)/len(dm.ax_list))
                if qaxi == 0:
                    qcolor = [qcolor[0], qcolor[1]+.5, qcolor[2], qcolor[3]]
                qax    = dm.ax_list[qaxi]
                qfpts  = cm.get_fpts(qcx, force_recomp=force_recomp)
                if feat_xy_bit:
                    qxys = qfpts[qfsel,0:2]
                    qax.plot(qxys[:,0],qxys[:,1],'o', markeredgecolor=map_color, transform=qtransData, markersize=2)
                if fpts_ell_bit:
                    qells = dm._get_fpt_ell_collection\
                            (qfpts[qfsel,:], qtransData, ell_alpha, map_color)
                    qax.add_collection(qells)
        # === 
        if bbox_bit:
            cxy = (0,0)
            (cw,ch) = cm.cx2_chip_size(cx)
            bbox = Rectangle(cxy,cw,ch,transform=transData) 
            bbox.set_fill(False)
            bbox.set_edgecolor(map_color)
            ax.add_patch(bbox)

            cid   = cm.cx2_cid[cx]
            name  = cm.cx2_name(cx)
            # Use the complimentary color as the text background
            _hsv = colorsys.rgb_to_hsv(textcolor[0],textcolor[1],textcolor[2])
            comp_hsv = [_hsv[0], _hsv[1], .2]
            #shift_color(_hsv, [0, 0,-.5])
            comp_rgb = list(colorsys.hsv_to_rgb(comp_hsv[0], comp_hsv[1], comp_hsv[2]))
            comp_rgb.append(.7)

            # Draw Orientation Backwards 
            degrees = -cm.cx2_theta[cx]*180/np.pi
            #chip_text =  'nid='+str(nid)+'\n'+'cid='+str(cid)
            chip_text =  'name='+name+'\n'+'cid='+str(cid)
            textcolor = [1,1,1]
            ax.text(1, 1, chip_text,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=transData,
                    color=textcolor,
                    rotation=degrees,
                    backgroundcolor=comp_rgb)

    def draw_chiprep2(dm, cx, axi=0, fsel=None, in_image_bit=False, axi_color=0, bbox_bit=None):
        cm = dm.hs.cm
        # Grab Preferences
        feat_xy_bit  = dm.draw_prefs.points_bit 
        fpts_ell_bit = dm.draw_prefs.ellipse_bit 
        bbox_bit     = dm.draw_prefs.bbox_bit if bbox_bit is None else bbox_bit
        ell_alpha    = dm.draw_prefs.ellipse_alpha
        if ell_alpha > 1: ell_alpha = 1.0
        if ell_alpha < 0: ell_alpha = 0.0
        map_color   = get_cmap('hsv')(float(axi_color)/len(dm.ax_list))
        if axi_color == 0: map_color = [map_color[0], map_color[1]+.5, map_color[2], map_color[3]]
        
        # Axis We are drawing to.
        ax        = dm.ax_list[axi]
        transData = ax.transData # data coordinates -> display coordinates
        # Data coordinates are chip coords
        if in_image_bit: 
            # data coords = chip coords -> image coords -> display coords
            transImg = Affine2D( cm.cx2_transImg(cx) ) 
        else: 
            # data coords = chip coords -> display coords
            transImg = Affine2D()

        (cw,ch) = cm.cx2_chip_size(cx) # This is not ok, because the size disagrees with roi after rotation
        if feat_xy_bit or fpts_ell_bit or qfsel != None:
            fpts = cm.get_fpts(cx)
            if in_image_bit:
                theta = cm.cx2_theta[cx]
                transRot = Affine2D().rotate_around(cw/2,ch/2,theta)
            else:
                transRot = Affine2D()
            trans_kpts = transRot + transImg + transData
            if fsel is None: fsel = range(len(fpts))
            if fpts_ell_bit and len(fpts) > 0: # Plot ellipses
                ells = dm._get_fpt_ell_collection(fpts[fsel,:], trans_kpts, ell_alpha, map_color)
                ax.add_collection(ells)
            if feat_xy_bit and len(fpts) > 0: # Plot xy points
                ax.plot(fpts[fsel,0], fpts[fsel,1], 'o',\
                        markeredgecolor=map_color,\
                        markerfacecolor=map_color,\
                        transform=trans_kpts,\
                        markersize=2)
        # === 
        if bbox_bit:
            trans_bbox = transImg + transData
            # Draw Bounding Rectangle
            cxy = (0,0)
            #(_,_,cw,ch) = cm.cx2_roi[cx]
            bbox = Rectangle(cxy,cw,ch,transform=trans_bbox) 
            bbox.set_fill(False)
            bbox.set_edgecolor(map_color)
            ax.add_patch(bbox)

            # Draw Text Annotation
            cid   = cm.cx2_cid[cx]
            name  = cm.cx2_name(cx)
            # Use the complimentary color as the text background
            _hsv = colorsys.rgb_to_hsv(map_color[0],map_color[1],map_color[2])
            comp_hsv = [_hsv[0], _hsv[1], .2]
            comp_rgb = list(colorsys.hsv_to_rgb(comp_hsv[0], comp_hsv[1], comp_hsv[2]))
            comp_rgb.append(.7)
            # Draw Orientation Backwards 
            degrees = -cm.cx2_theta[cx]*180/np.pi
            chip_text =  'name='+name+'\n'+'cid='+str(cid)
            ax.text(1, 1, chip_text,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=trans_bbox,
                    color=[1,1,1],
                    rotation=degrees,
                    backgroundcolor=comp_rgb)
