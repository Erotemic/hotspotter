import sys
import colorsys
from other.helpers import *
from other.logger import *
from matplotlib import gridspec
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from matplotlib.pyplot import \
  ginput, draw, figure, get_cmap, jet, gray
from numpy import array, uint32, round, sqrt, ceil, float32
from warnings import catch_warnings, simplefilter 

class DrawManager(AbstractManager):
    def __init__(dm, hs):
        super( DrawManager, dm ).__init__( hs )        
        dm.hs      =   hs
        dm.fignum =    0
        dm.ax_list =   []
    # ---
    def get_figure(dm):
        guifig = dm.hs.uim.get_gui_figure()
        if guifig != None and dm.fignum == 0: # Check to see if we have access to the gui
            return guifig
        fig = figure(num=dm.fignum, figsize=(5,5), dpi=72, facecolor='w', edgecolor='k')
        return fig
    # ---
    def select_roi(dm):
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
        except: 
            logmsg('Select ROI Failed')
            return None
    # ---
    def end_draw(dm):
        gray()
        logdbg('Finalizing Draw with '+str(len(dm.ax_list))+' axes')
        fig = dm.get_figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        if dm.hs.prefs['draw_in_cmd_bit']:
            from IPython.core.display import display
            display(fig)
        fig.show()
        dm.hs.uim.redraw_gui()
        draw() 
    # ---
    def save_fig(dm, save_file):
        dm.end_draw()
        fig.savefig(save_file, format='png')
    # ---
    def add_images(dm, img_list, title_list=[]):
        fig = dm.get_figure()
        fig.clf()
        num_images = len(img_list)
        #
        dm.ax_list     = [None]*num_images
        transData_list = [None]*num_images
        title_list     = title_list + ['NoTitle']*(num_images-len(title_list))
        #
        nr = int( ceil( float(num_images)/2 ) )
        nc = 2 if num_images >= 2 else 1
        #
        gs = gridspec.GridSpec( nr, nc )
        for i in range(num_images):
            #logdbg(' Adding the '+str(i)+'th Image')
            #logdbg('   * type(img_list[i]): %s'+str(type(img_list[i])))
            #logdbg('   * img_list[i].shape: %s'+str(img_list[i].shape))
            dm.ax_list[i] = fig.add_subplot(gs[i])
            dm.ax_list[i].imshow( img_list[i])
            dm.ax_list[i].get_xaxis().set_ticks([])
            dm.ax_list[i].get_yaxis().set_ticks([])
            dm.ax_list[i].set_title(title_list[i])
            transData_list[i] = dm.ax_list[i].transData
            # transData: data coordinates -> display coordinates
            # transAxes: axes coordinates -> display coordinates
            # transLimits: data - > axes
        #
        logdbg('Added '+str(num_images)+' images/axes')
        return transData_list
    # ---
    def _get_fpt_ell_collection(dm, fpts, transData, alpha, edgecolor):
        ell_patches = []
        for (x,y,a,c,d) in fpts: # Manually Calculated sqrtm(inv(A))
            with catch_warnings():
                simplefilter("ignore")
                aIS = 1/sqrt(a)
                cIS = (c/sqrt(a) - c/sqrt(d))/(a - d + eps)
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

        _default = lambda pref_val, default_val:\
                pref_val if pref_val != None else default_val
        hs = dm.hs
        feat_xy_bit  = _default(hs.prefs['fpts_xys_bit'], False)
        fpts_ell_bit = _default(hs.prefs['fpts_ell_bit'],  True)
        bbox_bit     = _default(hs.prefs['bbox_bit'],      True)
        ell_alpha    = _default(hs.prefs['ellipse_alpha'],   .2)

        colormap     = _default(hs.prefs['colormap'],   'hsv')
        shift_color  = lambda color, shift:\
                map(lambda (cc, shiftc): min(1,(cc+shiftc) % 1.001), zip(color, shift))

        map_color   = get_cmap(colormap)(float(axi)/len(dm.ax_list))
        #if len(map_color_) == 3:
        #    map_color_ = map_color_ + tuple(1)
        #map_color    = shift_color( map_color_, [0, .5, 0, 0] )
        if axi == 0:
            map_color = [map_color[0], map_color[1]+.5, map_color[2], map_color[3]]

        textcolor  = _default(hs.prefs['text_color'], map_color)


        mcolor_ell_bit = _default(hs.prefs['match_with_color_ell'],   False)
        mcolor_xys_bit = _default(hs.prefs['match_with_color_xys'],   False)
        #mlines_bit     = _default(hs.prefs['match_with_lines'],   False)

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
            nid   = cm.cx2_nid(cx)
            name  = cm.cx2_name(cx)
            # Use the complimentary color as the text background
            _hsv = colorsys.rgb_to_hsv(textcolor[0],textcolor[1],textcolor[2])
            comp_hsv = [_hsv[0], _hsv[1], .2]
            #shift_color(_hsv, [0, 0,-.5])
            comp_rgb = list(colorsys.hsv_to_rgb(comp_hsv[0], comp_hsv[1], comp_hsv[2]))
            comp_rgb.append(.7)

            #chip_text =  'nid='+str(nid)+'\n'+'cid='+str(cid)
            chip_text =  'name='+name+'\n'+'cid='+str(cid)
            textcolor = [1,1,1]
            ax.text(1, 1, chip_text,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=transData,
                    color=textcolor,
                    backgroundcolor=comp_rgb)
    # ---
    def draw_graph(dm, G):
        import networkx
        #fig = dm.get_current_figure()
        fig = figure(9001)
        fig.clf()
        ax = fig.gca()
        pos = networkx.spring_layout(G, dim=2, scale=1,iterations=100) 
        #pos = networkx.spectral_layout(G)
        #pos = networkx.circular_layout(G)
        node_labels=dict([(id,d['qnid']) for id,d in G.nodes(data=True)])
        #networkx.draw_networkx(G,pos,ax=ax)
        colormap = dm.hs.prefs['colormap']
        cmap = get_cmap(colormap)
        tot_num = dm.hs.nm.num_n+2
        for cid in pos.keys():
            cx = dm.hs.cm.cid2_cx[cid]
            nid = dm.hs.cm.cx2_nid(cx)
            color = cmap(float(nid)/tot_num)
            #print color
            networkx.draw_networkx_nodes(G,pos, nodelist=[cid], node_color=color, node_size=1000)
            pass
        #networkx.draw_networkx_nodes(G,pos)
        #networkx.draw_networkx_labels(pos,node_labels)
        networkx.draw_networkx_edges(G,pos,alpha=.5)
        networkx.draw_networkx_labels(G,pos)
            
        #labels=networkx.draw_networkx_labels(G,pos=pos)
        #edge_labels=dict([((u,v,),'%.1f' % d['weight']) for u,v,d in G.edges(data=True)])
        #edge_labels = {}
        #networkx.draw_networkx_edge_labels(G,pos,edge_labels,alpha=0.5)
        #trans = ax.transData.transform
        #trans2 = fig.transFigure.inverted().transform
        #for node in G.nodes():
        #    x,y = pos[node]
