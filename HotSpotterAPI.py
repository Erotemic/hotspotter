# Notes to developers
#   * Do all your running and debugging in IPython
#   * when you encounter an error, print hsl is your best friend
#   * You can use the func_log decorator
#   * Document the code with the log functions
#   * use logdbg to indicate progress and give a helpful message
#   * use logmsg if the user should see the message
#   * use logio if you are doing io
#   *
#   * Code Concepts: 
'''
Code Concepts: 

    HotSpotterAPI holds the managers. The managers hold the data. 
    Different functions need different data. They use hs.get_managers
    to get the managers they need. Try to keep as much data in place as 
    possible.

'''
from Experiments            import ExperimentManager
from PIL import Image
from PyQt4.Qt import QString
from PyQt4.QtGui import QFileDialog
from core.AlgorithmManager  import AlgorithmManager
from core.ChipManager       import ChipManager
from core.IOManager         import IOManager
from core.ImageManager      import ImageManager
from core.NameManager       import NameManager
from core.QueryManager      import QueryManager, QueryResult
from core.VisualModel       import VisualModel
from gui.DrawManager       import DrawManager
from gui.UIManager         import UIManager
from matplotlib.transforms import Affine2D
from numpy import asarray, append
from other.helpers           import *
from other.logger           import *
import cPickle
import os
import sys
import types

class HotSpotterAPI(AbstractPrintable):
    @func_log
    def read_prefs(hs, default_bit=False):
        'Loads preferences from the home directory'
        logdbg('Reading Prefs: ')    
        # None will usually map to a function default
        # 'none' will get rid of the preference
        #TODO: Alot of these should be drawmanager settings
        # Thumbnailing should be taken care of the draw manager
        # maybe
        default_prefs = {
            'database_dir'     :  None,
            'thumbnail_size'   :   128,        
            'thumbnail_bit'    : False,
            'plotwidget_bit'   : False,

            'fpts_ell_bit'     : False,
            'fpts_xys_bit'     : False,
            'bbox_bit'         :  True,
            'res_as_img'       : False, 

            'text_color'       :  None,
            'colormap'         : 'hsv',
            'ellipse_alpha'    :    .5,   

            'draw_in_cmd_bit'  :  False,
            'fignum'           :     0,

            #'match_with_lines'     : False,
            'match_with_color_ell' :  True,
            'match_with_color_xys' : False,

            'roi_beast_mode' : False
        }
        pref_fpath = hs.iom.get_prefs_fpath()
        if default_bit or not filecheck(pref_fpath):
            hs.prefs = default_prefs
        else:
            with open(pref_fpath, 'r') as f:
                hs.prefs = cPickle.load(f)
            if type(hs.prefs) != types.DictType:
                logerr('Preference file is corrupted')
            for key in default_prefs.keys(): # Add New Preferences Dynamically
                if not key in hs.prefs.keys(): 
                    hs.prefs[key] = default_prefs[key]
        return True

    def use_thumbnail(hs, thumb_bit=None):
        return (thumb_bit is not None and thumb_bit) or\
               (thumb_bit is None and hs.prefs['thumbnail_bit'])

    @func_log
    def write_prefs(hs):
        if type(hs.prefs) != types.DictType:
            logerr('API preference dictionary is corrupted!')
        with open(hs.iom.get_prefs_fpath(), 'w') as f:
            cPickle.dump(hs.prefs, f)

    @func_log
    def smartup_db_dpath(hs, db_dpath):
        ' Performs a smart update of the db_dpath '
        ' Trys a number of various  options to get it rightu'
        if db_dpath is None: # None = Open the last opened database
            db_dpath = str(hs.prefs['database_dir'])
            if db_dpath in [None, 'None'] or not dircheck(db_dpath, False):
                logwarn('Warning! Global preference database_dir is invalid')
                db_dpath = '' 
        if db_dpath == '': # Quotes = try user selection
            db_dpath = hs.uim.select_database()
        elif db_dpath == None:
            logerr('db_dpath cannot be None')
        if not os.path.exists(db_dpath):
            logerr('ERROR: db_dpath \"'+str(db_dpath)+'\" doesnt exist')
        # --
        db_dpath_files = os.listdir(db_dpath)
        if hs.iom.internal_dname in db_dpath_files:
            logmsg('Loading a HotSpotter database')
        elif 'images' in db_dpath_files or 'data' in db_dpath_files:
            logmsg('Loading a StripSpotter database')
        elif len(db_dpath_files) == 0:
            logmsg('Creating a new database')
        else:
            logdbg('Files in dir: '+str(db_dpath_files))
            logerr('Unknown database type. Select a HotSpotter, StripeSpotter, or empty directory.')
        # --
        if hs.prefs['database_dir'] != db_dpath:
            hs.prefs['database_dir'] = db_dpath
            hs.write_prefs()
        logdbg('setting db_dpath = '+str(db_dpath))
        hs.db_dpath = db_dpath

    def __init__(hs):
        super( HotSpotterAPI, hs ).__init__(['cm','gm','nm','em','qm','dm','am','vm','iom','uim'])
        #
        hs.db_dpath = None #Database directory.
        hs.data_loaded_bit = False
        hs.prefs = None
        # --- Managers ---
        hs.iom = None # IO Manager.
        #
        hs.am = None # Algorithm Manager
        hs.vm = None # Vocab Manager
        hs.qm = None # Vocab Manager
        hs.em = None # Experiment Manager
        hs.dm = None # Draw Manager
        #-
        hs.gm = None # Image Manager
        hs.cm = None # Instance Manager
        hs.nm = None # Name Manager
        #-
        hs.prev_draw = lambda : None
        hs.uim = UIManager(hs)
        hs.iom = IOManager(hs)
        hs.dm  = DrawManager(hs)
        hs.am = AlgorithmManager(hs)
        hs.read_prefs()
        # --- 

    @func_log
    def restart(hs, db_dpath=None, autoload=True):
        hs.data_loaded_bit = False
        if hs.db_dpath != None and db_dpath == None:
            db_dpath = hs.db_dpath
        hs.read_prefs()
        hs.smartup_db_dpath(db_dpath)
        
        hs.gm  = ImageManager(hs)
        hs.nm  = NameManager(hs)
        hs.cm  = ChipManager(hs)
        hs.vm  = VisualModel(hs)
        hs.qm  = QueryManager(hs)
        hs.em  = ExperimentManager(hs)
        if autoload == True:
            hs.load_tables()
        else: 
            logdbg('autoload is false.')
    # ---
    @func_log
    def set_pref(hs, pref_name, pref_val):
        if pref_val == 'toggle':
            if not hs.prefs[pref_name] in [True, False]:
                logerr('Cannot toggle a non-boolean pref: '+str(pref_name))
            hs.prefs[pref_name] = not hs.prefs[pref_name]
        else:
            hs.prefs[pref_name] = pref_val
        hs.write_prefs()
        hs.prev_draw()
    # ---
    @func_log
    def load_tables(hs):
        hs.iom.load_tables()
        hs.data_loaded_bit = True
    # ---
    @func_log
    def add_image_list(hs, image_list):
        num_add = len(image_list)
        logmsg('Selected '+str(num_add)+' images to import')
        prev_g = hs.gm.num_g
        logdbg('Prev #g=%d' % prev_g)
        for src_img in image_list:
            hs.gm.add_img(gid=None, gname=None, aif=False, src_img=src_img)
        post_g = hs.gm.num_g
        num_new = (post_g - prev_g)
        num_old = num_add - num_new
        logdbg('Post #g=%d' % post_g)
        logmsg('Imported '+str(num_new)+' new images')
        if num_old != 0:
            logmsg('%d Images had already been copied into the image directory' % num_old)
    # ---
    @func_log
    def delete_computed_directory(hs):
        'Unloads all features and models and deletes the computed directory'
        logmsg('Deleting the computed directory')
        all_cxs = hs.cm.get_valid_cxs()
        hs.cm.unload_features(all_cxs)
        hs.vm.reset()
        hs.iom.remove_computed_files_with_pattern('*')
    # ---
    @func_log
    def precompute_chips(hs):
        logmsg('Precomputing the chips')
        all_cxs = hs.cm.get_valid_cxs()
        for cx in all_cxs:
            hs.cm.compute_chip(cx)
    # ---
    @func_log
    def get_source_fpath(hs):
        import __init__
        return os.path.dirname(__init__.__file__)

    # ---
    @func_log
    def show_splash(hs): 
        hs.prev_draw = lambda: hs.show_splash()
        splash_fname = os.path.join(hs.get_source_fpath(), 'gui', 'splash.tif')
        splash_img = asarray(Image.open(splash_fname))
        hs.dm.add_images([splash_img],['Welcome to Hotspotter']);
        hs.dm.end_draw()
    # ---
    @func_log
    def show_image(hs, gx):
        hs.prev_draw = lambda: hs.show_image(gx)
        gm, cm, dm = hs.get_managers('gm','cm','dm')
        gid        = gm.gx2_gid[gx]
        img_list   = gm.gx2_img_list(gx)
        title_list = ['gid='+str(gid)+'   gname='+gm.gx2_gname[gx]]
        transData  = hs.dm.add_images(img_list, title_list)[0]
        cx_list        = gm.gx2_cx_list[gx]
        for cx in iter(cx_list):
            transImg = Affine2D( cm.cx2_transImg(cx) ) 
            trans    = transImg + transData
            dm.draw_chiprep(cx, trans, 0)
        dm.end_draw()
    # ---
    @func_log
    def show_chip(hs, cx):
        hs.prev_draw = lambda: hs.show_chip(cx)
        cm, dm = hs.get_managers('cm','dm')
        cid, name, chip = cm.cx2_(cx, 'cid', 'name', 'chip')
        transData = dm.add_images([chip], [name])[0]
        dm.draw_chiprep(cx, transData)
        dm.end_draw()
    # ---
    @func_log
    def show_query(hs, res):
        cm, nm, dm, am = hs.get_managers('cm','nm','dm','am')
        hs.prev_draw = lambda: hs.show_query(res)
        # Make sure draw is valid
        if res is None: hs.show_splash(); return
        # Get Chip Properties
        dynargs =\
        ('cx', 'cid', 'nid', 'name')
        (qcx , qcid , qnid , qname ) =  res.qcid2_(*dynargs)
        (tcx , tcid , tnid , tname , tscore ) = res.tcid2_(*dynargs+('score',))
        # Titles of the Plot
        qtitle = 'name: %s\nQuery cid=%d, nid=%d' % (qname, qcid, qnid)
        ttitle_func = lambda (name_, score_): 'name: %s\nscore=%.2f' % (name_, score_)
        ttile = map(ttitle_func, zip(tname, tscore))
        title_list = [qtitle] + ttile

        # Add the images to draw
        if hs.prefs['res_as_img']: # Show the results in their full image
            qimg = cm.cx2_img(qcx)
            timg = cm.cx2_img_list(tcx)
            transData_list = dm.add_images([qimg] + timg, title_list)
            for (ix, cx) in enumerate(append(qcx,tcx)):
                transImg = Affine2D( cm.cx2_transImg(cx) )
                transData_list[ix] = transImg + transData_list[ix]
        else: # Show just the chips
            qchip = cm.cx2_chip_list(qcx)
            tchip = cm.cx2_chip_list(tcx)
            transData_list = dm.add_images(qchip + tchip, title_list)

        # Draw the Query Chiprep
        qtransData = transData_list[0]
        qaxi       = 0; qfsel      = []
        dm.draw_chiprep(qcx, qtransData, qaxi, qfsel)
        # Draw the Top Result Chipreps
        for (tx, cx) in enumerate(tcx):
            fm    = res.rr.cx2_fm[cx]
            fs    = res.rr.cx2_fs[cx]
            axi   = tx+1
            qfsel = fm[fs > 0][:,0]
            fsel  = fm[fs > 0][:,1]
            transData = transData_list[tx+1]
            dm.draw_chiprep(cx,   transData,  axi,  fsel,\
                            qcx, qtransData, qaxi, qfsel)
        dm.end_draw()
    # --- 
    @func_log
    def show_matching_graph(hs, computed_cx2_rr=None):
        hs.prev_draw = lambda: hs.show_matching_graph(computed_cx2_rr)
        cm = hs.cm
        dm = hs.dm
        try: 
            computed_cx2_rr
        except NameError:
            computed_cx2_rr = None
        if computed_cx2_rr == None:
            computed_cx2_rr = vm.batch_query()
        import networkx
        G = networkx.DiGraph()
        for rr in computed_cx2_rr:
            if rr == []: continue
            res = QueryResult(hs,rr)
            qcid  = res.qcid
            qnid  = res.qnid
            G.add_node(qcid,qcid=qcid,qnid=qnid)
            rr = res.rr
            for (tscore,tcx) in zip(rr.cx2_cscore, range(len(rr.cx2_cscore))):
                tcid = cm.cx2_cid[tcx]
                if tscore > 0:
                    G.add_edge(qcid, tcid, weight=tscore**2-tscore)
        dm.draw_graph(G)
        #hs.dm.end_draw()
        pass
    # --- 
    def get_managers(hs, *manager_list):
        'quick access of managers eg: (am, cm, iom) = hs.managers("am","cm","iom")'
        return tuple([hs.__dict__[manager_name] for manager_name in manager_list])

    def dynget(hs, *dynargs):
        return tuple([hs.__dict__[arg] for arg in dynargs])

    def __getitem__(hs, *dynargs):
        return tuple([hs.__dict__[arg] for arg in dynargs])
