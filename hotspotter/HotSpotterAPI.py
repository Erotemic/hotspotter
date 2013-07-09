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
from hotspotter.AlgorithmManager import AlgorithmManager
from hotspotter.ChipManager import ChipManager
from hotspotter.Experiments import ExperimentManager
from hotspotter.IOManager import IOManager
from hotspotter.ImageManager import ImageManager
from hotspotter.NameManager import NameManager
from hotspotter.QueryManager import QueryManager, QueryResult
from hotspotter.VisualModel import VisualModel
from hotspotter.front.DrawManager import DrawManager
from hotspotter.front.UIManager import UIManager
from hotspotter.other.AbstractPrintable import AbstractPrintable
from hotspotter.other.ConcretePrintable import Pref
from hotspotter.other.logger import logdbg, logerr, logmsg, logwarn, func_log
import hotspotter.Parallelize
import hotspotter.ChipFunctions
import cPickle
import os.path
import types
class HotSpotterAPI(AbstractPrintable):

    def init_preferences(hs, default_bit=False):
        'All classe with preferences have an init_preference function'
        'TODO: Make a general way to automatically have this function in ConcretePrintable.Pref'
        iom = hs.iom
        # Create a Preference Object With a Save Path
        # Pref is a special class. __setattrib__ is overridden
        if hs.core_prefs == None:
            hs.core_prefs = Pref(fpath=iom.get_prefs_fpath('core_prefs'), hidden=False)
        hs.core_prefs.database_dpath = hs.db_dpath
        hs.core_prefs.legacy_bit = Pref(True)
        hs.core_prefs.num_procs  = Pref(hotspotter.Parallelize.cpu_count() + 1)

        if not default_bit:
            hs.core_prefs.load()

    def show_chips(hs, cid_list, titles=[], save_fpath=None, fignum=1):
        'Displays a chip or list of chips'
        cm = hs.cm
        hs.dm.fignum = fignum #TODO Set Fignum safer
        cx_list   = cm.cid2_cx[cid_list]
        chip_list = cm.cx2_chip_list(cx_list)
        if len(titles) == 0:
            titles = cm.cx2_name(cx_list)
        hs.dm.add_images(chip_list, titles)
        hs.dm.end_draw()
        if not save_fpath is None:
            hs.dm.save_fig(save_fpath)

    @func_log
    def on_cx_modified(hs, cx):
        # When a cx is modified mark dependents as dirty
        # TODO: Add conditional dirtyness
        cid = hs.cm.cx2_cid[cx]
        hs.vm.isDirty = True
        # Remove selection
        hs.uim.sel_res = None
        hs.cm.unload_features(cx)
        hs.cm.delete_computed_cid(cid)
        hs.cm.cx2_dirty_bit[cx] = True
        hs.iom.remove_computed_files_with_pattern('model*.npz')
        hs.iom.remove_computed_files_with_pattern('index*.flann')

    @func_log
    def is_valid_db_dpath(hs, db_dpath):
        'Checks to see if database conforms to expected conventions'
        if not os.path.exists(db_dpath): 
            logwarn('db_dpath \"'+str(db_dpath)+'\" doesnt exist')
            return False
        db_dpath_files = os.listdir(db_dpath)
        if hs.iom.internal_dname in db_dpath_files:
            logmsg('Opening a HotSpotter database: '+db_dpath)
        elif 'images' in db_dpath_files or\
             'data'   in db_dpath_files:
            logmsg('Opening a StripSpotter database: '+db_dpath)
        elif len(db_dpath_files) == 0:
            logmsg('Creating a new database: '+db_dpath)
        else:
            logwarn('Unknown database type: '+db_dpath)
            logdbg('Files in dir: '+str(db_dpath_files))
            return False
        return True

    @func_log
    def smartget_db_dpath(hs, db_dpath):
        ''' Performs a smart update of the db_dpath
        Trys a number of various  options to get it right

        None = Read from preferences
        ''   = Prompt the User For database
        '''
        if db_dpath is None: # If requested to read prefs
            db_dpath = str(hs.core_prefs.database_dpath)
        if db_dpath in [None, 'None'] or\
           not os.path.exists(db_dpath): # Check validity
            logwarn('db_dpath='+repr(db_dpath)+' is invalid')
            db_dpath = '' 
        if db_dpath == '': # Prompt The User. TODO Move this to Facade/UIManager
            logmsg('what database should I open?')
            try: 
                db_dpath = hs.uim.select_database()
            except: 
                logerr(' Was unable to prompt user with QT')
        return db_dpath

    @func_log
    def restart(hs, db_dpath=None, autoload=True, save_pref_bit=True):
        hs.data_loaded_bit = False
        if hs.db_dpath != None and db_dpath == None:
            db_dpath = hs.db_dpath
        db_dpath = hs.smartget_db_dpath(db_dpath)
        # --
        hs.db_dpath = None
        if hs.is_valid_db_dpath(db_dpath):
            hs.db_dpath = db_dpath
            if save_pref_bit:
                logdbg('Setting db_dpath = '+str(db_dpath))
                hs.core_prefs.update('database_dpath',db_dpath)
        if hs.db_dpath is None:
            logerr('Invalid Database. '+\
                   'Select an existing HotSpotter, StripeSpotter database. '+\
                   'To create a new database, select and empty directory. ')

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

    def get_database_stats(hs):
        return (hs.db_dpath, hs.cm.num_c, hs.gm.num_g, hs.nm.num_n)

    def get_database_stat_str(hs):
        if hs.data_loaded_bit:
            return ( '''
        Database Directory = %s
        #Chips = %d
        #Images = %d
        #Names = %d
            ''' % hs.get_database_stats())
        else: 
            return '''
        No database has been selected.

        * Use the open_database in the IPython window
        OR 
        * In the GUI select File->Open Database

        Valid databases are: 
            a HotSpotter Database Folder
            a StripeSpotter Database Folder
            or an empty folder for a new database
            '''

    def get_dbid(hs):
        return os.path.split(hs.db_dpath)[1]

    def __init__(hs, db_dpath=None, autoload=True, delete_home_dir_bit=False, save_pref_bit=False):
        super( HotSpotterAPI, hs ).__init__(['cm','gm','nm','em','qm','dm','am','vm','iom','uim'])
        #
        hs.db_dpath = None #Database directory.
        hs.data_loaded_bit = False
        hs.core_prefs = None
        # --- Managers ---
        hs.iom = IOManager(hs) # Maintains path structures
        if delete_home_dir_bit:
            # Developer hack to delete the home dir when big things change
            hs.delete_preferences()
        # CLASSES WITH PREFERENCES
        hs.uim = UIManager(hs) # Interface to the QtGui
        hs.dm = DrawManager(hs) # Matplotlib interface. Draws on a GUI
        hs.am = AlgorithmManager(hs) # Settings and Standalone algos
        hs.init_preferences()
        hs.all_pref = Pref()
        hs.all_pref.algo_prefs = hs.am.algo_prefs
        hs.all_pref.core_prefs = hs.core_prefs
        hs.all_pref.ui_prefs   = hs.uim.ui_prefs
        hs.all_pref.draw_prefs = hs.dm.draw_prefs

        # Data Managers
        hs.vm = None # Vocab Manager
        hs.qm = None # Vocab Manager
        hs.em = None # Experiment Manager
        hs.gm = None # Image Manager
        hs.cm = None # Instance Manager
        hs.nm = None # Name Manager
        #m
        if db_dpath != None:
            hs.restart(db_dpath, autoload, save_pref_bit=save_pref_bit)
        # --- 

    def save_database(hs):
        hs.iom.save_tables()
        hs.core_prefs.save()
        hs.dm.draw_prefs.save()
        hs.am.algo_prefs.save()
        hs.uim.ui_prefs.save()

    def merge_database(hs, db_dpath):
        #db_dpath = r'D:\data\work\Lionfish\LF_OPTIMIZADAS_NI_V_E'
        hs_other = HotSpotterAPI(db_dpath)
        gid_offset = hs.gm.max_gid
        cid_offset = hs.cm.max_cid
        for gx in iter(hs_other.gm.get_valid_gxs()):
            gid = hs_other.gm.gx2_gid[gx] + gid_offset
            aif = hs_other.gm.gx2_aif_bit[gx]
            gname = hs_other.gm.gx2_gname[gx]
            src_img = hs_other.gm.gx2_img_fpath(gx)
            # TODO: Have databases be able to handle adding images and not copy them
            #relpath = os.path.relpath(hs_other.iom.get_img_dpath(), hs.iom.get_img_dpath()) 
            #gname = os.path.normpath(relpath+'/'+hs_other.gm.gx2_gname[gx])
            hs.gm.add_img(gid=gid, gname=gname, aif=aif, src_img=src_img)
        for cx in  iter(hs_other.cm.get_valid_cxs()):
            cid, gid, name = hs_other.cm.cx2_(cx, 'cid', 'gid', 'name')
            roi = hs_other.cm.cx2_roi[cx]
            theta = hs_other.cm.cx2_theta[cx]
            nx = hs.nm.add_name(-1, name)
            gx = hs.gm.gid2_gx[gid + gid_offset]
            hs.cm.add_chip(cid + cid_offset, nx, gx, roi, theta)
    # ---
    @func_log
    def load_tables(hs):
        hs.iom.load_tables()
        hs.data_loaded_bit = True
    # --- 
    @func_log
    def batch_rename(hs, name1, name2):
        logmsg('Batch Renaming %s to %s' % (name1, name2))
        cm, nm = hs.get_managers('cm','nm')
        if name1 == nm.UNIDEN_NAME():
            logerr('Cannot batch rename '+str(name1)+'. It is UNIDENTIFIED and has special meaning')
        if name1 not in nm.name2_nx.keys():
            logerr('Cannot batch rename. '+str(name1)+' does not exist')
        cx_list = nm.name2_cx_list(name1)[:] # COPY BEFORE YOU CHANGE. Man, sneaky errors
        num_chips = len(cx_list)
        if num_chips == 0:
            logerr('Cannot batch rename. '+str(name1)+' has no chips')
        logmsg('Renaming '+str(num_chips)+' chips: '+str(cx_list))
        for cx in cx_list:
            logdbg('Batch Rename '+str(cx))
            cm.rename_chip(cx, name2)
        return True
    # --- 
  
    # Will eventaully be cleaned into new more efficient functions
    # Right now they just convert the new efficient format into the 
    # old one
    def add_name2(hs, name):
        nid = hs.nm.add_name(-1, name)
        nx  = hs.nm.nid2_nx[nid]
        return nx
    def add_img2(hs, src_img):
        gid = hs.gm.add_img(src_img=src_img, aif=True)
        gx  = hs.gm.gid2_gx[gid]
        return gx
    def add_chip2(hs, nx, gx, roi, theta):
        cid = hs.cm.add_chip(-1, nx, gx, roi, theta)
        cx  = hs.cm.cid2_cx[cid]
        return cx
    def add_img_list2(hs, img_list):
        return [hs.add_img2(_img) 
                for _img in iter(img_list)]
    def add_name_list2(hs, name_list):
        return [hs.add_name2(_name) 
                for _name in iter(name_list)]
    def add_chip_list2(hs, nx_list, gx_list, roi_list, theta_list):
        return [hs.add_chip2(*carg)  
                for carg in iter(zip(nx_list, gx_list, roi_list, theta_list))]
        
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
    def reload_preferences(hs):
        hs.am.init_preferences(default_bit=True)
        hs.dm.init_preferences(default_bit=True)
        hs.uim.init_preferences(default_bit=True)
        hs.init_preferences(default_bit=True)

    @func_log
    def delete_preferences(hs):
        'Deletes the preference files in the ~/.hotspotter directory'
        logmsg('Deleting the ~/.hotspotter preference directory')
        hs.iom.remove_settings_files_with_pattern('*')
    # ---
    @func_log
    def unload_all_features(hs):
        'Unloads all features and models'
        all_cxs = hs.cm.get_valid_cxs()
        hs.cm.unload_features(all_cxs)
        hs.vm.reset()
    # ---
    @func_log
    def delete_computed_directory(hs):
        'Unloads all features and models and deletes the computed directory'
        logmsg('Deleting the computed directory')
        hs.iom.remove_computed_files_with_pattern('*')
        hs.unload_all_features()
    # ---
    @func_log
    def delete_precomputed_results(hs):
        logmsg('Deleting precomputed results')
        hs.iom.remove_computed_files_with_pattern('rr_*')
    # ---
    def add_roi_to_all_images(hs):
        cm, gm, nm = hs.get_managers('cm','gm','nm')
        gx_list = gm.get_empty_gxs()
        logmsg('Adding '+str(len(gx_list))+' rois to empty images')
        for gx in gx_list:
            (gw, gh) = gm.gx2_img_size(gx)
            cm.add_chip(-1, nm.UNIDEN_NX(), gx, [0, 0, gw, gh], 0)
    # ---
    @func_log
    def precompute_chips(hs, num_procs=None):
        return hotspotter.ChipFunctions.precompute_chips(hs, num_procs)

    @func_log
    def precompute_chipreps(hs, num_procs=None):
        return hotspotter.ChipFunctions.precompute_chipreps(hs, num_procs)

    @func_log
    def ensure_model(hs):
        logdbg('Ensuring Computed Chips')
        hs.precompute_chips()
        logdbg('Ensuring Computed ChipsReps')
        hs.precompute_chipreps()
        logdbg('Ensuring Visual Model')
        hs.vm.build_model()

    @func_log
    def query(hs, qcid, qhs=None):
        '''
        Runs query against this database with cid. 
        If hsother is specified uses the chip from that database
        '''
        qhs = hs if qhs is None else qhs
        qcx = qhs.cm.cx(qcid)
        hs.ensure_model()
        rawres = hs.qm.cx2_rr(qcx, qhs)
        return QueryResult(hs, rawres, qhs)


    @func_log
    def get_source_fpath(hs):
        import __init__ as root_module
        return os.normpath(os.path.dirname(root_module.__file__))

    def get_managers(hs, *manager_list):
        'quick access of managers eg: (am, cm, iom) = hs.managers("am","cm","iom")'
        return tuple([hs.__dict__[manager_name] for manager_name in manager_list])

    def dynget(hs, *dynargs):
        return tuple([hs.__dict__[arg] for arg in dynargs])

    def __getitem__(hs, *dynargs):
        return tuple([hs.__dict__[arg] for arg in dynargs])
