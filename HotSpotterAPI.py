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
from back.tests.Experiments import ExperimentManager
from back.AlgorithmManager import AlgorithmManager
from back.ChipManager import ChipManager
from back.IOManager import IOManager
from back.ImageManager import ImageManager
from back.NameManager import NameManager
from back.QueryManager import QueryManager
from back.VisualModel import VisualModel
from front.DrawManager import DrawManager
from front.UIManager import UIManager
from other.AbstractPrintable import AbstractPrintable
from other.ConcretePrintable import PrefStruct
from other.logger import logdbg, logerr, logmsg, logwarn, func_log
from other.helpers import dircheck, filecheck
import cPickle
import types
import os.path

class HotSpotterAPI(AbstractPrintable):

    def init_preferences(hs):
        iom = hs.iom
        hs.core_prefs = PrefStruct(iom.get_prefs_fpath('core_prefs'))
        hs.core_prefs.database_dpath  = None
        hs.core_prefs.load()

    @func_log
    def is_valid_db_dpath(hs, db_dpath):
        'Checks to see if database conforms to expected conventions'
        if not os.path.exists(db_dpath): 
            logwarn('db_dpath \"'+str(db_dpath)+'\" doesnt exist')
            return False
        db_dpath_files = os.listdir(db_dpath)
        if hs.iom.internal_dname in db_dpath_files:
            logmsg('Loading a HotSpotter database')
        elif 'images' in db_dpath_files or\
             'data'   in db_dpath_files:
            logmsg('Loading a StripSpotter database')
        elif len(db_dpath_files) == 0:
            logmsg('Loading a new database')
        else:
            logwarn('Unknown database type')
            logdbg('Files in dir: '+str(db_dpath_files))
            return False
        return True

    @func_log
    def smartset_db_dpath(hs, db_dpath):
        ''' Performs a smart update of the db_dpath
        Trys a number of various  options to get it right

        None = Read from preferences
        ''   = Prompt the User For database
        '''
        if db_dpath is None: # If requested to read prefs
            db_dpath = str(hs.core_prefs.database_dpath)
        if db_dpath in [None, 'None'] or\
           not os.path.exists(db_dpath): # Check validity
            db_dpath = '' 
            logwarn('Saved database_dpath was invalid')
        if db_dpath == '': # Prompt The User
            db_dpath = hs.uim.select_database()
        # --
        if hs.is_valid_db_dpath(db_dpath):
            logdbg('Setting db_dpath = '+str(db_dpath))
            hs.db_dpath = db_dpath
            hs.core_prefs.update('database_dpath',db_dpath)
        else:
            logerr('Invalid Database. '+\
                   'Select an existing HotSpotter, StripeSpotter database. '+\
                   'To create a new database, select and empty directory. ')

    def __init__(hs):
        super( HotSpotterAPI, hs ).__init__(['cm','gm','nm','em','qm','dm','am','vm','iom','uim'])
        #
        hs.db_dpath = None #Database directory.
        hs.data_loaded_bit = False
        hs.core_prefs = None
        # --- Managers ---
        hs.iom = IOManager(hs) # Maintains path structures
        hs.uim = UIManager(hs) # Interface to the QtGui
        hs.dm = DrawManager(hs) # Matplotlib interface. Draws on a GUI
        hs.am = AlgorithmManager(hs) # Settings and Standalone algos
        # Data Managers
        hs.vm = None # Vocab Manager
        hs.qm = None # Vocab Manager
        hs.em = None # Experiment Manager
        hs.gm = None # Image Manager
        hs.cm = None # Instance Manager
        hs.nm = None # Name Manager
        #-
        hs.init_preferences()
        # --- 

    @func_log
    def restart(hs, db_dpath=None, autoload=True):
        hs.data_loaded_bit = False
        if hs.db_dpath != None and db_dpath == None:
            db_dpath = hs.db_dpath
        hs.smartset_db_dpath(db_dpath)
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
    @func_log
    def add_all_images_recursively(hs, image_list):
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
    def delete_home_pref_directory(hs):
        logmsg('Deleting the ~/.hotspotter preference directory')
        hs.iom.remove_settings_files_with_pattern('*')
    # ---
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
        hs.unload_all_features()
        hs.iom.remove_computed_files_with_pattern('*')
    # ---
    def add_roi_to_all_images(hs):
        cm, gm, nm = hs.get_managers('cm','gm','nm')
        gx_list = gm.get_empty_gxs()
        logmsg('Adding '+str(len(gx_list))+' rois to empty images')
        for gx in gx_list:
            (gw, gh) = gm.gx2_img_size(gx)
            cm.add_chip(-1, nm.UNIDEN_NX(), gx, [0, 0, gw, gh])
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
        import __init__ as root_module
        return os.path.dirname(root_module.__file__)

    def get_managers(hs, *manager_list):
        'quick access of managers eg: (am, cm, iom) = hs.managers("am","cm","iom")'
        return tuple([hs.__dict__[manager_name] for manager_name in manager_list])

    def dynget(hs, *dynargs):
        return tuple([hs.__dict__[arg] for arg in dynargs])

    def __getitem__(hs, *dynargs):
        return tuple([hs.__dict__[arg] for arg in dynargs])

