from HotSpotterAPI  import HotSpotterAPI
from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui  import QTreeWidgetItem
from numpy          import logical_and
from other.helpers     import *
from other.logger import *
import other.crossplat as crossplat
import other.messages as messages
import subprocess
import sys
import time

# Globals
clbls = ['cid','gid','nid','name','roi']
glbls = ['gid','gname','num_c','cids']
nlbls = ['nid','name','cids']

class Facade(QObject):
    'A friendlier interface into HotSpotter.'
    # Initialization, Opening, and Saving 
    def __init__(fac, use_gui=True, autoload=True, init_prefs={}):
        super( Facade, fac ).__init__()
        # Create API
        fac.hs = HotSpotterAPI()
        # Specify Command Line Prefs
        if len(init_prefs) > 0:
            fac.hs.read_prefs()
            for key in init_prefs.keys():
                fac.hs.prefs[key] = init_prefs[key]
            fac.hs.write_prefs()
        if use_gui: #Make GUI? 
            uim = fac.hs.uim

            uim.start_gui(fac)
        # Open previous database
        try:
            fac.open_db(None, autoload)
        except Exception as e:
            print "Error occurred in autoload"
            print str(e)
            print '<<<<<<   Traceback    >>>>>'
            traceback.print_exc()
            print "Error occurred in autoload"

    @pyqtSlot(name='run_experiments')
    def run_experiments(fac):
        print fac.hs.em

    @pyqtSlot(name='open_db')
    @func_log
    def open_db(fac, db_dpath='', autoload=True):
        'Opens the database db_dpath. Enters UIMode db_dpath is \'\' '
        uim = fac.hs.uim
        uim.update_state('open_database')
        fac.hs.restart(db_dpath, autoload)
        fac.unselect()
        uim.populate_tables()

    @func_log
    def unselect(fac):
        uim = fac.hs.uim
        uim.update_state('splash_view')
        uim.unselect_all()
        uim.draw()

    @pyqtSlot(name='save_database')
    @func_log
    def save_db(fac):
        iom, uim = fac.hs.get_managers('iom','uim')
        old_state = uim.update_state('save_database')
        'Saves the database chip, image, and name tables'
        fac.hs.iom.save_tables()
        uim.update_state(old_state)
    # ---------------
    @pyqtSlot(name='import_images')
    @func_log
    def import_images(fac):
        uim = fac.hs.uim
        old_state = uim.update_state('import_images')
        fac.hs.iom.save_tables()
        image_list = fac.hs.uim.select_images_on_disk()
        fac.hs.add_image_list(image_list)
        uim.populate_tables()
        uim.update_state(old_state)

    @pyqtSlot(name='add_chip')
    @func_log
    def add_chip(fac, gid=None):
        gm, nm, cm, uim = fac.hs.get_managers('gm', 'nm','cm', 'uim')
        if gid=='None' or gid == None:
            gx = uim.sel_gx()
        else:
            gx = gm.gx(gid)
            uim.select_gid(gid)
        uim.update_state('add_chip')
        new_roi = uim.select_roi()
        uim.update_state('image_view')
        new_cid = cm.add_chip(-1, nm.UNIDEN_NX(), gx, new_roi, delete_prev=True)
        uim.select_cid(new_cid)
        print 'New Chip: '+fac.hs.cm.info(new_cid, clbls)
        #If in beast mode, then move to the next ROI without drawing
        if fac.hs.prefs['roi_beast_mode'] and fac.next_empty_image():
            num_empty = len(fac.hs.gm.get_empty_gxs())
            print 'Only %d left to go!' % num_empty
        else:
            uim.populate_tables()
            uim.draw()

    @pyqtSlot(name='reselect_roi')
    @func_log
    def reselect_roi(fac):
        uim = fac.hs.uim
        new_roi = uim.select_roi()
        sel_cx = uim.sel_cx()
        fac.hs.cm.change_roi(sel_cx, new_roi)
        uim.draw()

    @pyqtSlot(str, int, name='rename_cid')
    @func_log
    def rename_cid(fac, new_name, cid=-1):
        cm, uim = fac.hs.get_managers('cm','uim')
        if cid == -1:
            cid = uim.sel_cid
        cm.rename_chip(cm.cx(cid), str(new_name))
        uim.populate_tables()

    @pyqtSlot(name='remove_cid')
    @func_log
    def remove_cid(fac, cid=None):
        uim = fac.hs.uim
        uim.update_state('image_view')
        if cid == 'None' or cid == None:
            cx = uim.sel_cx()
        else: 
            uim.select_cid(cid)
            cx = cm.cx(cid)
        fac.hs.cm.remove_chip(cx)
        uim.select_gid(uim.sel_gid)
        uim.populate_tables()
        uim.draw()

    @func_log
    @pyqtSlot(name='selc')
    def selc(fac, cid):
        uim = fac.hs.uim
        uim.update_state('chip_view')
        uim.select_cid(cid)
        uim.draw()

    @func_log
    @pyqtSlot(name='selg')
    def selg(fac, gid):
        uim = fac.hs.uim
        uim.update_state('image_view')
        uim.select_gid(gid)
        uim.draw()

    @pyqtSlot(int, name='change_view')
    @func_log
    def change_view(fac, new_state):
        uim = fac.hs.uim
        print new_state
        # THIS LIST IS IN THE ORDER OF THE TABS. 
        # THIS SHOULD CHANGE TO BE INDEPENDENT OF THAT FIXME
        valid_states = ['image_view','chip_view','result_view']
        if not new_state in valid_states:
            if new_state in range(len(valid_states)):
                new_state = valid_states[new_state]
            else:
                logerr('State is: '+str(new_state)+', but it must be one of: '+str(valid_states))
        uim.update_state(new_state)
        uim.draw()

    @pyqtSlot(name='query')
    @func_log
    def query(fac, qcid=None):
        uim, cm, qm, vm = fac.hs.get_managers('uim', 'cm','qm','vm')
        try:
            if qcid is None:
                qcx = uim.sel_cx()
            else: 
                uim.select_cid(qcid)
                qcx = cm.cx(qcid)
            uim.update_state('Querying')
            vm.sample_train_set()
            vm.build_model()
            print 'Querying Chip: '+fac.hs.cm.cx2_info(qcx, clbls)
            uim.sel_res = fac.hs.qm.cx2_res(qcx)
            uim.update_state('done_querying')
            logdbg(str(uim.sel_res))
            uim.update_state('result_view')
            uim.populate_result_table()
            uim.draw()
        except Exception as e: 
            uim.update_state('done_querying')
            uim.update_state('query_failed')
            raise

    @pyqtSlot(str, name='logdbg')
    def logdbg(fac, msg):
        logdbg(msg)

    @pyqtSlot(int, name='set_fignum')
    @func_log
    def set_fignum(fac, fignum):
        # This should be a preference
        uim, dm = fac.hs.get_managers('uim','dm')
        dm.fignum = fignum
        uim.draw()
    # Printing Functions --------------------------------------------
    # - print image/chip/name table
    def gtbl(fac):
        gm = fac.hs.gm
        print gm.gx2_info(lbls=glbls)
    def ctbl(fac):
        cm = fac.hs.cm
        print cm.cx2_info(lbls=clbls)
    def ntbl(fac):
        nm = fac.hs.nm
        print nm.nx2_info(lbls=nlbls)

    def print_database_stats(fac):
        if fac.hs.data_loaded_bit:
            print( '''
        Database Directory = %s
        #Chips = %d
        #Images = %d
        #Names = %d
            ''' % (fac.hs.db_dpath, fac.hs.cm.num_c, fac.hs.gm.num_g, fac.hs.nm.num_n))
        else: 
            print '''
        No database has been selected.

        * Use the open_database in the IPython window
        OR 
        * In the GUI select File->Open Database

        Valid databases are: 
            a HotSpotter Database Folder
            a StripeSpotter Database Folder
            or an empty folder for a new database
            '''

    def print_selected(fac):
        uim = fac.hs.uim
        print '''
        HotSpotter State: '''+uim.state+'''
        Selected CID: '''+str(uim.sel_cid)+'''
        Selected GID: '''+str(uim.sel_gid)
        if uim.sel_gid != None:
            gx = uim.sel_gx()
            lbls = ['gid','gname','num_c']
            print 'Image Info: \n'+fac.hs.gm.gx2_info(gx,lbls).replace('\n', '  \n')
        if uim.sel_cid != None: 
            cx = uim.sel_cx()
            lbls = ['cid','gid','nid','name','roi']
            print 'Chip Info: \n'+fac.hs.cm.cx2_info(cx, lbls).replace('\n', '  \n')

    def print_status(fac):
        print '\n\n  ___STATUS___'
        fac.print_database_stats()
        fac.print_selected()
        print '\n Need Help? type print_help()'
        sys.stdout.flush()

    @pyqtSlot(name='vdd')
    def vdd(fac):
        'Opens the database directory window'
        crossplat.view_directory(fac.hs.db_dpath)

    @pyqtSlot(name='vdi')
    def vdi(fac):
        'View the .hsInternal directory'
        crossplat.view_directory(fac.hs.iom.get_internal_dpath())

    @pyqtSlot(name='vd')
    def vd(fac, dname=None):
        'View a specific directory (defaults to source directory)'
        if dname == None:
            hssourcedir = os.path.join(os.path.dirname(crossplat.__file__))
            dname = hssourcedir
        crossplat.view_directory(dname)

    @pyqtSlot(name='select_next')
    @func_log
    def select_next(fac):
        uim = fac.hs.uim
        if uim.state == 'chip_view':
            fac.next_unident_chip()
        elif uim.state == 'image_view':
            fac.next_empty_image()
        else:
            logerr('Cannot goto next in state: '+uim.state)

    @func_log
    def next_empty_image(fac):
        empty_gxs = fac.hs.gm.get_empty_gxs()
        if len(empty_gxs) == 0:
            print 'There are no more empty images.'
            return False
        gx = empty_gxs[0]
        gid = fac.hs.gm.gx2_gid[gx]
        fac.selg(gid)
        return True

    @func_log
    def next_unident_chip(fac):
        empty_cxs = find(logical_and(fac.hs.cm.cx2_nx == fac.hs.nm.UNIDEN_NX(), fac.hs.cm.cx2_cid > 0))
        if len(empty_cxs) == 0:
            print 'There are no more empty images'
            return False
        cx = empty_cxs[0]
        cid = fac.hs.cm.cx2_cid[cx]
        fac.selc(cid)
        return True

    @pyqtSlot(QTreeWidgetItem, int, name='change_pref')
    def change_pref(fac, item, col):
        print item.data(0,Qt.DisplayRole)
        print item.data(1,Qt.DisplayRole)


    @func_log
    def toggle_pref(fac,pref_name):
        uim = fac.hs.uim
        uim.hs.set_pref(pref_name, 'toggle')
        uim.draw()

    def logs(fac, use_blacklist_bit=True):
        'Prints current logs to the screen'
        print hsl.hidden_logs(use_blacklist_bit)

    @pyqtSlot(name='write_logs')
    def write_logs(fac):
        'Write current logs to a timestamped file and open in an editor'
        timestamp = str(time.time())
        logfname  = 'hotspotter_logs_'+timestamp+'.txt'
        with open(logfname,'w') as logfile:
            logfile.write(str(hsl))
        crossplat.view_text_file(logfname)

    def print_help(fac):
        print messages.cmd_help

    @func_log
    def profile(fac, cmd='fac.query(1)'):
        # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
        import cProfile
        iom = fac.hs.iom
        logmsg('Profiling Command: '+cmd)
        cProfOut_fpath = iom.get_temp_fpath('OpenGLContext_'+cmd+'.profile')
        cProfile.runctx( cmd, globals(), locals(), filename=cProfOut_fpath )
        logdbg('Profiled Output: '+cProfOut_fpath)
        rsr_fpath = 'C:\\Python27\\Scripts\\runsnake.exe'
        view_cmd = rsr_fpath+' '+cProfOut_fpath
        os.system(view_cmd)

    @func_log
    def line_profile(fac, cmd='fac.query(1)'):
        # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
        import line_profiler
        iom = fac.hs.iom
        logmsg('Line Profiling Command: '+cmd)
        line_profile_fpath = iom.get_temp_fpath('line_profile.'+cmd+'.profile')
        lp = line_profiler.LineProfiler()
        from inspect import getmembers, isfunction, ismethod

        for module in [fac.hs, fac.hs.cm, fac.hs.gm, fac.hs.nm, fac.hs.qm, fac.hs.vm, fac.hs.am, fac.hs.dm]:
            for (method_name, method) in getmembers(module, ismethod):
                lp.add_function(method.im_func)
            #functions_list = [o for o in getmembers(module, isfunction)]
        
        lp.runctx( cmd, globals(), locals())
        lp.dump_stats(line_profile_fpath)
        lp.print_stats()
        rsr_fpath = 'C:\\Python27\\Scripts\\runsnake.exe'
        view_cmd = rsr_fpath+' '+line_profile_fpath
        os.system(view_cmd)
        return lp

    def call_graph(fac, cmd='fac.query(1)'):
        import pycallgraph
        import Image
        iom = fac.hs.iom
        logmsg('Call Graph Command: '+cmd)
        callgraph_fpath = iom.get_temp_fpath('callgraph'+cmd+'.png')
        pycallgraph.start_trace()
        eval(cmd)
        pycallgraph.stop_trace()
        pycallgraph.make_dot_graph(callgraph_fpath)
        Image.open(callgraph_fpath).show()

    def get_namespace(fac):
        ''' Alias facade functions so the command line can reference them by name
        the namespace eg: fac.select_cid(3) gets an alias select_cid(4)'''
