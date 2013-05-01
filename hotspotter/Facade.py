from HotSpotterAPI  import HotSpotterAPI
from PyQt4.Qt import QObject, pyqtSlot, QTreeWidgetItem
from numpy          import logical_and
from pylab import find
from other.logger import logdbg, logerr, logmsg, func_log, hsl
import other.crossplat as crossplat
import other.messages as messages
import sys
import time
import os.path

# Globals
clbls = ['cid', 'gid', 'nid', 'name', 'roi']
glbls = ['gid', 'gname', 'num_c', 'cids']
nlbls = ['nid', 'name', 'cids']

class Facade(QObject):
    'A friendlier interface into HotSpotter.'
    # Initialization, Opening, and Saving 
    def __init__(fac, use_gui=True, autoload=True):
        super( Facade, fac ).__init__()
        # Create API
        fac.hs = HotSpotterAPI()
        if use_gui: #Make GUI? 
            logdbg('Starting with gui')
            uim = fac.hs.uim
            uim.start_gui(fac)
            fac.show_main_window()
        else: #HACKY HACKY HACK
            logdbg('Starting without gui')
            fac.hs.dm.fignum = 1
            fac.hs.uim.start_gui(fac) #TODO: Remove
        try: # Open previous database
            fac.open_db(None, autoload)
        except Exception as ex:
            import traceback
            print "Error occurred in autoload"
            print str(ex)
            print '<<<<<<   Traceback    >>>>>'
            traceback.print_exc()
            print "Error occurred in autoload"

    @pyqtSlot(name='run_experiments')
    def run_experiments(fac):
       fac.hs.em.run_experiment()

    @pyqtSlot(name='open_db')
    @func_log
    def open_db(fac, db_dpath='', autoload=True):
        'Opens the database db_dpath. Enters UIMode db_dpath is \'\' '
        uim = fac.hs.uim
        uim.update_state('open_database')
        fac.hs.restart(db_dpath, autoload)
        fac.unselect()
        uim.populate_tables()

    def merge_db(fac, db_dpath):
        db_dpath1 = r'D:\data\work\LF_all\LF_OPTIMIZADAS_NI_V_E'
        db_dpath2 = r'D:\data\work\LF_all\LF_WEST_POINT_OPTIMIZADAS'
        db_dpath3 = r'D:\data\work\LF_all\LF_Bajo bonito'
        db_dpath4 = r'D:\data\work\LF_all\LF_Juan'
        db_dpath = db_dpath1
        fac.hs.merge_database(db_dpath)
        

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
        fac.hs.core_prefs.save()
        fac.hs.dm.draw_prefs.save()
        fac.hs.am.algo_prefs.save()
        fac.hs.uim.ui_prefs.save()
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
        if uim.ui_prefs.quick_roi_select and fac.next_empty_image():
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
        # THIS LIST IS IN THE ORDER OF THE TABS. 
        # THIS SHOULD CHANGE TO BE INDEPENDENT OF THAT FIXME
        if not new_state in uim.tab_order:
            if new_state in range(len(uim.tab_order)):
                new_state = uim.tab_order[new_state]+'_view'
            else:
                logerr('State is: '+str(new_state)+', but it must be one of: '+str(uim.tab_order))
        uim.update_state(new_state)
        uim.draw()

    @pyqtSlot(name='query')
    @func_log
    def query(fac, qcid=None):
        uim, cm, qm, vm, nm = fac.hs.get_managers('uim', 'cm','qm','vm', 'nm')
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
            try:
                cx1 = uim.sel_res.rr.qcx
                cx2 = uim.sel_res.top_cx()[0]
            except Exception as ex:
                logdbg('bad quick and dirty facade code: '+str(ex))
                pass
            if uim.ui_prefs.prompt_after_result and cm.cx2_nx[cx1] == nm.UNIDEN_NX() and cm.cx2_nx[cx2] != nm.UNIDEN_NX():
                logdbg('Quick and dirty prompting')
                fac._quick_and_dirty_result_prompt(uim.sel_res.rr.qcx, uim.sel_res.top_cx()[0])
            else:
                logdbg('No Quick and dirty prompting')
                    
        except Exception as ex: 
            uim.update_state('done_querying')
            uim.update_state('query_failed')
            raise



    def _quick_and_dirty_batch_rename(fac):
        from PyQt4.Qt import QDialog
        from front.ChangeNameDialog import Ui_changeNameDialog
        cm, nm, uim = fac.hs.get_managers('cm','nm', 'uim')
        try:
            if uim.sel_cid != None:
                name = cm.cx2_name(uim.sel_cx())
            else:
                name = ''
        except Exception as ex:
            print 'A quick and dirty exception was caught'
            logdbg(str(ex))
            name = ''
        class ChangeNameDialog(QDialog):
            def __init__(self, name, fac):
                super( ChangeNameDialog, self ).__init__()
                self.dlg_skel = Ui_changeNameDialog()
                self.dlg_skel.setupUi(self)
                self.dlg_skel.oldNameEdit.setText(name)
                def qad_batch_rename():
                    print 'qad batch renaming'
                    try:
                        name1 = str(self.dlg_skel.oldNameEdit.text())
                        name2 = str(self.dlg_skel.newNameEdit.text())
                        fac.hs.batch_rename(name1, name2)
                    except Exception as ex:
                        logerr(str(ex))
                    fac.hs.uim.populate_tables()
                    fac.hs.uim.draw()
                    self.close()
                self.dlg_skel.buttonBox.ApplyRole = self.dlg_skel.buttonBox.AcceptRole
                self.dlg_skel.buttonBox.accepted.connect(qad_batch_rename)
                self.show()
        changeNameDlg = ChangeNameDialog(name, fac)
        self = changeNameDlg

    def _quick_and_dirty_result_prompt(fac, cx_query, cx_result):
        from PyQt4.Qt import QDialog
        from front.ResultDialog import Ui_ResultDialog
        from tpl.other.matplotlibwidget import MatplotlibWidget
        cm, nm = fac.hs.get_managers('cm','nm')
        chip1 = cm.cx2_chip(cx_query)
        chip2 = cm.cx2_chip(cx_result)
        query_cid = cm.cid(cx_query)
        top_name = cm.cx2_name(cx_result)
        class ResultDialog(QDialog):
            def __init__(self, chip1, chip2, title1, title2, change_func, fac):
                super( ResultDialog, self ).__init__()
                self.dlg_skel = Ui_ResultDialog()
                self.dlg_skel.setupUi(self)
                self.pltWidget1 = MatplotlibWidget(self)
                self.pltWidget2 = MatplotlibWidget(self)
                self.dlg_skel.horizontalLayout.addWidget(self.pltWidget1)
                self.dlg_skel.horizontalLayout.addWidget(self.pltWidget2)
                def acceptSlot():
                    print 'Accepted QaD Match'
                    print change_func
                    change_func()
                    self.close()
                    fac.hs.uim.draw()
                def rejectSlot():
                    print 'Rejected QaD Match'
                    self.close()
                self.dlg_skel.buttonBox.accepted.connect(acceptSlot)
                self.dlg_skel.buttonBox.rejected.connect(rejectSlot)
                self.fig1 = self.pltWidget1.figure
                self.fig1.show = lambda: self.pltWidget1.show() #HACKY HACK HACK
                self.fig2 = self.pltWidget2.figure
                self.fig2.show = lambda: self.pltWidget2.show() #HACKY HACK HACK
                ax1 = self.fig1.add_subplot(111) 
                ax1.imshow(chip1)
                ax1.set_title(title1)
                ax2 = self.fig2.add_subplot(111) 
                ax2.imshow(chip2)
                ax2.set_title(title2)
                self.pltWidget1.show()
                self.pltWidget1.draw()
                self.pltWidget2.show()
                self.pltWidget2.draw()
                self.show()
        resdlg = ResultDialog(chip1, chip2, 'Unknown Query', 'Accept Match to '+str(top_name)+'?', lambda: fac.rename_cid(top_name, query_cid), fac)


    @pyqtSlot(str, name='logdbg')
    def logdbg(fac, msg):
        logdbg(msg)

    @pyqtSlot(int, name='set_fignum')
    @func_log
    def set_fignum(fac, fignum):
        # This should be a preference
        uim, dm = fac.hs.get_managers('uim','dm')
        dm.fignum = fignum
        uim.set_fignum(fignum)
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

    @pyqtSlot(name='toggle_ellipse')
    def toggle_ellipse(fac):
        dm, uim = fac.hs.get_managers('dm','uim')
        dm.draw_prefs.toggle('ellipse_bit')
        uim.draw()

    @pyqtSlot(name='toggle_points')
    def toggle_points(fac):
        dm, uim = fac.hs.get_managers('dm','uim')
        dm.draw_prefs.toggle('points_bit')
        uim.draw()

    def logs(fac, use_blacklist_bit=True):
        'Prints current logs to the screen'
        print hsl.hidden_logs(use_blacklist_bit)

    @pyqtSlot(name='write_logs')
    def write_logs(fac):
        'Write current logs to a timestamped file and open in an editor'
        timestamp = str(time.time())
        logfname  = fac.hs.iom.get_temp_fpath('hotspotter_logs_'+timestamp+'.txt')
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

    @pyqtSlot(name='convert_all_images_to_chips')
    def convert_all_images_to_chips(fac):
        uim = fac.hs.uim
        uim.update_state('working')
        fac.hs.add_roi_to_all_images()
        uim.populate_tables()
        uim.update_state('chip_view')
        uim.draw()

    def show_main_window(fac):
        uim = fac.hs.uim
        if uim.hsgui != None:
            uim.hsgui.show()
        else: 
            logerr('GUI does not exist')

    def show_edit_preference_widget(fac):
        uim = fac.hs.uim
        if uim.hsgui != None:
            uim.hsgui.epw.show()
        else: 
            logerr('GUI does not exist')

    def redraw(fac):
        uim = fac.hs.uim
        uim.draw()

    @pyqtSlot(name='default_prefs')
    def default_prefs(fac):
        uim = fac.hs.uim
        fac.hs.reload_preferences()
        logmsg('The change to defaults will not become permanent until you save or change one')
        if uim.hsgui != None:
            uim.hsgui.epw.pref_model.dataChanged.emit()

    def unload_features_and_models(fac):
        fac.hs.unload_all_features()

    def write_database_stats(fac):
        'Writes some statistics to disk and returns them'
        import numpy as np
        cm, nm, gm, iom = fac.hs.get_managers('cm','nm','gm','iom')
        num_images = gm.num_g
        num_chips  = cm.num_c
        num_names  = nm.num_n

        vgx2_nChips = [] # Num Chips Per Image
        for gx in iter(gm.get_valid_gxs()):
            vgx2_nChips += [len(gm.gx2_cx_list[gx])]
        vgx2_nChips = np.array(vgx2_nChips)

        chips_per_image_mean = np.mean(vgx2_nChips)
        chips_per_image_std  = np.std(vgx2_nChips)

        chips_per_image_mean_gt0 = np.mean(vgx2_nChips[vgx2_nChips > 0])
        chips_per_image_std_gt0  = np.std(vgx2_nChips[vgx2_nChips > 0])

        db_stats = \
        [
            'Num Images:           %d' % num_images ,
            'Num Chips:            %d' % num_chips,
            'Num Names:            %d' % num_names,
            'Num Tagged Images:    %d' % (vgx2_nChips >= 1).sum(),
            'Num Untagged Images:  %d' % (vgx2_nChips == 0).sum(),
            'Num Chips/TaggedImage:  %.2f += %.2f ' % ( chips_per_image_mean_gt0, chips_per_image_std_gt0 ),
            'Num Chips/Image:        %.2f += %.2f ' % ( chips_per_image_mean, chips_per_image_std ),
        ]
        db_stats_str = '\n'.join(db_stats)
        iom.write_to_user_fpath('database_stats.txt', db_stats_str)
        return db_stats_str
