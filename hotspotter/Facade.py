from HotSpotterAPI  import HotSpotterAPI
from PyQt4.Qt import QObject, pyqtSlot, QTreeWidgetItem, QDialog, QInputDialog
from pylab import find
from other.logger import logdbg, logerr, logmsg, func_log, hsl
import other.crossplat as crossplat
import other.messages as messages
import sys
import time
import os.path
import numpy as np

# Globals
clbls = ['cid', 'gid', 'nid', 'name', 'roi', 'theta']
glbls = ['gid', 'gname', 'num_c', 'cids']
nlbls = ['nid', 'name', 'cids']

class Facade(QObject):
    'A friendlier interface into HotSpotter.'
    # Initialization, Opening, and Saving 
    def __init__(fac, use_gui=True, autoload=True):
        super( Facade, fac ).__init__()
        # Create API
        fac.hs = HotSpotterAPI(autoload=False)
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
            print("Error occurred in autoload")
            print(str(ex))
            print('<<<<<<   Traceback    >>>>>')
            traceback.print_exc()
            print("Error occurred in autoload")

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
        fac.hs.merge_database(db_dpath)
        #fac.hs.merge_database(r'D:\data\work\Lionfish\LF_OPTIMIZADAS_NI_V_E')
        #fac.hs.merge_database(r'D:\data\work\Lionfish\LF_WEST_POINT_OPTIMIZADAS')
        #fac.hs.merge_database(r'D:\data\work\Lionfish\LF_Bajo_bonito')
        #fac.hs.merge_database(r'D:\data\work\Lionfish\LF_Juan')
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
        'Saves the database chip, image, and name tables'
        iom, uim = fac.hs.get_managers('iom','uim')
        old_state = uim.update_state('save_database')
        fac.hs.save_database()
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
        new_roi = uim.annotate_roi()
        theta = 0
        uim.update_state('image_view')
        new_cid = cm.add_chip(-1, nm.UNIDEN_NX(), gx, new_roi, theta, delete_prev=True)
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
        new_roi = uim.annotate_roi()
        sel_cx = uim.sel_cx()
        fac.hs.cm.change_roi(sel_cx, new_roi)
        uim.draw()

    @pyqtSlot(name='reselect_orientation')
    @func_log
    def reselect_orientation(fac):
        uim = fac.hs.uim
        new_theta = uim.annotate_orientation()
        sel_cx = uim.sel_cx()
        fac.hs.cm.change_orientation(sel_cx, new_theta)
        uim.draw()

    @pyqtSlot(str, int, name='rename_cid')
    @func_log
    def rename_cid(fac, new_name, cid=-1):
        cm, uim = fac.hs.get_managers('cm','uim')
        if cid == -1:
            cid = uim.sel_cid
        cm.rename_chip(cm.cx(cid), str(new_name))
        uim.populate_tables()

    @pyqtSlot(str, int, name='change_chip_prop')
    @func_log
    def change_chip_prop(fac, propname, newval, cid=-1):
        cm, uim = fac.hs.get_managers('cm','uim')
        if cid == -1:
            cid = uim.sel_cid
        cx = cm.cx(cid)
        cm.user_props[str(propname)][cx] = str(newval).replace('\n','\t').replace(',',';;')
        uim.populate_tables()

    @pyqtSlot(name='add_new_prop')
    @func_log
    def add_new_prop(fac, propname=None):
        'add a new property to keep track of'
        if propname is None:
            # User ask
            dlg = QInputDialog()
            textres = dlg.getText(None, 'New Metadata Property','What is the new property name? ')
            if not textres[1]:
                logmsg('Cancelled new property')
                return
            propname = str(textres[0])

        logmsg('Adding property '+propname)
        fac.hs.cm.add_user_prop(propname)
        fac.hs.uim.populate_tables()


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
        prevBlock = uim.hsgui.main_skel.tablesTabWidget.blockSignals(True)
        # THIS LIST IS IN THE ORDER OF THE TABS. 
        # THIS SHOULD CHANGE TO BE INDEPENDENT OF THAT FIXME
        if not new_state in uim.tab_order:
            if new_state in xrange(len(uim.tab_order)):
                new_state = uim.tab_order[new_state]+'_view'
            else:
                logerr('State is: '+str(new_state)+', but it must be one of: '+str(uim.tab_order))
        uim.update_state(new_state)
        uim.draw()
        uim.hsgui.main_skel.tablesTabWidget.blockSignals(prevBlock)
        

    @pyqtSlot(name='query')
    @func_log
    def query(fac, qcid=None):
        'Performs a query'
        uim, cm, qm, vm, nm = fac.hs.get_managers('uim', 'cm','qm','vm', 'nm')
        try:
            if qcid is None:
                qcid = uim.sel_cid
            else: 
                uim.select_cid(qcid)
            qcx = cm.cx(qcid)
            uim.update_state('Querying')
            print('Querying Chip: '+cm.cx2_info(qcx, clbls))
            logdbg('\n\nQuerying Chip: '+cm.cx2_info(qcx, clbls))
            uim.sel_res = fac.hs.query(qcid)
            logmsg('\n\nFinished Query')
            uim.update_state('done_querying')
            logmsg(str(uim.sel_res))
            logdbg('\n\n*** Populating Results Tables ***')
            uim.populate_result_table()
            logdbg('\n\n*** Switching To Result Views ***')
            uim.update_state('result_view')
            logdbg('\n\n*** Redrawing UI ***')
            uim.draw()
            logdbg('\n\n*** Done Redrawing UI ***')
            # QUICK AND DIRTY CODE. PLEASE FIXME
            try:
                cx1 = uim.sel_res.rr.qcx
                cx2 = uim.sel_res.top_cx()[0]
                if uim.ui_prefs.prompt_after_result and cm.cx2_nx[cx1] == nm.UNIDEN_NX() and cm.cx2_nx[cx2] != nm.UNIDEN_NX():
                    logdbg('Quick and dirty prompting')
                    fac._quick_and_dirty_result_prompt(uim.sel_res.rr.qcx, uim.sel_res.top_cx()[0])
                else:
                    logdbg('No Quick and dirty prompting')
            except Exception as ex:
                logdbg('bad quick and dirty facade code: '+str(ex))
                pass
            logdbg('\n\n-----------Query OVER-------------\n\n')
                    
        except Exception as ex: 
            uim.update_state('done_querying')
            uim.update_state('query_failed')
            raise



    def _quick_and_dirty_batch_rename(fac):
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
        print( fac.hs.get_database_stat_str() )

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
        print('\n\n  ___STATUS___')
        fac.print_database_stats()
        fac.print_selected()
        print('\n Need Help? type print_help()')
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
            dname = fac.hs.iom.hsroot()
        crossplat.view_directory(dname)

    @pyqtSlot(name='select_next')
    @func_log
    def select_next(fac):
        uim = fac.hs.uim
        if uim.state == 'chip_view':
            fac.next_unident_chip()
        elif uim.state == 'image_view':
            if not fac.next_empty_image():
                if not fac.next_equal_size_chip():
                    fac.next_0_theta_chip()

        else:
            logerr('Cannot goto next in state: '+uim.state)

    @func_log
    def next_unident_chip(fac):
        'Next call that finds a chip that is unidentified'
        empty_cxs = find(np.logical_and(fac.hs.cm.cx2_nx == fac.hs.nm.UNIDEN_NX(), fac.hs.cm.cx2_cid > 0))
        if len(empty_cxs) == 0:
            print 'There are no more empty images'
            return False
        cx = empty_cxs[0]
        cid = fac.hs.cm.cx2_cid[cx]
        fac.selc(cid)
        return True

    @func_log
    def next_empty_image(fac):
        'Next call that finds an image without a chip'
        empty_gxs = fac.hs.gm.get_empty_gxs()
        if len(empty_gxs) == 0:
            print 'There are no more empty images.'
            return False
        gx = empty_gxs[0]
        gid = fac.hs.gm.gx2_gid[gx]
        fac.selg(gid)
        return True

    @func_log
    def next_equal_size_chip(fac):
        'Next call that finds a chip where the entire image is the roi'
        cm = fac.hs.cm
        gm = fac.hs.gm
        valid_cxs = cm.get_valid_cxs()
        fac.hs.cm.cx2_nx
        gid = -1
        for cx in iter(valid_cxs):
            (gw,gh) = gm.gx2_img_size(cm.cx2_gx[cx])
            (_,_,cw,ch) = cm.cx2_roi[cx]
            if gw == cw and ch == gh:
                gid = fac.hs.cm.cx2_gid(cx)
                break
        if gid == -1: 
            print 'There are no more unrefined rois'
            return False
        fac.selg(gid)

    @func_log
    def next_0_theta_chip(fac):
        'Next call that finds a chip without an orientation'
        cm = fac.hs.cm
        gm = fac.hs.gm
        valid_cxs = cm.get_valid_cxs()
        fac.hs.cm.cx2_nx
        gid = -1
        for cx in iter(valid_cxs):
            (gw,gh) = gm.gx2_img_size(cm.cx2_gx[cx])
            (_,_,cw,ch) = cm.cx2_roi[cx]
            if cm.cx2_theta[cx] == 0:
                gid = fac.hs.cm.cx2_gid(cx)
                break
        if gid == -1: 
            print 'There are no more 0 theta rois'
            return False
        fac.selg(gid)

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

    def profile(cmd):
        # Meliae # from meliae import loader # om = loader.load('filename.json') # s = om.summarize();
        import cProfile, sys, os
        print('Profiling Command: '+cmd)
        cProfOut_fpath = 'OpenGLContext.profile'
        cProfile.runctx( cmd, globals(), locals(), filename=cProfOut_fpath )
        # RUN SNAKE
        print('Profiled Output: '+cProfOut_fpath)
        if sys.platform == 'win32':
            rsr_fpath = 'C:/Python27/Scripts/runsnake.exe'
        else:
            rsr_fpath = 'runsnake'
        view_cmd = rsr_fpath+' "'+cProfOut_fpath+'"'
        os.system(view_cmd)
        #import pstat
        #stats = pstats.Stats(cProfOut_fpath)
        #stats.print()

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
        if not uim.hsgui is None:
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
            uim.hsgui.epw.pref_model.layoutChanged.emit()

    def unload_features_and_models(fac):
        fac.hs.unload_all_features()


    def figure_for_paper(fac):
        fac.set_fignum(1)
        fac.hs.dm.draw_prefs.ellipse_bit = True
        fac.hs.dm.draw_prefs.points_bit = False
        #fac.selg(7)
        import random
        random.seed(0)
        fsel_ret = fac.hs.dm.show_chip(1, in_raw_chip=True, fsel='rand', ell_alpha=1, bbox_bit=False, color=[0,0,1], ell_bit=True, xy_bit=False)
        #fsel_ret = fac.hs.dm.show_chip(1, in_raw_chip=True, fsel=fsel_ret, ell_alpha=1, bbox_bit=False)
        return fsel_ret




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

    def SetNamesFromLionfishGroundTruth(fac):
        import os.path
        import re
        cm = fac.hs.cm
        nm = fac.hs.nm
        gm = fac.hs.gm
        name_fn = lambda path: os.path.splitext(os.path.split(path)[1])[0]
        re_lfname = re.compile(r'(?P<DATASET>.*)-(?P<NAMEID>\d\d)-(?P<SIGHTINGID>[a-z])')
        for cx in iter(cm.get_valid_cxs()):
            gx = cm.cx2_gx[cx]
            name = name_fn(gm.gx2_gname[gx])
            match_obj = re_lfname.match(name)
            if match_obj != None:
                match_dict = match_obj.groupdict()
                name_id = match_dict['NAMEID']
                dataset = match_dict['DATASET']
                sightingid = match_dict['SIGHTINGID']
                if dataset.find('BB') > -1:
                    dataset = 'BB'
                if dataset.find('SA-WP') > -1:
                    dataset = 'WP'
                if dataset.find('SA-NV') > -1:
                    dataset = 'NV'
                if dataset.find('SA-VE') > -1:
                    dataset = 'VE'
                new_name = 'Lionfish_'+str(dataset)+str(name_id)
                cm.rename_chip(cx, new_name)

    @pyqtSlot(name='expand_rois')
    @func_log
    def expand_rois(fac, percent_increase=None):
        'expand rois by a percentage of the diagonal'
        if percent_increase == None:
            # User ask
            dlg = QInputDialog()
            percentres = dlg.getText(None, 'ROI Expansion Factor', 
                                    'Enter the percentage to expand the ROIs.\n'+
                                'The percentage is in terms of diagonal length')
            if not percentres[1]:
                logmsg('Cancelled all match')
                return
            try:
                percent_increase = float(str(percentres[0]))
            except ValueError: 
                logerr('The percentage must be a number')
        cm = fac.hs.cm
        gm = fac.hs.gm
        logmsg('Resizing all chips')
        for cx in iter(cm.get_valid_cxs()):
            logmsg('Resizing cx='+str(cx))
            # Get ROI size and Image size
            [rx, ry, rw, rh] = cm.cx2_roi[cx]
            [gw, gh] = gm.gx2_img_size(cm.cx2_gx[cx])
            # Find Diagonal Increase 
            diag = np.sqrt(rw**2 + rh**2)
            scale_factor = percent_increase/100.0
            diag_increase = scale_factor * diag
            target_diag = diag + diag_increase
            # Find Width/Height Increase 
            ar = float(rw)/float(rh)
            w_increase = np.sqrt(ar**2 * diag_increase**2 / (ar**2 + 1))
            h_increase = w_increase / ar
            # Find New xywh within image constriants 
            new_x = int(max(0, round(rx - w_increase / 2.0)))
            new_y = int(max(0, round(ry - h_increase / 2.0)))
            new_w = int(min(gw - new_x, round(rw + w_increase)))
            new_h = int(min(gh - new_y, round(rh + h_increase)))
            new_roi = [new_x, new_y, new_w, new_h]
            logmsg('Old Roi: '+repr([rx, ry, rw, rh]))
            cm.change_roi(cx, new_roi)
            logmsg('\n')
        logmsg('Done resizing all chips')

    @pyqtSlot(name='match_all_above_thresh')
    def match_all_above_thresh(fac, threshold=None):
        'do matching and assign all above thresh'
        if threshold == None:
            # User ask
            dlg = QInputDialog()
            threshres = dlg.getText(None, 'Threshold Selector', 
                                    'Enter a matching threshold.\n'+
             'The system will query each chip and assign all matches above this thresh')
            if not threshres[1]:
                logmsg('Cancelled all match')
                return
            try:
                threshold = float(str(threshres[0]))
            except ValueError: 
                logerr('The threshold must be a number')
        qm = fac.hs.qm
        cm = fac.hs.cm
        nm = fac.hs.nm
        vm = fac.hs.vm
        # Get model ready
        vm.sample_train_set()
        fac.hs.ensure_model()
        # Do all queries
        for qcx in iter(cm.get_valid_cxs()):
            qcid = cm.cx2_cid[qcx]
            logmsg('Querying CID='+str(qcid))
            query_name = cm.cx2_name(qcx)
            logdbg(str(qcx))
            logdbg(str(type(qcx)))
            cm.load_features(qcx)
            res = fac.hs.query(qcid)
            # Match only those above a thresh
            res.num_top_min = 0
            res.num_extra_return = 0
            res.top_thresh = threshold
            top_cx = res.top_cx()
            if len(top_cx) == 0:
                print('No matched for cid='+str(qcid))
                continue
            top_names = cm.cx2_name(top_cx)
            all_names = np.append(top_names,[query_name])
            if all([nm.UNIDEN_NAME() == name for name in all_names]):
                # If all names haven't been identified, make a new one 
                new_name = nm.get_new_name()
            else:
                # Rename to the most frequent non ____ name seen
                from collections import Counter
                name_freq = Counter(np.append(top_names,[query_name])).most_common()
                new_name = name_freq[0][0] 
                if new_name == nm.UNIDEN_NAME():
                    new_name = name_freq[1][0]
            # Do renaming
            cm.rename_chip(qcx, new_name)
            for cx in top_cx:
                cm.rename_chip(cx, new_name)
        fac.hs.uim.populate_tables()

    @pyqtSlot(str, name='logdbgSlot')
    def logdbgSlot(fac, msg):
        # This function is a hack so MainWin can call logdbg
        logdbg(msg)

    @pyqtSlot(name='run_matching_experiment')
    @func_log
    def run_matching_experiment(fac):
        fac.hs.em.run_matching_experiment()

    @pyqtSlot(name='run_name_consistency_experiment')
    @func_log
    def run_name_consistency_experiment(fac):
        fac.hs.em.run_name_consistency_experiment()

    @pyqtSlot(name='view_documentation')
    @func_log
    def view_documentation(fac):
        import os.path
        pdf_name = 'HotSpotterUserGuide.pdf'
        doc_path = os.path.join(fac.hs.iom.hsroot(), 'documentation')
        pdf_fpath = os.path.join(doc_path, pdf_name)
        # Tries to open pdf, if it fails it opens the documentation folder
        if os.system('open '+pdf_fpath) == 1:
            if os.system(pdf_fpath) == 1:
                crossplat.view_directory(doc_path)

    @pyqtSlot(name='precompute')
    @func_log
    def precompute(fac):
        fac.hs.ensure_model()
