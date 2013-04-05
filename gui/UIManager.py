from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui  import QFileDialog
from gui.HotspotterMainWindow import HotspotterMainWindow
from numpy import setdiff1d
from other.helpers   import *
from other.logger import *
# The UIManager should be running in the same thread as 
# the Facade functions. It should talk to the hsgui with 
# signals and slots
class UIManager(QObject):
    # --- UIManager things that deal with the GUI Directly
    # Signals that call GUI slots
    populateChipTblSignal   = pyqtSignal(list, list, list, list)
    populateImageTblSignal  = pyqtSignal(list, list, list, list)
    populateResultTblSignal = pyqtSignal(list, list, list, list)
    populatePrefTreeSignal  = pyqtSignal(dict)
    updateStateSignal       = pyqtSignal(str)
    selectionSignal         = pyqtSignal(int, int)
    redrawGuiSignal         = pyqtSignal()
    changeTabSignal         = pyqtSignal(int)


    # --- UIManager talks to the main thread
    def __init__(uim, hs):
        super( UIManager, uim ).__init__()
        uim.hs = hs
        uim.hsgui = None
        # User Interface State
        uim.sel_cid = None
        uim.sel_gid = None
        uim.sel_res = None
        uim.state = 'splash_view'
        uim.tab_order = ['image','chip','result']

    def start_gui(uim, fac): # Currently needs facade access
        logdbg('Creating the GUI')
        uim.hsgui = HotspotterMainWindow(fac)

        logdbg('Connecting Facade >> to >> GUI')
        uim.populateChipTblSignal.connect( uim.hsgui.populateChipTblSlot )
        uim.populateImageTblSignal.connect( uim.hsgui.populateImageTblSlot )
        uim.populateResultTblSignal.connect( uim.hsgui.populateResultTblSlot )
        uim.updateStateSignal.connect( uim.hsgui.updateStateSlot ) 
        uim.selectionSignal.connect( uim.hsgui.updateSelSpinsSlot )
        uim.redrawGuiSignal.connect( uim.hsgui.redrawGuiSlot )
        uim.populatePrefTreeSignal.connect( uim.hsgui.epw.populatePrefTreeSlot )
        uim.changeTabSignal.connect( uim.hsgui.main_skel.tablesTabWidget.setCurrentIndex )
        uim.populate_algo_settings()

    def get_gui_figure(uim):
        'returns the matplotlib.pyplot.figure'
        if uim.hsgui != None and uim.hsgui.plotWidget != None:
            fig = uim.hsgui.plotWidget.figure
            fig.show = lambda: uim.hsgui.plotWidget.show() #HACKY HACK HACK
            return fig
        return None

    @func_log
    def draw(uim):
        'Tells the HotSpotterAPI to draw the current selection in the current mode'
        cm, gm = uim.hs.get_managers('cm','gm')
        current_tab = uim.hsgui.main_skel.tablesTabWidget.currentIndex
        if uim.state in ['splash_view']:
            uim.hs.dm.show_splash()
        elif uim.state in ['select_roi']:
            if gm.is_valid(uim.sel_gid):
                uim.hs.dm.show_image(gm.gx(uim.sel_gid))
        elif uim.state in ['chip_view']:
            if cm.is_valid(uim.sel_cid):
                uim.hs.dm.show_chip(cm.cx(uim.sel_cid))
                uim.changeTabSignal.emit(uim.tab_order.index('chip'))
        elif uim.state in ['image_view']:
            if gm.is_valid(uim.sel_gid):
                uim.hs.dm.show_image(gm.gx(uim.sel_gid))
                uim.changeTabSignal.emit(uim.tab_order.index('image'))
        elif uim.state in ['result_view']:
            uim.hs.dm.show_query(uim.sel_res)
            uim.changeTabSignal.emit(uim.tab_order.index('result'))
        else:
            logerr('I dont know how to draw in state: '+str(uim.state))

    @func_log
    def select_images_on_disk(uim, start_path=None):
        dlg = QFileDialog()
        logmsg('Select one or more images to add.')
        image_list = dlg.getOpenFileNames(caption='Select one or more images to add.',\
                                          directory=uim.hs.db_dpath)
        image_list = [str(fpath) for fpath in image_list]
        return image_list

    @func_log
    def select_database(uim):
        dlg = QFileDialog()
        opt = QFileDialog.ShowDirsOnly
        if uim.hs.db_dpath is None:
            db_dpath = str(dlg.getExistingDirectory(caption='Open/New Database', options=opt))
        else:
            db_dpath = str(dlg.getExistingDirectory(\
                       caption='Open/New Database', options=opt, directory=uim.hs.db_dpath))
        return db_dpath

    def populate_tables(uim):
        uim.populate_chip_table()
        uim.populate_image_table()
        uim.populate_result_table()

    @func_log
    def update_state(uim, new_state):
        old_state   = uim.state
        logdbg('Updating to State: '+str(new_state)+', from: '+str(old_state))
        if old_state == 'select_roi':
            if new_state != 'select_roi_done':
                logerr('Cannot enter new state while selecting an ROI.')
        elif old_state == 'querying':
            if new_state != 'done_querying':
                logerr('Cannot enter new state while querying')
        uim.state = new_state
        uim.updateStateSignal.emit(new_state)
        return old_state

    @func_log
    def update_selection(uim):
        gid = uim.sel_gid if uim.sel_gid != None else -1
        cid = uim.sel_cid if uim.sel_cid != None else -1
        uim.selectionSignal.emit(cid, gid)

    def sel_cx(uim):
        return uim.hs.cm.cx(uim.sel_cid)
    def sel_gx(uim):
        return uim.hs.gm.gx(uim.sel_gid)

    @func_log
    def redraw_gui(uim):
        uim.redrawGuiSignal.emit()

    @func_log
    def unselect_all(uim):
        uim.sel_res = None
        uim.sel_cid = None
        uim.sel_gid = None
        uim.update_selection()

    @func_log
    def select_cid(uim, cid):
        cm = uim.hs.cm
        uim.sel_cid = cid
        uim.sel_gid = cm.gid(cid)
        uim.update_selection()

    @func_log
    def select_gid(uim, gid, cid_x=0):
        'selects an image and the cid_x^th chip in it'
        gm = uim.hs.gm
        cid_list = gm.cid_list(gid)
        uim.sel_cid = cid_list[cid_x] if cid_x < len(cid_list) else None
        uim.sel_gid = gid
        uim.update_selection()

    @func_log
    def select_roi(uim):
        if not uim.hs.gm.is_valid(uim.sel_gid):
            logerr('Select an Image before you draw an ROI')
        prev_state = uim.update_state('select_roi')
        uim.draw()
        new_roi = uim.hs.dm.select_roi()
        uim.update_state('select_roi_done')
        uim.update_state(prev_state)
        return new_roi

    # --- UIManager things that deal with the GUI Through Signals
    @func_log
    def populate_chip_table(uim):
        cm = uim.hs.cm
        col_headers  = ['Chip ID', 'Chip Name', 'Name ID', 'Image ID', 'Other CIDS']
        col_editable = [ False  ,   True      ,  False   ,   False   ,   False     ]
        cx_list  = cm.get_valid_cxs()
        data_list = [None]*len(cx_list)
        row_list = range(len(cx_list))
        for (i,cx) in enumerate(cx_list):
            cid  =  cm.cx2_cid[cx]
            gid  =  cm.cx2_gid(cx)
            nid  =  cm.cx2_nid(cx)
            name = cm.cx2_name(cx)
            other_cxs_ = setdiff1d(cm.cx2_other_cxs([cx])[0], cx)
            other_cids = cm.cx2_cid[other_cxs_]
            data_list[i] = (cid, name, nid, gid, other_cids)
        uim.populateChipTblSignal.emit(col_headers, col_editable, row_list, data_list)

    @func_log
    def populate_image_table(uim):
        cm, gm = uim.hs.get_managers('cm','gm')
        col_headers  = ['Image ID', 'Image Name', 'Chip IDs', 'Chip Names']
        col_editable = [ False  ,   False      ,  False     ,    False]
        gx_list  = gm.get_valid_gxs()
        data_list = [None]*len(gx_list)
        row_list = range(len(gx_list))
        for (i,gx) in enumerate(gx_list):
            gid   = gm.gx2_gid[gx]
            gname = gm.gx2_gname[gx]
            cid_list  = gm.gx2_cids(gx)
            name_list = str([cm.cid2_(cid, 'name') for cid in cid_list])
            data_list[i] = (gid, gname, cid_list, name_list)
        uim.populateImageTblSignal.emit(col_headers, col_editable, row_list, data_list)
        
    @func_log
    def populate_result_table(uim):
        res = uim.sel_res
        if res is None:
            logdbg('Requested populate_results, but there are no results to populate.')
            return None
        logmsg(res) 
        gm, cm, am = uim.hs.get_managers('gm','cm','am')
        col_headers  = ['Rank', 'Chip ID', 'Chip Name', 'score']
        col_editable = [False ,   False  ,     True   ,  False ]
        dynargs =\
        ('cid', 'name' )
        (qcid , qname  ) =  res.qcid2_(*dynargs)
        (tcid , tname , tscore ) = res.tcid2_(*dynargs+('score',))
        num_top = len(tcid)
        
        data_list = [None]*(num_top+1)
        row_list = range(num_top+1)
        qcx = res.rr.qcx
        data_list[0] = [0,  qcid, qname, 'Queried Chip']
        for (ix, (cid, name, score)) in enumerate(zip(tcid, tname, tscore)):
            rank   = ix+1 
            data_list[ix+1] = (rank, cid, name, score)
        uim.populateResultTblSignal.emit(col_headers, col_editable, row_list, data_list)

    def populate_algo_settings(uim):
        hs = uim.hs
        am = hs.am
        all_sett = DynStruct()
        all_sett.algorithm_settings = am.settings
        all_sett.core_settings = hs.prefs
        uim.populatePrefTreeSignal.emit(all_sett.to_dict())

