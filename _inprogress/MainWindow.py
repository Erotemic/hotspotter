from PyQt4 import QtCore, QtGui
from PyQt4.Qt import QObject, pyqtSignal, QFileDialog
from MainSkel import Ui_mainSkel
import multiprocessing
from PyQt4.Qt import QMainWindow, QTableWidgetItem, QMessageBox, \
        QAbstractItemView,  QWidget, Qt, pyqtSlot, pyqtSignal, \
        QStandardItem, QStandardItemModel, QString, QObject
from _tpl.other.matplotlibwidget import MatplotlibWidget

# http://stackoverflow.com/questions/2312210/window-icon-of-exe-in-pyqt4


#-------------------------------------------
def gui_log(fn):
    'log what happens in the GUI for debugging'
    def gui_log_wrapper(hsgui, *args, **kwargs):
        try:
            function_name = fn.func_name
            into_str = 'In  hsgui.'+function_name
            outo_str = 'Out hsgui.'+function_name+'\n'
            hsgui.logdbgSignal.emit(into_str)
            ret = fn(hsgui, *args, **kwargs)
            hsgui.logdbgSignal.emit(outo_str)
            return ret
        except Exception as ex:
            import traceback
            logmsg('\n\n *!!* HotSpotter GUI Raised Exception: '+str(ex))
            logmsg('\n\n *!!* HotSpotter GUI Exception Traceback: \n\n'+traceback.format_exc())
    return gui_log_wrapper

class EditPrefWidget(QWidget):
    'The Settings Pane; Subclass of Main Windows.'
    def __init__(epw, fac):
        super( EditPrefWidget, epw ).__init__()
        epw.pref_skel = Ui_editPrefSkel()
        epw.pref_skel.setupUi(epw)
        epw.pref_model = None
        epw.pref_skel.redrawBUT.clicked.connect(fac.redraw)
        epw.pref_skel.defaultPrefsBUT.clicked.connect(fac.default_prefs)
        epw.pref_skel.unloadFeaturesAndModelsBUT.clicked.connect(fac.unload_features_and_models)

    @pyqtSlot(Pref, name='populatePrefTreeSlot')
    def populatePrefTreeSlot(epw, pref_struct):
        'Populates the Preference Tree Model'
        logdbg('Bulding Preference Model of: '+repr(pref_struct))
        epw.pref_model = pref_struct.createQPreferenceModel()
        logdbg('Built: '+repr(epw.pref_model))
        epw.pref_skel.prefTreeView.setModel(epw.pref_model)
        epw.pref_skel.prefTreeView.header().resizeSection(0,250)

class MainWindow(QtGui.QMainWindow):
    populateChipTblSignal   = pyqtSignal(list, list, list, list)
    def __init__(self, hs=None):
        super(HotSpotterMainWindow, self).__init__()
        self.hs = None
        self.ui=Ui_mainSkel()
        self.ui.setupUi(self)
        self.show()
        if hs is None:
            self.connect_api(hs)
    def connect_api(self, hs):
        print('[win] connecting api')
        self.hs = hs
        hsgui.epw = EditPrefWidget(fac)
        hsgui.plotWidget = MatplotlibWidget(hsgui.main_skel.centralwidget)
        hsgui.plotWidget.setObjectName(_fromUtf8('plotWidget'))
        hsgui.main_skel.root_hlayout.addWidget(hsgui.plotWidget)
        hsgui.prev_tbl_item = None
        hsgui.prev_cid = None
        hsgui.prev_gid = None 
        hsgui.non_modal_qt_handles = []

    def connectSignals(hsgui, fac):
        'Connects GUI signals to Facade Actions'
        logdbg('Connecting GUI >> to >> Facade')
        # Base Signals
        hsgui.selectCidSignal.connect(fac.selc)
        hsgui.selectGidSignal.connect(fac.selg)
        hsgui.renameChipIdSignal.connect(fac.rename_cid)
        hsgui.changeChipPropSignal.connect(fac.change_chip_prop)
        hsgui.logdbgSignal.connect(fac.logdbgSlot)
        # SKEL SIGNALS
        main_skel = hsgui.main_skel
        # Widget
        hsgui.main_skel.fignumSPIN.valueChanged.connect(
            fac.set_fignum)
        # File
        main_skel.actionOpen_Database.triggered.connect(
            fac.open_db)
        main_skel.actionSave_Database.triggered.connect(
            fac.save_db)
        main_skel.actionImport_Images.triggered.connect(
            fac.import_images)
        main_skel.actionQuit.triggered.connect(
            hsgui.close)
        # Actions
        main_skel.actionQuery.triggered.connect(
            fac.query)
        main_skel.actionAdd_ROI.triggered.connect(
            fac.add_chip)
        main_skel.actionReselect_Orientation.triggered.connect(
            fac.reselect_orientation)
        main_skel.actionReselect_ROI.triggered.connect(
            fac.reselect_roi)
        main_skel.actionRemove_Chip.triggered.connect(
            fac.remove_cid)
        main_skel.actionNext.triggered.connect(
            fac.select_next)
        # Options
        main_skel.actionTogEll.triggered.connect(
            fac.toggle_ellipse)
        main_skel.actionTogPts.triggered.connect(
            fac.toggle_points)
        main_skel.actionTogPlt.triggered.connect(
            hsgui.setPlotWidgetVisibleSlot)
        main_skel.actionPreferences.triggered.connect(
            hsgui.epw.show )
        # Help
        main_skel.actionView_Documentation.triggered.connect(
            fac.view_documentation)
        main_skel.actionHelpCMD.triggered.connect(
            lambda:hsgui.msgbox('Command Line Help', cmd_help))
        main_skel.actionHelpWorkflow.triggered.connect(
            lambda:hsgui.msgbox('Workflow HOWTO', workflow_help))
        main_skel.actionHelpTroubles.triggered.connect(
            lambda:hsgui.msgbox('Troubleshooting Help', troubles_help))
        main_skel.actionWriteLogs.triggered.connect(
            fac.write_logs)
        # Convinience
        main_skel.actionOpen_Source_Directory.triggered.connect(
            fac.vd)
        main_skel.actionOpen_Data_Directory.triggered.connect(
            fac.vdd)
        main_skel.actionOpen_Internal_Directory.triggered.connect(
            fac.vdi)
        main_skel.actionConvertImage2Chip.triggered.connect(
            fac.convert_all_images_to_chips)
        main_skel.actionBatch_Change_Name.triggered.connect(
            fac._quick_and_dirty_batch_rename)
        main_skel.actionAdd_Metadata_Property.triggered.connect(
            fac.add_new_prop)
        main_skel.actionAssign_Matches_Above_Threshold.triggered.connect(
            fac.match_all_above_thresh)
        main_skel.actionIncrease_ROI_Size.triggered.connect(
            fac.expand_rois)
        # Experiments
        main_skel.actionMatching_Experiment.triggered.connect(
            fac.run_matching_experiment)
        main_skel.actionName_Consistency_Experiment.triggered.connect(
            fac.run_name_consistency_experiment)
        # 
        # Gui Components
        # Tables Widgets
        main_skel.chip_TBL.itemClicked.connect(
            hsgui.chipTableClickedSlot)
        main_skel.chip_TBL.itemChanged.connect(
            hsgui.chipTableChangedSlot)
        main_skel.image_TBL.itemClicked.connect(
            hsgui.imageTableClickedSlot)
        main_skel.res_TBL.itemChanged.connect(
            hsgui.resultTableChangedSlot)
        # Tab Widget
        # This signal slot setup is very bad. Needs rewrite
        main_skel.tablesTabWidget.currentChanged.connect(
            fac.change_view)
        main_skel.chip_TBL.sortByColumn(0, Qt.AscendingOrder)
        main_skel.res_TBL.sortByColumn(0, Qt.AscendingOrder)
        main_skel.image_TBL.sortByColumn(0, Qt.AscendingOrder)

    @pyqtSlot(name='setPlotWidgetVisible')
    def setPlotWidgetVisibleSlot(hsgui, bit=None): #None = toggle
        if hsgui.plotWidget != None:
            logdbg('Disabling Plot Widget')
            if bit is None: bit = not hsgui.plotWidget.isVisible()
            was_visible = hsgui.plotWidget.setVisible(bit)
            if was_visible != bit: 
                if bit:
                    hsgui.main_skel.fignumSPIN.setValue(0)
                else:
                    hsgui.main_skel.fignumSPIN.setValue(1)
                #hsgui.setFignumSignal.emit(int(1 - bit)) # plotwidget fignum = 0

    # Internal GUI Functions
    def populate_tbl_helper(hsgui, tbl, col_headers, col_editable, row_list, row2_data_tup ):
        #tbl = main_skel.chip_TBL
        hheader = tbl.horizontalHeader()
        sort_col = hheader.sortIndicatorSection()
        sort_ord = hheader.sortIndicatorOrder()
        tbl.sortByColumn(0, Qt.AscendingOrder) # Basic Sorting
        prevBlockSignals = tbl.blockSignals(True)
        tbl.clear()
        tbl.setColumnCount(len(col_headers))
        tbl.setRowCount(len(row_list))
        tbl.verticalHeader().hide()
        tbl.setHorizontalHeaderLabels(col_headers)
        tbl.setSelectionMode( QAbstractItemView.SingleSelection )
        tbl.setSelectionBehavior( QAbstractItemView.SelectRows)
        tbl.setSortingEnabled(False)
        for row in iter(row_list):
            data_tup = row2_data_tup[row]
            for col, data in enumerate(data_tup):
                item = QTableWidgetItem()
                try:
                    int_data = int(data)
                    item.setData(Qt.DisplayRole, int_data)
                except ValueError: # for strings
                    item.setText(str(data))
                except TypeError: #for lists
                    item.setText(str(data))
                item.setTextAlignment(Qt.AlignHCenter)
                if col_editable[col]: item.setFlags(item.flags() | Qt.ItemIsEditable)
                else: item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                tbl.setItem(row, col, item)
        tbl.setSortingEnabled(True)
        tbl.sortByColumn(sort_col,sort_ord) # Move back to old sorting
        tbl.show()
        tbl.blockSignals(prevBlockSignals)


    @pyqtSlot(dict, name='updateDBStatsSlot')
    @gui_log
    def updateDBStatsSlot(hsgui, stats):
        hsgui.setWindowTitle(stats['title'])
        

    def updateSelSpinsSlot(hsgui, cid, gid):
        hsgui.prev_cid = cid
        hsgui.prev_gid = gid
        hsgui.main_skel.sel_cid_SPIN.setValue(cid)
        hsgui.main_skel.sel_gid_SPIN.setValue(gid)

    def redrawGuiSlot(hsgui):
        hsgui.show()
        if hsgui.plotWidget != None and\
           hsgui.plotWidget.isVisible(): 
            hsgui.plotWidget.show()
            hsgui.plotWidget.draw()

    def updateStateLabelSlot(hsgui, state):
        hsgui.main_skel.state_LBL.setText(state)

    @pyqtSlot(list, list, list, list, name='populateChipTblSlot')
    def populateChipTblSlot(hsgui, col_headers, col_editable, row_list, row2_data_tup):
        hsgui.populate_tbl_helper(hsgui.main_skel.chip_TBL, col_headers, col_editable, row_list, row2_data_tup)
    @pyqtSlot(list, list, list, list, name='populateImageTblSlot')
    def populateImageTblSlot(hsgui, col_headers, col_editable, row_list, row2_data_tup):
        hsgui.populate_tbl_helper(hsgui.main_skel.image_TBL, col_headers, col_editable, row_list, row2_data_tup)
    @pyqtSlot(list, list, list, list, name='populateResultTblSlot')
    def populateResultTblSlot(hsgui, col_headers, col_editable, row_list, row2_data_tup):
        hsgui.populate_tbl_helper(hsgui.main_skel.res_TBL, col_headers, col_editable, row_list, row2_data_tup)

    @gui_log
    def chipTableChangedSlot(hsgui, item):
        'A Chip had a data member changed '
        hsgui.logdbgSignal.emit('chip table changed')
        sel_row = item.row()
        sel_col = item.column()
        sel_cid = int(hsgui.main_skel.chip_TBL.item(sel_row,0).text())
        new_val = str(item.text()).replace(',',';;')
        header_lbl = str(hsgui.main_skel.chip_TBL.horizontalHeaderItem(sel_col).text())
        hsgui.selectCidSignal.emit(sel_cid)
        # Rename the chip!
        if header_lbl == 'Chip Name':
            hsgui.renameChipIdSignal.emit(new_val, sel_cid)
        # Change the user property instead
        else:
            hsgui.changeChipPropSignal.emit(header_lbl, new_val, sel_cid)

    @gui_log
    def resultTableChangedSlot(hsgui, item):
        'A Chip was Renamed in Result View'
        hsgui.logdbgSignal.emit('result table changed')
        sel_row  = item.row()
        sel_cid  = int(hsgui.main_skel.res_TBL.item(sel_row,1).text())
        new_name = str(item.text())
        hsgui.renameChipIdSignal.emit(new_name, int(sel_cid))

    def imageTableClickedSlot(hsgui, item):
        'Select Image ID'
        if item == hsgui.prev_tbl_item: return
        hsgui.prev_tbl_item = item
        sel_row = item.row()
        sel_gid = int(hsgui.main_skel.image_TBL.item(sel_row,0).text())
        hsgui.selectGidSignal.emit(sel_gid)

    def chipTableClickedSlot(hsgui, item):
        'Select Chip ID'
        hsgui.logdbgSignal.emit('chip table clicked')
        if item == hsgui.prev_tbl_item: return
        hsgui.prev_tbl_item = item
        sel_row = item.row()
        sel_cid = int(hsgui.main_skel.chip_TBL.item(sel_row,0).text())
        hsgui.selectCidSignal.emit(sel_cid)

    def update_image_table(self):
        uim.populateImageTblSignal.connect( uim.hsgui.populateImageTblSlot )
        pass
    def select_tab(uim, tabname, block_draw=False):
        logdbg('Selecting the '+tabname+' Tab')
        if block_draw:
            prevBlock = uim.hsgui.main_skel.tablesTabWidget.blockSignals(True)
        tab_index = uim.tab_order.index(tabname)
        uim.selectTabSignal.emit(tab_index)
        if block_draw:
            uim.hsgui.main_skel.tablesTabWidget.blockSignals(prevBlock)

    def get_gui_figure(uim):
        'returns the matplotlib.pyplot.figure'
        if uim.hsgui != None and uim.hsgui.plotWidget != None:
            fig = uim.hsgui.plotWidget.figure
            fig.show = lambda: uim.hsgui.plotWidget.show() #HACKY HACK HACK
            return fig
        return None
    @func_log
    def redraw_gui(uim):
        if not uim.hsgui is None and uim.hsgui.isVisible():
            uim.redrawGuiSignal.emit()
    # --- UIManager things that deal with the GUI Through Signals
    @func_log
    def populate_chip_table(uim):
        #tbl = uim.hsgui.main_skel.chip_TBL
        cm = uim.hs.cm
        col_headers  = ['Chip ID', 'Chip Name', 'Name ID', 'Image ID', 'Other CIDS']
        col_editable = [ False  ,   True      ,  False   ,   False   ,   False     ]
        # Add User Properties to headers
        col_headers += cm.user_props.keys()
        col_editable += [True for key in cm.user_props.keys()]
        # Create Data List
        cx_list  = cm.get_valid_cxs()
        data_list = [None]*len(cx_list)
        row_list = range(len(cx_list))
        for (i,cx) in enumerate(cx_list):
            # Get Indexing Data
            cid  =  cm.cx2_cid[cx]
            gid  =  cm.cx2_gid(cx)
            nid  =  cm.cx2_nid(cx)
            # Get Useful Data
            name = cm.cx2_name(cx)
            other_cxs_ = setdiff1d(cm.cx2_other_cxs([cx])[0], cx)
            other_cids = cm.cx2_cid[other_cxs_]
            # Get User Data
            cm.user_props.keys()
            user_data = [cm.user_props[key][cx] for key in
                         cm.user_props.iterkeys()]
            # Pack data to sent to Qt
            data_list[i] = (cid, name, nid, gid, other_cids)+tuple(user_data)
            #(cid, name, nid, gid, other_cids, *user_data)
        uim.populateChipTblSignal.emit(col_headers, col_editable, row_list, data_list)

    @func_log
    def populate_image_table(uim):
        col_headers  = ['Image ID', 'Image Name', 'Chip IDs', 'Chip Names']
        col_editable = [ False    ,  False      ,  False    ,  False      ]
        # Populate table with valid image indexes
        cm, gm = uim.hs.get_managers('cm','gm')
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
        col_headers  = ['Rank', 'Chip ID', 'Chip Name', 'score']
        col_editable = [False ,   False  ,     True   ,  False ]
        # Check to see if results exist
        res = uim.sel_res
        if res is None:
            logdbg('Not populating. selected results are None.')
            return None
        logmsg(res) 
        gm, cm, am = uim.hs.get_managers('gm','cm','am')
        dynargs =\
        ('cid', 'name' )
        (qcid , qname  ) =  res.qcid2_(*dynargs)
        (tcid , tname , tscore ) = res.tcid2_(*dynargs+('score',))
        num_results = len(tcid)
        
        data_list = [None]*(num_results+1)
        row_list = range(num_results+1)
        data_list[0] = [0,  qcid, qname, 'Queried Chip']
        for (ix, (cid, name, score)) in enumerate(zip(tcid, tname, tscore)):
            rank   = ix+1 
            data_list[ix+1] = (rank, cid, name, score)
        uim.populateResultTblSignal.emit(col_headers, col_editable, row_list, data_list)

    def populate_algo_settings(uim):
        logdbg('Populating the Preference Tree... Sending Signal')
        uim.populatePrefTreeSignal.emit(uim.hs.all_pref)

    def set_fignum(uim, fignum):
        if uim.hsgui != None:
            prevBlockSignals = uim.hsgui.main_skel.fignumSPIN.blockSignals(True)
            uim.setfignumSignal.emit(fignum)
            uim.hsgui.main_skel.fignumSPIN.blockSignals(prevBlockSignals)

if __name__ == '__main__':
    import sys
    multiprocessing.freeze_support()
    def test():
        app = QtGui.QApplication(sys.argv)
        main_win = HotSpotterMainWindow()
        app.setActiveWindow(main_win)
        sys.exit(app.exec_())
    test()

