from hotspotter.other.ConcretePrintable import Pref
from PyQt4.Qt import QMainWindow, QTableWidgetItem, QMessageBox, \
        QAbstractItemView,  QWidget, Qt, pyqtSlot, pyqtSignal, \
        QStandardItem, QStandardItemModel, QString, QObject
from hotspotter.front.EditPrefSkel import Ui_editPrefSkel
from hotspotter.front.MainSkel import Ui_mainSkel
from hotspotter.tpl.other.matplotlibwidget import MatplotlibWidget
from hotspotter.other.logger import logmsg, logdbg, func_log
from hotspotter.other.messages import workflow_help, cmd_help, troubles_help
import types
#from weakref import ref

# --- QtMainWindow Thread --- # 
# Talk to this only with signals and slots
try:
    _fromUtf8 = QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

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

#-------------------------------------------
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

#-------------------------------------------
# TODO: Remove the .ui files all together and code the gui 
# entirely in this file. 
class HotspotterMainWindow(QMainWindow):
    'The GUI guts of the skeletons in the hsgui directory'
    # Signals that call Facade Slots
    selectCidSignal      = pyqtSignal(int)
    selectGidSignal      = pyqtSignal(int)
    renameChipIdSignal   = pyqtSignal(str, int)
    changeChipPropSignal = pyqtSignal(str, str, int)
    logdbgSignal         = pyqtSignal(str)

    def __init__(hsgui, fac):
        super( HotspotterMainWindow, hsgui ).__init__()
        # Setup main window
        hsgui.main_skel = Ui_mainSkel()
        hsgui.main_skel.setupUi(hsgui)
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

    def msgbox(hsgui, title, msg):
        # Make a non modal critical QMessageBox
        msgBox = QMessageBox( hsgui );
        msgBox.setAttribute( Qt.WA_DeleteOnClose )
        msgBox.setStandardButtons( QMessageBox.Ok )
        msgBox.setWindowTitle( title )
        msgBox.setText( msg )
        msgBox.setModal( False )
        msgBox.open( msgBox.close )
        msgBox.show()
        hsgui.non_modal_qt_handles.append(msgBox)
        # Old Modal Version: QMessageBox.critical(None, 'ERROR', msg)

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


