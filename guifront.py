from __future__ import division, print_function
import __builtin__
import os
import os.path
import sys
import PyQt4
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.Qt import (QMainWindow, QApplication, QCoreApplication,
                      QTableWidgetItem, QAbstractItemView, QWidget, Qt,
                      pyqtSlot, pyqtSignal, QStandardItem, QStandardItemModel,
                      QString, QObject, QInputDialog, QDialog, QTreeWidgetItem)
import guitools

# Dynamic module reloading
def reload_module():
    import imp, sys
    print('[*front] reloading '+__name__)
    imp.reload(sys.modules[__name__])
rrr = reload_module

IS_INIT = False

def init_plotWidget(win):
    from _tpl.other.matplotlibwidget import MatplotlibWidget
    plotWidget = MatplotlibWidget(win.ui.centralwidget)
    plotWidget.setObjectName(guitools._fromUtf8('plotWidget'))
    plotWidget.setFocus()
    win.ui.root_hlayout.addWidget(plotWidget)
    return plotWidget

def init_ui(win):
    from _frontend.MainSkel import Ui_mainSkel
    ui = Ui_mainSkel()
    ui.setupUi(win)
    return ui

def connect_file_signals(win):
    ui = win.ui; backend = win.backend
    ui.actionNew_Database.triggered.connect(backend.new_database)
    ui.actionOpen_Database.triggered.connect(backend.open_database)
    ui.actionSave_Database.triggered.connect(backend.save_database)
    ui.actionImport_Img_file.triggered.connect(backend.import_images_from_file)
    ui.actionImport_Img_dir.triggered.connect(backend.add_images_from_dir)
    ui.actionQuit.triggered.connect(backend.quit)

def connect_action_signals(win):
    ui = win.ui; backend = win.backend
    ui.actionAdd_Chip.triggered.connect(backend.add_chip)
    ui.actionNew_Chip_Property.triggered.connect(backend.new_prop)
    ui.actionQuery.triggered.connect(backend.query)
    ui.actionReselect_Ori.triggered.connect(backend.reselect_ori)
    ui.actionReselect_ROI.triggered.connect(backend.reselect_roi)
    ui.actionDelete_Chip.triggered.connect(backend.delete_chip)
    ui.actionNext.triggered.connect(backend.select_next)

def connect_option_signals(win):
    ui = win.ui; backend = win.backend
    ui.actionLayout_Figures.triggered.connect(backend.layout_figures)
    #ui.actionTogEll.triggered.connect(backend.toggle_ellipse)
    #ui.actionTogPts.triggered.connect(backend.toggle_points)
    #ui.actionTogPlt.triggered.connect(backend.toggle_plotWidget)
    #ui.actionPreferences.triggered.connect(backend.select_next)

def connect_help_signals(win):
    ui = win.ui; backend = win.backend
    msg_event = lambda title, msg: lambda: guitools.msgbox(title, msg)
    #ui.actionView_Docs.triggered.connect(backend.view_docs)
    ui.actionView_DBDir.triggered.connect(backend.view_database_dir)
    ui.actionView_Computed_Dir.triggered.connect(backend.view_computed_dir)
    ui.actionView_Global_Dir.triggered.connect(backend.view_global_dir)

    ui.actionAbout.triggered.connect(msg_event('About', 'hotspotter'))
    ui.actionDelete_computed_directory.triggered.connect(backend.delete_computed_dir)
    ui.actionDelete_global_preferences.triggered.connect(backend.delete_global_prefs)
    ui.actionDev_Mode_IPython.triggered.connect(backend.dev_mode)
    #ui.actionWriteLogs.triggered.connect(backend.write_logs)

def connect_batch_signals(win):
    ui = win.ui; backend = win.backend
    #ui.actionBatch_Change_Name.triggered.connect(backend.batch_rename)
    ui.actionPrecomputeChipsFeatures.triggered.connect(backend.precompute_feats)
    ui.actionPrecompute_Queries.triggered.connect(backend.precompute_queries)
    #ui.actionScale_all_ROIS.triggered.connect(backend.expand_rois)
    #ui.actionConvert_all_images_into_chips.triggered.connect(backend.convert_images2chips)
    #ui.actionAddMetaProp.triggered.connect(backend.add_chip_property)
    #ui.actionAutoassign.triggered.connect(backend.autoassign)

def connect_experimental_signals(win):
    ui = win.ui; backend = win.backend
    ui.actionMatching_Experiment.triggered.connect(backend.actionRankErrorExpt)
    ui.actionName_Consistency_Experiment.triggered.connect(backend.autoassign)

class MainWindowFrontend(QtGui.QMainWindow):
    printSignal     = pyqtSignal(str)
    selectGxSignal  = pyqtSignal(int)
    selectCidSignal = pyqtSignal(int)

    def __init__(self, backend, use_plot_widget=True):
        super(MainWindowFrontend, self).__init__()
        print('[*front] creating frontend')
        self.prev_tbl_item = None
        self.backend = backend
        self.ui = init_ui(self)
        if use_plot_widget:
            self.plotWidget = init_plotWidget(self)
        self.connect_signals()

    def connect_signals(self):
        # Connect signals to slots
        backend = self.backend
        ui = self.ui
        # Frontend Signals
        self.printSignal.connect(backend.backend_print)
        self.selectGxSignal.connect(backend.select_gx)
        self.selectCidSignal.connect(backend.select_cid)

        # Menubar signals
        connect_file_signals(self)
        connect_action_signals(self)
        connect_option_signals(self)
        connect_batch_signals(self)
        #connect_experimental_signals(self)
        connect_help_signals(self)
        # 
        # Gui Components
        # Tables Widgets
        ui.chip_TBL.itemClicked.connect(self.chip_tbl_clicked)
        ui.chip_TBL.itemChanged.connect(self.chip_tbl_changed)
        ui.image_TBL.itemClicked.connect(self.img_tbl_clicked)
        ui.image_TBL.itemChanged.connect(self.img_tbl_changed)
        ui.res_TBL.itemClicked.connect(self.res_tbl_clicked)
        ui.res_TBL.itemChanged.connect(self.res_tbl_changed)
        # Tab Widget
        ui.tablesTabWidget.currentChanged.connect(self.change_view)
        ui.chip_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.res_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.image_TBL.sortByColumn(0, Qt.AscendingOrder)

    def print(self, msg):
        #print('[*front*] '+msg)
        self.printSignal.emit('[*front] '+msg)

    #def popup(self, pos):
        #for i in self.ui.image_TBL.selectionModel().selection().indexes():
            #self.print(repr((i.row(), i.column())))
        #menu = QtGui.QMenu()
        #action1 = menu.addAction("action1")
        #action2 = menu.addAction("action2")
        #action3 = menu.addAction("action2")
        #action = menu.exec_(self.ui.image_TBL.mapToGlobal(pos))
        #self.print('action = %r ' % action)

    @pyqtSlot(bool, name='setPlotWidgetEnabled')
    def setPlotWidgetEnabled(self, flag):
        flag = bool(flag)
        self.print('setPlotWidgetEnabled(%r)' % flag)
        print(self.plotWidget)
        self.plotWidget.setVisible(flag)

    @pyqtSlot(bool, name='setEnabled')
    def setEnabled(self, flag):
        self.print('setEnabled(%r)' % flag)
        ui = self.ui
        # Enable or disable all actions
        for uikey in ui.__dict__.keys():
            if uikey.find('action') == 0:
                ui.__dict__[uikey].setEnabled(flag)
        ui.actionOpen_Database.setEnabled(True) # always allowed
        ui.actionNew_Database.setEnabled(True) # always allowed
        ui.actionQuit.setEnabled(True) # always allowed
        ui.actionAbout.setEnabled(True) # always allowed
        ui.actionView_Docs.setEnabled(True) # always allowed

        # These are not yet useful disable them
        actions = [item for list_ in [
            #ui.menuFile.children(),
            #ui.menuActions.children(),
            #ui.menuBatch.children(),
            #ui.menuOptions.children(),
            #ui.menuHelp.children(),
        ] for item in list_]
        for item in actions:
            item.setEnabled(False)

        #for uikey in ui.__dict__.keys():
            #if uikey.find('action') == 0:
                #ui.__dict__[uikey].setEnabled(flag)

    def _populate_table(self, tbl, col_headers, col_editable, row_list, row2_datatup):
        self.print('_populate_table()')
        #tbl = ui.chip_TBL
        hheader = tbl.horizontalHeader()
        def set_header_context_menu(hheader):
            hheader.setContextMenuPolicy(Qt.CustomContextMenu)
            opt2_callback = [
                ('header', lambda: print('finishme')),
                ('cancel', lambda: print('cancel')), ]
            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            hheader.customContextMenuRequested.connect(popup_slot)
        def set_table_context_menu(tbl):
            tbl.setContextMenuPolicy(Qt.CustomContextMenu)
            opt2_callback = [
                ('item',  lambda: print('finishme')),
                ('cancel', lambda: print('cancel')), ]
            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            tbl.customContextMenuRequested.connect(popup_slot)
        set_header_context_menu(hheader)
        set_table_context_menu(tbl)

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
            data_tup = row2_datatup[row]
            for col, data in enumerate(data_tup):
                item = QtGui.QTableWidgetItem()
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

    @pyqtSlot(str, list, list, list, list)
    def populate_tbl(self, table_name, col_headers, col_editable,
                          row_list, row2_datatup):
        table_name = str(table_name)
        self.print('populate_tbl('+table_name+')')
        try:
            tbl = self.ui.__dict__[table_name+'_TBL']
        except KeyError as ex:
            valid_table_names = [key for key in self.ui.__dict__.keys() 
                                 if key.find('_TBL') >= 0]
            msg = '\n'.join(['Invalid table_name = '+table_name+'_TBL',
                             'valid names:\n  '+'\n  '.join(valid_table_names)])
            raise Exception(msg)
        self._populate_table(tbl, col_headers, col_editable, row_list, row2_datatup)

    @pyqtSlot(QtGui.QTableWidgetItem)
    def res_tbl_changed(self, item):
        self.print('res_tbl_changed()')
        raise NotImplementedError('res_tbl_changed()')

    @pyqtSlot(QtGui.QTableWidgetItem)
    def img_tbl_changed(self, item):
        self.print('img_tbl_changed()')
        raise NotImplementedError('img_tbl_changed()')

    @pyqtSlot(QtGui.QTableWidgetItem)
    def img_tbl_changed(self, item):
        self.print('img_tbl_changed()')
        raise NotImplementedError('img_tbl_changed()')

    @pyqtSlot(QtGui.QTableWidgetItem)
    def chip_tbl_changed(self, item):
        'A Chip had a data member changed '
        self.print('chip_tbl_changed()')
        sel_row = item.row()
        sel_col = item.column()
        sel_cid = int(self.ui.chip_TBL.item(sel_row,0).text())
        new_val = str(item.text()).replace(',',';;')
        header_lbl = str(self.ui.chip_TBL.horizontalHeaderItem(sel_col).text())
        self.selectCidSignal.emit(sel_cid)
        # Rename the chip!
        if header_lbl == 'Chip Name':
            self.renameChipIdSignal.emit(new_val, sel_cid)
        # Change the user property instead
        else:
            self.changeChipPropSignal.emit(header_lbl, new_val, sel_cid)

    @pyqtSlot(QtGui.QTableWidgetItem)
    def res_tbl_clicked(self, item):
        'A Chip was Renamed in Result View'
        self.print('res_tbl_clicked()')
        sel_row  = item.row()
        sel_cid  = int(self.ei.res_TBL.item(sel_row,1).text())
        new_name = str(item.text())
        self.renameChipIdSignal.emit(new_name, int(sel_cid))

    @pyqtSlot(QtGui.QTableWidgetItem)
    def img_tbl_clicked(self, item):
        self.print('img_tbl_clicked()')
        if item == self.prev_tbl_item: return
        self.prev_tbl_item = item
        sel_row = item.row()
        sel_gid = int(self.ui.image_TBL.item(sel_row,0).text())
        self.selectGxSignal.emit(sel_gid)

    @pyqtSlot(QtGui.QTableWidgetItem)
    def chip_tbl_clicked(self, item):
        self.print('chip_tbl_clicked()')
        if item == self.prev_tbl_item: return
        self.prev_tbl_item = item
        sel_row = item.row()
        sel_cid = int(self.ui.chip_TBL.item(sel_row,0).text())
        self.selectCidSignal.emit(sel_cid)

    @pyqtSlot(int, name='change_view')
    def change_view(self, new_state):
        self.print('change_view()')
        prevBlock = self.ui.tablesTabWidget.blockSignals(True)
        self.ui.tablesTabWidget.blockSignals(prevBlock)

    @pyqtSlot(str, str, list)
    def modal_useroption(self, msg, title, options):
        pass

