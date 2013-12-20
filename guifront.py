from __future__ import division, print_function
from PyQt4 import QtGui
from PyQt4.Qt import (QAbstractItemView, pyqtSlot, pyqtSignal, Qt)
import guitools
import tools

IS_INIT = False


def rrr():
    'Dynamic module reloading'
    import imp
    import sys
    print('[*front] reloading %s' % __name__)
    imp.reload(sys.modules[__name__])


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


def tblheader_text(tbl, col):
    header_item = str(tbl.horizontalHeaderItem(2))
    return header_item.text()


def connect_file_signals(win):
    ui = win.ui
    backend = win.backend
    ui.actionNew_Database.triggered.connect(backend.new_database)
    ui.actionOpen_Database.triggered.connect(backend.open_database)
    ui.actionSave_Database.triggered.connect(backend.save_database)
    ui.actionImport_Img_file.triggered.connect(backend.import_images_from_file)
    ui.actionImport_Img_dir.triggered.connect(backend.import_images_from_dir)
    ui.actionQuit.triggered.connect(backend.quit)


def connect_action_signals(win):
    ui = win.ui
    backend = win.backend
    ui.actionAdd_Chip.triggered.connect(backend.add_chip)
    ui.actionNew_Chip_Property.triggered.connect(backend.new_prop)
    ui.actionQuery.triggered.connect(backend.query)
    ui.actionReselect_Ori.triggered.connect(backend.reselect_ori)
    ui.actionReselect_ROI.triggered.connect(backend.reselect_roi)
    ui.actionDelete_Chip.triggered.connect(backend.delete_chip)
    ui.actionNext.triggered.connect(backend.select_next)


def connect_option_signals(win):
    ui = win.ui
    backend = win.backend
    ui.actionLayout_Figures.triggered.connect(backend.layout_figures)
    ui.actionPreferences.triggered.connect(backend.edit_preferences)
    #ui.actionTogPts.triggered.connect(backend.toggle_points)
    #ui.actionTogPlt.triggered.connect(backend.toggle_plotWidget)


def connect_help_signals(win):
    ui = win.ui
    backend = win.backend
    msg_event = lambda title, msg: lambda: guitools.msgbox(title, msg)
    #ui.actionView_Docs.triggered.connect(backend.view_docs)
    ui.actionView_DBDir.triggered.connect(backend.view_database_dir)
    ui.actionView_Computed_Dir.triggered.connect(backend.view_computed_dir)
    ui.actionView_Global_Dir.triggered.connect(backend.view_global_dir)

    ui.actionAbout.triggered.connect(msg_event('About', 'hotspotter'))
    ui.actionDelete_computed_directory.triggered.connect(backend.delete_computed_dir)
    ui.actionDelete_global_preferences.triggered.connect(backend.delete_global_prefs)
    ui.actionDev_Mode_IPython.triggered.connect(backend.dev_help)
    #ui.actionWriteLogs.triggered.connect(backend.write_logs)


def connect_batch_signals(win):
    ui = win.ui
    backend = win.backend
    #ui.actionBatch_Change_Name.triggered.connect(backend.batch_rename)
    ui.actionPrecomputeChipsFeatures.triggered.connect(backend.precompute_feats)
    ui.actionPrecompute_Queries.triggered.connect(backend.precompute_queries)
    #ui.actionScale_all_ROIS.triggered.connect(backend.expand_rois)
    #ui.actionConvert_all_images_into_chips.triggered.connect(backend.convert_images2chips)
    #ui.actionAddMetaProp.triggered.connect(backend.add_chip_property)
    #ui.actionAutoassign.triggered.connect(backend.autoassign)


def connect_experimental_signals(win):
    ui = win.ui
    backend = win.backend
    ui.actionMatching_Experiment.triggered.connect(backend.actionRankErrorExpt)
    ui.actionName_Consistency_Experiment.triggered.connect(backend.autoassign)


class MainWindowFrontend(QtGui.QMainWindow):
    printSignal     = pyqtSignal(str)
    quitSignal      = pyqtSignal()
    selectGxSignal  = pyqtSignal(int)
    selectCidSignal = pyqtSignal(int)
    selectResSignal = pyqtSignal(int)
    changeCidSignal = pyqtSignal(int, str, str)
    querySignal = pyqtSignal()

    def __init__(self, backend, use_plot_widget=True):
        super(MainWindowFrontend, self).__init__()
        print('[*front] creating frontend')
        self.prev_tbl_item = None
        self.backend = backend
        self.ui = init_ui(self)
        if use_plot_widget:
            self.plotWidget = init_plotWidget(self)
        # Progress bar is not hooked up yet
        self.ui.progressBar.setVisible(False)
        self.connect_signals()

    @pyqtSlot(name='closeEvent')
    def closeEvent(self, event):
        self.printSignal.emit('[*front] closeEvent')
        event.accept()
        self.quitSignal.emit()

    def connect_signals(self):
        # Connect signals to slots
        backend = self.backend
        ui = self.ui
        # Frontend Signals
        self.printSignal.connect(backend.backend_print)
        self.quitSignal.connect(backend.quit)
        self.selectGxSignal.connect(backend.select_gx)
        self.selectCidSignal.connect(backend.select_cid)
        self.changeCidSignal.connect(backend.change_chip_property)
        self.selectResSignal.connect(backend.select_res_cid)
        self.querySignal.connect(backend.query)

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
        print('[*front*] ' + msg)
        #self.printSignal.emit('[*front] ' + msg)

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

        # The following options are always enabled
        ui.actionOpen_Database.setEnabled(True)
        ui.actionNew_Database.setEnabled(True)
        ui.actionQuit.setEnabled(True)
        ui.actionAbout.setEnabled(True)
        ui.actionView_Docs.setEnabled(True)
        ui.actionDelete_global_preferences.setEnabled(True)

        # The following options are no implemented. Disable them
        ui.actionConvert_all_images_into_chips.setEnabled(False)
        ui.actionBatch_Change_Name.setEnabled(False)
        ui.actionScale_all_ROIS.setEnabled(False)
        ui.actionWriteLogs.setEnabled(False)
        ui.actionAbout.setEnabled(False)
        ui.actionView_Docs.setEnabled(False)

    def _populate_table(self, tbl, col_headers, col_editable, row_list, row2_datatup):
        self.print('_populate_table()')
        #tbl = ui.chip_TBL
        hheader = tbl.horizontalHeader()

        def set_header_context_menu(hheader):
            hheader.setContextMenuPolicy(Qt.CustomContextMenu)
            opt2_callback = [
                ('header', lambda: print('finishme')),
                ('cancel', lambda: print('cancel')), ]
            # HENDRIK / JASON TODO:
            # I have a small right-click context menu working
            # Can one of you put some useful functions in these?
            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            hheader.customContextMenuRequested.connect(popup_slot)

        def set_table_context_menu(tbl):
            tbl.setContextMenuPolicy(Qt.CustomContextMenu)
            opt2_callback = [
                ('Query', self.querySignal.emit), ]
                #('item',  lambda: print('finishme')),
                #('cancel', lambda: print('cancel')), ]

            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            tbl.customContextMenuRequested.connect(popup_slot)
        #set_header_context_menu(hheader)
        set_table_context_menu(tbl)

        sort_col = hheader.sortIndicatorSection()
        sort_ord = hheader.sortIndicatorOrder()
        tbl.sortByColumn(0, Qt.AscendingOrder)  # Basic Sorting
        prevBlockSignals = tbl.blockSignals(True)
        tbl.clear()
        tbl.setColumnCount(len(col_headers))
        tbl.setRowCount(len(row_list))
        tbl.verticalHeader().hide()
        tbl.setHorizontalHeaderLabels(col_headers)
        tbl.setSelectionMode( QAbstractItemView.SingleSelection)
        tbl.setSelectionBehavior( QAbstractItemView.SelectRows)
        tbl.setSortingEnabled(False)
        for row in iter(row_list):
            data_tup = row2_datatup[row]
            for col, data in enumerate(data_tup):
                item = QtGui.QTableWidgetItem()

                if tools.is_int(data):
                    item.setData(Qt.DisplayRole, int(data))
                elif tools.is_float(data):
                    item.setData(Qt.DisplayRole, float(data))
                else:
                    item.setText(str(data))

                item.setTextAlignment(Qt.AlignHCenter)
                if col_editable[col]:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                tbl.setItem(row, col, item)
        tbl.setSortingEnabled(True)
        tbl.sortByColumn(sort_col, sort_ord)  # Move back to old sorting
        tbl.show()
        tbl.blockSignals(prevBlockSignals)

    @pyqtSlot(str, list, list, list, list)
    def populate_tbl(self, table_name, col_headers, col_editable,
                     row_list, row2_datatup):
        table_name = str(table_name)
        self.print('populate_tbl(%s)' % table_name)
        try:
            tbl = self.ui.__dict__['%s_TBL' % table_name]
        except KeyError:
            valid_table_names = [key for key in self.ui.__dict__.keys()
                                 if key.find('_TBL') >= 0]
            msg = '\n'.join(['Invalid table_name = %s_TBL' % table_name,
                             'valid names:\n  ' + '\n  '.join(valid_table_names)])
            raise Exception(msg)
        self._populate_table(tbl, col_headers, col_editable, row_list, row2_datatup)

    def get_tbl_header(self, tbl, col):
        return str(tbl.horizontalHeaderItem(col).text())

    def get_tbl_cid(self, tbl, row, cid_col):
        cid_header = self.get_tbl_header(tbl, cid_col)
        assert cid_header == 'Chip ID', 'Header is %s' % cid_header
        cid = int(tbl.item(row, cid_col).text())
        return cid

    def get_tbl_gid(self, tbl, row, gid_col):
        gid_header = self.get_tbl_header(tbl, gid_col)
        assert gid_header == 'Image Index', 'Header is %s' % gid_header
        gid = int(tbl.item(row, gid_col).text())
        return gid

    # Table Changed Functions
    @pyqtSlot(QtGui.QTableWidgetItem)
    def img_tbl_changed(self, item):
        self.print('img_tbl_changed()')
        raise NotImplementedError('img_tbl_changed()')

    @pyqtSlot(QtGui.QTableWidgetItem)
    def chip_tbl_changed(self, item):
        'Chip Table Chip Changed'
        self.print('chip_tbl_changed()')
        tbl = self.ui.chip_TBL
        row, col = (item.row(), item.column())
        # Get selected chipid
        sel_cid = self.get_tbl_cid(tbl, row, 0)
        # Get the changed property key and value
        new_val = str(item.text()).replace(',', ';;')  # sanatize for csv
        # Get which column is being changed
        header_lbl = self.get_tbl_header(tbl, col)
        # Tell the backend about the change
        self.changeCidSignal.emit(sel_cid, header_lbl, new_val)

    @pyqtSlot(QtGui.QTableWidgetItem)
    def res_tbl_changed(self, item):
        'Result Table Chip Changed'
        self.print('res_tbl_changed()')
        tbl = self.ui.res_TBL
        row, col = (item.row(), item.column())
        sel_cid  = self.get_tbl_cid(tbl, row, 2)  # The changed row's chip id
        # Get which column is being changed
        header_lbl = self.get_tbl_header(tbl, col)
        # The changed items's value
        new_val = str(item.text()).replace(',', ';;')  # sanatize for csv
        # Tell the backend about the change
        self.changeCidSignal.emit(sel_cid, header_lbl, new_val)

    # Table Clicked Functions
    @pyqtSlot(QtGui.QTableWidgetItem)
    def res_tbl_clicked(self, item):
        'Result Table Clicked'
        tbl = self.ui.res_TBL
        row, col = (item.row(), item.column())
        self.print('res_tbl_clicked(%r, %r)' % (row, col))
        if self.get_tbl_header(tbl, col) == 'Matching Name':
            self.print('[front] does not select when clicking name column')
            return
        if item == self.prev_tbl_item:
            return
        self.prev_tbl_item = item
        # Get the clicked Chip ID (from res tbl)
        sel_cid = self.get_tbl_cid(tbl, row, 2)
        self.selectResSignal.emit(sel_cid)

    @pyqtSlot(QtGui.QTableWidgetItem)
    def img_tbl_clicked(self, item):
        'Image Table Clicked'
        tbl = self.ui.image_TBL
        row = item.row()
        self.print('img_tbl_clicked(%r)' % (row))
        if item == self.prev_tbl_item:
            return
        self.prev_tbl_item = item
        # Get the clicked Image ID
        sel_gid = self.get_tbl_gid(tbl, row, 0)
        self.selectGxSignal.emit(sel_gid)

    @pyqtSlot(QtGui.QTableWidgetItem)
    def chip_tbl_clicked(self, item):
        'Chip Table Clicked'
        tbl = self.ui.chip_TBL
        row, col = (item.row(), item.column())
        self.print('chip_tbl_clicked(%r, %r)' % (row, col))
        if self.get_tbl_header(tbl, col) == 'Name':
            self.print('[front] does not select when clicking name column')
            return
        if item == self.prev_tbl_item:
            return
        self.prev_tbl_item = item
        # Get the clicked Chip ID (from chip tbl)
        sel_cid = self.get_tbl_cid(tbl, row, 0)
        self.selectCidSignal.emit(sel_cid)

    @pyqtSlot(int, name='change_view')
    def change_view(self, new_state):
        self.print('change_view()')
        prevBlock = self.ui.tablesTabWidget.blockSignals(True)
        self.ui.tablesTabWidget.blockSignals(prevBlock)

    @pyqtSlot(str, str, list)
    def modal_useroption(self, msg, title, options):
        pass
