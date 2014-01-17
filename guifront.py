from __future__ import division, print_function
import __common__
(print, print_, print_on, print_off,
 rrr, profile) = __common__.init(__name__, '[front]')
from PyQt4 import QtGui, QtCore
from PyQt4.Qt import (QAbstractItemView, pyqtSignal, Qt)
import guitools
import tools
from guitools import slot_
from guitools import frontblocking as blocking
import sys
from _frontend.MainSkel import Ui_mainSkel

IS_INIT = False
NOSTEAL_OVERRIDE = True


class StreamStealer(QtCore.QObject):
    write_ = QtCore.pyqtSignal(str)
    flush_ =  QtCore.pyqtSignal()

    def __init__(self, parent=None, stolen=None, share=False):
        super(StreamStealer, self).__init__(parent)
        if stolen is not None:
            self.stolen = stolen
        self.write = self.write_both if share else self.write_gui

    def write_both(self, msg):
        msg_ = str(msg)
        self.stolen.write(msg_)
        self.write_.emit(msg_)

    def write_gui(self, msg):
        self.write_.emit(str(msg))

    def flush(self):
        self.flush_.emit()


def init_plotWidget(front):
    from _tpl.other.matplotlibwidget import MatplotlibWidget
    plotWidget = MatplotlibWidget(front.ui.centralwidget)
    plotWidget.setObjectName(guitools._fromUtf8('plotWidget'))
    plotWidget.setFocus()
    front.ui.root_hlayout.addWidget(plotWidget)
    return plotWidget


def init_ui(front):
    ui = Ui_mainSkel()
    ui.setupUi(front)
    return ui


def connect_file_signals(front):
    ui = front.ui
    back = front.back
    ui.actionNew_Database.triggered.connect(back.new_database)
    ui.actionOpen_Database.triggered.connect(back.open_database)
    ui.actionSave_Database.triggered.connect(back.save_database)
    ui.actionImport_Img_file.triggered.connect(back.import_images_from_file)
    ui.actionImport_Img_dir.triggered.connect(back.import_images_from_dir)
    ui.actionQuit.triggered.connect(back.quit)


def connect_action_signals(front):
    ui = front.ui
    back = front.back
    ui.actionAdd_Chip.triggered.connect(back.add_chip)
    ui.actionNew_Chip_Property.triggered.connect(back.new_prop)
    ui.actionQuery.triggered.connect(back.query)
    ui.actionReselect_Ori.triggered.connect(back.reselect_ori)
    ui.actionReselect_ROI.triggered.connect(back.reselect_roi)
    ui.actionDelete_Chip.triggered.connect(back.delete_chip)
    ui.actionNext.triggered.connect(back.select_next)


def connect_option_signals(front):
    ui = front.ui
    back = front.back
    ui.actionLayout_Figures.triggered.connect(back.layout_figures)
    ui.actionPreferences.triggered.connect(back.edit_preferences)
    #ui.actionTogPts.triggered.connect(back.toggle_points)
    #ui.actionTogPlt.triggered.connect(back.toggle_plotWidget)


def connect_help_signals(front):
    ui = front.ui
    back = front.back
    msg_event = lambda title, msg: lambda: guitools.msgbox(title, msg)
    #ui.actionView_Docs.triggered.connect(back.view_docs)
    ui.actionView_DBDir.triggered.connect(back.view_database_dir)
    ui.actionView_Computed_Dir.triggered.connect(back.view_computed_dir)
    ui.actionView_Global_Dir.triggered.connect(back.view_global_dir)

    ui.actionAbout.triggered.connect(msg_event('About', 'hotspotter'))
    ui.actionDelete_computed_directory.triggered.connect(back.delete_cache)
    ui.actionDelete_global_preferences.triggered.connect(back.delete_global_prefs)
    ui.actionDelete_Precomputed_Results.triggered.connect(back.delete_queryresults_dir)
    ui.actionDev_Mode_IPython.triggered.connect(back.dev_mode)
    ui.actionDeveloper_Reload.triggered.connect(back.dev_reload)
    #ui.actionWriteLogs.triggered.connect(back.write_logs)


def connect_batch_signals(front):
    ui = front.ui
    back = front.back
    #ui.actionBatch_Change_Name.triggered.connect(back.batch_rename)
    ui.actionPrecomputeChipsFeatures.triggered.connect(back.precompute_feats)
    ui.actionPrecompute_Queries.triggered.connect(back.precompute_queries)
    #ui.actionScale_all_ROIS.triggered.connect(back.expand_rois)
    #ui.actionConvert_all_images_into_chips.triggered.connect(back.convert_images2chips)
    #ui.actionAddMetaProp.triggered.connect(back.add_chip_property)
    #ui.actionAutoassign.triggered.connect(back.autoassign)


def connect_experimental_signals(front):
    ui = front.ui
    back = front.back
    ui.actionMatching_Experiment.triggered.connect(back.actionRankErrorExpt)
    ui.actionName_Consistency_Experiment.triggered.connect(back.autoassign)


def csv_sanatize(str_):
    return str(str_).replace(',', ';;')


def clicked(func):
    def clicked_wrapper(front, item, *args, **kwargs):
        if front.isItemEditable(item):
            front.print('[front] does not select when clicking editable column')
            return
        if item == front.prev_tbl_item:
            return
        front.prev_tbl_item = item
        return func(front, item, *args, **kwargs)
    clicked_wrapper.func_name = func.func_name
    # Hacky decorator
    return clicked_wrapper

#def popup(front, pos):
    #for i in front.ui.image_TBL.selectionModel().selection().indexes():
        #front.print(repr((i.row(), i.column())))
    #menu = QtGui.QMenu()
    #action1 = menu.addAction("action1")
    #action2 = menu.addAction("action2")
    #action3 = menu.addAction("action2")
    #action = menu.exec_(front.ui.image_TBL.mapToGlobal(pos))
    #front.print('action = %r ' % action)


#@slot_(bool)
#def setPlotWidgetEnabled(front, flag):
    #flag = bool(flag)
    ##front.printDBG('setPlotWidgetEnabled(%r)' % flag)
    #front.plotWidget.setVisible(flag)

class MainWindowFrontend(QtGui.QMainWindow):
    printSignal     = pyqtSignal(str)
    quitSignal      = pyqtSignal()
    selectGxSignal  = pyqtSignal(int)
    selectCidSignal = pyqtSignal(int)
    selectResSignal = pyqtSignal(int)
    selectNameSignal = pyqtSignal(str)
    changeCidSignal = pyqtSignal(int, str, str)
    changeGxSignal  = pyqtSignal(int, str, bool)
    querySignal = pyqtSignal()

    def __init__(front, back, use_plot_widget=True):
        super(MainWindowFrontend, front).__init__()
        #print('[*front] creating frontend')
        front.prev_tbl_item = None
        front.ostream = None
        front.back = back
        front.ui = init_ui(front)
        if use_plot_widget:
            front.plotWidget = init_plotWidget(front)
        # Progress bar is not hooked up yet
        front.ui.progressBar.setVisible(False)
        front.connect_signals()
        front.steal_stdout()

    def steal_stdout(front):
        #front.ui.outputEdit.setPlainText(sys.stdout)
        hs = front.back.hs
        nosteal = hs.args.nosteal
        noshare = hs.args.noshare
        if NOSTEAL_OVERRIDE or (nosteal and noshare):
            return
        print('[front] stealing standard out')
        if front.ostream is None:
            front.ostream = StreamStealer(stolen=sys.stdout, share=not noshare)
            front.ostream.write_.connect(front.gui_write)
            front.ostream.flush_.connect(front.gui_flush)
            sys.stdout = front.ostream
        else:
            print('[front] stream already stolen')

    def return_stdout(front):
        #front.ui.outputEdit.setPlainText(sys.stdout)
        print('[front] returning standard out')
        if front.ostream is not None:
            sys.stdout = front.ostream.stolen
            front.ostream = None
            return True
        else:
            print('[front] stream has not been stolen')
            return False

    # TODO: this code is duplicated in back
    def user_info(front, *args, **kwargs):
        return guitools.user_info(front, *args, **kwargs)

    def user_input(front, *args, **kwargs):
        return guitools.user_input(front, *args, **kwargs)

    def user_option(front, *args, **kwargs):
        return guitools.user_option(front, *args, **kwargs)

    @slot_()
    def closeEvent(front, event):
        #front.printSignal.emit('[*front] closeEvent')
        event.accept()
        front.quitSignal.emit()

    def connect_signals(front):
        # Connect signals to slots
        back = front.back
        ui = front.ui
        # Frontend Signals
        front.printSignal.connect(back.backend_print)
        front.quitSignal.connect(back.quit)
        front.selectGxSignal.connect(back.select_gx)
        front.selectCidSignal.connect(back.select_cid)
        front.selectResSignal.connect(back.select_res_cid)
        front.selectNameSignal.connect(back.select_name)
        front.changeCidSignal.connect(back.change_chip_property)
        front.changeGxSignal.connect(back.change_image_property)
        front.querySignal.connect(back.query)

        # Menubar signals
        connect_file_signals(front)
        connect_action_signals(front)
        connect_option_signals(front)
        connect_batch_signals(front)
        #connect_experimental_signals(front)
        connect_help_signals(front)
        #
        # Gui Components
        # Tables Widgets
        ui.chip_TBL.itemClicked.connect(front.chip_tbl_clicked)
        ui.chip_TBL.itemChanged.connect(front.chip_tbl_changed)
        ui.image_TBL.itemClicked.connect(front.img_tbl_clicked)
        ui.image_TBL.itemChanged.connect(front.img_tbl_changed)
        ui.res_TBL.itemClicked.connect(front.res_tbl_clicked)
        ui.res_TBL.itemChanged.connect(front.res_tbl_changed)
        ui.name_TBL.itemClicked.connect(front.name_tbl_clicked)
        # Tab Widget
        ui.tablesTabWidget.currentChanged.connect(front.change_view)
        ui.chip_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.res_TBL.sortByColumn(0, Qt.AscendingOrder)
        ui.image_TBL.sortByColumn(0, Qt.AscendingOrder)

    def print(front, msg):
        print('[*front*] ' + msg)
        #front.printSignal.emit('[*front] ' + msg)

    @slot_(bool)
    def setEnabled(front, flag):
        #front.printDBG('setEnabled(%r)' % flag)
        ui = front.ui
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

    def _populate_table(front, tbl, col_headers, col_editable, row_list, row2_datatup):
        #front.printDBG('_populate_table()')
        hheader = tbl.horizontalHeader()

        def set_header_context_menu(hheader):
            hheader.setContextMenuPolicy(Qt.CustomContextMenu)
            # TODO: for chip table: delete metedata column
            opt2_callback = [
                ('header', lambda: print('finishme')),
                ('cancel', lambda: print('cancel')), ]
            # HENDRIK / JASON TODO:
            # I have a small right-click context menu working
            # Maybe one of you can put some useful functions in these?
            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            hheader.customContextMenuRequested.connect(popup_slot)

        def set_table_context_menu(tbl):
            tbl.setContextMenuPolicy(Qt.CustomContextMenu)
            # RCOS TODO: How do we get the clicked item on a right click?
            opt2_callback = [
                ('Query', front.querySignal.emit), ]
                #('item',  lambda: print('finishme')),
                #('cancel', lambda: print('cancel')), ]

            popup_slot = guitools.popup_menu(tbl, opt2_callback)
            tbl.customContextMenuRequested.connect(popup_slot)
        #set_header_context_menu(hheader)
        set_table_context_menu(tbl)

        sort_col = hheader.sortIndicatorSection()
        sort_ord = hheader.sortIndicatorOrder()
        tbl.sortByColumn(0, Qt.AscendingOrder)  # Basic Sorting
        tblWasBlocked = tbl.blockSignals(True)
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
                # RCOS TODO: Pass in datatype here.
                #if col_headers[col] == 'AIF':
                    #print('col=%r dat=%r, %r' % (col, data, type(data)))
                if tools.is_bool(data) or data == 'True' or data == 'False':
                    bit = bool(data)
                    #print(bit)
                    if bit:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)
                    #item.setData(Qt.DisplayRole, bool(data))
                elif tools.is_int(data):
                    item.setData(Qt.DisplayRole, int(data))
                elif tools.is_float(data):
                    item.setData(Qt.DisplayRole, float(data))
                else:
                    item.setText(str(data))

                item.setTextAlignment(Qt.AlignHCenter)
                if col_editable[col]:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    #print(item.getBackground())
                    item.setBackground(QtGui.QColor(250, 240, 240))
                else:
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                tbl.setItem(row, col, item)
        tbl.setSortingEnabled(True)
        tbl.sortByColumn(sort_col, sort_ord)  # Move back to old sorting
        tbl.show()
        tbl.blockSignals(tblWasBlocked)

    @slot_(str, list, list, list, list)
    @blocking
    def populate_tbl(front, table_name, col_headers, col_editable,
                     row_list, row2_datatup):
        table_name = str(table_name)
        #front.printDBG('populate_tbl(%s)' % table_name)
        try:
            tbl = front.ui.__dict__['%s_TBL' % table_name]
        except KeyError:
            valid_table_names = [key for key in front.ui.__dict__.keys()
                                 if key.find('_TBL') >= 0]
            msg = '\n'.join(['Invalid table_name = %s_TBL' % table_name,
                             'valid names:\n  ' + '\n  '.join(valid_table_names)])
            raise Exception(msg)
        front._populate_table(tbl, col_headers, col_editable, row_list, row2_datatup)

    def isItemEditable(self, item):
        return int(Qt.ItemIsEditable & item.flags()) == int(Qt.ItemIsEditable)

    #=======================
    # General Table Getters
    #=======================

    def get_tbl_header(front, tbl, col):
        return str(tbl.horizontalHeaderItem(col).text())

    def get_tbl_int(front, tbl, row, col):
        return int(tbl.item(row, col).text())

    def get_tbl_str(front, tbl, row, col):
        return str(tbl.item(row, col).text())

    def get_header_val(front, tbl, header, row):
        # RCOS TODO: This is hacky. These just need to be
        # in dicts to begin with.
        tblname = str(tbl.objectName()).replace('_TBL', '')
        tblname = tblname.replace('image', 'img')  # Sooooo hack
        col = front.back.__dict__[tblname + 'tbl_headers'].index(header)
        return tbl.item(row, col).text()

    #=======================
    # Specific Item Getters
    #=======================

    def get_chiptbl_header(front, col):
        return front.get_tbl_header(front.ui.chip_TBL, col)

    def get_imgtbl_header(front, col):
        return front.get_tbl_header(front.ui.image_TBL, col)

    def get_restbl_header(front, col):
        return front.get_tbl_header(front.ui.res_TBL, col)

    def get_nametbl_header(front, col):
        return front.get_tbl_header(front.ui.name_TBL, col)

    def get_restbl_cid(front, row):
        return int(front.get_header_val(front.ui.res_TBL, 'Chip ID', row))

    def get_chiptbl_cid(front, row):
        return int(front.get_header_val(front.ui.chip_TBL, 'Chip ID', row))

    def get_nametbl_name(front, row):
        return str(front.get_header_val(front.ui.name_TBL, 'Name', row))

    def get_imgtbl_gx(front, row):
        return int(front.get_header_val(front.ui.image_TBL, 'Image Index', row))

    #=======================
    # Table Changed Functions
    #=======================

    @slot_(QtGui.QTableWidgetItem)
    def img_tbl_changed(front, item):
        front.print('img_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_gx = front.get_imgtbl_gx(row)
        header_lbl = front.get_imgtbl_header(col)
        new_val = item.checkState() == Qt.Checked
        front.changeGxSignal.emit(sel_gx, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def chip_tbl_changed(front, item):
        front.print('chip_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_cid = front.get_chiptbl_cid(row)  # Get selected chipid
        new_val = csv_sanatize(item.text())   # sanatize for csv
        header_lbl = front.get_chiptbl_header(col)  # Get changed column
        front.changeCidSignal.emit(sel_cid, header_lbl, new_val)

    @slot_(QtGui.QTableWidgetItem)
    def res_tbl_changed(front, item):
        front.print('res_tbl_changed()')
        row, col = (item.row(), item.column())
        sel_cid  = front.get_restbl_cid(row)  # The changed row's chip id
        new_val  = csv_sanatize(item.text())  # sanatize val for csv
        header_lbl = front.get_restbl_header(col)  # Get changed column
        front.changeCidSignal.emit(sel_cid, header_lbl, new_val)

    #=======================
    # Table Clicked Functions
    #=======================
    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def img_tbl_clicked(front, item):
        row = item.row()
        front.print('img_tbl_clicked(%r)' % (row))
        sel_gx = front.get_imgtbl_gx(row)
        front.selectGxSignal.emit(sel_gx)

    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def chip_tbl_clicked(front, item):
        row, col = (item.row(), item.column())
        front.print('chip_tbl_clicked(%r, %r)' % (row, col))
        sel_cid = front.get_chiptbl_cid(row)
        front.selectCidSignal.emit(sel_cid)

    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def res_tbl_clicked(front, item):
        row, col = (item.row(), item.column())
        front.print('res_tbl_clicked(%r, %r)' % (row, col))
        sel_cid = front.get_restbl_cid(row)
        front.selectResSignal.emit(sel_cid)

    @slot_(QtGui.QTableWidgetItem)
    @clicked
    def name_tbl_clicked(front, item):
        row, col = (item.row(), item.column())
        front.print('name_tbl_clicked(%r, %r)' % (row, col))
        sel_name = front.get_nametbl_name(row)
        front.selectNameSignal.emit(sel_name)

    #=======================
    # Other
    #=======================

    @slot_(int)
    def change_view(front, new_state):
        front.print('change_view()')
        prevBlock = front.ui.tablesTabWidget.blockSignals(True)
        front.ui.tablesTabWidget.blockSignals(prevBlock)

    @slot_(str, str, list)
    def modal_useroption(front, msg, title, options):
        pass

    @slot_(str)
    def gui_write(front, msg_):
        app = front.back.app
        outputEdit = front.ui.outputEdit
        # Write msg to text area
        outputEdit.moveCursor(QtGui.QTextCursor.End)
        # TODO: Find out how to do backspaces in textEdit
        msg = str(msg_)
        if msg.find('\b') != -1:
            msg = msg.replace('\b', '') + '\n'
        outputEdit.insertPlainText(msg)
        if app is not None:
            app.processEvents()

    @slot_()
    def gui_flush(front):
        app = front.back.app
        if app is not None:
            app.processEvents()
        #front.ui.outputEdit.moveCursor(QtGui.QTextCursor.End)
        #front.ui.outputEdit.insertPlainText(msg)
